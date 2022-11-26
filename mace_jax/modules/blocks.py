from typing import Callable, Optional, Tuple

import e3nn_jax as e3nn
import haiku as hk
import jax
import jax.numpy as jnp

from .message_passing import MessagePassingConvolution
from .symmetric_contraction import SymmetricContraction


class LinearNodeEmbeddingBlock(hk.Module):
    def __init__(self, num_species: int, irreps_out: e3nn.Irreps):
        super().__init__()
        self.num_species = num_species
        self.irreps_out = e3nn.Irreps(irreps_out).filter("0e").regroup()

    def __call__(self, node_specie: jnp.ndarray) -> e3nn.IrrepsArray:
        w = hk.get_parameter(
            f"embeddings",
            shape=(self.num_species, self.irreps_out.dim),
            dtype=jnp.float32,
            init=hk.initializers.RandomNormal(),
        )
        return e3nn.IrrepsArray(self.irreps_out, w[node_specie])


class LinearReadoutBlock(hk.Module):
    def __init__(
        self,
        output_irreps: e3nn.Irreps,
    ):
        super().__init__()
        self.output_irreps = output_irreps

    def __call__(self, x: e3nn.IrrepsArray) -> e3nn.IrrepsArray:
        # x = [n_nodes, irreps]
        return e3nn.Linear(self.output_irreps)(x)  # [n_nodes, output_irreps]


class NonLinearReadoutBlock(hk.Module):
    def __init__(
        self,
        hidden_irreps: e3nn.Irreps,
        output_irreps: e3nn.Irreps,
        *,
        activation: Optional[Callable] = None,
        gate: Optional[Callable] = None,
    ):
        super().__init__()
        self.hidden_irreps = hidden_irreps
        self.output_irreps = output_irreps
        self.activation = activation
        self.gate = gate

    def __call__(self, x: e3nn.IrrepsArray) -> e3nn.IrrepsArray:
        # x = [n_nodes, irreps]
        num_vectors = (
            self.hidden_irreps.num_irreps
            - self.hidden_irreps.filter(["0e", "0o"]).num_irreps
        )  # Multiplicity of (l > 0) irreps
        x = e3nn.Linear(
            (self.hidden_irreps + e3nn.Irreps(f"{num_vectors}x0e")).simplify()
        )(x)
        x = e3nn.gate(x, even_act=self.activation, even_gate_act=self.gate)
        return e3nn.Linear(self.output_irreps)(x)  # [n_nodes, output_irreps]


class RadialEmbeddingBlock:
    def __init__(
        self,
        *,
        r_max: float,
        avg_r_min: Optional[float] = None,
        basis_functions: Callable[[jnp.ndarray], jnp.ndarray],
        envelope_function: Callable[[jnp.ndarray], jnp.ndarray],
    ):
        self.r_max = r_max
        self.avg_r_min = avg_r_min
        self.basis_functions = basis_functions
        self.envelope_function = envelope_function

    def __call__(
        self,
        edge_lengths: jnp.ndarray,  # [n_edges]
    ) -> e3nn.IrrepsArray:  # [n_edges, num_basis]
        def func(lengths):
            basis = self.basis_functions(lengths, self.r_max)  # [n_edges, num_basis]
            cutoff = self.envelope_function(lengths, self.r_max)  # [n_edges]
            return basis * cutoff[:, None]  # [n_edges, num_basis]

        with jax.ensure_compile_time_eval():
            if self.avg_r_min is None:
                factor = 1.0
            else:
                samples = jnp.linspace(self.avg_r_min, self.r_max, 1000)
                factor = jnp.mean(func(samples) ** 2) ** -0.5

        embedding = factor * func(edge_lengths)  # [n_edges, num_basis]
        return e3nn.IrrepsArray(f"{embedding.shape[-1]}x0e", embedding)


class EquivariantProductBasisBlock(hk.Module):
    def __init__(
        self,
        target_irreps: e3nn.Irreps,
        correlation: int,
        num_species: int,
        max_poly_order: Optional[int] = None,
        input_poly_order: int = 0,
        symmetric_tensor_product_basis: bool = True,
        off_diagonal: bool = False,
    ) -> None:
        super().__init__()
        self.target_irreps = e3nn.Irreps(target_irreps)
        self.symmetric_contractions = SymmetricContraction(
            keep_irrep_out={ir for _, ir in self.target_irreps},
            correlation=correlation,
            max_poly_order=max_poly_order,
            input_poly_order=input_poly_order,
            num_species=num_species,
            gradient_normalization="element",  # NOTE: This is to copy mace-torch
            symmetric_tensor_product_basis=symmetric_tensor_product_basis,
            off_diagonal=off_diagonal,
        )

    def __call__(
        self,
        node_feats: e3nn.IrrepsArray,  # [n_nodes, feature * irreps]
        node_specie: jnp.ndarray,  # [n_nodes, ] int
    ) -> e3nn.IrrepsArray:
        node_feats = node_feats.mul_to_axis().remove_nones()
        node_feats = self.symmetric_contractions(node_feats, node_specie)
        node_feats = node_feats.axis_to_mul()
        return e3nn.Linear(self.target_irreps)(node_feats)


class InteractionBlock(hk.Module):
    def __init__(
        self,
        *,
        target_irreps: e3nn.Irreps,
        avg_num_neighbors: float,
        activation: Callable,
    ) -> None:
        super().__init__()
        self.target_irreps = target_irreps
        self.avg_num_neighbors = avg_num_neighbors
        self.activation = activation

    def __call__(
        self,
        node_feats: e3nn.IrrepsArray,  # [n_nodes, irreps]
        edge_attrs: e3nn.IrrepsArray,  # [n_edges, irreps]
        senders: jnp.ndarray,  # [n_edges, ]
        receivers: jnp.ndarray,  # [n_edges, ]
    ) -> Tuple[e3nn.IrrepsArray, e3nn.IrrepsArray]:
        assert node_feats.ndim == 2

        node_feats = e3nn.Linear(node_feats.irreps, name="linear_up")(node_feats)

        node_feats = MessagePassingConvolution(
            self.avg_num_neighbors, self.target_irreps, self.activation
        )(node_feats, edge_attrs, senders, receivers)

        node_feats = e3nn.Linear(self.target_irreps, name="linear_down")(node_feats)

        assert node_feats.ndim == 2
        return node_feats  # [n_nodes, target_irreps]


class ScaleShiftBlock(hk.Module):
    def __init__(self, scale: float, shift: float):
        super().__init__()
        self.scale = scale
        self.shift = shift

    def __call__(self, x: e3nn.IrrepsArray) -> e3nn.IrrepsArray:
        return self.scale * x + self.shift

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(scale={self.scale:.6f}, shift={self.shift:.6f})"
        )
