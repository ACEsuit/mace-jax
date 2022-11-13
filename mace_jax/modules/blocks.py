from typing import Callable, Optional, Tuple

import e3nn_jax as e3nn
import haiku as hk
import jax
import jax.numpy as jnp

from .message_passing import MessagePassingConvolution
from .symmetric_contraction import SymmetricContraction


class LinearNodeEmbeddingBlock(hk.Module):
    def __init__(self, num_species: int, num_features: int, irreps_out: e3nn.Irreps):
        super().__init__()
        self.num_species = num_species
        self.num_features = num_features
        self.irreps_out = e3nn.Irreps(irreps_out)

    def __call__(
        self,
        node_specie: jnp.ndarray,  # [n_nodes, ]
    ) -> e3nn.IrrepsArray:
        new_list = []

        for i, (mul, ir) in enumerate(self.irreps_out):
            if ir == "0e":
                w = hk.get_parameter(
                    f"embeddings_{i}",
                    shape=(self.num_species, self.num_features, mul, ir.dim),
                    dtype=jnp.float32,
                    init=hk.initializers.RandomNormal(),
                )
                new_list.append(w[node_specie])  # [n_nodes, num_features, mul, ir.dim]
            else:
                new_list.append(None)

        return e3nn.IrrepsArray.from_list(
            self.irreps_out, new_list, (node_specie.shape[0], self.num_features)
        )


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
        avg_r_min: float = 0.0,
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
    ) -> jnp.ndarray:  # [n_edges, num_basis]
        def func(lengths):
            basis = self.basis_functions(lengths / self.r_max)  # [n_edges, num_basis]
            cutoff = self.envelope_function(lengths / self.r_max)  # [n_edges]
            return basis * cutoff[:, None]  # [n_edges, num_basis]

        with jax.ensure_compile_time_eval():
            samples = jnp.linspace(self.avg_r_min, self.r_max, 1000)
            factor = jnp.mean(func(samples) ** 2) ** -0.5

        return factor * func(edge_lengths)  # [n_edges, num_basis]


class EquivariantProductBasisBlock(hk.Module):
    def __init__(
        self,
        num_features: int,
        target_irreps: e3nn.Irreps,
        correlation: int,
        num_species: int,
        max_poly_order: Optional[int] = None,
        input_poly_order: int = 0,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.target_irreps = e3nn.Irreps(target_irreps)
        self.symmetric_contractions = SymmetricContraction(
            keep_irrep_out={ir for _, ir in self.target_irreps},
            correlation=correlation,
            max_poly_order=max_poly_order,
            input_poly_order=input_poly_order,
            num_species=num_species,
        )

    def __call__(
        self,
        node_feats: e3nn.IrrepsArray,  # [n_nodes, feature, irreps]
        node_specie: jnp.ndarray,  # [n_nodes, ] int
    ) -> e3nn.IrrepsArray:
        node_feats = node_feats.remove_nones()
        node_feats = self.symmetric_contractions(node_feats, node_specie)
        return e3nn.Linear(self.target_irreps, self.num_features)(node_feats)


class AgnosticResidualInteractionBlock(hk.Module):
    def __init__(
        self,
        *,
        num_species: int,
        num_features: int,
        target_irreps: e3nn.Irreps,
        hidden_irreps: e3nn.Irreps,
        avg_num_neighbors: float,
        activation: Callable,
    ) -> None:
        super().__init__()
        self.num_species = num_species
        self.num_features = num_features
        self.target_irreps = target_irreps
        self.hidden_irreps = hidden_irreps
        self.avg_num_neighbors = avg_num_neighbors
        self.activation = activation

    def __call__(
        self,
        node_specie: e3nn.IrrepsArray,  # [n_nodes] int
        node_feats: e3nn.IrrepsArray,  # [n_nodes, feature, irreps]
        edge_attrs: e3nn.IrrepsArray,  # [n_edges, irreps]
        edge_feats: e3nn.IrrepsArray,  # [n_edges, irreps]
        senders: jnp.ndarray,  # [n_edges, ]
        receivers: jnp.ndarray,  # [n_edges, ]
    ) -> Tuple[e3nn.IrrepsArray, e3nn.IrrepsArray]:
        assert node_specie.ndim == 1
        assert node_feats.ndim == 3

        sc = e3nn.Linear(
            self.hidden_irreps,
            self.num_features,
            num_weights=self.num_species,
            name="linear_sc",
        )(
            node_specie, node_feats
        )  # [n_nodes, feature, hidden_irreps]

        node_feats = e3nn.Linear(
            node_feats.irreps, self.num_features, name="linear_premp"
        )(node_feats)

        m = MessagePassingConvolution(
            self.avg_num_neighbors, self.target_irreps, self.activation
        )
        node_feats = hk.vmap(
            lambda x: m(x, edge_attrs, edge_feats, senders, receivers),
            in_axes=1,
            out_axes=1,
            split_rng=False,
        )(node_feats)

        node_feats = e3nn.Linear(
            self.target_irreps, self.num_features, name="linear_postmp"
        )(node_feats)

        assert node_feats.ndim == 3
        return (
            node_feats,  # [n_nodes, feature, target_irreps]
            sc,  # [n_nodes, feature, hidden_irreps]
        )


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
