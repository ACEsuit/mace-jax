from abc import ABC, abstractmethod
from typing import Callable, Optional, Tuple, Union

import e3nn_jax as e3nn
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

from .irreps_tools import tp_out_irreps_with_instructions
from .symmetric_contraction import SymmetricContraction


class LinearNodeEmbeddingBlock(hk.Module):
    def __init__(self, irreps_out: e3nn.Irreps):
        super().__init__()
        self.linear = e3nn.Linear(irreps_out=irreps_out)

    def __call__(
        self,
        node_attrs: e3nn.IrrepsArray,  # [n_nodes, irreps]
    ) -> e3nn.IrrepsArray:
        return self.linear(node_attrs)


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


class AtomicEnergiesBlock(hk.Module):
    atomic_energies: jnp.ndarray

    def __init__(self, atomic_energies: Union[np.ndarray, jnp.ndarray]):
        super().__init__()
        assert atomic_energies.ndim == 1
        self.atomic_energies = atomic_energies  # [n_elements, ]

    def __call__(
        self, x: jnp.ndarray  # one-hot of elements [..., n_elements]
    ) -> jnp.ndarray:  # [..., ]
        return x @ self.atomic_energies

    def __repr__(self):
        formatted_energies = ", ".join([f"{x:.4f}" for x in self.atomic_energies])
        return f"{self.__class__.__name__}(energies=[{formatted_energies}])"


class RadialEmbeddingBlock:
    def __init__(
        self,
        *,
        r_max: float,
        num_bessel: int,
        num_deriv_in_zero: Optional[int],
        num_deriv_in_one: Optional[int],
    ):
        self.num_bessel = num_bessel
        self.r_max = r_max
        self.num_deriv_in_zero = num_deriv_in_zero
        self.num_deriv_in_one = num_deriv_in_one

    def __call__(
        self,
        edge_lengths: jnp.ndarray,  # [n_edges, 1]
    ) -> jnp.ndarray:  # [n_edges, num_bessel]
        bessel = e3nn.bessel(
            edge_lengths[..., 0], self.num_bessel, x_max=self.r_max
        )  # [n_edges, num_bessel]

        if self.num_deriv_in_zero is None and self.num_deriv_in_one is None:
            cutoff = e3nn.soft_envelope(edge_lengths, x_max=self.r_max)  # [n_edges, 1]
        else:
            cutoff = e3nn.poly_envelope(
                self.num_deriv_in_zero, self.num_deriv_in_one, x_max=self.r_max
            )(
                edge_lengths
            )  # [n_edges, 1]
        return bessel * cutoff  # [n_edges, num_bessel]


class EquivariantProductBasisBlock(hk.Module):
    def __init__(
        self,
        target_irreps: e3nn.Irreps,
        correlation: int,
    ) -> None:
        super().__init__()
        self.target_irreps = e3nn.Irreps(target_irreps)
        self.symmetric_contractions = SymmetricContraction(
            keep_irrep_out={ir for _, ir in self.target_irreps},
            correlation=correlation,
        )

    def __call__(
        self,
        node_feats: e3nn.IrrepsArray,  # [n_nodes, irreps] with identical multiplicities
        node_attrs: e3nn.IrrepsArray,  # [n_nodes, irreps] with only scalars
    ) -> e3nn.IrrepsArray:
        assert node_attrs.irreps.is_scalar()
        node_feats = self.symmetric_contractions(
            node_feats.factor_mul_to_last_axis(), node_attrs.array
        ).repeat_mul_by_last_axis()
        return e3nn.Linear(self.target_irreps)(node_feats)


class InteractionBlock(ABC, hk.Module):
    def __init__(
        self,
        *,
        target_irreps: e3nn.Irreps,
        hidden_irreps: e3nn.Irreps,
        avg_num_neighbors: float,
    ) -> None:
        super().__init__()
        self.target_irreps = target_irreps
        self.hidden_irreps = hidden_irreps
        self.avg_num_neighbors = avg_num_neighbors

    @abstractmethod
    def __call__(
        self,
        node_attrs: e3nn.IrrepsArray,  # [n_nodes, irreps]
        node_feats: e3nn.IrrepsArray,  # [n_nodes, irreps]
        edge_attrs: e3nn.IrrepsArray,  # [n_edges, irreps]
        edge_feats: e3nn.IrrepsArray,  # [n_edges, irreps]
        senders: jnp.ndarray,  # [n_edges, ]
        receivers: jnp.ndarray,  # [n_edges, ]
    ) -> Tuple[e3nn.IrrepsArray, Optional[e3nn.IrrepsArray]]:
        raise NotImplementedError


class AgnosticResidualInteractionBlock(InteractionBlock):
    def __call__(
        self,
        node_attrs: e3nn.IrrepsArray,  # [n_nodes, irreps]
        node_feats: e3nn.IrrepsArray,  # [n_nodes, irreps]
        edge_attrs: e3nn.IrrepsArray,  # [n_edges, irreps]
        edge_feats: e3nn.IrrepsArray,  # [n_edges, irreps]
        senders: jnp.ndarray,  # [n_edges, ]
        receivers: jnp.ndarray,  # [n_edges, ]
    ) -> Tuple[e3nn.IrrepsArray, e3nn.IrrepsArray]:
        sc = e3nn.Linear(self.hidden_irreps)(
            e3nn.tensor_product(node_feats, node_attrs)
        )  # [n_nodes, hidden_irreps]
        # First linear
        node_feats = e3nn.Linear(node_feats.irreps)(node_feats)
        # Tensor product
        irreps_mid, instructions = tp_out_irreps_with_instructions(
            node_feats.irreps,
            edge_attrs.irreps,
            self.target_irreps,
        )
        # assert irreps_mid.simplify() == self.target_irreps.simplify()
        conv_tp = e3nn.FunctionalTensorProduct(
            node_feats.irreps,
            edge_attrs.irreps,
            irreps_mid,
            instructions=instructions,
        )
        del irreps_mid
        weight_numel = sum(
            np.prod(i.path_shape) for i in conv_tp.instructions if i.has_weight
        )

        # Learnable Radial
        assert edge_feats.irreps.is_scalar()
        tp_weights = e3nn.MultiLayerPerceptron(3 * [64] + [weight_numel], jax.nn.silu)(
            edge_feats.array
        )  # [n_edges, weight_numel]
        assert tp_weights.shape == (edge_feats.shape[0], weight_numel)

        # Convolution weights
        mji = jax.vmap(conv_tp.left_right)(
            tp_weights, node_feats[senders], edge_attrs
        )  # [n_edges, irreps]
        mji = mji.simplify()

        # Scatter sum
        message = e3nn.IrrepsArray.zeros(mji.irreps, (node_feats.shape[0],))
        message = message.at[receivers].add(mji) / jnp.sqrt(
            self.avg_num_neighbors
        )  # [n_nodes, irreps]

        # Linear
        message = e3nn.Linear(self.target_irreps)(message)

        return (
            message,  # [n_nodes, target_irreps]
            sc,  # [n_nodes, hidden_irreps]
        )


class AgnosticInteractionBlock(InteractionBlock):
    def __call__(
        self,
        node_attrs: e3nn.IrrepsArray,  # [n_nodes, irreps]
        node_feats: e3nn.IrrepsArray,  # [n_nodes, irreps]
        edge_attrs: e3nn.IrrepsArray,  # [n_edges, irreps]
        edge_feats: e3nn.IrrepsArray,  # [n_edges, irreps]
        senders: jnp.ndarray,  # [n_edges, ]
        receivers: jnp.ndarray,  # [n_edges, ]
    ) -> Tuple[e3nn.IrrepsArray, Optional[e3nn.IrrepsArray]]:
        message, _ = AgnosticResidualInteractionBlock(
            target_irreps=self.target_irreps,
            hidden_irreps=self.hidden_irreps,
            avg_num_neighbors=self.avg_num_neighbors,
        )(node_attrs, node_feats, edge_attrs, edge_feats, senders, receivers)

        # Selector TensorProduct
        message = e3nn.Linear(self.target_irreps)(
            e3nn.tensor_product(message, node_attrs)
        )
        return (
            message,  # [n_nodes, target_irreps]
            None,
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
