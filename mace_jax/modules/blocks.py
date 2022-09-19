from abc import ABC, abstractmethod
from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np

import e3nn_jax as e3nn
import haiku as hk
import jax
import jax.numpy as jnp


from .irreps_tools import tp_out_irreps_with_instructions
from .radial import BesselBasis, PolynomialCutoff
from .symmetric_contraction import SymmetricContraction


class LinearNodeEmbeddingBlock(hk.Module):
    def __init__(self, irreps_out: e3nn.Irreps):
        super().__init__()
        self.linear = e3nn.Linear(irreps_out=irreps_out)

    def __call__(
        self, node_attrs: e3nn.IrrepsArray,  # [n_nodes, irreps]
    ) -> e3nn.IrrepsArray:
        return self.linear(node_attrs)


class LinearReadoutBlock(hk.Module):
    def __call__(
        self, x: e3nn.IrrepsArray
    ) -> e3nn.IrrepsArray:  # [n_nodes, irreps]  # [..., ]
        return e3nn.Linear(e3nn.Irreps("0e"))(x)  # [n_nodes, 1]


class NonLinearReadoutBlock(hk.nn.Module):
    def __init__(self, MLP_irreps: e3nn.Irreps, gate: Optional[Callable]):
        super().__init__()
        assert len(MLP_irreps) == 1
        self.hidden_irreps = MLP_irreps
        self.gate = gate

    def __call__(
        self, x: e3nn.IrrepsArray
    ) -> e3nn.IrrepsArray:  # [n_nodes, irreps]  # [..., ]
        x = e3nn.Linear(self.hidden_irreps)(x)
        x = e3nn.scalar_activation(x)([self.gate])
        return e3nn.Linear(e3nn.Irreps("0e"))(x)  # [n_nodes, 1]


class AtomicEnergiesBlock(hk.Module):
    atomic_energies: jnp.ndarray

    def __init__(self, atomic_energies: Union[np.ndarray, jnp.ndarray]):
        super().__init__()
        assert len(atomic_energies.shape) == 1
        self.atomic_energies = atomic_energies  # [n_elements, ]

    def __call__(
        self, x: e3nn.IrrepsArray  # one-hot of elements [..., n_elements]
    ) -> e3nn.IrrepsArray:  # [..., ]
        return x @ self.atomic_energies

    def __repr__(self):
        formatted_energies = ", ".join([f"{x:.4f}" for x in self.atomic_energies])
        return f"{self.__class__.__name__}(energies=[{formatted_energies}])"


class RadialEmbeddingBlock(hk.Module):
    def __init__(self, r_max: float, num_bessel: int, num_polynomial_cutoff: int):
        super().__init__()
        self.bessel_fn = BesselBasis(r_max=r_max, num_basis=num_bessel)
        self.cutoff_fn = PolynomialCutoff(r_max=r_max, p=num_polynomial_cutoff)

    def __call__(
        self, edge_lengths: e3nn.IrrepsArray,  # [n_edges, 1]
    ):
        bessel = self.bessel_fn(edge_lengths)  # [n_edges, n_basis]
        cutoff = self.cutoff_fn(edge_lengths)  # [n_edges, 1]
        return bessel * cutoff  # [n_edges, n_basis]


class EquivariantProductBasisBlock(hk.Module):
    def __init__(self, target_irreps: e3nn.Irreps, correlation: int,) -> None:
        super().__init__()

        self.symmetric_contractions = SymmetricContraction(
            keep_irrep_out=target_irreps, correlation=correlation,
        )

    def __call__(
        self,
        node_feats: e3nn.IrrepsArray,
        node_attrs: e3nn.IrrepsArray,
        sc: Optional[e3nn.IrrepsArray],
    ) -> e3nn.IrrepsArray:
        node_feats = self.symmetric_contractions(
            node_feats.factor_mul_to_last_axis(), node_attrs
        )
        node_feats = e3nn.Linear(node_feats.irreps)(node_feats)
        return node_feats + sc if sc is not None else node_feats


class InteractionBlock(ABC, hk.Module):
    def __init__(
        self,
        target_irreps: e3nn.Irreps,
        hidden_irreps: e3nn.Irreps,
        avg_num_neighbors: float,
    ) -> None:
        super().__init__()
        self.target_irreps = target_irreps
        self.hidden_irreps = hidden_irreps
        self.avg_num_neighbors = avg_num_neighbors

        self._setup()

    @abstractmethod
    def _setup(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def __call__(
        self,
        node_attrs: e3nn.IrrepsArray,  # [n_nodes, irreps]
        node_feats: e3nn.IrrepsArray,
        edge_attrs: e3nn.IrrepsArray,
        edge_feats: e3nn.IrrepsArray,
        senders: jnp.ndarray,
        receivers: jnp.ndarray,
    ) -> Tuple[e3nn.IrrepsArray, Optional[e3nn.IrrepsArray]]:
        raise NotImplementedError


class AgnosticResidualInteractionBlock(InteractionBlock):
    def __call__(
        self,
        node_attrs: e3nn.IrrepsArray,
        node_feats: e3nn.IrrepsArray,
        edge_attrs: e3nn.IrrepsArray,
        edge_feats: e3nn.IrrepsArray,
        senders: jnp.ndarray,
        receivers: jnp.ndarray,
    ) -> Tuple[e3nn.IrrepsArray, e3nn.IrrepsArray]:
        sc = e3nn.Linear(self.hidden_irreps)(
            e3nn.tensor_product(node_feats, node_attrs)
        )  # [n_nodes, hidden_irreps]
        # First linear
        node_feats = e3nn.Linear(node_feats.irreps)(node_feats)
        # Tensor product
        irreps_mid, instructions = tp_out_irreps_with_instructions(
            node_feats.irreps, edge_attrs.irreps, self.target_irreps,
        )
        self.conv_tp = e3nn.FunctionalTensorProduct(
            node_feats.irreps, edge_attrs.irreps, irreps_mid, instructions=instructions,
        )
        # Learnable Radial
        tp_weights = self.e3nn.MultiLayerPerceptron(
            3 * [64] + [self.conv_tp.weight_numel], jax.nn.silu
        )(
            edge_feats
        )  # [n_edges, n_basis, 1]

        # Convolution weights
        mji = self.conv_tp(
            node_feats[senders], edge_attrs, tp_weights
        )  # [n_edges, irreps]

        # Scatter sum
        message = jnp.zeros(
            (node_feats.shape[0], self.target_irreps.dim)
        )  # [n_nodes, irreps]
        message = e3nn.IrrepsArray(
            self.target_irreps, message.at[receivers].add(mji.array)
        )
        # Linear
        self.irreps_out = self.target_irreps
        message = e3nn.Linear(self.target_irreps)(message) / self.avg_num_neighbors

        return (
            message,
            sc,
        )  # [n_nodes, channels, (lmax + 1)**2]


class AgnosticInteractionBlock(InteractionBlock):
    def __call__(
        self,
        node_attrs: e3nn.IrrepsArray,
        node_feats: e3nn.IrrepsArray,
        edge_attrs: e3nn.IrrepsArray,
        edge_feats: e3nn.IrrepsArray,
        senders: jnp.ndarray,
        receivers: jnp.ndarray,
    ) -> Tuple[e3nn.IrrepsArray, Optional[e3nn.IrrepsArray]]:
        interaction_block = AgnosticResidualInteractionBlock(
            node_attrs=node_attrs,
            node_feats=node_feats,
            edge_attrs=edge_attrs,
            edge_feats=edge_feats,
            senders=senders,
            receivers=receivers,
            target_irreps=self.target_irreps,
            hidden_irreps=self.hidden_irreps,
            avg_num_neighbors=self.avg_num_neighbors,
        )
        message, _ = interaction_block(
            node_attrs, node_feats, edge_attrs, edge_feats, senders, receivers
        )

        # Selector TensorProduct
        message = e3nn.Linear(self.target_irreps)(
            e3nn.tensor_product(message, node_attrs)
        )
        return (
            message,
            None,
        )  # [n_nodes, channels, (lmax + 1)**2]


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
