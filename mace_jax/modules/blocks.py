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
        self,
        node_attrs: e3nn.IrrepsArray,  # [n_nodes, irreps]
    ) -> e3nn.IrrepsArray:
        return self.linear(node_attrs)


class LinearReadoutBlock(hk.Module):
    def __call__(
        self, x: e3nn.IrrepsArray
    ) -> e3nn.IrrepsArray:  # [n_nodes, irreps]  # [..., ]
        return e3nn.Linear(e3nn.Irreps("0e"))(x)  # [n_nodes, 1]


class NonLinearReadoutBlock(hk.Module):
    def __init__(self, MLP_irreps: e3nn.Irreps, gate: Optional[Callable]):
        super().__init__()
        assert len(MLP_irreps) == 1
        self.hidden_irreps = MLP_irreps
        self.gate = gate

    def __call__(
        self, x: e3nn.IrrepsArray
    ) -> e3nn.IrrepsArray:  # [n_nodes, irreps]  # [..., ]
        x = e3nn.Linear(self.hidden_irreps)(x)
        x = e3nn.scalar_activation(x, [self.gate])
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
        self,
        edge_lengths: e3nn.IrrepsArray,  # [n_edges, 1]
    ):
        bessel = self.bessel_fn(edge_lengths)  # [n_edges, n_basis]
        cutoff = self.cutoff_fn(edge_lengths)  # [n_edges, 1]
        return bessel * cutoff  # [n_edges, n_basis]


class EquivariantProductBasisBlock(hk.Module):
    def __init__(
        self,
        target_irreps: e3nn.Irreps,
        correlation: int,
    ) -> None:
        super().__init__()
        target_irreps = e3nn.Irreps(target_irreps)
        self.symmetric_contractions = SymmetricContraction(
            keep_irrep_out={ir for _, ir in target_irreps},
            correlation=correlation,
        )

    def __call__(
        self,
        node_feats: e3nn.IrrepsArray,
        node_attrs: e3nn.IrrepsArray,
        sc: Optional[e3nn.IrrepsArray],
    ) -> e3nn.IrrepsArray:
        assert {ir for _, ir in node_attrs.irreps} == {e3nn.Irrep("0e")}
        node_feats = self.symmetric_contractions(
            node_feats.factor_mul_to_last_axis(), node_attrs.array
        ).repeat_mul_by_last_axis()
        node_feats = e3nn.Linear(node_feats.irreps)(node_feats)
        return node_feats + sc if sc is not None else node_feats


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
        assert {ir for _, ir in edge_feats.irreps} == {
            e3nn.Irrep("0e")
        }, edge_feats.irreps
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
        message = e3nn.IrrepsArray.zeros(
            mji.irreps, (node_feats.shape[0],)
        )  # [n_nodes, irreps]
        message = message.at[receivers].add(mji)
        # Linear
        message = e3nn.Linear(self.target_irreps)(message) / self.avg_num_neighbors

        return (
            message,
            sc,
        )  # [n_nodes, channels, (lmax + 1)**2]


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
