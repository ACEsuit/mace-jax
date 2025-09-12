import abc
from typing import Callable, List, Optional, Tuple

import e3nn_jax as e3nn
import haiku as hk
import jax
import jax.numpy as jnp
from e3nn_jax import Irreps, IrrepsArray

from mace_jax.modules.wrapper_ops import (
    CuEquivarianceConfig,
    FullyConnectedTensorProduct,
    Linear,
    OEQConfig,
    TensorProduct,
)
from mace_jax.tools.scatter import scatter_sum

from .irreps_tools import reshape_irreps, tp_out_irreps_with_instructions
from .message_passing import MessagePassingConvolution
from .symmetric_contraction import SymmetricContraction


class LinearNodeEmbeddingBlock(hk.Module):
    def __init__(self, num_species: int, irreps_out: Irreps):
        super().__init__()
        self.num_species = num_species
        self.irreps_out = Irreps(irreps_out).filter("0e").regroup()

    def __call__(self, node_specie: jnp.ndarray) -> IrrepsArray:
        w = hk.get_parameter(
            "embeddings",
            shape=(self.num_species, self.irreps_out.dim),
            dtype=jnp.float32,
            init=hk.initializers.RandomNormal(),
        )
        return IrrepsArray(self.irreps_out, w[node_specie])


class LinearReadoutBlock(hk.Module):
    def __init__(
        self,
        output_irreps: Irreps,
    ):
        super().__init__()
        self.output_irreps = output_irreps

    def __call__(self, x: IrrepsArray) -> IrrepsArray:
        # x = [n_nodes, irreps]
        return e3nn.haiku.Linear(self.output_irreps)(x)  # [n_nodes, output_irreps]


class NonLinearReadoutBlock(hk.Module):
    def __init__(
        self,
        hidden_irreps: Irreps,
        output_irreps: Irreps,
        *,
        activation: Optional[Callable] = None,
        gate: Optional[Callable] = None,
    ):
        super().__init__()
        self.hidden_irreps = hidden_irreps
        self.output_irreps = output_irreps
        self.activation = activation
        self.gate = gate

    def __call__(self, x: IrrepsArray) -> IrrepsArray:
        # x = [n_nodes, irreps]
        num_vectors = self.hidden_irreps.filter(
            drop=["0e", "0o"]
        ).num_irreps  # Multiplicity of (l > 0) irreps
        x = e3nn.haiku.Linear(
            (self.hidden_irreps + Irreps(f"{num_vectors}x0e")).simplify()
        )(x)
        x = e3nn.gate(x, even_act=self.activation, even_gate_act=self.gate)
        return e3nn.haiku.Linear(self.output_irreps)(x)  # [n_nodes, output_irreps]


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
    ) -> IrrepsArray:  # [n_edges, num_basis]
        def func(lengths):
            basis = self.basis_functions(lengths, self.r_max)  # [n_edges, num_basis]
            cutoff = self.envelope_function(lengths, self.r_max)  # [n_edges]
            return basis * cutoff[:, None]  # [n_edges, num_basis]

        with jax.ensure_compile_time_eval():
            if self.avg_r_min is None:
                factor = 1.0
            else:
                samples = jnp.linspace(
                    self.avg_r_min, self.r_max, 1000, dtype=jnp.float64
                )
                factor = jnp.mean(func(samples) ** 2).item() ** -0.5

        embedding = factor * jnp.where(
            (edge_lengths == 0.0)[:, None], 0.0, func(edge_lengths)
        )  # [n_edges, num_basis]
        return IrrepsArray(f"{embedding.shape[-1]}x0e", embedding)


class EquivariantProductBasisBlock(hk.Module):
    def __init__(
        self,
        target_irreps: Irreps,
        correlation: int,
        num_species: int,
        symmetric_tensor_product_basis: bool = True,
        off_diagonal: bool = False,
    ) -> None:
        super().__init__()
        self.target_irreps = Irreps(target_irreps)
        self.symmetric_contractions = SymmetricContraction(
            keep_irrep_out={ir for _, ir in self.target_irreps},
            correlation=correlation,
            num_species=num_species,
            gradient_normalization="element",  # NOTE: This is to copy mace-torch
            symmetric_tensor_product_basis=symmetric_tensor_product_basis,
            off_diagonal=off_diagonal,
        )

    def __call__(
        self,
        node_feats: IrrepsArray,  # [n_nodes, feature * irreps]
        node_specie: jnp.ndarray,  # [n_nodes, ] int
    ) -> IrrepsArray:
        node_feats = node_feats.mul_to_axis().remove_nones()
        node_feats = self.symmetric_contractions(node_feats, node_specie)
        node_feats = node_feats.axis_to_mul()
        return e3nn.haiku.Linear(self.target_irreps)(node_feats)


class InteractionBlock(hk.Module):
    def __init__(
        self,
        *,
        target_irreps: Irreps,
        avg_num_neighbors: float,
        max_ell: int,
        activation: Callable,
    ) -> None:
        super().__init__()
        self.target_irreps = target_irreps
        self.avg_num_neighbors = avg_num_neighbors
        self.max_ell = max_ell
        self.activation = activation

    def __call__(
        self,
        vectors: IrrepsArray,  # [n_edges, 3]
        node_feats: IrrepsArray,  # [n_nodes, irreps]
        radial_embedding: jnp.ndarray,  # [n_edges, radial_embedding_dim]
        senders: jnp.ndarray,  # [n_edges, ]
        receivers: jnp.ndarray,  # [n_edges, ]
    ) -> Tuple[IrrepsArray, IrrepsArray]:
        assert node_feats.ndim == 2
        assert vectors.ndim == 2
        assert radial_embedding.ndim == 2

        node_feats = e3nn.haiku.Linear(node_feats.irreps, name="linear_up")(node_feats)

        node_feats = MessagePassingConvolution(
            self.avg_num_neighbors, self.target_irreps, self.max_ell, self.activation
        )(vectors, node_feats, radial_embedding, senders, receivers)

        node_feats = e3nn.haiku.Linear(self.target_irreps, name="linear_down")(
            node_feats
        )

        assert node_feats.ndim == 2
        return node_feats  # [n_nodes, target_irreps]


class InteractionBlock(hk.Module, metaclass=abc.ABCMeta):
    """
    Abstract base class for interaction blocks in equivariant GNNs.

    Subclasses must implement:
        - _setup(self): module initialization
        - __call__(...): forward pass
    """

    def __init__(
        self,
        node_attrs_irreps: Irreps,
        node_feats_irreps: Irreps,
        edge_attrs_irreps: Irreps,
        edge_feats_irreps: Irreps,
        target_irreps: Irreps,
        hidden_irreps: Irreps,
        avg_num_neighbors: float,
        edge_irreps: Optional[Irreps] = None,
        radial_MLP: Optional[List[int]] = None,
        cueq_config: Optional["CuEquivarianceConfig"] = None,
        oeq_config: Optional["OEQConfig"] = None,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self.node_attrs_irreps = Irreps(node_attrs_irreps)
        self.node_feats_irreps = Irreps(node_feats_irreps)
        self.edge_attrs_irreps = Irreps(edge_attrs_irreps)
        self.edge_feats_irreps = Irreps(edge_feats_irreps)
        self.target_irreps = Irreps(target_irreps)
        self.hidden_irreps = Irreps(hidden_irreps)
        self.avg_num_neighbors = avg_num_neighbors

        if radial_MLP is None:
            radial_MLP = [64, 64, 64]
        if edge_irreps is None:
            edge_irreps = self.node_feats_irreps

        self.radial_MLP = radial_MLP
        self.edge_irreps = Irreps(edge_irreps)
        self.cueq_config = cueq_config
        self.oeq_config = oeq_config

        # Handle conv_fusion flag
        self.conv_fusion = None
        if self.oeq_config and getattr(self.oeq_config, "conv_fusion", None):
            self.conv_fusion = self.oeq_config.conv_fusion
        if self.cueq_config and getattr(self.cueq_config, "conv_fusion", None):
            self.conv_fusion = self.cueq_config.conv_fusion

        # Call subclass-defined setup
        self._setup()

    @abc.abstractmethod
    def _setup(self) -> None:
        """Subclasses implement module setup here."""
        raise NotImplementedError

    def truncate_ghosts(
        self, tensor: jnp.ndarray, n_real: Optional[int] = None
    ) -> jnp.ndarray:
        """Truncate to real atoms (remove ghost atoms)."""
        return tensor[:n_real] if n_real is not None else tensor

    @abc.abstractmethod
    def __call__(
        self,
        node_attrs: jnp.ndarray,
        node_feats: jnp.ndarray,
        edge_attrs: jnp.ndarray,
        edge_feats: jnp.ndarray,
        edge_index: jnp.ndarray,
    ) -> jnp.ndarray:
        """Forward pass (subclasses implement)."""
        raise NotImplementedError


class RealAgnosticInteractionBlock(InteractionBlock):
    def _setup(self) -> None:
        # First linear
        self.linear_up = Linear(
            self.node_feats_irreps,
            self.edge_irreps,
            internal_weights=True,
            shared_weights=True,
            cueq_config=self.cueq_config,
        )

        # TensorProduct
        irreps_mid, instructions = tp_out_irreps_with_instructions(
            self.edge_irreps,
            self.edge_attrs_irreps,
            self.target_irreps,
        )
        self.conv_tp = TensorProduct(
            self.edge_irreps,
            self.edge_attrs_irreps,
            irreps_mid,
            instructions=instructions,
            shared_weights=False,
            internal_weights=False,
            cueq_config=self.cueq_config,
            oeq_config=self.oeq_config,
        )

        # Convolution weights MLP
        self.conv_tp_weights = hk.nets.MLP(
            output_sizes=[*self.radial_MLP, self.conv_tp.weight_numel],
            activation=jax.nn.silu,
            activate_final=False,
            name="conv_tp_weights",
        )

        # Linear
        self.irreps_out = self.target_irreps
        self.linear = Linear(
            irreps_mid,
            self.irreps_out,
            internal_weights=True,
            shared_weights=True,
            cueq_config=self.cueq_config,
        )

        # Selector TensorProduct
        self.skip_tp = FullyConnectedTensorProduct(
            self.irreps_out,
            self.node_attrs_irreps,
            self.irreps_out,
            cueq_config=self.cueq_config,
        )
        self.reshape = reshape_irreps(self.irreps_out, cueq_config=self.cueq_config)

    def __call__(
        self,
        node_attrs: jnp.ndarray,
        node_feats: jnp.ndarray,
        edge_attrs: jnp.ndarray,
        edge_feats: jnp.ndarray,
        edge_index: jnp.ndarray,
        cutoff: Optional[jnp.ndarray] = None,
        n_real: Optional[int] = None,
    ) -> Tuple[jnp.ndarray, None]:
        # First linear projection
        node_feats = self.linear_up(node_feats)

        # Radial MLP for convolution weights
        tp_weights = self.conv_tp_weights(edge_feats)
        if cutoff is not None:
            tp_weights = tp_weights * cutoff

        # Message passing
        if hasattr(self, "conv_fusion"):
            message = self.conv_tp(node_feats, edge_attrs, tp_weights, edge_index)
        else:
            mji = self.conv_tp(node_feats[edge_index[0]], edge_attrs, tp_weights)
            message = scatter_sum(
                src=mji, index=edge_index[1], dim=0, dim_size=node_feats.shape[0]
            )

        # Truncate ghost atoms (noop if n_real is None)
        if n_real is not None:
            message = message[:n_real]
            node_attrs = node_attrs[:n_real]

        # Linear + skip connection
        message = self.linear(message) / self.avg_num_neighbors
        message = self.skip_tp(message, node_attrs)

        return self.reshape(message), None


class ScaleShiftBlock(hk.Module):
    def __init__(self, scale: float, shift: float):
        super().__init__()
        self.scale = scale
        self.shift = shift

    def __call__(self, x: IrrepsArray) -> IrrepsArray:
        return self.scale * x + self.shift

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(scale={self.scale:.6f}, shift={self.shift:.6f})"
        )
