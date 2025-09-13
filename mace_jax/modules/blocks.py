import abc
from typing import Callable, List, Optional, Tuple

import e3nn_jax as e3nn
import haiku as hk
import jax
import jax.numpy as jnp
from e3nn_jax import Irreps, IrrepsArray

from mace_jax.e3nn import nn
from mace_jax.modules.wrapper_ops import (
    CuEquivarianceConfig,
    FullyConnectedTensorProduct,
    Linear,
    OEQConfig,
    TensorProduct,
)
from mace_jax.tools.scatter import scatter_sum

from .irreps_tools import mask_head, reshape_irreps, tp_out_irreps_with_instructions
from .symmetric_contraction import SymmetricContraction


class LinearNodeEmbeddingBlock(hk.Module):
    def __init__(
        self, num_species: int, irreps_out: Irreps, name: Optional[str] = None
    ):
        super().__init__(name=name)
        self.num_species = num_species
        self.irreps_out = Irreps(irreps_out).filter("0e").regroup()

    def __call__(self, node_specie: jnp.ndarray) -> IrrepsArray:
        w = hk.get_parameter(
            "embeddings",
            shape=(self.num_species, self.irreps_out.dim),
            dtype=jnp.float32,
            init=hk.initializers.RandomNormal(),
            name="linear",
        )
        return IrrepsArray(self.irreps_out, w[node_specie])


class LinearReadoutBlock(hk.Module):
    def __init__(
        self,
        irreps_in: Irreps,
        irrep_out: Irreps = Irreps("0e"),
        cueq_config: Optional[CuEquivarianceConfig] = None,
        oeq_config: Optional[OEQConfig] = None,  # pylint: disable=unused-argument
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self.irreps_in = irreps_in
        self.irrep_out = irrep_out
        self.cueq_config = cueq_config

    def __call__(
        self,
        x: IrrepsArray,
        heads: Optional[jnp.ndarray] = None,  # pylint: disable=unused-argument
    ) -> IrrepsArray:
        # x = [n_nodes, irreps]
        return Linear(
            irreps_in=self.irreps_in,
            irreps_out=self.irrep_out,
            cueq_config=self.cueq_config,
            name="linear",
        )(x)  # [n_nodes, output_irreps]


class NonLinearReadoutBlock(hk.Module):
    def __init__(
        self,
        irreps_in: Irreps,
        MLP_irreps: Irreps,
        gate: Optional[Callable],
        irrep_out: Irreps = Irreps("0e"),
        num_heads: int = 1,
        cueq_config: Optional[CuEquivarianceConfig] = None,
        oeq_config: Optional[OEQConfig] = None,  # unused
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self.hidden_irreps = MLP_irreps
        self.num_heads = num_heads

        self.linear_1 = Linear(
            irreps_in=irreps_in,
            irreps_out=self.hidden_irreps,
            cueq_config=cueq_config,
            name="linear_1",
        )
        self.non_linearity = nn.Activation(
            irreps_in=self.hidden_irreps,
            acts=[gate],
            name="non_linearity",
        )
        self.linear_2 = Linear(
            irreps_in=self.hidden_irreps,
            irreps_out=irrep_out,
            cueq_config=cueq_config,
            name="linear_2",
        )

    def __call__(
        self, x: IrrepsArray, heads: Optional[jnp.ndarray] = None
    ) -> IrrepsArray:
        # First linear + nonlinearity
        x = self.non_linearity(self.linear_1(x))

        # Optional multi-head masking
        if hasattr(self, "num_heads"):
            if self.num_heads > 1 and heads is not None:
                x = mask_head(x, heads, self.num_heads)

        # Final linear projection
        return self.linear_2(x)


class NonLinearBiasReadoutBlock(hk.Module):
    """
    Non-linear readout with intermediate bias linear layers and optional multi-head masking.
    """

    def __init__(
        self,
        irreps_in: Irreps,
        MLP_irreps: Irreps,
        gate: Optional[Callable],
        irrep_out: Irreps = Irreps("0e"),
        num_heads: int = 1,
        cueq_config: Optional["CuEquivarianceConfig"] = None,
        oeq_config: Optional["OEQConfig"] = None,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self.hidden_irreps = MLP_irreps
        self.num_heads = num_heads
        self.linear_1 = Linear(
            irreps_in=irreps_in,
            irreps_out=self.hidden_irreps,
            cueq_config=cueq_config,
            name="linear_1",
        )
        self.non_linearity_1 = nn.Activation(
            irreps_in=self.hidden_irreps,
            acts=[gate],
            name="activation_1",
        )
        self.linear_mid = Linear(
            irreps_in=self.hidden_irreps,
            irreps_out=self.hidden_irreps,
            biases=True,
            name="linear_mid",
        )
        self.non_linearity_2 = nn.Activation(
            irreps_in=self.hidden_irreps,
            acts=[gate],
            name="activation_2",
        )
        self.linear_2 = Linear(
            irreps_in=self.hidden_irreps,
            irreps_out=irrep_out,
            biases=True,
            name="linear_2",
        )

    def __call__(
        self, x: IrrepsArray, heads: Optional[jnp.ndarray] = None
    ) -> IrrepsArray:
        # First linear + non-linearity
        x = self.non_linearity_1(self.linear_1(x))
        # Mid linear + non-linearity
        x = self.non_linearity_2(self.linear_mid(x))
        # Optional multi-head masking
        if self.num_heads > 1 and heads is not None:
            x = mask_head(x, heads, self.num_heads)
        # Final linear projection
        return self.linear_2(x)


class LinearDipoleReadoutBlock(hk.Module):
    """
    Linear readout block for dipoles or scalar+dipole.
    """

    def __init__(
        self,
        irreps_in: Irreps,
        dipole_only: bool = False,
        cueq_config: Optional["CuEquivarianceConfig"] = None,
        oeq_config: Optional["OEQConfig"] = None,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)

        # Output irreps
        if dipole_only:
            self.irreps_out = Irreps("1x1o")
        else:
            self.irreps_out = Irreps("1x0e + 1x1o")

        # Linear mapping
        self.linear = Linear(
            irreps_in=irreps_in,
            irreps_out=self.irreps_out,
            cueq_config=cueq_config,
            name="linear",
        )

    def __call__(self, x: IrrepsArray) -> IrrepsArray:
        return self.linear(x)  # [n_nodes, 1] or [n_nodes, irreps_out]


class NonLinearDipoleReadoutBlock(hk.Module):
    """
    Non-linear readout block for dipoles or scalar+dipole, with gated nonlinearity.
    """

    def __init__(
        self,
        irreps_in: Irreps,
        MLP_irreps: Irreps,
        gate: Callable,
        dipole_only: bool = False,
        cueq_config: Optional["CuEquivarianceConfig"] = None,
        oeq_config: Optional["OEQConfig"] = None,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)

        self.hidden_irreps = MLP_irreps

        # Output irreps
        if dipole_only:
            self.irreps_out = Irreps("1x1o")
        else:
            self.irreps_out = Irreps("1x0e + 1x1o")

        # Partition hidden irreps into scalars and gated irreps
        irreps_scalars = Irreps([
            (mul, ir) for mul, ir in MLP_irreps if ir.l == 0 and ir in self.irreps_out
        ])
        irreps_gated = Irreps([
            (mul, ir) for mul, ir in MLP_irreps if ir.l > 0 and ir in self.irreps_out
        ])
        irreps_gates = Irreps([(mul, Irreps("0e")[0][1]) for mul, _ in irreps_gated])

        # Gated nonlinearity
        self.equivariant_nonlin = nn.Gate(
            irreps_scalars=irreps_scalars,
            act_scalars=[gate for _, ir in irreps_scalars],
            irreps_gates=irreps_gates,
            act_gates=[gate] * len(irreps_gates),
            irreps_gated=irreps_gated,
        )

        # Input to nonlinearity
        self.irreps_nonlin = self.equivariant_nonlin.irreps_in.simplify()

        # Linear layers
        self.linear_1 = Linear(
            irreps_in=irreps_in,
            irreps_out=self.irreps_nonlin,
            cueq_config=cueq_config,
            name="linear_1",
        )
        self.linear_2 = Linear(
            irreps_in=self.hidden_irreps,
            irreps_out=self.irreps_out,
            cueq_config=cueq_config,
            name="linear_2",
        )

    def __call__(self, x: IrrepsArray) -> IrrepsArray:
        x = self.equivariant_nonlin(self.linear_1(x))
        return self.linear_2(x)  # [n_nodes, irreps_out]


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
            name="linear_up",
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
            name="conv_tp",
        )

        # Convolution weights network
        self.conv_tp_weights = nn.FullyConnectedNet(
            hs=[self.edge_feats_irreps.num_irreps]
            + self.radial_MLP
            + [self.conv_tp.weight_numel],
            act=jax.nn.silu,
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
            name="linear",
        )

        # Selector TensorProduct
        self.skip_tp = FullyConnectedTensorProduct(
            self.irreps_out,
            self.node_attrs_irreps,
            self.irreps_out,
            cueq_config=self.cueq_config,
            name="skip_tp",
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
