import abc
from collections.abc import Sequence
from typing import Callable, Optional, Union

import flax.linen as fnn
import jax
import jax.numpy as jnp
import numpy as np
from e3nn_jax import Irrep, Irreps, IrrepsArray

from mace_jax.adapters.e3nn import nn
from mace_jax.adapters.flax.torch import (
    auto_import_from_torch_flax,
    register_flax_module,
)
from mace_jax.modules.wrapper_ops import (
    CuEquivarianceConfig,
    FullyConnectedTensorProduct,
    Linear,
    SymmetricContractionWrapper,
    TensorProduct,
    TransposeIrrepsLayoutWrapper,
)
from mace_jax.tools.dtype import default_dtype
from mace_jax.tools.scatter import scatter_sum

from .irreps_tools import mask_head, reshape_irreps, tp_out_irreps_with_instructions
from .radial import (
    AgnesiTransform,
    BesselBasis,
    ChebychevBasis,
    GaussianBasis,
    PolynomialCutoff,
    RadialMLP,
    SoftTransform,
)


@register_flax_module('mace.modules.blocks.LinearNodeEmbeddingBlock')
@auto_import_from_torch_flax(allow_missing_mapper=True)
class LinearNodeEmbeddingBlock(fnn.Module):
    """Flax version of LinearNodeEmbeddingBlock."""

    irreps_in: Irreps
    irreps_out: Irreps
    cueq_config: Optional[CuEquivarianceConfig] = None

    def setup(self) -> None:
        self.linear = Linear(
            irreps_in=self.irreps_in,
            irreps_out=self.irreps_out,
            cueq_config=self.cueq_config,
            name='linear',
        )

    def __call__(self, node_attrs: jnp.ndarray) -> jnp.ndarray:
        return self.linear(node_attrs)

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}(irreps_in={self.irreps_in}, '
            f'irreps_out={self.irreps_out}, cueq_config={self.cueq_config})'
        )


@register_flax_module('mace.modules.blocks.LinearReadoutBlock')
@auto_import_from_torch_flax(allow_missing_mapper=True)
class LinearReadoutBlock(fnn.Module):
    """Flax version of LinearReadoutBlock."""

    irreps_in: Irreps
    irrep_out: Irreps = Irreps('0e')
    cueq_config: Optional[CuEquivarianceConfig] = None

    def setup(self) -> None:
        self.linear = Linear(
            irreps_in=self.irreps_in,
            irreps_out=self.irrep_out,
            cueq_config=self.cueq_config,
            name='linear',
        )

    def __call__(
        self,
        x: jnp.ndarray,
        heads: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        del heads  # maintained for Torch parity
        return self.linear(x)

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}(irreps_in={self.irreps_in}, '
            f'irrep_out={self.irrep_out}, cueq_config={self.cueq_config})'
        )


@register_flax_module('mace.modules.blocks.NonLinearReadoutBlock')
@auto_import_from_torch_flax(allow_missing_mapper=True)
class NonLinearReadoutBlock(fnn.Module):
    """Two-layer readout with optional multi-head masking."""

    irreps_in: Irreps
    MLP_irreps: Irreps
    gate: Optional[Callable]
    irrep_out: Irreps = Irreps('0e')
    num_heads: int = 1
    cueq_config: Optional[CuEquivarianceConfig] = None

    def setup(self) -> None:
        self.hidden_irreps = self.MLP_irreps
        self.linear_1 = Linear(
            irreps_in=self.irreps_in,
            irreps_out=self.hidden_irreps,
            cueq_config=self.cueq_config,
            name='linear_1',
        )
        self.non_linearity = nn.Activation(
            irreps_in=self.hidden_irreps,
            acts=[self.gate],
            name='non_linearity',
        )
        self.linear_2 = Linear(
            irreps_in=self.hidden_irreps,
            irreps_out=self.irrep_out,
            cueq_config=self.cueq_config,
            name='linear_2',
        )

    def __call__(
        self,
        x: IrrepsArray,
        heads: Optional[jnp.ndarray] = None,
    ) -> IrrepsArray:
        x = self.non_linearity(self.linear_1(x))
        if self.num_heads > 1 and heads is not None:
            x = mask_head(x, heads, self.num_heads)
        return self.linear_2(x)


@register_flax_module('mace.modules.blocks.NonLinearBiasReadoutBlock')
@auto_import_from_torch_flax(allow_missing_mapper=True)
class NonLinearBiasReadoutBlock(fnn.Module):
    """Non-linear readout with intermediate bias linear layers."""

    irreps_in: Irreps
    MLP_irreps: Irreps
    gate: Optional[Callable]
    irrep_out: Irreps = Irreps('0e')
    num_heads: int = 1
    cueq_config: Optional[CuEquivarianceConfig] = None

    def setup(self) -> None:
        self.hidden_irreps = self.MLP_irreps
        self.linear_1 = Linear(
            irreps_in=self.irreps_in,
            irreps_out=self.hidden_irreps,
            cueq_config=self.cueq_config,
            name='linear_1',
        )
        self.non_linearity_1 = nn.Activation(
            irreps_in=self.hidden_irreps,
            acts=[self.gate],
            name='activation_1',
        )
        self.linear_mid = Linear(
            irreps_in=self.hidden_irreps,
            irreps_out=self.hidden_irreps,
            cueq_config=self.cueq_config,
            name='linear_mid',
        )
        self.non_linearity_2 = nn.Activation(
            irreps_in=self.hidden_irreps,
            acts=[self.gate],
            name='activation_2',
        )
        self.linear_2 = Linear(
            irreps_in=self.hidden_irreps,
            irreps_out=self.irrep_out,
            cueq_config=self.cueq_config,
            name='linear_2',
        )

    def __call__(
        self,
        x: IrrepsArray,
        heads: Optional[jnp.ndarray] = None,
    ) -> IrrepsArray:
        x = self.non_linearity_1(self.linear_1(x))
        x = self.non_linearity_2(self.linear_mid(x))
        if self.num_heads > 1 and heads is not None:
            x = mask_head(x, heads, self.num_heads)
        return self.linear_2(x)


@register_flax_module('mace.modules.blocks.LinearDipoleReadoutBlock')
@auto_import_from_torch_flax(allow_missing_mapper=True)
class LinearDipoleReadoutBlock(fnn.Module):
    """Linear readout block for dipoles or scalar+dipole."""

    irreps_in: Irreps
    dipole_only: bool = False
    cueq_config: Optional[CuEquivarianceConfig] = None

    def setup(self) -> None:
        if self.dipole_only:
            self.irreps_out = Irreps('1x1o')
        else:
            self.irreps_out = Irreps('1x0e + 1x1o')

        self.linear = Linear(
            irreps_in=self.irreps_in,
            irreps_out=self.irreps_out,
            cueq_config=self.cueq_config,
            name='linear',
        )

    def __call__(self, x: IrrepsArray) -> IrrepsArray:
        return self.linear(x)


@register_flax_module('mace.modules.blocks.NonLinearDipoleReadoutBlock')
@auto_import_from_torch_flax(allow_missing_mapper=True)
class NonLinearDipoleReadoutBlock(fnn.Module):
    """Non-linear readout block for dipoles or scalar+dipole."""

    irreps_in: Irreps
    MLP_irreps: Irreps
    gate: Callable
    dipole_only: bool = False
    cueq_config: Optional[CuEquivarianceConfig] = None

    def setup(self) -> None:
        self.hidden_irreps = self.MLP_irreps
        if self.dipole_only:
            self.irreps_out = Irreps('1x1o')
        else:
            self.irreps_out = Irreps('1x0e + 1x1o')

        irreps_scalars = Irreps(
            [
                (mul, ir)
                for mul, ir in self.hidden_irreps
                if ir.l == 0 and ir in self.irreps_out
            ]
        )
        irreps_gated = Irreps(
            [
                (mul, ir)
                for mul, ir in self.hidden_irreps
                if ir.l > 0 and ir in self.irreps_out
            ]
        )
        irreps_gates = Irreps([(mul, Irreps('0e')[0][1]) for mul, _ in irreps_gated])

        self.equivariant_nonlin = nn.Gate(
            irreps_scalars=irreps_scalars,
            act_scalars=[self.gate for _ in irreps_scalars],
            irreps_gates=irreps_gates,
            act_gates=[self.gate] * len(irreps_gates),
            irreps_gated=irreps_gated,
        )
        self.irreps_nonlin = self.equivariant_nonlin.irreps_in.simplify()

        self.linear_1 = Linear(
            irreps_in=self.irreps_in,
            irreps_out=self.irreps_nonlin,
            cueq_config=self.cueq_config,
            name='linear_1',
        )
        self.linear_2 = Linear(
            irreps_in=self.hidden_irreps,
            irreps_out=self.irreps_out,
            cueq_config=self.cueq_config,
            name='linear_2',
        )

    def __call__(self, x: IrrepsArray) -> IrrepsArray:
        x = self.equivariant_nonlin(self.linear_1(x))
        return self.linear_2(x)


@register_flax_module('mace.modules.blocks.LinearDipolePolarReadoutBlock')
@auto_import_from_torch_flax(allow_missing_mapper=True)
class LinearDipolePolarReadoutBlock(fnn.Module):
    """Linear readout for dipole and polarizability."""

    irreps_in: Irreps
    use_polarizability: bool = True
    cueq_config: Optional[CuEquivarianceConfig] = None

    def setup(self) -> None:
        if not self.use_polarizability:
            raise ValueError(
                'LinearDipolePolarReadoutBlock requires use_polarizability=True.'
            )
        self.irreps_out = Irreps('2x0e + 1x1o + 1x2e')
        self.linear = Linear(
            irreps_in=self.irreps_in,
            irreps_out=self.irreps_out,
            cueq_config=self.cueq_config,
            name='linear',
        )

    def __call__(self, x: IrrepsArray) -> IrrepsArray:
        return self.linear(x)


@register_flax_module('mace.modules.blocks.NonLinearDipolePolarReadoutBlock')
@auto_import_from_torch_flax(allow_missing_mapper=True)
class NonLinearDipolePolarReadoutBlock(fnn.Module):
    """Non-linear readout for dipole and polarizability with equivariant gate."""

    irreps_in: Irreps
    MLP_irreps: Irreps
    gate: Callable
    use_polarizability: bool = True
    cueq_config: Optional[CuEquivarianceConfig] = None

    def setup(self) -> None:
        self.hidden_irreps = self.MLP_irreps
        if not self.use_polarizability:
            raise ValueError(
                'NonLinearDipolePolarReadoutBlock requires use_polarizability=True.'
            )
        self.irreps_out = Irreps('2x0e + 1x1o + 1x2e')

        irreps_scalars = Irreps(
            [
                (mul, ir)
                for mul, ir in self.hidden_irreps
                if ir.l == 0 and ir in self.irreps_out
            ]
        )
        irreps_gated = Irreps(
            [
                (mul, ir)
                for mul, ir in self.hidden_irreps
                if ir.l > 0 and ir in self.irreps_out
            ]
        )
        irreps_gates = Irreps([(mul, '0e') for mul, _ in irreps_gated])

        self.equivariant_nonlin = nn.Gate(
            irreps_scalars=irreps_scalars,
            act_scalars=[self.gate for _ in irreps_scalars],
            irreps_gates=irreps_gates,
            act_gates=[self.gate] * len(irreps_gates),
            irreps_gated=irreps_gated,
        )
        self.irreps_nonlin = self.equivariant_nonlin.irreps_in.simplify()

        self.linear_1 = Linear(
            irreps_in=self.irreps_in,
            irreps_out=self.irreps_nonlin,
            cueq_config=self.cueq_config,
            name='linear_1',
        )
        self.linear_2 = Linear(
            irreps_in=self.hidden_irreps,
            irreps_out=self.irreps_out,
            cueq_config=self.cueq_config,
            name='linear_2',
        )

    def __call__(self, x: IrrepsArray) -> IrrepsArray:
        x = self.equivariant_nonlin(self.linear_1(x))
        return self.linear_2(x)


@register_flax_module('mace.modules.blocks.AtomicEnergiesBlock')
@auto_import_from_torch_flax(allow_missing_mapper=True)
class AtomicEnergiesBlock(fnn.Module):
    """Block that returns atomic energies from one-hot element vectors."""

    atomic_energies_init: Union[np.ndarray, jnp.ndarray]

    @fnn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        init_values = jnp.asarray(self.atomic_energies_init, dtype=default_dtype())

        atomic_energies = self.param(
            'atomic_energies',
            lambda rng: init_values,
        )

        energies = jnp.atleast_2d(atomic_energies)
        return jnp.matmul(x, energies.T)

    def __repr__(self) -> str:
        energies_np = np.array(self.atomic_energies_init)
        formatted_energies = ', '.join(
            '[' + ', '.join(f'{value:.4f}' for value in row) + ']'
            for row in np.atleast_2d(energies_np)
        )
        return f'{self.__class__.__name__}(energies=[{formatted_energies}])'


@register_flax_module('mace.modules.blocks.RadialEmbeddingBlock')
@auto_import_from_torch_flax(allow_missing_mapper=True)
class RadialEmbeddingBlock(fnn.Module):
    """Radial basis embedding block for edges."""

    r_max: float
    num_bessel: int
    num_polynomial_cutoff: int
    radial_type: str = 'bessel'
    distance_transform: str = 'None'
    apply_cutoff: bool = True

    def setup(self) -> None:
        if self.radial_type == 'bessel':
            self.basis_fn = BesselBasis(r_max=self.r_max, num_basis=self.num_bessel)
        elif self.radial_type == 'gaussian':
            self.basis_fn = GaussianBasis(r_max=self.r_max, num_basis=self.num_bessel)
        elif self.radial_type == 'chebyshev':
            self.basis_fn = ChebychevBasis(r_max=self.r_max, num_basis=self.num_bessel)
        else:
            raise ValueError(f'Unknown radial_type: {self.radial_type}')

        if self.distance_transform == 'Agnesi':
            self.distance_transform_module = AgnesiTransform()
        elif self.distance_transform == 'Soft':
            self.distance_transform_module = SoftTransform()
        else:
            self.distance_transform_module = None

        self.cutoff_fn = PolynomialCutoff(
            r_max=self.r_max, p=self.num_polynomial_cutoff
        )
        self.out_dim = self.num_bessel

    def __call__(
        self,
        edge_lengths: jnp.ndarray,
        node_attrs: jnp.ndarray,
        edge_index: jnp.ndarray,
        atomic_numbers: jnp.ndarray,
    ):
        cutoff = self.cutoff_fn(edge_lengths)

        transformed_lengths = edge_lengths
        if self.distance_transform_module is not None:
            transformed_lengths = self.distance_transform_module(
                edge_lengths,
                node_attrs,
                edge_index,
                atomic_numbers,
            )

        radial = self.basis_fn(transformed_lengths)

        if self.apply_cutoff:
            return radial * cutoff, None
        return radial, cutoff


@register_flax_module('mace.modules.blocks.EquivariantProductBasisBlock')
@auto_import_from_torch_flax(allow_missing_mapper=True)
class EquivariantProductBasisBlock(fnn.Module):
    node_feats_irreps: Irreps
    target_irreps: Irreps
    correlation: int
    use_sc: bool = True
    num_elements: Optional[int] = None
    use_agnostic_product: bool = False
    use_reduced_cg: Optional[bool] = None
    cueq_config: Optional[CuEquivarianceConfig] = None

    def setup(self) -> None:
        num_elements = self.num_elements
        if self.use_agnostic_product:
            num_elements = 1

        self.symmetric_contractions = SymmetricContractionWrapper(
            irreps_in=self.node_feats_irreps,
            irreps_out=self.target_irreps,
            correlation=self.correlation,
            num_elements=num_elements,
            use_reduced_cg=self.use_reduced_cg,
            cueq_config=self.cueq_config,
            name='symmetric_contractions',
        )

        self.linear = Linear(
            irreps_in=self.target_irreps,
            irreps_out=self.target_irreps,
            internal_weights=True,
            shared_weights=True,
            cueq_config=self.cueq_config,
            name='linear',
        )

    def __call__(
        self,
        node_feats: jnp.ndarray,
        sc: Optional[jnp.ndarray],
        node_attrs: jnp.ndarray,
    ) -> jnp.ndarray:
        if self.use_agnostic_product:
            node_attrs = jnp.ones((node_feats.shape[0], 1), dtype=node_feats.dtype)

        use_cueq = False
        use_cueq_mul_ir = False
        if self.cueq_config is not None:
            if self.cueq_config.enabled and (
                self.cueq_config.optimize_all or self.cueq_config.optimize_symmetric
            ):
                use_cueq = True
            if getattr(self.cueq_config, 'layout_str', None) == 'mul_ir':
                use_cueq_mul_ir = True

        if use_cueq:
            if use_cueq_mul_ir:
                node_feats = jnp.transpose(node_feats, (0, 2, 1))
            index_attrs = jnp.nonzero(node_attrs, size=node_attrs.shape[0])[1]
            node_attrs = jax.nn.one_hot(
                index_attrs,
                self.symmetric_contractions.num_elements,
                dtype=node_feats.dtype,
            )
            node_feats = self.symmetric_contractions(node_feats, node_attrs)
        else:
            node_feats = self.symmetric_contractions(node_feats, node_attrs)

        if self.use_sc and sc is not None:
            return self.linear(node_feats) + sc
        return self.linear(node_feats)


class InteractionBlock(fnn.Module, metaclass=abc.ABCMeta):
    """Abstract base class for interaction blocks in equivariant GNNs."""

    node_attrs_irreps: Irreps
    node_feats_irreps: Irreps
    edge_attrs_irreps: Irreps
    edge_feats_irreps: Irreps
    target_irreps: Irreps
    hidden_irreps: Irreps
    avg_num_neighbors: float
    edge_irreps: Optional[Irreps] = None
    radial_MLP: Optional[Sequence[int]] = None
    cueq_config: Optional[CuEquivarianceConfig] = None

    def setup(self) -> None:
        object.__setattr__(self, 'node_attrs_irreps', Irreps(self.node_attrs_irreps))
        object.__setattr__(self, 'node_feats_irreps', Irreps(self.node_feats_irreps))
        object.__setattr__(self, 'edge_attrs_irreps', Irreps(self.edge_attrs_irreps))
        object.__setattr__(self, 'edge_feats_irreps', Irreps(self.edge_feats_irreps))
        object.__setattr__(self, 'target_irreps', Irreps(self.target_irreps))
        object.__setattr__(self, 'hidden_irreps', Irreps(self.hidden_irreps))

        if self.radial_MLP is not None:
            radial = list(self.radial_MLP)
        else:
            radial = [64, 64, 64]
        object.__setattr__(self, 'radial_MLP', radial)

        edge_irreps = (
            Irreps(self.edge_irreps)
            if self.edge_irreps is not None
            else Irreps(self.node_feats_irreps)
        )
        object.__setattr__(self, 'edge_irreps', edge_irreps)

        if self.cueq_config and getattr(self.cueq_config, 'conv_fusion', None):
            self.conv_fusion = self.cueq_config.conv_fusion

        self._setup()

    @abc.abstractmethod
    def _setup(self) -> None:
        raise NotImplementedError

    def truncate_ghosts(
        self,
        tensor: jnp.ndarray,
        n_real: Optional[int] = None,
    ) -> jnp.ndarray:
        return tensor[:n_real] if n_real is not None else tensor

    @abc.abstractmethod
    @fnn.compact
    @fnn.compact
    def __call__(
        self,
        node_attrs: jnp.ndarray,
        node_feats: jnp.ndarray,
        edge_attrs: jnp.ndarray,
        edge_feats: jnp.ndarray,
        edge_index: jnp.ndarray,
    ) -> jnp.ndarray:
        raise NotImplementedError


@register_flax_module('mace.modules.blocks.RealAgnosticInteractionBlock')
@auto_import_from_torch_flax(allow_missing_mapper=True)
class RealAgnosticInteractionBlock(InteractionBlock):
    def _setup(self) -> None:
        # First linear
        self.linear_up = Linear(
            self.node_feats_irreps,
            self.edge_irreps,
            internal_weights=True,
            shared_weights=True,
            cueq_config=self.cueq_config,
            name='linear_up',
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
            name='conv_tp',
        )

        # Convolution weights network
        self.conv_tp_weights = nn.FullyConnectedNet(
            hs=[self.edge_feats_irreps.num_irreps]
            + list(self.radial_MLP)
            + [self.conv_tp.weight_numel],
            act=jax.nn.silu,
            name='conv_tp_weights',
        )

        # Linear
        self.irreps_out = self.target_irreps
        self.linear = Linear(
            irreps_mid,
            self.irreps_out,
            internal_weights=True,
            shared_weights=True,
            cueq_config=self.cueq_config,
            name='linear',
        )

        # Selector TensorProduct
        self.skip_tp = FullyConnectedTensorProduct(
            self.irreps_out,
            self.node_attrs_irreps,
            self.irreps_out,
            cueq_config=self.cueq_config,
            name='skip_tp',
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
        first_layer: bool = False,
    ) -> tuple[jnp.ndarray, None]:
        # First linear projection
        node_feats = self.linear_up(node_feats)

        # Radial MLP for convolution weights
        tp_weights = self.conv_tp_weights(edge_feats)
        if cutoff is not None:
            tp_weights = tp_weights * cutoff

        # Message passing
        if hasattr(self, 'conv_fusion'):
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


@register_flax_module('mace.modules.blocks.RealAgnosticResidualInteractionBlock')
@auto_import_from_torch_flax(allow_missing_mapper=True)
class RealAgnosticResidualInteractionBlock(InteractionBlock):
    def _setup(self) -> None:
        # First linear
        self.linear_up = Linear(
            self.node_feats_irreps,
            self.edge_irreps,
            internal_weights=True,
            shared_weights=True,
            cueq_config=self.cueq_config,
            name='linear_up',
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
            name='conv_tp',
        )

        # Convolution weights network
        self.conv_tp_weights = nn.FullyConnectedNet(
            hs=[self.edge_feats_irreps.num_irreps]
            + list(self.radial_MLP)
            + [self.conv_tp.weight_numel],
            act=jax.nn.silu,
            name='conv_tp_weights',
        )

        # Linear
        self.irreps_out = self.target_irreps
        self.linear = Linear(
            irreps_mid,
            self.irreps_out,
            internal_weights=True,
            shared_weights=True,
            cueq_config=self.cueq_config,
            name='linear',
        )

        # Selector TensorProduct (skip connection)
        self.skip_tp = FullyConnectedTensorProduct(
            self.node_feats_irreps,
            self.node_attrs_irreps,
            self.hidden_irreps,
            cueq_config=self.cueq_config,
            name='skip_tp',
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
        first_layer: bool = False,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        # Skip connection
        sc = self.skip_tp(node_feats, node_attrs)

        # First linear projection
        node_feats = self.linear_up(node_feats)

        # Radial MLP for convolution weights
        tp_weights = self.conv_tp_weights(edge_feats)
        if cutoff is not None:
            tp_weights = tp_weights * cutoff

        # Message passing
        if hasattr(self, 'conv_fusion'):
            message = self.conv_tp(node_feats, edge_attrs, tp_weights, edge_index)
        else:
            mji = self.conv_tp(node_feats[edge_index[0]], edge_attrs, tp_weights)
            message = scatter_sum(
                src=mji,
                index=edge_index[1],
                dim=0,
                dim_size=node_feats.shape[0],
            )

        # Truncate ghost atoms (noop if n_real is None)
        if n_real is not None:
            message = message[:n_real]
            node_attrs = node_attrs[:n_real]
            sc = sc[:n_real]

        # Linear + normalization
        message = self.linear(message) / self.avg_num_neighbors

        return self.reshape(message), sc


@register_flax_module('mace.modules.blocks.RealAgnosticDensityInteractionBlock')
@auto_import_from_torch_flax(allow_missing_mapper=True)
class RealAgnosticDensityInteractionBlock(InteractionBlock):
    def _setup(self) -> None:
        # First linear
        self.linear_up = Linear(
            self.node_feats_irreps,
            self.edge_irreps,
            internal_weights=True,
            shared_weights=True,
            cueq_config=self.cueq_config,
            name='linear_up',
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
            name='conv_tp',
        )

        # Convolution weights network
        self.conv_tp_weights = nn.FullyConnectedNet(
            hs=[self.edge_feats_irreps.num_irreps]
            + list(self.radial_MLP)
            + [self.conv_tp.weight_numel],
            act=jax.nn.silu,
            name='conv_tp_weights',
        )

        # Linear projection
        self.irreps_out = self.target_irreps
        self.linear = Linear(
            irreps_mid,
            self.irreps_out,
            internal_weights=True,
            shared_weights=True,
            cueq_config=self.cueq_config,
            name='linear',
        )

        # Selector TensorProduct (skip connection)
        self.skip_tp = FullyConnectedTensorProduct(
            self.irreps_out,
            self.node_attrs_irreps,
            self.irreps_out,
            cueq_config=self.cueq_config,
            name='skip_tp',
        )

        # Density normalization network
        self.density_fn = nn.FullyConnectedNet(
            hs=[self.edge_feats_irreps.num_irreps, 1],
            act=jax.nn.silu,
            name='density_fn',
        )

        # Reshape output
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
        first_layer: bool = False,
    ) -> tuple[jnp.ndarray, None]:
        receiver = edge_index[1]
        num_nodes = node_feats.shape[0]

        # Linear projection
        node_feats = self.linear_up(node_feats)

        # Convolution weights
        tp_weights = self.conv_tp_weights(edge_feats)

        # Edge density
        edge_density = jnp.tanh(self.density_fn(edge_feats) ** 2)

        if cutoff is not None:
            tp_weights = tp_weights * cutoff
            edge_density = edge_density * cutoff

        # Aggregate density per node
        density = scatter_sum(
            edge_density, receiver, dim=0, dim_size=num_nodes
        )  # [n_nodes, 1]

        # Message passing
        if hasattr(self, 'conv_fusion'):
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
            density = density[:n_real]

        # Normalize messages by density
        message = self.linear(message) / (density + 1)
        message = self.skip_tp(message, node_attrs)

        return self.reshape(message), None


@register_flax_module('mace.modules.blocks.RealAgnosticDensityResidualInteractionBlock')
@auto_import_from_torch_flax(allow_missing_mapper=True)
class RealAgnosticDensityResidualInteractionBlock(InteractionBlock):
    def _setup(self) -> None:
        # First linear
        self.linear_up = Linear(
            self.node_feats_irreps,
            self.edge_irreps,
            internal_weights=True,
            shared_weights=True,
            cueq_config=self.cueq_config,
            name='linear_up',
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
            name='conv_tp',
        )

        # Convolution weights network
        self.conv_tp_weights = nn.FullyConnectedNet(
            hs=[self.edge_feats_irreps.num_irreps]
            + list(self.radial_MLP)
            + [self.conv_tp.weight_numel],
            act=jax.nn.silu,
            name='conv_tp_weights',
        )

        # Linear projection
        self.irreps_out = self.target_irreps
        self.linear = Linear(
            irreps_mid,
            self.irreps_out,
            internal_weights=True,
            shared_weights=True,
            cueq_config=self.cueq_config,
            name='linear',
        )

        # Selector TensorProduct (skip connection)
        self.skip_tp = FullyConnectedTensorProduct(
            self.node_feats_irreps,
            self.node_attrs_irreps,
            self.hidden_irreps,
            cueq_config=self.cueq_config,
            name='skip_tp',
        )

        # Density normalization network
        self.density_fn = nn.FullyConnectedNet(
            hs=[self.edge_feats_irreps.num_irreps, 1],
            act=jax.nn.silu,
            name='density_fn',
        )

        # Reshape output
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
        first_layer: bool = False,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        receiver = edge_index[1]
        num_nodes = node_feats.shape[0]

        # Skip connection
        sc = self.skip_tp(node_feats, node_attrs)

        # Linear projection
        node_feats = self.linear_up(node_feats)

        # Convolution weights
        tp_weights = self.conv_tp_weights(edge_feats)

        # Edge density
        edge_density = jnp.tanh(self.density_fn(edge_feats) ** 2)

        if cutoff is not None:
            tp_weights = tp_weights * cutoff
            edge_density = edge_density * cutoff

        # Aggregate density per node
        density = scatter_sum(
            src=edge_density, index=receiver, dim=0, dim_size=num_nodes
        )  # [n_nodes, 1]

        # Message passing
        if hasattr(self, 'conv_fusion'):
            message = self.conv_tp(node_feats, edge_attrs, tp_weights, edge_index)
        else:
            mji = self.conv_tp(
                node_feats[edge_index[0]], edge_attrs, tp_weights
            )  # [n_nodes, irreps]
            message = scatter_sum(
                src=mji, index=edge_index[1], dim=0, dim_size=node_feats.shape[0]
            )

        message = self.truncate_ghosts(message, n_real)
        node_attrs = self.truncate_ghosts(node_attrs, n_real)
        density = self.truncate_ghosts(density, n_real)
        sc = self.truncate_ghosts(sc, n_real)

        # Normalize messages by density
        message = self.linear(message) / (density + 1)

        return self.reshape(message), sc


@register_flax_module('mace.modules.blocks.RealAgnosticAttResidualInteractionBlock')
@auto_import_from_torch_flax(allow_missing_mapper=True)
class RealAgnosticAttResidualInteractionBlock(InteractionBlock):
    def _setup(self) -> None:
        # Downsample irreps
        object.__setattr__(self, 'node_feats_down_irreps', Irreps('64x0e'))

        # First linear (up)
        self.linear_up = Linear(
            self.node_feats_irreps,
            self.edge_irreps,
            internal_weights=True,
            shared_weights=True,
            cueq_config=self.cueq_config,
            name='linear_up',
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
            name='conv_tp',
        )

        # Linear (down)
        self.linear_down = Linear(
            self.node_feats_irreps,
            self.node_feats_down_irreps,
            internal_weights=True,
            shared_weights=True,
            cueq_config=self.cueq_config,
            name='linear_down',
        )

        # Convolution weights network
        input_dim = (
            self.edge_feats_irreps.num_irreps
            + 2 * self.node_feats_down_irreps.num_irreps
        )
        self.conv_tp_weights = nn.FullyConnectedNet(
            hs=[input_dim] + [256, 256, 256] + [self.conv_tp.weight_numel],
            act=jax.nn.silu,
            name='conv_tp_weights',
        )

        # Linear output
        self.irreps_out = self.target_irreps
        self.linear = Linear(
            irreps_mid,
            self.irreps_out,
            internal_weights=True,
            shared_weights=True,
            cueq_config=self.cueq_config,
            name='linear',
        )

        # Output reshape
        self.reshape = reshape_irreps(self.irreps_out, cueq_config=self.cueq_config)

        # Skip connection
        self.skip_linear = Linear(
            self.node_feats_irreps,
            self.hidden_irreps,
            cueq_config=self.cueq_config,
            name='skip_linear',
        )

    def __call__(
        self,
        node_attrs: jnp.ndarray,
        node_feats: jnp.ndarray,
        edge_attrs: jnp.ndarray,
        edge_feats: jnp.ndarray,
        edge_index: jnp.ndarray,
        cutoff: Optional[jnp.ndarray] = None,
        n_real: Optional[int] = None,
        first_layer: bool = False,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        sender = edge_index[0]
        receiver = edge_index[1]

        # Skip connection
        sc = self.skip_linear(node_feats)

        # Linear projections
        node_feats_up = self.linear_up(node_feats)
        node_feats_down = self.linear_down(node_feats)

        # Augmented edge features for convolution
        augmented_edge_feats = jnp.concatenate(
            [edge_feats, node_feats_down[sender], node_feats_down[receiver]], axis=-1
        )

        # TensorProduct weights
        tp_weights = self.conv_tp_weights(augmented_edge_feats)
        if cutoff is not None:
            tp_weights = tp_weights * cutoff

        # Message passing
        if hasattr(self, 'conv_fusion'):
            message = self.conv_tp(node_feats_up, edge_attrs, tp_weights, edge_index)
        else:
            mji = self.conv_tp(
                node_feats_up[edge_index[0]], edge_attrs, tp_weights
            )  # [n_nodes, irreps]
            message = scatter_sum(
                src=mji, index=edge_index[1], dim=0, dim_size=node_feats.shape[0]
            )

        # Linear projection and normalization
        message = self.linear(message) / self.avg_num_neighbors

        return self.reshape(message), sc


@register_flax_module(
    'mace.modules.blocks.RealAgnosticResidualNonLinearInteractionBlock'
)
@auto_import_from_torch_flax(allow_missing_mapper=True)
class RealAgnosticResidualNonLinearInteractionBlock(InteractionBlock):
    def _setup(self) -> None:
        # Compute scalar irreps
        node_scalar_irreps = Irreps(
            [(self.node_feats_irreps.count(Irrep(0, 1)), (0, 1))]
        )

        # Source/target embeddings
        self.source_embedding = Linear(
            self.node_attrs_irreps,
            node_scalar_irreps,
            internal_weights=True,
            shared_weights=True,
            cueq_config=self.cueq_config,
            name='source_embedding',
        )
        self.target_embedding = Linear(
            self.node_attrs_irreps,
            node_scalar_irreps,
            internal_weights=True,
            shared_weights=True,
            cueq_config=self.cueq_config,
            name='target_embedding',
        )

        # First linear
        self.linear_up = Linear(
            self.node_feats_irreps,
            self.edge_irreps,
            internal_weights=True,
            shared_weights=True,
            cueq_config=self.cueq_config,
            name='linear_up',
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
            name='conv_tp',
        )

        # Convolution weights (Radial MLP)
        input_dim = self.edge_feats_irreps.num_irreps
        self.conv_tp_weights = RadialMLP(
            [input_dim + 2 * node_scalar_irreps.dim]
            + self.radial_MLP
            + [self.conv_tp.weight_numel],
            name='conv_tp_weights',
        )

        # Output irreps
        self.irreps_out = self.target_irreps

        # Selector skip connection
        self.skip_tp = Linear(
            self.node_feats_irreps,
            self.hidden_irreps,
            cueq_config=self.cueq_config,
            name='skip_tp',
        )

        # Reshape
        self.reshape = reshape_irreps(self.irreps_out, cueq_config=self.cueq_config)

        # Equivariant non-linearity
        irreps_scalars = Irreps([(mul, ir) for mul, ir in self.irreps_out if ir.l == 0])
        irreps_gated = Irreps([(mul, ir) for mul, ir in self.irreps_out if ir.l > 0])
        irreps_gates = Irreps([(mul, (0, 1)) for mul, _ in irreps_gated])
        self.equivariant_nonlin = nn.Gate(
            irreps_scalars=irreps_scalars,
            act_scalars=[jax.nn.silu] * len(irreps_scalars),
            irreps_gates=irreps_gates,
            act_gates=[jax.nn.sigmoid] * len(irreps_gates),
            irreps_gated=irreps_gated,
            name='equivariant_nonlin',
        )
        self.irreps_nonlin = self.equivariant_nonlin.irreps_in.simplify()

        # Linear residual
        self.linear_res = Linear(
            self.edge_irreps,
            self.irreps_nonlin,
            internal_weights=True,
            shared_weights=True,
            cueq_config=self.cueq_config,
            name='linear_res',
        )

        # Linear blocks
        self.linear_1 = Linear(
            irreps_mid,
            self.irreps_nonlin,
            internal_weights=True,
            shared_weights=True,
            cueq_config=self.cueq_config,
            name='linear_1',
        )
        self.linear_2 = Linear(
            self.irreps_out,
            self.irreps_out,
            internal_weights=True,
            shared_weights=True,
            cueq_config=self.cueq_config,
            name='linear_2',
        )

        # Density normalization
        self.density_fn = RadialMLP(
            [input_dim + 2 * node_scalar_irreps.dim, 64, 1], name='density_fn'
        )

        # Transpose wrappers
        self.transpose_mul_ir = TransposeIrrepsLayoutWrapper(
            irreps=self.irreps_nonlin,
            source='ir_mul',
            target='mul_ir',
            cueq_config=self.cueq_config,
        )
        self.transpose_ir_mul = TransposeIrrepsLayoutWrapper(
            irreps=self.irreps_out,
            source='mul_ir',
            target='ir_mul',
            cueq_config=self.cueq_config,
        )

    @fnn.compact
    def __call__(
        self,
        node_attrs: jnp.ndarray,
        node_feats: jnp.ndarray,
        edge_attrs: jnp.ndarray,
        edge_feats: jnp.ndarray,
        edge_index: jnp.ndarray,
        cutoff: Optional[jnp.ndarray] = None,
        n_real: Optional[int] = None,
        first_layer: bool = False,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        num_nodes = node_feats.shape[0]

        # Skip connection
        sc = self.skip_tp(node_feats)

        # Linear projections
        node_feats = self.linear_up(node_feats)
        node_feats_res = self.linear_res(node_feats)

        # Source/target embeddings for edges
        source_embedding = self.source_embedding(node_attrs)
        target_embedding = self.target_embedding(node_attrs)
        edge_feats = jnp.concatenate(
            [
                edge_feats,
                source_embedding[edge_index[0]],
                target_embedding[edge_index[1]],
            ],
            axis=-1,
        )

        # Convolution weights
        tp_weights = self.conv_tp_weights(edge_feats)
        edge_density = jnp.tanh(self.density_fn(edge_feats) ** 2)
        if cutoff is not None:
            tp_weights = tp_weights * cutoff
            edge_density = edge_density * cutoff

        # Density sum per node
        density = scatter_sum(
            src=edge_density, index=edge_index[1], dim=0, dim_size=num_nodes
        )

        # Message passing
        if hasattr(self, 'conv_fusion'):
            message = self.conv_tp(node_feats, edge_attrs, tp_weights, edge_index)
        else:
            mji = self.conv_tp(
                node_feats[edge_index[0]], edge_attrs, tp_weights
            )  # [n_edges, irreps]
            message = scatter_sum(
                src=mji, index=edge_index[1], dim=0, dim_size=num_nodes
            )  # [n_nodes, irreps]

        # Truncate ghosts
        message = self.truncate_ghosts(message, n_real)
        density = self.truncate_ghosts(density, n_real)
        sc = self.truncate_ghosts(sc, n_real)
        node_feats_res = self.truncate_ghosts(node_feats_res, n_real)

        # Linear + normalization
        alpha_var = self.variable(
            'params',
            'alpha',
            lambda: jnp.array(20.0, dtype=default_dtype()),
        )
        beta_var = self.variable(
            'params',
            'beta',
            lambda: jnp.zeros((), dtype=default_dtype()),
        )
        alpha = alpha_var.value
        beta = beta_var.value

        message = self.linear_1(message) / (density * beta + alpha)
        message = message + node_feats_res

        # Equivariant non-linearity
        if self.transpose_mul_ir is not None:
            message = self.transpose_mul_ir(message)
        message = self.equivariant_nonlin(message)
        if self.transpose_ir_mul is not None:
            tensor = message.array if isinstance(message, IrrepsArray) else message
            message = self.transpose_ir_mul(tensor)

        # Linear output
        message = self.linear_2(message)

        return self.reshape(message), sc


@register_flax_module('mace.modules.blocks.ScaleShiftBlock')
@auto_import_from_torch_flax(allow_missing_mapper=True)
class ScaleShiftBlock(fnn.Module):
    scale: Union[float, jnp.ndarray]
    shift: Union[float, jnp.ndarray]

    def setup(self) -> None:
        self.scale_array = jnp.asarray(self.scale, dtype=default_dtype())
        self.shift_array = jnp.asarray(self.shift, dtype=default_dtype())

    def __call__(self, x: jnp.ndarray, head: jnp.ndarray) -> jnp.ndarray:
        scale_h = jnp.atleast_1d(self.scale_array)[head]
        shift_h = jnp.atleast_1d(self.shift_array)[head]
        return scale_h * x + shift_h

    def __repr__(self) -> str:
        scale_vals = jnp.atleast_1d(self.scale_array)
        shift_vals = jnp.atleast_1d(self.shift_array)
        formatted_scale = ', '.join(f'{float(val):.4f}' for val in scale_vals)
        formatted_shift = ', '.join(f'{float(val):.4f}' for val in shift_vals)
        return f'{self.__class__.__name__}(scale={formatted_scale}, shift={formatted_shift})'
