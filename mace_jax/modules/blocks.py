import abc
from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from e3nn_jax import Irrep, Irreps, IrrepsArray
from flax import nnx

from mace_jax.adapters.e3nn import nn
from mace_jax.adapters.nnx.torch import (
    nxx_auto_import_from_torch,
    nxx_register_module,
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


@nxx_register_module('mace.modules.blocks.LinearNodeEmbeddingBlock')
@nxx_auto_import_from_torch(allow_missing_mapper=True)
class LinearNodeEmbeddingBlock(nnx.Module):
    """Flax version of LinearNodeEmbeddingBlock."""

    irreps_in: Irreps
    irreps_out: Irreps
    cueq_config: CuEquivarianceConfig | None = None

    def __init__(
        self,
        irreps_in: Irreps,
        irreps_out: Irreps,
        cueq_config: CuEquivarianceConfig | None = None,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.irreps_in = Irreps(irreps_in)
        self.irreps_out = Irreps(irreps_out)
        self.cueq_config = cueq_config
        self.linear = Linear(
            irreps_in=self.irreps_in,
            irreps_out=self.irreps_out,
            cueq_config=self.cueq_config,
            rngs=rngs,
        )

    def __call__(self, node_attrs: jnp.ndarray) -> jnp.ndarray:
        return self.linear(node_attrs)

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}(irreps_in={self.irreps_in}, '
            f'irreps_out={self.irreps_out}, cueq_config={self.cueq_config})'
        )


@nxx_register_module('mace.modules.blocks.LinearReadoutBlock')
@nxx_auto_import_from_torch(allow_missing_mapper=True)
class LinearReadoutBlock(nnx.Module):
    """Flax version of LinearReadoutBlock."""

    irreps_in: Irreps
    irrep_out: Irreps = Irreps('0e')
    cueq_config: CuEquivarianceConfig | None = None

    def __init__(
        self,
        irreps_in: Irreps,
        irrep_out: Irreps = Irreps('0e'),
        cueq_config: CuEquivarianceConfig | None = None,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.irreps_in = Irreps(irreps_in)
        self.irrep_out = Irreps(irrep_out)
        self.cueq_config = cueq_config
        self.linear = Linear(
            irreps_in=self.irreps_in,
            irreps_out=self.irrep_out,
            cueq_config=self.cueq_config,
            rngs=rngs,
        )

    def __call__(
        self,
        x: jnp.ndarray,
        heads: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        del heads  # maintained for Torch parity
        return self.linear(x)

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}(irreps_in={self.irreps_in}, '
            f'irrep_out={self.irrep_out}, cueq_config={self.cueq_config})'
        )


@nxx_register_module('mace.modules.blocks.NonLinearReadoutBlock')
@nxx_auto_import_from_torch(allow_missing_mapper=True)
class NonLinearReadoutBlock(nnx.Module):
    """Two-layer readout with optional multi-head masking."""

    irreps_in: Irreps
    MLP_irreps: Irreps
    gate: Callable | None
    irrep_out: Irreps = Irreps('0e')
    num_heads: int = 1
    cueq_config: CuEquivarianceConfig | None = None

    def __init__(
        self,
        irreps_in: Irreps,
        MLP_irreps: Irreps,
        gate: Callable | None,
        irrep_out: Irreps = Irreps('0e'),
        num_heads: int = 1,
        cueq_config: CuEquivarianceConfig | None = None,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.irreps_in = Irreps(irreps_in)
        self.MLP_irreps = Irreps(MLP_irreps)
        self.gate = gate
        self.irrep_out = Irreps(irrep_out)
        self.num_heads = num_heads
        self.cueq_config = cueq_config
        self.hidden_irreps = self.MLP_irreps
        self.linear_1 = Linear(
            irreps_in=self.irreps_in,
            irreps_out=self.hidden_irreps,
            cueq_config=self.cueq_config,
            rngs=rngs,
        )
        self.non_linearity = nn.Activation(
            irreps_in=self.hidden_irreps,
            acts=[self.gate],
        )
        self.linear_2 = Linear(
            irreps_in=self.hidden_irreps,
            irreps_out=self.irrep_out,
            cueq_config=self.cueq_config,
            rngs=rngs,
        )

    def __call__(
        self,
        x: IrrepsArray,
        heads: jnp.ndarray | None = None,
    ) -> IrrepsArray:
        x = self.non_linearity(self.linear_1(x))
        if self.num_heads > 1 and heads is not None:
            x = mask_head(x, heads, self.num_heads)
        return self.linear_2(x)


@nxx_register_module('mace.modules.blocks.NonLinearBiasReadoutBlock')
@nxx_auto_import_from_torch(allow_missing_mapper=True)
class NonLinearBiasReadoutBlock(nnx.Module):
    """Non-linear readout with intermediate bias linear layers."""

    irreps_in: Irreps
    MLP_irreps: Irreps
    gate: Callable | None
    irrep_out: Irreps = Irreps('0e')
    num_heads: int = 1
    cueq_config: CuEquivarianceConfig | None = None

    def __init__(
        self,
        irreps_in: Irreps,
        MLP_irreps: Irreps,
        gate: Callable | None,
        irrep_out: Irreps = Irreps('0e'),
        num_heads: int = 1,
        cueq_config: CuEquivarianceConfig | None = None,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.irreps_in = Irreps(irreps_in)
        self.MLP_irreps = Irreps(MLP_irreps)
        self.gate = gate
        self.irrep_out = Irreps(irrep_out)
        self.num_heads = num_heads
        self.cueq_config = cueq_config
        self.hidden_irreps = self.MLP_irreps
        self.linear_1 = Linear(
            irreps_in=self.irreps_in,
            irreps_out=self.hidden_irreps,
            cueq_config=self.cueq_config,
            rngs=rngs,
        )
        self.non_linearity_1 = nn.Activation(
            irreps_in=self.hidden_irreps,
            acts=[self.gate],
        )
        self.linear_mid = Linear(
            irreps_in=self.hidden_irreps,
            irreps_out=self.hidden_irreps,
            cueq_config=self.cueq_config,
            rngs=rngs,
        )
        self.non_linearity_2 = nn.Activation(
            irreps_in=self.hidden_irreps,
            acts=[self.gate],
        )
        self.linear_2 = Linear(
            irreps_in=self.hidden_irreps,
            irreps_out=self.irrep_out,
            cueq_config=self.cueq_config,
            rngs=rngs,
        )

    def __call__(
        self,
        x: IrrepsArray,
        heads: jnp.ndarray | None = None,
    ) -> IrrepsArray:
        x = self.non_linearity_1(self.linear_1(x))
        x = self.non_linearity_2(self.linear_mid(x))
        if self.num_heads > 1 and heads is not None:
            x = mask_head(x, heads, self.num_heads)
        return self.linear_2(x)


@nxx_register_module('mace.modules.blocks.LinearDipoleReadoutBlock')
@nxx_auto_import_from_torch(allow_missing_mapper=True)
class LinearDipoleReadoutBlock(nnx.Module):
    """Linear readout block for dipoles or scalar+dipole."""

    irreps_in: Irreps
    dipole_only: bool = False
    cueq_config: CuEquivarianceConfig | None = None

    def __init__(
        self,
        irreps_in: Irreps,
        dipole_only: bool = False,
        cueq_config: CuEquivarianceConfig | None = None,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.irreps_in = Irreps(irreps_in)
        self.dipole_only = dipole_only
        self.cueq_config = cueq_config
        if self.dipole_only:
            self.irreps_out = Irreps('1x1o')
        else:
            self.irreps_out = Irreps('1x0e + 1x1o')

        self.linear = Linear(
            irreps_in=self.irreps_in,
            irreps_out=self.irreps_out,
            cueq_config=self.cueq_config,
            rngs=rngs,
        )

    def __call__(self, x: IrrepsArray) -> IrrepsArray:
        return self.linear(x)


@nxx_register_module('mace.modules.blocks.NonLinearDipoleReadoutBlock')
@nxx_auto_import_from_torch(allow_missing_mapper=True)
class NonLinearDipoleReadoutBlock(nnx.Module):
    """Non-linear readout block for dipoles or scalar+dipole."""

    irreps_in: Irreps
    MLP_irreps: Irreps
    gate: Callable
    dipole_only: bool = False
    cueq_config: CuEquivarianceConfig | None = None

    def __init__(
        self,
        irreps_in: Irreps,
        MLP_irreps: Irreps,
        gate: Callable,
        dipole_only: bool = False,
        cueq_config: CuEquivarianceConfig | None = None,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.irreps_in = Irreps(irreps_in)
        self.MLP_irreps = Irreps(MLP_irreps)
        self.gate = gate
        self.dipole_only = dipole_only
        self.cueq_config = cueq_config
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
            rngs=rngs,
        )
        self.linear_2 = Linear(
            irreps_in=self.hidden_irreps,
            irreps_out=self.irreps_out,
            cueq_config=self.cueq_config,
            rngs=rngs,
        )

    def __call__(self, x: IrrepsArray) -> IrrepsArray:
        x = self.equivariant_nonlin(self.linear_1(x))
        return self.linear_2(x)


@nxx_register_module('mace.modules.blocks.LinearDipolePolarReadoutBlock')
@nxx_auto_import_from_torch(allow_missing_mapper=True)
class LinearDipolePolarReadoutBlock(nnx.Module):
    """Linear readout for dipole and polarizability."""

    irreps_in: Irreps
    use_polarizability: bool = True
    cueq_config: CuEquivarianceConfig | None = None

    def __init__(
        self,
        irreps_in: Irreps,
        use_polarizability: bool = True,
        cueq_config: CuEquivarianceConfig | None = None,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.irreps_in = Irreps(irreps_in)
        self.use_polarizability = use_polarizability
        self.cueq_config = cueq_config
        if not self.use_polarizability:
            raise ValueError(
                'LinearDipolePolarReadoutBlock requires use_polarizability=True.'
            )
        self.irreps_out = Irreps('2x0e + 1x1o + 1x2e')
        self.linear = Linear(
            irreps_in=self.irreps_in,
            irreps_out=self.irreps_out,
            cueq_config=self.cueq_config,
            rngs=rngs,
        )

    def __call__(self, x: IrrepsArray) -> IrrepsArray:
        return self.linear(x)


@nxx_register_module('mace.modules.blocks.NonLinearDipolePolarReadoutBlock')
@nxx_auto_import_from_torch(allow_missing_mapper=True)
class NonLinearDipolePolarReadoutBlock(nnx.Module):
    """Non-linear readout for dipole and polarizability with equivariant gate."""

    irreps_in: Irreps
    MLP_irreps: Irreps
    gate: Callable
    use_polarizability: bool = True
    cueq_config: CuEquivarianceConfig | None = None

    def __init__(
        self,
        irreps_in: Irreps,
        MLP_irreps: Irreps,
        gate: Callable,
        use_polarizability: bool = True,
        cueq_config: CuEquivarianceConfig | None = None,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.irreps_in = Irreps(irreps_in)
        self.MLP_irreps = Irreps(MLP_irreps)
        self.gate = gate
        self.use_polarizability = use_polarizability
        self.cueq_config = cueq_config
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
            rngs=rngs,
        )
        self.linear_2 = Linear(
            irreps_in=self.hidden_irreps,
            irreps_out=self.irreps_out,
            cueq_config=self.cueq_config,
            rngs=rngs,
        )

    def __call__(self, x: IrrepsArray) -> IrrepsArray:
        x = self.equivariant_nonlin(self.linear_1(x))
        return self.linear_2(x)


@nxx_register_module('mace.modules.blocks.AtomicEnergiesBlock')
@nxx_auto_import_from_torch(allow_missing_mapper=True)
class AtomicEnergiesBlock(nnx.Module):
    """Block that returns atomic energies from one-hot element vectors."""

    atomic_energies_init: np.ndarray | jnp.ndarray

    def __init__(
        self,
        atomic_energies_init: np.ndarray | jnp.ndarray,
        *,
        rngs: nnx.Rngs | None = None,
    ) -> None:
        self.atomic_energies_init = atomic_energies_init
        init_values = jnp.asarray(self.atomic_energies_init, dtype=default_dtype())
        self.atomic_energies = nnx.Param(init_values)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        atomic_energies = self.atomic_energies
        # Prevent atomic reference energies from receiving gradients during training.
        atomic_energies = jax.lax.stop_gradient(atomic_energies)
        energies = jnp.atleast_2d(atomic_energies)
        return jnp.matmul(x, energies.T)

    def __repr__(self) -> str:
        energies_np = np.array(self.atomic_energies_init)
        formatted_energies = ', '.join(
            '[' + ', '.join(f'{value:.4f}' for value in row) + ']'
            for row in np.atleast_2d(energies_np)
        )
        return f'{self.__class__.__name__}(energies=[{formatted_energies}])'


@nxx_register_module('mace.modules.blocks.RadialEmbeddingBlock')
@nxx_auto_import_from_torch(allow_missing_mapper=True)
class RadialEmbeddingBlock(nnx.Module):
    """Radial basis embedding block for edges."""

    r_max: float
    num_bessel: int
    num_polynomial_cutoff: int
    radial_type: str = 'bessel'
    distance_transform: str = 'None'
    apply_cutoff: bool = True

    def __init__(
        self,
        r_max: float,
        num_bessel: int,
        num_polynomial_cutoff: int,
        radial_type: str = 'bessel',
        distance_transform: str = 'None',
        apply_cutoff: bool = True,
        *,
        rngs: nnx.Rngs | None = None,
    ) -> None:
        self.r_max = r_max
        self.num_bessel = num_bessel
        self.num_polynomial_cutoff = num_polynomial_cutoff
        self.radial_type = radial_type
        self.distance_transform = distance_transform
        self.apply_cutoff = apply_cutoff
        if self.radial_type == 'bessel':
            self.basis_fn = BesselBasis(
                r_max=self.r_max,
                num_basis=self.num_bessel,
                rngs=rngs,
            )
        elif self.radial_type == 'gaussian':
            self.basis_fn = GaussianBasis(
                r_max=self.r_max,
                num_basis=self.num_bessel,
                rngs=rngs,
            )
        elif self.radial_type == 'chebyshev':
            self.basis_fn = ChebychevBasis(r_max=self.r_max, num_basis=self.num_bessel)
        else:
            raise ValueError(f'Unknown radial_type: {self.radial_type}')

        if self.distance_transform == 'Agnesi':
            self.distance_transform_module = AgnesiTransform(rngs=rngs)
        elif self.distance_transform == 'Soft':
            self.distance_transform_module = SoftTransform(rngs=rngs)
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
        node_attrs_index: jnp.ndarray | None = None,
    ):
        cutoff = self.cutoff_fn(edge_lengths)

        transformed_lengths = edge_lengths
        if self.distance_transform_module is not None:
            transformed_lengths = self.distance_transform_module(
                edge_lengths,
                node_attrs,
                edge_index,
                atomic_numbers,
                node_attrs_index=node_attrs_index,
            )

        radial = self.basis_fn(transformed_lengths)

        if self.apply_cutoff:
            return radial * cutoff, None
        return radial, cutoff


@nxx_register_module('mace.modules.blocks.EquivariantProductBasisBlock')
@nxx_auto_import_from_torch(allow_missing_mapper=True)
class EquivariantProductBasisBlock(nnx.Module):
    node_feats_irreps: Irreps
    target_irreps: Irreps
    correlation: int
    use_sc: bool = True
    num_elements: int | None = None
    use_agnostic_product: bool = False
    use_reduced_cg: bool | None = None
    cueq_config: CuEquivarianceConfig | None = None

    def __init__(
        self,
        node_feats_irreps: Irreps,
        target_irreps: Irreps,
        correlation: int,
        use_sc: bool = True,
        num_elements: int | None = None,
        use_agnostic_product: bool = False,
        use_reduced_cg: bool | None = None,
        cueq_config: CuEquivarianceConfig | None = None,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.node_feats_irreps = Irreps(node_feats_irreps)
        self.target_irreps = Irreps(target_irreps)
        self.correlation = correlation
        self.use_sc = use_sc
        self.num_elements = num_elements
        self.use_agnostic_product = use_agnostic_product
        self.use_reduced_cg = use_reduced_cg
        self.cueq_config = cueq_config

        num_elements_local = self.num_elements
        if self.use_agnostic_product:
            num_elements_local = 1

        self.symmetric_contractions = SymmetricContractionWrapper(
            irreps_in=self.node_feats_irreps,
            irreps_out=self.target_irreps,
            correlation=self.correlation,
            num_elements=num_elements_local,
            use_reduced_cg=self.use_reduced_cg,
            cueq_config=self.cueq_config,
            rngs=rngs,
        )

        self.linear = Linear(
            irreps_in=self.target_irreps,
            irreps_out=self.target_irreps,
            internal_weights=True,
            shared_weights=True,
            cueq_config=self.cueq_config,
            rngs=rngs,
        )

    def __call__(
        self,
        node_feats: jnp.ndarray,
        sc: jnp.ndarray | None,
        node_attrs: jnp.ndarray,
        node_attrs_index: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        if self.use_agnostic_product:
            node_attrs = jnp.ones((node_feats.shape[0], 1), dtype=node_feats.dtype)
            node_attrs_index = None

        use_cueq = False
        layout_str = getattr(self.cueq_config, 'layout_str', 'mul_ir')
        if self.cueq_config is not None:
            if self.cueq_config.enabled and (
                self.cueq_config.optimize_all or self.cueq_config.optimize_symmetric
            ):
                use_cueq = True

        if use_cueq:
            if (
                node_attrs_index is not None
                and getattr(node_attrs_index, 'ndim', 1) != 1
            ):
                node_attrs_index = None
            if node_attrs_index is None:
                index_attrs = jnp.argmax(node_attrs, axis=1).astype(jnp.int32)
            else:
                index_attrs = jnp.asarray(node_attrs_index, dtype=jnp.int32).reshape(-1)
            features = node_feats
            if layout_str == 'mul_ir':
                features = jnp.transpose(features, (0, 2, 1))
            features = features.reshape(features.shape[0], -1)
            node_feats = self.symmetric_contractions(features, index_attrs)
        else:
            node_feats = self.symmetric_contractions(node_feats, node_attrs)

        if self.use_sc and sc is not None:
            return self.linear(node_feats) + sc
        return self.linear(node_feats)


class InteractionBlock(nnx.Module, metaclass=abc.ABCMeta):
    """Abstract base class for interaction blocks in equivariant GNNs."""

    node_attrs_irreps: Irreps
    node_feats_irreps: Irreps
    edge_attrs_irreps: Irreps
    edge_feats_irreps: Irreps
    target_irreps: Irreps
    hidden_irreps: Irreps
    avg_num_neighbors: float
    edge_irreps: Irreps | None = None
    radial_MLP: Sequence[int] | None = None
    cueq_config: CuEquivarianceConfig | None = None

    def __init__(
        self,
        node_attrs_irreps: Irreps,
        node_feats_irreps: Irreps,
        edge_attrs_irreps: Irreps,
        edge_feats_irreps: Irreps,
        target_irreps: Irreps,
        hidden_irreps: Irreps,
        avg_num_neighbors: float,
        edge_irreps: Irreps | None = None,
        radial_MLP: Sequence[int] | None = None,
        cueq_config: CuEquivarianceConfig | None = None,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.node_attrs_irreps = Irreps(node_attrs_irreps)
        self.node_feats_irreps = Irreps(node_feats_irreps)
        self.edge_attrs_irreps = Irreps(edge_attrs_irreps)
        self.edge_feats_irreps = Irreps(edge_feats_irreps)
        self.target_irreps = Irreps(target_irreps)
        self.hidden_irreps = Irreps(hidden_irreps)
        self.avg_num_neighbors = avg_num_neighbors
        self.cueq_config = cueq_config

        if radial_MLP is not None:
            self.radial_MLP = list(radial_MLP)
        else:
            self.radial_MLP = [64, 64, 64]

        if edge_irreps is not None:
            self.edge_irreps = Irreps(edge_irreps)
        else:
            self.edge_irreps = Irreps(self.node_feats_irreps)

        if self.cueq_config and getattr(self.cueq_config, 'conv_fusion', None):
            self.conv_fusion = self.cueq_config.conv_fusion

        self._setup(rngs)

    @abc.abstractmethod
    def _setup(self, rngs: nnx.Rngs) -> None:
        raise NotImplementedError

    def truncate_ghosts(
        self,
        tensor: jnp.ndarray,
        n_real: int | None = None,
    ) -> jnp.ndarray:
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
        raise NotImplementedError


@nxx_register_module('mace.modules.blocks.RealAgnosticInteractionBlock')
@nxx_auto_import_from_torch(allow_missing_mapper=True)
class RealAgnosticInteractionBlock(InteractionBlock):
    def _setup(self, rngs: nnx.Rngs) -> None:
        # First linear
        self.linear_up = Linear(
            self.node_feats_irreps,
            self.edge_irreps,
            internal_weights=True,
            shared_weights=True,
            cueq_config=self.cueq_config,
            rngs=rngs,
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
            rngs=rngs,
        )

        # Convolution weights network
        self.conv_tp_weights = nn.FullyConnectedNet(
            hs=[self.edge_feats_irreps.num_irreps]
            + list(self.radial_MLP)
            + [self.conv_tp.weight_numel],
            act=jax.nn.silu,
            rngs=rngs,
        )

        # Linear
        self.irreps_out = self.target_irreps
        self.linear = Linear(
            irreps_mid,
            self.irreps_out,
            internal_weights=True,
            shared_weights=True,
            cueq_config=self.cueq_config,
            rngs=rngs,
        )

        # Selector TensorProduct
        self.skip_tp = FullyConnectedTensorProduct(
            self.irreps_out,
            self.node_attrs_irreps,
            self.irreps_out,
            cueq_config=self.cueq_config,
            rngs=rngs,
        )
        self.reshape = reshape_irreps(self.irreps_out, cueq_config=self.cueq_config)

    def __call__(
        self,
        node_attrs: jnp.ndarray,
        node_feats: jnp.ndarray,
        edge_attrs: jnp.ndarray,
        edge_feats: jnp.ndarray,
        edge_index: jnp.ndarray,
        cutoff: jnp.ndarray | None = None,
        n_real: int | None = None,
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


@nxx_register_module('mace.modules.blocks.RealAgnosticResidualInteractionBlock')
@nxx_auto_import_from_torch(allow_missing_mapper=True)
class RealAgnosticResidualInteractionBlock(InteractionBlock):
    def _setup(self, rngs: nnx.Rngs) -> None:
        # First linear
        self.linear_up = Linear(
            self.node_feats_irreps,
            self.edge_irreps,
            internal_weights=True,
            shared_weights=True,
            cueq_config=self.cueq_config,
            rngs=rngs,
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
            rngs=rngs,
        )

        # Convolution weights network
        self.conv_tp_weights = nn.FullyConnectedNet(
            hs=[self.edge_feats_irreps.num_irreps]
            + list(self.radial_MLP)
            + [self.conv_tp.weight_numel],
            act=jax.nn.silu,
            rngs=rngs,
        )

        # Linear
        self.irreps_out = self.target_irreps
        self.linear = Linear(
            irreps_mid,
            self.irreps_out,
            internal_weights=True,
            shared_weights=True,
            cueq_config=self.cueq_config,
            rngs=rngs,
        )

        # Selector TensorProduct (skip connection)
        self.skip_tp = FullyConnectedTensorProduct(
            self.node_feats_irreps,
            self.node_attrs_irreps,
            self.hidden_irreps,
            cueq_config=self.cueq_config,
            rngs=rngs,
        )
        self.reshape = reshape_irreps(self.irreps_out, cueq_config=self.cueq_config)

    def __call__(
        self,
        node_attrs: jnp.ndarray,
        node_feats: jnp.ndarray,
        edge_attrs: jnp.ndarray,
        edge_feats: jnp.ndarray,
        edge_index: jnp.ndarray,
        cutoff: jnp.ndarray | None = None,
        n_real: int | None = None,
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


@nxx_register_module('mace.modules.blocks.RealAgnosticDensityInteractionBlock')
@nxx_auto_import_from_torch(allow_missing_mapper=True)
class RealAgnosticDensityInteractionBlock(InteractionBlock):
    def _setup(self, rngs: nnx.Rngs) -> None:
        # First linear
        self.linear_up = Linear(
            self.node_feats_irreps,
            self.edge_irreps,
            internal_weights=True,
            shared_weights=True,
            cueq_config=self.cueq_config,
            rngs=rngs,
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
            rngs=rngs,
        )

        # Convolution weights network
        self.conv_tp_weights = nn.FullyConnectedNet(
            hs=[self.edge_feats_irreps.num_irreps]
            + list(self.radial_MLP)
            + [self.conv_tp.weight_numel],
            act=jax.nn.silu,
            rngs=rngs,
        )

        # Linear projection
        self.irreps_out = self.target_irreps
        self.linear = Linear(
            irreps_mid,
            self.irreps_out,
            internal_weights=True,
            shared_weights=True,
            cueq_config=self.cueq_config,
            rngs=rngs,
        )

        # Selector TensorProduct (skip connection)
        self.skip_tp = FullyConnectedTensorProduct(
            self.irreps_out,
            self.node_attrs_irreps,
            self.irreps_out,
            cueq_config=self.cueq_config,
            rngs=rngs,
        )

        # Density normalization network
        self.density_fn = nn.FullyConnectedNet(
            hs=[self.edge_feats_irreps.num_irreps, 1],
            act=jax.nn.silu,
            rngs=rngs,
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
        cutoff: jnp.ndarray | None = None,
        n_real: int | None = None,
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


@nxx_register_module('mace.modules.blocks.RealAgnosticDensityResidualInteractionBlock')
@nxx_auto_import_from_torch(allow_missing_mapper=True)
class RealAgnosticDensityResidualInteractionBlock(InteractionBlock):
    def _setup(self, rngs: nnx.Rngs) -> None:
        # First linear
        self.linear_up = Linear(
            self.node_feats_irreps,
            self.edge_irreps,
            internal_weights=True,
            shared_weights=True,
            cueq_config=self.cueq_config,
            rngs=rngs,
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
            rngs=rngs,
        )

        # Convolution weights network
        self.conv_tp_weights = nn.FullyConnectedNet(
            hs=[self.edge_feats_irreps.num_irreps]
            + list(self.radial_MLP)
            + [self.conv_tp.weight_numel],
            act=jax.nn.silu,
            rngs=rngs,
        )

        # Linear projection
        self.irreps_out = self.target_irreps
        self.linear = Linear(
            irreps_mid,
            self.irreps_out,
            internal_weights=True,
            shared_weights=True,
            cueq_config=self.cueq_config,
            rngs=rngs,
        )

        # Selector TensorProduct (skip connection)
        self.skip_tp = FullyConnectedTensorProduct(
            self.node_feats_irreps,
            self.node_attrs_irreps,
            self.hidden_irreps,
            cueq_config=self.cueq_config,
            rngs=rngs,
        )

        # Density normalization network
        self.density_fn = nn.FullyConnectedNet(
            hs=[self.edge_feats_irreps.num_irreps, 1],
            act=jax.nn.silu,
            rngs=rngs,
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
        cutoff: jnp.ndarray | None = None,
        n_real: int | None = None,
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


@nxx_register_module('mace.modules.blocks.RealAgnosticAttResidualInteractionBlock')
@nxx_auto_import_from_torch(allow_missing_mapper=True)
class RealAgnosticAttResidualInteractionBlock(InteractionBlock):
    def _setup(self, rngs: nnx.Rngs) -> None:
        # Downsample irreps
        object.__setattr__(self, 'node_feats_down_irreps', Irreps('64x0e'))

        # First linear (up)
        self.linear_up = Linear(
            self.node_feats_irreps,
            self.edge_irreps,
            internal_weights=True,
            shared_weights=True,
            cueq_config=self.cueq_config,
            rngs=rngs,
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
            rngs=rngs,
        )

        # Linear (down)
        self.linear_down = Linear(
            self.node_feats_irreps,
            self.node_feats_down_irreps,
            internal_weights=True,
            shared_weights=True,
            cueq_config=self.cueq_config,
            rngs=rngs,
        )

        # Convolution weights network
        input_dim = (
            self.edge_feats_irreps.num_irreps
            + 2 * self.node_feats_down_irreps.num_irreps
        )
        self.conv_tp_weights = nn.FullyConnectedNet(
            hs=[input_dim] + [256, 256, 256] + [self.conv_tp.weight_numel],
            act=jax.nn.silu,
            rngs=rngs,
        )

        # Linear output
        self.irreps_out = self.target_irreps
        self.linear = Linear(
            irreps_mid,
            self.irreps_out,
            internal_weights=True,
            shared_weights=True,
            cueq_config=self.cueq_config,
            rngs=rngs,
        )

        # Output reshape
        self.reshape = reshape_irreps(self.irreps_out, cueq_config=self.cueq_config)

        # Skip connection
        self.skip_linear = Linear(
            self.node_feats_irreps,
            self.hidden_irreps,
            cueq_config=self.cueq_config,
            rngs=rngs,
        )

    def __call__(
        self,
        node_attrs: jnp.ndarray,
        node_feats: jnp.ndarray,
        edge_attrs: jnp.ndarray,
        edge_feats: jnp.ndarray,
        edge_index: jnp.ndarray,
        cutoff: jnp.ndarray | None = None,
        n_real: int | None = None,
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


@nxx_register_module(
    'mace.modules.blocks.RealAgnosticResidualNonLinearInteractionBlock'
)
@nxx_auto_import_from_torch(allow_missing_mapper=True)
class RealAgnosticResidualNonLinearInteractionBlock(InteractionBlock):
    def _setup(self, rngs: nnx.Rngs) -> None:
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
            rngs=rngs,
        )
        self.target_embedding = Linear(
            self.node_attrs_irreps,
            node_scalar_irreps,
            internal_weights=True,
            shared_weights=True,
            cueq_config=self.cueq_config,
            rngs=rngs,
        )

        # First linear
        self.linear_up = Linear(
            self.node_feats_irreps,
            self.edge_irreps,
            internal_weights=True,
            shared_weights=True,
            cueq_config=self.cueq_config,
            rngs=rngs,
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
            rngs=rngs,
        )

        # Convolution weights (Radial MLP)
        input_dim = self.edge_feats_irreps.num_irreps
        self.conv_tp_weights = RadialMLP(
            [input_dim + 2 * node_scalar_irreps.dim]
            + self.radial_MLP
            + [self.conv_tp.weight_numel],
            rngs=rngs,
        )

        # Output irreps
        self.irreps_out = self.target_irreps

        # Selector skip connection
        self.skip_tp = Linear(
            self.node_feats_irreps,
            self.hidden_irreps,
            cueq_config=self.cueq_config,
            rngs=rngs,
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
        )
        self.irreps_nonlin = self.equivariant_nonlin.irreps_in.simplify()

        # Linear residual
        self.linear_res = Linear(
            self.edge_irreps,
            self.irreps_nonlin,
            internal_weights=True,
            shared_weights=True,
            cueq_config=self.cueq_config,
            rngs=rngs,
        )

        # Linear blocks
        self.linear_1 = Linear(
            irreps_mid,
            self.irreps_nonlin,
            internal_weights=True,
            shared_weights=True,
            cueq_config=self.cueq_config,
            rngs=rngs,
        )
        self.linear_2 = Linear(
            self.irreps_out,
            self.irreps_out,
            internal_weights=True,
            shared_weights=True,
            cueq_config=self.cueq_config,
            rngs=rngs,
        )

        # Density normalization
        self.density_fn = RadialMLP(
            [input_dim + 2 * node_scalar_irreps.dim, 64, 1],
            rngs=rngs,
        )

        self.alpha = nnx.Param(jnp.array(20.0, dtype=default_dtype()))
        self.beta = nnx.Param(jnp.zeros((), dtype=default_dtype()))

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

    def __call__(
        self,
        node_attrs: jnp.ndarray,
        node_feats: jnp.ndarray,
        edge_attrs: jnp.ndarray,
        edge_feats: jnp.ndarray,
        edge_index: jnp.ndarray,
        cutoff: jnp.ndarray | None = None,
        n_real: int | None = None,
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
        alpha = jnp.asarray(self.alpha, dtype=message.dtype)
        beta = jnp.asarray(self.beta, dtype=message.dtype)

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


@nxx_register_module('mace.modules.blocks.ScaleShiftBlock')
@nxx_auto_import_from_torch(allow_missing_mapper=True)
class ScaleShiftBlock(nnx.Module):
    def __init__(self, scale: float | jnp.ndarray, shift: float | jnp.ndarray) -> None:
        self.scale = nnx.Param(jnp.asarray(scale, dtype=default_dtype()))
        self.shift = nnx.Param(jnp.asarray(shift, dtype=default_dtype()))

    def __call__(self, x: jnp.ndarray, head: jnp.ndarray) -> jnp.ndarray:
        # Match Torch behaviour (buffers) by keeping scale/shift constants during training.
        scale = jax.lax.stop_gradient(self.scale)
        shift = jax.lax.stop_gradient(self.shift)

        scale_h = jnp.atleast_1d(scale)[head]
        shift_h = jnp.atleast_1d(shift)[head]
        return scale_h * x + shift_h

    def __repr__(self) -> str:
        scale_vals = jnp.atleast_1d(jnp.asarray(self.scale))
        shift_vals = jnp.atleast_1d(jnp.asarray(self.shift))
        formatted_scale = ', '.join(f'{float(val):.4f}' for val in scale_vals)
        formatted_shift = ', '.join(f'{float(val):.4f}' for val in shift_vals)
        return f'{self.__class__.__name__}(scale={formatted_scale}, shift={formatted_shift})'
