"""
Wrapper class for o3.Linear that optionally uses cuet.Linear
"""

import dataclasses
import warnings
from typing import Optional

import cuequivariance as cue
import jax.numpy as jnp
from e3nn_jax import Irreps  # type: ignore
from flax import nnx

from mace_jax.adapters.cuequivariance import (
    FullyConnectedTensorProduct as CueFullyConnectedTensorProduct,
)
from mace_jax.adapters.cuequivariance import Linear as CueLinear
from mace_jax.adapters.cuequivariance import (
    SymmetricContraction as CueSymmetricContraction,
)
from mace_jax.adapters.cuequivariance import TensorProduct as CueTensorProduct
from mace_jax.tools.cg import O3_e3nn

_SUPPORTED_CUE_GROUPS = {'O3', 'O3_e3nn'}


def _group_name(group_value: object | None) -> str | None:
    if group_value is None:
        return None
    if isinstance(group_value, str):
        return group_value
    return getattr(group_value, '__name__', None) or str(group_value)


def _resolve_cue_group(cueq_config: 'CuEquivarianceConfig | None') -> object:
    if cueq_config is None:
        return cue.O3
    group_value = getattr(cueq_config, 'group', None)
    if group_value is None:
        return cue.O3
    if isinstance(group_value, str):
        if group_value == 'O3_e3nn':
            return O3_e3nn
        try:
            return getattr(cue, group_value)
        except AttributeError as exc:
            raise ValueError(
                f"Unsupported cuequivariance group '{group_value}'."
            ) from exc
    return group_value


def _validate_cue_group(group_value: object | None, *, context: str) -> None:
    name = _group_name(group_value)
    if name is None:
        return
    if name not in _SUPPORTED_CUE_GROUPS:
        raise ValueError(
            f"{context} only supports the 'O3' or 'O3_e3nn' groups; "
            f'received {group_value!r}.'
        )


@dataclasses.dataclass
class OpenEquivarianceConfig:
    """Configuration for the supported OpenEquivariance JAX operations.

    Fully connected tensor products are intentionally unavailable.  The
    OpenEquivariance 0.6.8 backward kernel is order-dependent when multiple
    FCTP problems execute in one process and can silently corrupt force
    derivatives.  Keep ``optimize_fctp`` in the serialized schema so affected
    bundles fail explicitly instead of changing their meaning while loading.
    """

    enabled: bool = False
    group: str = 'O3_e3nn'
    optimize_all: bool = False
    optimize_linear: bool = False
    optimize_channelwise: bool = False
    optimize_symmetric: bool = False
    optimize_fctp: bool = False
    conv_fusion: bool = False

    def __post_init__(self) -> None:
        unsupported = []
        if self.optimize_linear:
            unsupported.append('optimize_linear')
        if self.optimize_symmetric:
            unsupported.append('optimize_symmetric')
        if self.optimize_fctp:
            unsupported.append('optimize_fctp (OpenEquivariance 0.6.8 backward '
                               'is not safe for general FCTP execution)')
        if unsupported:
            raise ValueError(
                'OpenEquivariance does not accelerate these operations: '
                f"{', '.join(unsupported)}."
            )
        if self.group != 'O3_e3nn':
            raise ValueError(
                "OpenEquivariance v1 requires group='O3_e3nn'; "
                f'received {self.group!r}.'
            )

    @property
    def channelwise_fusion(self) -> bool:
        return bool(
            self.enabled
            and self.conv_fusion
            and (self.optimize_all or self.optimize_channelwise)
        )

    def to_dict(self) -> dict[str, object]:
        return {
            'enabled': bool(self.enabled),
            'group': 'O3_e3nn',
            'optimize_all': bool(self.optimize_all),
            'optimize_linear': bool(self.optimize_linear),
            'optimize_channelwise': bool(self.optimize_channelwise),
            'optimize_symmetric': bool(self.optimize_symmetric),
            'optimize_fctp': bool(self.optimize_fctp),
            'conv_fusion': bool(self.conv_fusion),
        }


@dataclasses.dataclass
class CuEquivarianceConfig:
    """Configuration for cuequivariance acceleration"""

    enabled: bool = False
    group: str = 'O3'
    optimize_all: bool = False  # Set to True to enable all optimizations
    optimize_linear: bool = False
    optimize_channelwise: bool = False
    optimize_symmetric: bool = False
    optimize_fctp: bool = False
    conv_fusion: bool = False  # Set to True to enable conv fusion

    def __post_init__(self):
        if self.enabled:
            self.group = (
                O3_e3nn if self.group == 'O3_e3nn' else getattr(cue, self.group)
            )

    def to_dict(self) -> dict[str, object]:
        """Return a stable bundle representation without backend class objects."""
        return {
            'enabled': bool(self.enabled),
            'group': _group_name(self.group) or 'O3',
            'optimize_all': bool(self.optimize_all),
            'optimize_linear': bool(self.optimize_linear),
            'optimize_channelwise': bool(self.optimize_channelwise),
            'optimize_symmetric': bool(self.optimize_symmetric),
            'optimize_fctp': bool(self.optimize_fctp),
            'conv_fusion': bool(self.conv_fusion),
        }


@dataclasses.dataclass
class EquivarianceConfig:
    """Backend-neutral equivariance acceleration configuration."""

    layout: str = 'mul_ir'
    cueq_config: CuEquivarianceConfig | dict[str, object] | None = None
    openeq_config: OpenEquivarianceConfig | dict[str, object] | None = None

    def __post_init__(self) -> None:
        if self.layout not in {'mul_ir', 'ir_mul'}:
            raise ValueError(
                "layout must be either 'mul_ir' or 'ir_mul'; "
                f'received {self.layout!r}.'
            )
        if isinstance(self.cueq_config, dict):
            self.cueq_config = CuEquivarianceConfig(**self.cueq_config)
        if isinstance(self.openeq_config, dict):
            self.openeq_config = OpenEquivarianceConfig(**self.openeq_config)
        if self.openeq_config is not None and self.openeq_config.channelwise_fusion:
            cueq = self.cueq_config
            cue_channelwise = cueq is not None and (
                cueq.conv_fusion
                or (cueq.enabled and (cueq.optimize_all or cueq.optimize_channelwise))
            )
            if cue_channelwise:
                raise ValueError(
                    'OpenEquivariance and cuEquivariance cannot both be selected '
                    'for the channel-wise convolution.'
                )

    @property
    def layout_str(self) -> str:
        """Compatibility spelling used by existing layout-aware modules."""
        return self.layout

    def to_dict(self) -> dict[str, object]:
        """Return the canonical, backend-neutral bundle representation."""
        return {
            'layout': self.layout,
            'cueq_config': (
                self.cueq_config.to_dict() if self.cueq_config is not None else None
            ),
            'openeq_config': (
                self.openeq_config.to_dict() if self.openeq_config is not None else None
            ),
        }


def resolve_equivariance_config(
    equivariance_config: EquivarianceConfig | dict[str, object] | None = None,
    *,
    cueq_config: CuEquivarianceConfig | dict[str, object] | None = None,
) -> EquivarianceConfig | None:
    """Resolve the canonical config and migrate the deprecated model argument."""
    if equivariance_config is not None and cueq_config is not None:
        raise ValueError(
            'Specify only equivariance_config; cueq_config is a deprecated alias.'
        )
    if equivariance_config is not None:
        if isinstance(equivariance_config, EquivarianceConfig):
            return equivariance_config
        return EquivarianceConfig(**equivariance_config)
    if cueq_config is None:
        return None

    warnings.warn(
        'cueq_config is deprecated; pass equivariance_config instead.',
        DeprecationWarning,
        stacklevel=2,
    )
    if isinstance(cueq_config, CuEquivarianceConfig):
        return EquivarianceConfig(cueq_config=cueq_config)

    legacy = dict(cueq_config)
    nested_openeq = legacy.pop('openeq_config', None)
    layout = legacy.pop('layout', None)
    layout_str = legacy.pop('layout_str', None)
    if layout is not None and layout_str is not None and layout != layout_str:
        raise ValueError(
            'Legacy cueq_config has conflicting layout and layout_str values.'
        )
    layout = str(layout if layout is not None else layout_str or 'mul_ir')
    return EquivarianceConfig(
        layout=layout,
        cueq_config=CuEquivarianceConfig(**legacy),
        openeq_config=nested_openeq,
    )


class Linear:
    """Returns either a cuet.Linear or o3.Linear based on config"""

    def __new__(
        cls,
        irreps_in: Irreps,
        irreps_out: Irreps,
        shared_weights: bool = True,
        internal_weights: bool = True,
        equivariance_config: EquivarianceConfig | None = None,
        rngs: nnx.Rngs | None = None,
    ):
        cueq_config = (
            equivariance_config.cueq_config if equivariance_config is not None else None
        )
        group_value = getattr(cueq_config, 'group', None) if cueq_config else None
        _validate_cue_group(group_value, context='Linear')
        group = _resolve_cue_group(cueq_config) if cueq_config else None
        layout = (
            equivariance_config.layout
            if equivariance_config is not None
            else 'mul_ir'
        )

        linear_kwargs = dict(
            shared_weights=shared_weights,
            internal_weights=internal_weights,
            layout=layout,
        )
        if group is not None:
            linear_kwargs['group'] = group

        return CueLinear(
            irreps_in,
            irreps_out,
            rngs=rngs,
            **linear_kwargs,
        )


class TensorProduct:
    """Wrapper around o3.TensorProduct / cuequivariance_jax.segmented_polynomial"""

    def __new__(
        cls,
        irreps_in1: Irreps,
        irreps_in2: Irreps,
        irreps_out: Irreps,
        instructions=None,
        shared_weights: bool = False,
        internal_weights: bool = False,
        equivariance_config: EquivarianceConfig | None = None,
        rngs: nnx.Rngs | None = None,
    ):
        conv_fusion = False
        cueq_config = (
            equivariance_config.cueq_config if equivariance_config is not None else None
        )
        openeq_config = (
            equivariance_config.openeq_config
            if equivariance_config is not None
            else None
        )
        use_openeq = bool(
            openeq_config is not None
            and openeq_config.enabled
            and (openeq_config.optimize_all or openeq_config.optimize_channelwise)
        )
        if use_openeq:
            # Keep this import lazy: OpenEquivariance is an optional CUDA backend.
            from mace_jax.adapters.openequivariance import (
                TensorProduct as OpenEqTensorProduct,
            )

            return OpenEqTensorProduct(
                irreps_in1,
                irreps_in2,
                irreps_out,
                instructions=instructions,
                shared_weights=shared_weights,
                internal_weights=internal_weights,
                conv_fusion=openeq_config.conv_fusion,
                layout=equivariance_config.layout,
                group=openeq_config.group,
                rngs=rngs,
            )
        group_value = getattr(cueq_config, 'group', None) if cueq_config else None
        _validate_cue_group(group_value, context='TensorProduct')
        group = _resolve_cue_group(cueq_config) if cueq_config else None
        if cueq_config is not None:
            conv_fusion = bool(getattr(cueq_config, 'conv_fusion', False))
        tp_kwargs = dict(
            instructions=instructions,
            shared_weights=shared_weights,
            internal_weights=internal_weights,
            conv_fusion=conv_fusion,
        )
        if group is not None:
            tp_kwargs['group'] = group

        return CueTensorProduct(
            irreps_in1,
            irreps_in2,
            irreps_out,
            rngs=rngs,
            **tp_kwargs,
        )


def FullyConnectedTensorProduct(
    irreps_in1: Irreps,
    irreps_in2: Irreps,
    irreps_out: Irreps,
    shared_weights: bool = True,
    internal_weights: bool = True,
    equivariance_config: EquivarianceConfig | None = None,
    rngs: nnx.Rngs | None = None,
):
    """
    Wrapper around o3.FullyConnectedTensorProduct (JAX version).
    When CuEquivariance acceleration is requested, this raises since a JAX binding
    is not yet available; otherwise defaults to the e3nn_jax implementation.
    """
    cueq_config = (
        equivariance_config.cueq_config if equivariance_config is not None else None
    )
    use_cue = (
        cueq_config is not None
        and getattr(cueq_config, 'enabled', False)
        and (
            getattr(cueq_config, 'optimize_all', False)
            or getattr(cueq_config, 'optimize_symmetric', False)
        )
    )
    # conv_fusion can be toggled independently (enabled stays False) so that the
    # tensor product backend switches to cue while symmetric contraction remains
    # on the pure-JAX implementation, matching the Torch wrapper semantics.
    group_value = getattr(cueq_config, 'group', None) if cueq_config else None
    if use_cue or cueq_config is not None:
        _validate_cue_group(group_value, context='FullyConnectedTensorProduct')
    group = _resolve_cue_group(cueq_config) if cueq_config else None

    fctp_kwargs = dict(
        shared_weights=shared_weights,
        internal_weights=internal_weights,
    )
    if group is not None:
        fctp_kwargs['group'] = group

    return CueFullyConnectedTensorProduct(
        irreps_in1,
        irreps_in2,
        irreps_out,
        rngs=rngs,
        **fctp_kwargs,
    )


def SymmetricContractionWrapper(
    irreps_in: Irreps,
    irreps_out: Irreps,
    correlation: int,
    num_elements: int | None = None,
    equivariance_config: Optional['EquivarianceConfig'] = None,
    use_reduced_cg: bool = True,
    rngs: nnx.Rngs | None = None,
):
    """
    JAX implementation of SymmetricContraction powered by cuequivariance-jax.
    """

    cueq_config = (
        equivariance_config.cueq_config if equivariance_config is not None else None
    )
    use_cue = cueq_config is not None and getattr(cueq_config, 'enabled', False)

    group_value = getattr(cueq_config, 'group', None) if cueq_config else None
    if cueq_config is not None:
        _validate_cue_group(group_value, context='SymmetricContraction')
    group = _resolve_cue_group(cueq_config) if cueq_config else None
    if equivariance_config is not None and equivariance_config.layout not in {
        'mul_ir',
        'ir_mul',
    }:
        raise ValueError(
            f"Unsupported equivariance layout '{equivariance_config.layout}'."
        )

    input_layout = (
        equivariance_config.layout
        if equivariance_config is not None
        else 'mul_ir'
    )

    sc_kwargs = dict(
        correlation=correlation,
        num_elements=num_elements,
        use_reduced_cg=use_reduced_cg,
        input_layout=input_layout,
    )
    if group is not None:
        sc_kwargs['group'] = group

    return CueSymmetricContraction(
        irreps_in=irreps_in,
        irreps_out=irreps_out,
        rngs=rngs,
        **sc_kwargs,
    )


class TransposeIrrepsLayoutWrapper:
    """Wrapper around cuex.TransposeIrrepsLayout"""

    def __new__(
        cls,
        irreps: Irreps,
        source: str,
        target: str,
        equivariance_config: EquivarianceConfig | None = None,
    ):
        # These boundary adapters are needed only while the model-wide
        # representation is ir_mul.  A mul_ir model already matches the
        # canonical layout consumed and produced by e3nn nonlinearities.
        if (
            equivariance_config is None
            or equivariance_config.layout != 'ir_mul'
        ):
            return None

        source = source.lower()
        target = target.lower()

        if source == target:
            return _IdentityTranspose()

        if {source, target} != {'mul_ir', 'ir_mul'}:
            raise ValueError(
                "TransposeIrrepsLayoutWrapper only supports conversions between 'mul_ir' and 'ir_mul' layouts"
                f' (got source={source!r}, target={target!r}).'
            )

        return _IrrepsLayoutTranspose(irreps=Irreps(irreps), swap_to=target)


class _IdentityTranspose:
    def __call__(self, tensor: jnp.ndarray) -> jnp.ndarray:
        return tensor


class _IrrepsLayoutTranspose:
    def __init__(self, *, irreps: Irreps, swap_to: str) -> None:
        self.irreps = irreps
        self.swap_to = swap_to

    def __call__(self, tensor: jnp.ndarray) -> jnp.ndarray:
        leading_shape = tensor.shape[:-1]
        offset = 0
        pieces = []

        for mul, ir in self.irreps:
            dim = ir.dim
            block_size = mul * dim
            if block_size == 0:
                continue

            segment = tensor[..., offset : offset + block_size]
            offset += block_size

            if self.swap_to == 'ir_mul':
                segment = segment.reshape(leading_shape + (mul, dim))
                segment = jnp.swapaxes(segment, -2, -1)
            else:  # target is 'mul_ir'
                segment = segment.reshape(leading_shape + (dim, mul))
                segment = jnp.swapaxes(segment, -2, -1)

            segment = segment.reshape(leading_shape + (block_size,))
            pieces.append(segment)

        if not pieces:
            return tensor

        return jnp.concatenate(pieces, axis=-1)
