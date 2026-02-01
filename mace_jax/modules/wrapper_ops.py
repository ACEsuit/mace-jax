"""
Wrapper class for o3.Linear that optionally uses cuet.Linear
"""

import dataclasses
from typing import Optional

import cuequivariance as cue
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
class CuEquivarianceConfig:
    """Configuration for cuequivariance acceleration"""

    enabled: bool = False
    layout: str = 'ir_mul'  # One of: mul_ir, ir_mul
    layout_str: str = 'ir_mul'
    group: str = 'O3'
    optimize_all: bool = False  # Set to True to enable all optimizations
    optimize_linear: bool = False
    optimize_channelwise: bool = False
    optimize_symmetric: bool = False
    optimize_fctp: bool = False
    conv_fusion: bool = False  # Set to True to enable conv fusion

    def __post_init__(self):
        if isinstance(self.layout, str):
            self.layout_str = self.layout
        else:
            self.layout_str = getattr(self.layout, 'name', None) or getattr(
                self.layout, '__name__', None
            )
            if self.layout_str is None:
                self.layout_str = str(self.layout)
        if self.enabled:
            if isinstance(self.layout, str):
                self.layout = getattr(cue, self.layout)
            self.group = (
                O3_e3nn if self.group == 'O3_e3nn' else getattr(cue, self.group)
            )


class Linear:
    """Returns either a cuet.Linear or o3.Linear based on config"""

    def __new__(
        cls,
        irreps_in: Irreps,
        irreps_out: Irreps,
        shared_weights: bool = True,
        internal_weights: bool = True,
        cueq_config: CuEquivarianceConfig | None = None,
        rngs: nnx.Rngs | None = None,
    ):
        group_value = getattr(cueq_config, 'group', None) if cueq_config else None
        _validate_cue_group(group_value, context='Linear')
        group = _resolve_cue_group(cueq_config) if cueq_config else None
        layout = (
            getattr(cueq_config, 'layout', 'mul_ir')
            if cueq_config is not None
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
        cueq_config=None,
        rngs: nnx.Rngs | None = None,
    ):
        conv_fusion = False
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
        if cueq_config is not None and getattr(cueq_config, 'layout', None) is not None:
            tp_kwargs['layout'] = getattr(cueq_config, 'layout')
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
    cueq_config: CuEquivarianceConfig | None = None,
    rngs: nnx.Rngs | None = None,
):
    """
    Wrapper around o3.FullyConnectedTensorProduct (JAX version).
    When CuEquivariance acceleration is requested, this raises since a JAX binding
    is not yet available; otherwise defaults to the e3nn_jax implementation.
    """
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
    if cueq_config is not None and getattr(cueq_config, 'layout', None) is not None:
        fctp_kwargs['layout'] = getattr(cueq_config, 'layout')
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
    cueq_config: Optional['CuEquivarianceConfig'] = None,
    use_reduced_cg: bool = True,
    rngs: nnx.Rngs | None = None,
):
    """
    JAX implementation of SymmetricContraction powered by cuequivariance-jax.
    """

    use_cue = cueq_config is not None and getattr(cueq_config, 'enabled', False)

    group_value = getattr(cueq_config, 'group', None) if cueq_config else None
    if cueq_config is not None:
        _validate_cue_group(group_value, context='SymmetricContraction')
    group = _resolve_cue_group(cueq_config) if cueq_config else None
    if cueq_config is not None and use_cue:
        if cueq_config.layout_str not in {'mul_ir', 'ir_mul'}:
            raise ValueError(
                f"Unsupported cuequivariance layout '{cueq_config.layout_str}'."
            )

    input_layout = 'mul_ir'
    if use_cue:
        # cuet.SymmetricContraction expects ir_mul inputs regardless of output layout.
        input_layout = 'ir_mul'
    output_layout = (
        getattr(cueq_config, 'layout_str', 'mul_ir') if cueq_config else 'mul_ir'
    )

    sc_kwargs = dict(
        correlation=correlation,
        num_elements=num_elements,
        use_reduced_cg=use_reduced_cg,
        input_layout=input_layout,
        output_layout=output_layout,
    )
    if group is not None:
        sc_kwargs['group'] = group

    return CueSymmetricContraction(
        irreps_in=irreps_in,
        irreps_out=irreps_out,
        rngs=rngs,
        **sc_kwargs,
    )
