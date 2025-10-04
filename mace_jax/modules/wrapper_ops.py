"""
Wrapper class for o3.Linear that optionally uses cuet.Linear
"""

import dataclasses
from typing import Optional

import cuequivariance as cue
import jax.numpy as jnp
from e3nn_jax import Irreps

from mace_jax.cuequivariance import (
    FullyConnectedTensorProduct as CueFullyConnectedTensorProduct,
)
from mace_jax.cuequivariance import (
    Linear as CueLinear,
)
from mace_jax.cuequivariance import (
    TensorProduct as CueTensorProduct,
)
from mace_jax.e3nn import _linear
from mace_jax.e3nn import _tensor_product as _tp
from mace_jax.modules.symmetric_contraction import SymmetricContraction
from mace_jax.tools.cg import O3_e3nn


@dataclasses.dataclass
class CuEquivarianceConfig:
    """Configuration for cuequivariance acceleration"""

    enabled: bool = False
    layout: str = 'mul_ir'  # One of: mul_ir, ir_mul
    layout_str: str = 'mul_ir'
    group: str = 'O3'
    optimize_all: bool = False  # Set to True to enable all optimizations
    optimize_linear: bool = False
    optimize_channelwise: bool = False
    optimize_symmetric: bool = False
    optimize_fctp: bool = False
    conv_fusion: bool = False  # Set to True to enable conv fusion

    def __post_init__(self):
        if self.enabled:
            self.layout_str = self.layout
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
        cueq_config: Optional[CuEquivarianceConfig] = None,
        name: Optional[str] = None,
    ):
        if cueq_config is not None and cueq_config.enabled:
            return CueLinear(
                irreps_in,
                irreps_out,
                shared_weights=shared_weights,
                internal_weights=internal_weights,
                cueq_config=cueq_config,
                name=name,
            )

        return _linear.Linear(
            irreps_in,
            irreps_out,
            shared_weights=shared_weights,
            internal_weights=internal_weights,
            name=name,
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
        name: Optional[str] = None,
    ):
        # --- Case 1: CuEquivariance backend ---
        if cueq_config is not None and cueq_config.enabled:
            if getattr(cueq_config, 'conv_fusion', False):
                raise NotImplementedError(
                    'conv_fusion is not supported by the cuequivariance tensor product backend.'
                )

            return CueTensorProduct(
                irreps_in1,
                irreps_in2,
                irreps_out,
                instructions=instructions,
                shared_weights=shared_weights,
                internal_weights=internal_weights,
                cueq_config=cueq_config,
                name=name,
            )

        # --- Default: fallback to e3nn_jax.TensorProduct ---
        return _tp.TensorProduct(
            irreps_in1,
            irreps_in2,
            irreps_out,
            instructions=instructions,
            shared_weights=shared_weights,
            internal_weights=internal_weights,
            name=name,
        )


def FullyConnectedTensorProduct(
    irreps_in1: Irreps,
    irreps_in2: Irreps,
    irreps_out: Irreps,
    shared_weights: bool = True,
    internal_weights: bool = True,
    cueq_config: Optional[CuEquivarianceConfig] = None,
    name: Optional[str] = None,
):
    """
    Wrapper around o3.FullyConnectedTensorProduct (JAX version).
    When CuEquivariance acceleration is requested, this raises since a JAX binding
    is not yet available; otherwise defaults to the e3nn_jax implementation.
    """
    if cueq_config is not None and cueq_config.enabled:
        if getattr(cueq_config, 'conv_fusion', False):
            raise NotImplementedError(
                'conv_fusion is not supported by the cuequivariance tensor product backend.'
            )
        return CueFullyConnectedTensorProduct(
            irreps_in1,
            irreps_in2,
            irreps_out,
            shared_weights=shared_weights,
            internal_weights=internal_weights,
            cueq_config=cueq_config,
            name=name,
        )

    # Default: e3nn_jax implementation
    return _tp.FullyConnectedTensorProduct(
        irreps_in1,
        irreps_in2,
        irreps_out,
        shared_weights=shared_weights,
        internal_weights=internal_weights,
        name=name,
    )


def SymmetricContractionWrapper(
    irreps_in: Irreps,
    irreps_out: Irreps,
    correlation: int,
    num_elements: Optional[int] = None,
    cueq_config: Optional['CuEquivarianceConfig'] = None,
    use_reduced_cg: bool = False,
    name: Optional[str] = None,
):
    """
    JAX implementation of SymmetricContraction powered by cuequivariance-jax.

    ``use_reduced_cg`` is accepted for API compatibility but ignored because the
    cue backend always operates on the full CG tables.
    """

    if use_reduced_cg:
        raise NotImplementedError(
            'use_reduced_cg is not supported by the JAX symmetric contraction backend.'
        )

    method = 'naive'
    if cueq_config is not None and cueq_config.enabled:
        if cueq_config.layout_str not in {'mul_ir', 'ir_mul'}:
            raise ValueError(
                f"Unsupported cuequivariance layout '{cueq_config.layout_str}'."
            )

    return SymmetricContraction(
        irreps_in=irreps_in,
        irreps_out=irreps_out,
        correlation=correlation,
        num_elements=num_elements,
        method=method,
        name=name,
    )


class TransposeIrrepsLayoutWrapper:
    """Wrapper around cuex.TransposeIrrepsLayout"""

    def __new__(
        cls,
        irreps: Irreps,
        source: str,
        target: str,
        cueq_config: Optional[CuEquivarianceConfig] = None,
    ):
        if cueq_config is None or not cueq_config.enabled:
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
