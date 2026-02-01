# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utilities for working with dict[Irrep, Array] representation.

Vendored from cuequivariance_jax.ir_dict, with the segmented_polynomial import
routed through cuequivariance_jax to support older versions that lack the
ir_dict module.
"""

from string import ascii_lowercase, ascii_uppercase, digits
from typing import Any

import cuequivariance_jax as cuex
import jax
import jax.numpy as jnp
from jax import Array

import cuequivariance as cue
from cuequivariance import Irrep

segmented_polynomial = cuex.segmented_polynomial

__all__ = [
    'segmented_polynomial_uniform_1d',
    'assert_mul_ir_dict',
    'mul_ir_dict',
    'flat_to_dict',
    'dict_to_flat',
    'irreps_add',
    'irreps_zeros_like',
]


def _ensure_letter_capacity(min_len: int = 1024) -> None:
    """Extend cuequivariance operand letters to avoid repr crashes on errors."""
    try:
        from cuequivariance.segmented_polynomials import operation as sp_operation
    except Exception:
        return

    alphabet = ascii_lowercase + ascii_uppercase + digits
    if len(sp_operation.IVARS) < min_len:
        repeats = (min_len - len(sp_operation.IVARS) + len(alphabet) - 1) // len(
            alphabet
        )
        sp_operation.IVARS = sp_operation.IVARS + (alphabet * repeats)
    if len(sp_operation.OVARS) < min_len:
        repeats = (min_len - len(sp_operation.OVARS) + len(alphabet) - 1) // len(
            alphabet
        )
        sp_operation.OVARS = sp_operation.OVARS + (alphabet * repeats)


_ensure_letter_capacity()


def _ensure_letter_capacity(min_len: int = 1024) -> None:
    """Extend cuequivariance operand letters to avoid repr crashes on errors."""
    try:
        from cuequivariance.segmented_polynomials import operation as sp_operation
    except Exception:
        return

    alphabet = ascii_lowercase + ascii_uppercase + digits
    if len(sp_operation.IVARS) < min_len:
        repeats = (min_len - len(sp_operation.IVARS) + len(alphabet) - 1) // len(
            alphabet
        )
        sp_operation.IVARS = sp_operation.IVARS + (alphabet * repeats)
    if len(sp_operation.OVARS) < min_len:
        repeats = (min_len - len(sp_operation.OVARS) + len(alphabet) - 1) // len(
            alphabet
        )
        sp_operation.OVARS = sp_operation.OVARS + (alphabet * repeats)


_ensure_letter_capacity()


def segmented_polynomial_uniform_1d(
    polynomial: cue.SegmentedPolynomial,
    inputs: Any,
    outputs: Any = None,
    input_indices: Any = None,
    output_indices: Any = None,
    *,
    math_dtype: Any = None,
    name: str | None = None,
) -> Any:
    """Execute a segmented polynomial with uniform 1D method on tree-structured inputs/outputs.

    This function wraps cuex.segmented_polynomial with method="uniform_1d", handling
    the flattening/unflattening of pytree-structured inputs and outputs. It's designed
    to work with dict[Irrep, Array] representations where each array has shape
    (..., num_segments, *segment_shape).
    """

    def is_none(x):
        return x is None

    assert len(jax.tree.leaves(inputs, is_none)) == polynomial.num_inputs
    assert len(jax.tree.leaves(outputs, is_none)) == polynomial.num_outputs

    input_indices = jax.tree.broadcast(input_indices, inputs, is_none)
    output_indices = jax.tree.broadcast(output_indices, outputs, is_none)

    def flatten_input(i: int, desc: cue.SegmentedOperand, x: Array) -> Array:
        if not desc.all_same_segment_shape():
            raise ValueError(
                f'Input operand {i}: segments must have uniform shape.\n'
                f'  Descriptor: {desc}\n'
                f'  Segment shapes: {desc.segments}'
            )
        expected_suffix = (desc.num_segments,) + desc.segment_shape
        min_ndim = 1 + desc.ndim
        if x.ndim < min_ndim:
            raise ValueError(
                f'Input operand {i}: array has too few dimensions.\n'
                f'  Expected at least {min_ndim} dims (batch... + {expected_suffix})\n'
                f'  Got shape {x.shape} with {x.ndim} dims\n'
                f'  Descriptor: num_segments={desc.num_segments}, '
                f'segment_shape={desc.segment_shape}'
            )
        actual_suffix = x.shape[-(1 + desc.ndim) :]
        if actual_suffix != expected_suffix:
            raise ValueError(
                f'Input operand {i}: shape mismatch in trailing dimensions.\n'
                f'  Expected trailing dims: {expected_suffix} '
                f'(num_segments={desc.num_segments}, segment_shape={desc.segment_shape})\n'
                f'  Got trailing dims: {actual_suffix}\n'
                f'  Full array shape: {x.shape}\n'
                f'  Descriptor: {desc}'
            )
        return jnp.reshape(x, x.shape[: -(1 + desc.ndim)] + (desc.size,))

    list_inputs = jax.tree.leaves(inputs, is_none)
    assert all(isinstance(x, Array) for x in list_inputs)
    list_inputs = [
        flatten_input(i, desc, x)
        for i, (desc, x) in enumerate(zip(polynomial.inputs, list_inputs))
    ]

    shapes = []
    dtypes = []
    for x, i in zip(list_inputs, jax.tree.leaves(input_indices, is_none)):
        if i is None:
            shapes.append(x.shape[:-1])
        else:
            shapes.append(i.shape + x.shape[1:-1])
        dtypes.append(x.dtype)

    default_shape = jnp.broadcast_shapes(*shapes)
    default_dtype = jnp.result_type(*dtypes)

    def flatten_output(
        desc: cue.SegmentedOperand, x: Array | jax.ShapeDtypeStruct | None
    ) -> Array | None:
        assert desc.all_same_segment_shape()
        if isinstance(x, jax.ShapeDtypeStruct):
            x = jnp.zeros(x.shape, x.dtype)
        if x is None:
            x = jnp.zeros(
                default_shape + (desc.num_segments,) + desc.segment_shape, default_dtype
            )
        assert x.ndim >= 1 + desc.ndim, f'desc: {desc}, x.shape: {x.shape}'
        assert (
            x.shape[-(1 + desc.ndim) :] == (desc.num_segments,) + desc.segment_shape
        ), f'desc: {desc}, x.shape: {x.shape}'
        return jnp.reshape(x, x.shape[: -(1 + desc.ndim)] + (desc.size,))

    list_outputs = jax.tree.leaves(outputs, is_none)
    list_outputs = [
        flatten_output(desc, x) for desc, x in zip(polynomial.outputs, list_outputs)
    ]

    list_indices = jax.tree.leaves(input_indices, is_none) + jax.tree.leaves(
        output_indices, is_none
    )
    list_outputs = segmented_polynomial(
        polynomial,
        list_inputs,
        list_outputs,
        list_indices,
        method='uniform_1d',
        math_dtype=math_dtype,
        name=name,
    )

    def unflatten_output(desc: cue.SegmentedOperand, x: Array) -> Array:
        return jnp.reshape(x, x.shape[:-1] + (desc.num_segments,) + desc.segment_shape)

    list_outputs = [
        unflatten_output(desc, x) for desc, x in zip(polynomial.outputs, list_outputs)
    ]
    return jax.tree.unflatten(jax.tree.structure(outputs, is_none), list_outputs)


def assert_mul_ir_dict(irreps: cue.Irreps, x: dict[Irrep, Array]) -> None:
    """Assert that a dict[Irrep, Array] matches the expected irreps structure."""
    error_msg = (
        f'Dict {jax.tree.map(lambda v: v.shape, x)} does not match irreps {irreps}'
    )
    for (expected_mul, expected_ir), (actual_ir, actual_v) in zip(irreps, x.items()):
        assert actual_ir == expected_ir, error_msg
        assert actual_v.shape[-2:] == (expected_mul, expected_ir.dim), error_msg


def mul_ir_dict(irreps: cue.Irreps, data: Any) -> dict[Irrep, Any]:
    """Create a dict[Irrep, data] by broadcasting data to match irreps structure."""
    return jax.tree.broadcast(data, {ir: None for _, ir in irreps}, lambda v: v is None)


def flat_to_dict(
    irreps: cue.Irreps, data: Array, *, layout: str = 'mul_ir'
) -> dict[Irrep, Array]:
    """Convert a flat array to dict[Irrep, Array] with shape (..., mul, ir.dim)."""
    assert layout in ('mul_ir', 'ir_mul')
    result = {}
    offset = 0
    for mul, ir in irreps:
        size = mul * ir.dim
        segment = data[..., offset : offset + size]
        if layout == 'mul_ir':
            result[ir] = jnp.reshape(segment, data.shape[:-1] + (mul, ir.dim))
        else:  # ir_mul
            result[ir] = jnp.reshape(segment, data.shape[:-1] + (ir.dim, mul))
            result[ir] = jnp.swapaxes(result[ir], -2, -1)
        offset += size
    return result


def dict_to_flat(irreps: cue.Irreps, x: dict[Irrep, Array]) -> Array:
    """Convert dict[Irrep, Array] back to a flat contiguous array."""
    arrays = []
    for mul, ir in irreps:
        v = x[ir]
        arrays.append(jnp.reshape(v, v.shape[:-2] + (mul * ir.dim,)))
    return jnp.concatenate(arrays, axis=-1)


def irreps_add(x: dict[Irrep, Array], y: dict[Irrep, Array]) -> dict[Irrep, Array]:
    """Element-wise addition of two dict[Irrep, Array] representations."""
    assert x.keys() == y.keys()
    return {ir: x[ir] + y[ir] for ir in x.keys()}


def irreps_zeros_like(x: dict[Irrep, Array]) -> dict[Irrep, Array]:
    """Create a dict[Irrep, Array] of zeros with the same structure."""
    return {ir: jnp.zeros_like(v) for ir, v in x.items()}
