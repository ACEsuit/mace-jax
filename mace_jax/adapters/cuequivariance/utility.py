"""Cue-equivariant tensor product implemented via segmented polynomials."""

from __future__ import annotations

import jax.numpy as jnp
from e3nn_jax import Irreps  # type: ignore


def _ensure_tuple_ints(value: tuple[int, ...], name: str) -> tuple[int, ...]:
    """Return ``value`` if it is a tuple of ints, otherwise raise ``TypeError``.

    The helper keeps validation for descriptor metadata in one place so callers
    can assume segment shapes are well-formed.  It preserves the original tuple
    to avoid extra allocations and fails loudly when a descriptor contains
    unexpected types.
    """

    if not isinstance(value, tuple) or not all(isinstance(x, int) for x in value):
        raise TypeError(f'{name} must be a tuple of ints, got {value!r}')
    return value


def collapse_ir_mul_segments(
    array: jnp.ndarray,
    descriptor_irreps: Irreps,
    target_irreps: Irreps,
    segment_shapes: tuple[tuple[int, ...], ...],
) -> jnp.ndarray:
    """Map ``array`` from descriptor layout to ``target_irreps`` ``ir_mul`` layout.

    Channel-wise cue descriptors expand each irrep block into
    ``(ir_dim, mul_in1, mul_in2)`` segments so that every u–v combination is
    explicit.  e3nn, however, expects multiplicities that correspond to the
    output irreps only.  This helper reshapes each block to
    ``(..., ir_dim, mul_in1, mul_in2)``, reduces across the redundant axis when
    necessary (normalising by ``sqrt(multiplicity)`` to preserve norms), and
    flattens back to ``(..., ir_dim * mul_out)`` so the tensor matches the
    target irreps while remaining in ``ir_mul`` order.
    """

    if descriptor_irreps == target_irreps:
        return array

    leading_shape = array.shape[:-1]
    flat = array.reshape(*leading_shape, descriptor_irreps.dim)

    if len(descriptor_irreps) != len(segment_shapes):
        raise ValueError(
            'Descriptor irreps and segment_shapes must have the same length '
            f'(got {len(descriptor_irreps)} and {len(segment_shapes)})'
        )

    if len(descriptor_irreps) != len(target_irreps):
        raise ValueError(
            'Descriptor and target irreps must have the same number of entries '
            f'(got {len(descriptor_irreps)} and {len(target_irreps)})'
        )

    blocks: list[jnp.ndarray] = []
    offset = 0

    for index, ((desc_mul, desc_ir), (target_mul, target_ir)) in enumerate(
        zip(descriptor_irreps, target_irreps)
    ):
        if desc_ir != target_ir:
            raise ValueError(
                'Descriptor and target irreps mismatch at index '
                f'{index}: {desc_ir} vs {target_ir}'
            )

        ir_dim = desc_ir.dim
        block_size = ir_dim * desc_mul
        block = flat[..., offset : offset + block_size]
        offset += block_size

        seg_shape = _ensure_tuple_ints(segment_shapes[index], 'segment_shapes')
        if len(seg_shape) != 3:
            raise ValueError(
                'Expected output segment shapes with three axes (ir, mul1, mul2), '
                f'got {seg_shape} at index {index}'
            )

        ir_axis, mul1, mul2 = seg_shape
        if ir_axis != ir_dim:
            raise ValueError(
                f'Irrep dimension mismatch at index {index}: segment has {ir_axis}, '
                f'expected {ir_dim}'
            )

        block = block.reshape(*leading_shape, ir_dim, mul1, mul2)

        if desc_mul == target_mul:
            if mul1 * mul2 != target_mul:
                raise ValueError(
                    f'Unexpected multiplicity {mul1 * mul2} for index {index}, '
                    f'expected {target_mul}'
                )
            block_ir_mul = block.reshape(*leading_shape, ir_dim, target_mul)
        elif desc_mul == target_mul * mul2:
            block_ir_mul = block.sum(axis=-1)
            block_ir_mul = block_ir_mul / jnp.sqrt(float(mul2))
            if block_ir_mul.shape[-1] != target_mul:
                raise ValueError(
                    f'Failed to collapse v-axis at index {index}: '
                    f'got {block_ir_mul.shape[-1]}, expected {target_mul}'
                )
        elif desc_mul == target_mul * mul1:
            block_ir_mul = block.sum(axis=-2)
            block_ir_mul = block_ir_mul / jnp.sqrt(float(mul1))
            if block_ir_mul.shape[-1] != target_mul:
                raise ValueError(
                    f'Failed to collapse u-axis at index {index}: '
                    f'got {block_ir_mul.shape[-1]}, expected {target_mul}'
                )
        else:
            raise ValueError(
                'Cannot map descriptor multiplicity to target multiplicity '
                f'at index {index}: desc_mul={desc_mul}, target_mul={target_mul}, '
                f'segment_shape={seg_shape}'
            )

        blocks.append(block_ir_mul.reshape(*leading_shape, ir_dim * target_mul))

    if offset != flat.shape[-1]:
        raise ValueError(
            'Processed elements do not cover the descriptor output '
            f'(covered {offset}, total {flat.shape[-1]})'
        )

    return jnp.concatenate(blocks, axis=-1) if blocks else flat[..., :0]


def mul_ir_to_ir_mul(array: jnp.ndarray, irreps: Irreps) -> jnp.ndarray:
    """Reorder the last axis of ``array`` from ``mul_ir`` to ``ir_mul`` layout.

    e3nn stores each irrep block as ``(mul, ir_dim)`` whereas cue expects
    ``(ir_dim, mul)``.  This routine performs the reshape–transpose–reshape dance
    for every block described by ``irreps`` so that downstream cue calls receive
    data in the correct memory order.
    """

    if irreps.dim == 0:
        return array

    leading_shape = array.shape[:-1]
    array = array.reshape(*leading_shape, irreps.dim)
    segments: list[jnp.ndarray] = []
    offset = 0
    for mul, ir in irreps:
        block = array[..., offset : offset + mul * ir.dim]
        offset += mul * ir.dim
        block = block.reshape(*leading_shape, mul, ir.dim)
        block = jnp.swapaxes(block, -1, -2)  # -> (..., ir_dim, mul)
        block = block.reshape(*leading_shape, ir.dim * mul)
        segments.append(block)
    return jnp.concatenate(segments, axis=-1) if segments else array


def ir_mul_to_mul_ir(array: jnp.ndarray, irreps: Irreps) -> jnp.ndarray:
    """Reorder the last axis of ``array`` from ``ir_mul`` back to ``mul_ir``.

    This is the inverse transformation of :func:`mul_ir_to_ir_mul`.  It restores
    e3nn’s block layout after a cue backend produced results in ``ir_mul``
    order, ensuring the shapes can flow directly into e3nn-style modules.
    """

    if irreps.dim == 0:
        return array

    leading_shape = array.shape[:-1]
    array = array.reshape(*leading_shape, irreps.dim)
    segments: list[jnp.ndarray] = []
    offset = 0
    for mul, ir in irreps:
        block = array[..., offset : offset + mul * ir.dim]
        offset += mul * ir.dim
        block = block.reshape(*leading_shape, ir.dim, mul)
        block = jnp.swapaxes(block, -1, -2)  # -> (..., mul, ir_dim)
        block = block.reshape(*leading_shape, mul * ir.dim)
        segments.append(block)
    return jnp.concatenate(segments, axis=-1) if segments else array
