"""Thin wrappers around cuequivariance_jax.ir_dict helpers."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
from e3nn_jax import Irreps

import cuequivariance as cue

from . import ir_dict_vendor as ir_dict_local
from .utility import ir_mul_to_mul_ir, mul_ir_to_ir_mul

IrDict = dict[cue.Irrep, jnp.ndarray]

IR_DICT = ir_dict_local


def _layout(layout_str: str) -> str:
    layout = layout_str.lower()
    if layout not in {'mul_ir', 'ir_mul'}:
        raise ValueError(f"Unsupported layout_str '{layout_str}'.")
    return layout


def _cue_irreps(irreps: Irreps, *, group: object) -> cue.Irreps:
    return cue.Irreps(group, Irreps(irreps))


def is_ir_dict(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    if not value:
        return False
    return all(isinstance(k, cue.Irrep) for k in value.keys())


def mul_ir_to_ir_dict(
    irreps: Irreps,
    array: jnp.ndarray,
    *,
    group: object = cue.O3,
    layout_str: str = 'mul_ir',
) -> IrDict:
    tensor = getattr(array, 'array', array)
    cue_irreps = _cue_irreps(irreps, group=group)
    layout = _layout(layout_str)
    if layout == 'ir_mul':
        # Convert ir_mul -> mul_ir for consistent block handling.
        tensor = ir_mul_to_mul_ir(jnp.asarray(tensor), Irreps(irreps))
        layout = 'mul_ir'

    return IR_DICT.flat_to_dict(cue_irreps, tensor, layout=layout)


def ir_dict_to_mul_ir(
    irreps: Irreps,
    feats: IrDict,
    *,
    group: object = cue.O3,
    layout_str: str = 'mul_ir',
) -> jnp.ndarray:
    cue_irreps = _cue_irreps(irreps, group=group)
    layout = _layout(layout_str)
    if not feats:
        return jnp.zeros((0, cue_irreps.dim))

    IR_DICT.assert_mul_ir_dict(cue_irreps, feats)
    want_ir_mul = layout == 'ir_mul'
    if want_ir_mul:
        layout = 'mul_ir'

    out = IR_DICT.dict_to_flat(cue_irreps, feats)
    return mul_ir_to_ir_mul(out, Irreps(irreps)) if want_ir_mul else out
