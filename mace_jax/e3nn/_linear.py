from math import prod
from typing import List, NamedTuple, Optional

import jax.numpy as jnp
from e3nn_jax import Irreps

from ._tensor_product._codegen import _sum_tensors


class Instruction(NamedTuple):
    i_in: int
    i_out: int
    path_shape: tuple
    path_weight: float


def _codegen_linear(
    x: jnp.ndarray,
    ws: jnp.ndarray,
    bs: Optional[jnp.ndarray],
    irreps_in: Irreps,
    irreps_out: Irreps,
    instructions: List["Instruction"],
    f_in: Optional[int] = None,
    f_out: Optional[int] = None,
    shared_weights: bool = False,
) -> jnp.ndarray:
    """
    JAX version of the linear codegen.
    Unlike the torch.fx version, this directly evaluates the computation.
    """

    if f_in is None:
        size = x.shape[:-1]
        outsize = size + (irreps_out.dim,)
    else:
        size = x.shape[:-2]
        outsize = size + (f_out, irreps_out.dim)

    # count how many bias elements are needed
    bias_numel = sum(irreps_out[i.i_out].dim for i in instructions if i.i_in == -1)
    if bias_numel > 0 and bs is not None:
        if f_out is None:
            bs = bs.reshape(-1, bias_numel)
        else:
            bs = bs.reshape(-1, f_out, bias_numel)

    # filter out empty instructions
    instructions = [ins for ins in instructions if 0 not in ins.path_shape]

    # short-circuit: nothing to do
    if len(instructions) == 0 and bias_numel == 0:
        return jnp.zeros(outsize, dtype=x.dtype)

    # reshape input
    if f_in is None:
        x = x.reshape(-1, irreps_in.dim)
    else:
        x = x.reshape(-1, f_in, irreps_in.dim)
    batch_out = x.shape[0]

    # reshape weights
    weight_numel = sum(prod(ins.path_shape) for ins in instructions if ins.i_in != -1)
    if weight_numel > 0:
        if f_in is None:
            ws = ws.reshape(-1, weight_numel)
        else:
            ws = ws.reshape(-1, f_in, f_out, weight_numel)

    # extract individual input irreps
    if len(irreps_in) == 1:
        x_list = [
            x.reshape(
                batch_out,
                *(() if f_in is None else (f_in,)),
                irreps_in[0].mul,
                irreps_in[0].ir.dim,
            )
        ]
    else:
        x_list = [
            x[..., i.start : i.start + mul_ir.dim].reshape(
                batch_out,
                *(() if f_in is None else (f_in,)),
                mul_ir.mul,
                mul_ir.ir.dim,
            )
            for i, mul_ir in zip(irreps_in.slices(), irreps_in)
        ]

    z = "" if shared_weights else "z"

    flat_weight_index = 0
    flat_bias_index = 0

    out_list = []

    for ins in instructions:
        mul_ir_out = irreps_out[ins.i_out]

        if ins.i_in == -1:
            # bias
            b = bs[..., flat_bias_index : flat_bias_index + prod(ins.path_shape)]
            flat_bias_index += prod(ins.path_shape)
            out_list.append(
                (ins.path_weight * b).reshape(
                    1, *(() if f_out is None else (f_out,)), mul_ir_out.dim
                )
            )
        else:
            mul_ir_in = irreps_in[ins.i_in]

            # skip empty irreps
            if mul_ir_in.dim == 0 or mul_ir_out.dim == 0:
                continue

            # extract weight
            path_nweight = prod(ins.path_shape)
            if len(instructions) == 1:
                w = ws
            else:
                w = ws[..., flat_weight_index : flat_weight_index + path_nweight]
            flat_weight_index += path_nweight

            w = w.reshape(
                (() if shared_weights else (-1,))
                + (() if f_in is None else (f_in, f_out))
                + ins.path_shape
            )

            # einsum
            if f_in is None:
                ein_out = jnp.einsum(f"{z}uw,zui->zwi", w, x_list[ins.i_in])
            else:
                ein_out = jnp.einsum(f"{z}xyuw,zxui->zywi", w, x_list[ins.i_in])

            ein_out = ins.path_weight * ein_out

            out_list.append(
                ein_out.reshape(
                    batch_out, *(() if f_out is None else (f_out,)), mul_ir_out.dim
                )
            )

    # === collect outputs per irreps_out slot ===
    out = [
        _sum_tensors(
            [out for ins, out in zip(instructions, out_list) if ins.i_out == i_out],
            shape=(batch_out, *(() if f_out is None else (f_out,)), mul_ir_out.dim),
            like=x,
        )
        for i_out, mul_ir_out in enumerate(irreps_out)
        if mul_ir_out.mul > 0
    ]

    if len(out) > 1:
        out = jnp.concatenate(out, axis=-1)
    else:
        out = out[0]

    out = out.reshape(outsize)

    return out
