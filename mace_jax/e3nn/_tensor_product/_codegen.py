from math import prod, sqrt
from typing import List

import jax
import jax.numpy as jnp
from e3nn.o3 import wigner_3j
from e3nn_jax import Irreps

from ._instruction import Instruction


def _sum_tensors(xs: list[jax.Array], shape: tuple[int, ...]) -> jax.Array:
    if len(xs) > 0:
        out = xs[0]
        for x in xs[1:]:
            out = out + x
        # Ensure output has the full shape
        return jnp.reshape(out, shape)
    # Return zeros with the correct 3D shape
    return jnp.zeros(shape, dtype=jnp.float32)


def codegen_tensor_product_left_right(
    x1: jax.Array,
    x2: jax.Array,
    weights: jax.Array,
    irreps_in1: Irreps,
    irreps_in2: Irreps,
    irreps_out: Irreps,
    instructions: List[Instruction],
    shared_weights: bool = False,
    specialized_code: bool = True,
) -> jax.Array:
    # Broadcast shapes
    if shared_weights:
        output_shape = jnp.broadcast_shapes(x1[..., :1].shape, x2[..., :1].shape)[:-1]
    else:
        output_shape = jnp.broadcast_shapes(
            x1[..., :1].shape, x2[..., :1].shape, weights[..., :1].shape
        )[:-1]

    # Short-circuit
    instructions = [ins for ins in instructions if 0 not in ins.path_shape]
    if len(instructions) == 0:
        return jnp.zeros(output_shape + (irreps_out.dim,))

    # Flatten inputs
    x1s = jnp.reshape(x1, (-1, irreps_in1.dim))
    x2s = jnp.reshape(x2, (-1, irreps_in2.dim))
    batch_numel = x1s.shape[0]

    # Extract irreps (pseudo-code, depends on how irreps_in1/2 are defined)
    x1_list = [
        x1s[:, i].reshape(batch_numel, mul_ir.mul, mul_ir.ir.dim)
        for i, mul_ir in zip(irreps_in1.slices(), irreps_in1)
    ]
    x2_list = [
        x2s[:, i].reshape(batch_numel, mul_ir.mul, mul_ir.ir.dim)
        for i, mul_ir in zip(irreps_in2.slices(), irreps_in2)
    ]

    weight_numel = sum(prod(ins.path_shape) for ins in instructions if ins.has_weight)

    if weight_numel > 0:
        # Flatten weights into (batch, total_weight) if not shared
        weights = jnp.reshape(
            weights, (-1, weight_numel)
        )  # shape: (batch?, total_weight)

    # If weights are shared (no batch dimension), add fake batch dimension
    if shared_weights and weights.ndim == 1:
        weights = weights[None, :]  # shape: (1, total_weight)

    # Cache of input irrep pairs whose outer products (xx) have already been computed
    xx_dict = dict()

    # Current index in the flat weight tensor
    flat_weight_index = 0

    outputs = []

    for ins in instructions:
        mul_ir_in1 = irreps_in1[ins.i_in1]
        mul_ir_in2 = irreps_in2[ins.i_in2]
        mul_ir_out = irreps_out[ins.i_out]

        # sanity checks
        assert mul_ir_in1.ir.p * mul_ir_in2.ir.p == mul_ir_out.ir.p
        assert (
            abs(mul_ir_in1.ir.l - mul_ir_in2.ir.l)
            <= mul_ir_out.ir.l
            <= mul_ir_in1.ir.l + mul_ir_in2.ir.l
        )

        if mul_ir_in1.dim == 0 or mul_ir_in2.dim == 0 or mul_ir_out.dim == 0:
            continue

        x1 = x1_list[ins.i_in1]
        x2 = x2_list[ins.i_in2]

        # weights
        if ins.has_weight:
            n_w = prod(ins.path_shape)
            w = jnp.reshape(
                weights[:, flat_weight_index : flat_weight_index + n_w],
                (() if shared_weights else (-1,)) + tuple(ins.path_shape),
            )
            flat_weight_index += n_w
        else:
            w = None

        # xx cache
        key = (ins.i_in1, ins.i_in2, ins.connection_mode[:2])
        if key not in xx_dict:
            if ins.connection_mode[:2] == "uu":
                xx_dict[key] = jnp.einsum("zui,zuj->zuij", x1, x2)
            else:
                xx_dict[key] = jnp.einsum("zui,zvj->zuvij", x1, x2)
        xx = xx_dict[key]

        # The einsum string index to prepend to the weights if the weights are not shared and have a batch dimension
        z = "" if shared_weights else "z"

        # Wigner 3j lookup
        l1, l2, l3 = mul_ir_in1.ir.l, mul_ir_in2.ir.l, mul_ir_out.ir.l
        w3j = jnp.array(wigner_3j(l1, l2, l3))

        # contraction modes
        if ins.connection_mode == "uvw":
            assert ins.has_weight
            if specialized_code and (l1, l2, l3) == (0, 0, 0):
                result = jnp.einsum(
                    f"{z}uvw,zu,zv->zw",
                    w,
                    jnp.reshape(x1, (batch_numel, mul_ir_in1.dim)),
                    jnp.reshape(x2, (batch_numel, mul_ir_in2.dim)),
                )
            elif specialized_code and mul_ir_in1.ir.l == 0:
                result = jnp.einsum(
                    f"{z}uvw,zu,zvj->zwj",
                    w,
                    jnp.reshape(x1, (batch_numel, mul_ir_in1.dim)),
                    x2,
                ) / sqrt(mul_ir_out.ir.dim)
            elif specialized_code and mul_ir_in2.ir.l == 0:
                result = jnp.einsum(
                    f"{z}uvw,zui,zv->zwi",
                    w,
                    x1,
                    jnp.reshape(x2, (batch_numel, mul_ir_in2.dim)),
                ) / sqrt(mul_ir_out.ir.dim)
            elif specialized_code and mul_ir_out.ir.l == 0:
                result = jnp.einsum(f"{z}uvw,zui,zvi->zw", w, x1, x2) / sqrt(
                    mul_ir_in1.ir.dim
                )
            else:
                result = jnp.einsum(f"{z}uvw,ijk,zuvij->zwk", w, w3j, xx)

        elif ins.connection_mode == "uvu":
            assert mul_ir_in1.mul == mul_ir_out.mul
            if ins.has_weight:
                if specialized_code and (l1, l2, l3) == (0, 0, 0):
                    result = jnp.einsum(
                        f"{z}uv,zu,zv->zu",
                        w,
                        jnp.reshape(x1, (batch_numel, mul_ir_in1.dim)),
                        jnp.reshape(x2, (batch_numel, mul_ir_in2.dim)),
                    )
                elif specialized_code and mul_ir_in1.ir.l == 0:
                    result = jnp.einsum(
                        f"{z}uv,zu,zvj->zuj",
                        w,
                        jnp.reshape(x1, (batch_numel, mul_ir_in1.dim)),
                        x2,
                    ) / jnp.sqrt(mul_ir_out.ir.dim)
                elif specialized_code and mul_ir_in2.ir.l == 0:
                    result = jnp.einsum(
                        f"{z}uv,zui,zv->zui",
                        w,
                        x1,
                        jnp.reshape(x2, (batch_numel, mul_ir_in2.dim)),
                    ) / jnp.sqrt(mul_ir_out.ir.dim)
                elif specialized_code and mul_ir_out.ir.l == 0:
                    result = jnp.einsum(f"{z}uv,zui,zvi->zu", w, x1, x2) / jnp.sqrt(
                        mul_ir_in1.ir.dim
                    )
                else:
                    result = jnp.einsum(f"{z}uv,ijk,zuvij->zuk", w, w3j, xx)
            else:
                # not so useful operation because v is summed
                result = jnp.einsum("ijk,zuvij->zuk", w3j, xx)

        elif ins.connection_mode == "uvv":
            assert mul_ir_in2.mul == mul_ir_out.mul
            if ins.has_weight:
                if specialized_code and (l1, l2, l3) == (0, 0, 0):
                    result = jnp.einsum(
                        f"{z}uv,zu,zv->zv",
                        w,
                        jnp.reshape(x1, (batch_numel, mul_ir_in1.dim)),
                        jnp.reshape(x2, (batch_numel, mul_ir_in2.dim)),
                    )
                elif specialized_code and mul_ir_in1.ir.l == 0:
                    result = jnp.einsum(
                        f"{z}uv,zu,zvj->zvj",
                        w,
                        jnp.reshape(x1, (batch_numel, mul_ir_in1.dim)),
                        x2,
                    ) / jnp.sqrt(mul_ir_out.ir.dim)
                elif specialized_code and mul_ir_in2.ir.l == 0:
                    result = jnp.einsum(
                        f"{z}uv,zui,zv->zvi",
                        w,
                        x1,
                        jnp.reshape(x2, (batch_numel, mul_ir_in2.dim)),
                    ) / jnp.sqrt(mul_ir_out.ir.dim)
                elif specialized_code and mul_ir_out.ir.l == 0:
                    result = jnp.einsum(f"{z}uv,zui,zvi->zv", w, x1, x2) / jnp.sqrt(
                        mul_ir_in1.ir.dim
                    )
                else:
                    result = jnp.einsum(f"{z}uv,ijk,zuvij->zvk", w, w3j, xx)
            else:
                # not so useful operation because u is summed
                if specialized_code and (l1, l2, l3) == (0, 0, 0):
                    result = jnp.einsum(
                        "zu,zv->zv",
                        jnp.reshape(x1, (batch_numel, mul_ir_in1.dim)),
                        jnp.reshape(x2, (batch_numel, mul_ir_in2.dim)),
                    )
                elif specialized_code and mul_ir_in1.ir.l == 0:
                    result = jnp.einsum(
                        "zu,zvj->zvj",
                        jnp.reshape(x1, (batch_numel, mul_ir_in1.dim)),
                        x2,
                    ) / jnp.sqrt(mul_ir_out.ir.dim)
                elif specialized_code and mul_ir_in2.ir.l == 0:
                    result = jnp.einsum(
                        "zui,zv->zvi",
                        x1,
                        jnp.reshape(x2, (batch_numel, mul_ir_in2.dim)),
                    ) / jnp.sqrt(mul_ir_out.ir.dim)
                elif specialized_code and mul_ir_out.ir.l == 0:
                    result = jnp.einsum("zui,zvi->zv", x1, x2) / jnp.sqrt(
                        mul_ir_in1.ir.dim
                    )
                else:
                    result = jnp.einsum("ijk,zuvij->zvk", w3j, xx)

        elif ins.connection_mode == "uuw":
            assert mul_ir_in1.mul == mul_ir_in2.mul
            if ins.has_weight:
                if specialized_code and (l1, l2, l3) == (0, 0, 0):
                    result = jnp.einsum(
                        f"{z}uw,zu,zu->zw",
                        w,
                        jnp.reshape(x1, (batch_numel, mul_ir_in1.dim)),
                        jnp.reshape(x2, (batch_numel, mul_ir_in2.dim)),
                    )
                elif specialized_code and mul_ir_in1.ir.l == 0:
                    result = jnp.einsum(
                        f"{z}uw,zu,zuj->zwj",
                        w,
                        jnp.reshape(x1, (batch_numel, mul_ir_in1.dim)),
                        x2,
                    ) / jnp.sqrt(mul_ir_out.ir.dim)
                elif specialized_code and mul_ir_in2.ir.l == 0:
                    result = jnp.einsum(
                        f"{z}uw,zui,zu->zwi",
                        w,
                        x1,
                        jnp.reshape(x2, (batch_numel, mul_ir_in2.dim)),
                    ) / jnp.sqrt(mul_ir_out.ir.dim)
                elif specialized_code and mul_ir_out.ir.l == 0:
                    result = jnp.einsum(f"{z}uw,zui,zui->zw", w, x1, x2) / jnp.sqrt(
                        mul_ir_in1.ir.dim
                    )
                else:
                    result = jnp.einsum(f"{z}uw,ijk,zuij->zwk", w, w3j, xx)
            else:
                # equivalent to tp(x, y, 'uuu').sum('u')
                assert mul_ir_out.mul == 1
                result = jnp.einsum("ijk,zuij->zk", w3j, xx)

        elif ins.connection_mode == "uuu":
            assert mul_ir_in1.mul == mul_ir_in2.mul == mul_ir_out.mul
            if ins.has_weight:
                if specialized_code and (l1, l2, l3) == (0, 0, 0):
                    result = jnp.einsum(
                        f"{z}u,zu,zu->zu",
                        w,
                        jnp.reshape(x1, (batch_numel, mul_ir_in1.dim)),
                        jnp.reshape(x2, (batch_numel, mul_ir_in2.dim)),
                    )
                elif specialized_code and (l1, l2, l3) == (1, 1, 1):
                    result = jnp.einsum(
                        f"{z}u,zui->zui", w, jnp.cross(x1, x2, axis=2)
                    ) / jnp.sqrt(2 * 3)
                elif specialized_code and mul_ir_in1.ir.l == 0:
                    result = jnp.einsum(
                        f"{z}u,zu,zuj->zuj",
                        w,
                        jnp.reshape(x1, (batch_numel, mul_ir_in1.dim)),
                        x2,
                    ) / jnp.sqrt(mul_ir_out.ir.dim)
                elif specialized_code and mul_ir_in2.ir.l == 0:
                    result = jnp.einsum(
                        f"{z}u,zui,zu->zui",
                        w,
                        x1,
                        jnp.reshape(x2, (batch_numel, mul_ir_in2.dim)),
                    ) / jnp.sqrt(mul_ir_out.ir.dim)
                elif specialized_code and mul_ir_out.ir.l == 0:
                    result = jnp.einsum(f"{z}u,zui,zui->zu", w, x1, x2) / jnp.sqrt(
                        mul_ir_in1.ir.dim
                    )
                else:
                    result = jnp.einsum(f"{z}u,ijk,zuij->zuk", w, w3j, xx)
            else:
                if specialized_code and (l1, l2, l3) == (0, 0, 0):
                    result = jnp.einsum(
                        "zu,zu->zu",
                        jnp.reshape(x1, (batch_numel, mul_ir_in1.dim)),
                        jnp.reshape(x2, (batch_numel, mul_ir_in2.dim)),
                    )
                elif specialized_code and (l1, l2, l3) == (1, 1, 1):
                    result = jnp.cross(x1, x2, axis=2) * (1.0 / jnp.sqrt(2 * 3))
                elif specialized_code and mul_ir_in1.ir.l == 0:
                    result = jnp.einsum(
                        "zu,zuj->zuj",
                        jnp.reshape(x1, (batch_numel, mul_ir_in1.dim)),
                        x2,
                    ) / jnp.sqrt(mul_ir_out.ir.dim)
                elif specialized_code and mul_ir_in2.ir.l == 0:
                    result = jnp.einsum(
                        "zui,zu->zui",
                        x1,
                        jnp.reshape(x2, (batch_numel, mul_ir_in2.dim)),
                    ) / jnp.sqrt(mul_ir_out.ir.dim)
                elif specialized_code and mul_ir_out.ir.l == 0:
                    result = jnp.einsum("zui,zui->zu", x1, x2) / jnp.sqrt(
                        mul_ir_in1.ir.dim
                    )
                else:
                    result = jnp.einsum("ijk,zuij->zuk", w3j, xx)

        elif ins.connection_mode == "uvuv":
            assert mul_ir_in1.mul * mul_ir_in2.mul == mul_ir_out.mul
            if ins.has_weight:
                # TODO implement specialized code
                result = jnp.einsum(f"{z}uv,ijk,zuvij->zuvk", w, w3j, xx)
            else:
                # TODO implement specialized code
                result = jnp.einsum("ijk,zuvij->zuvk", w3j, xx)

        elif ins.connection_mode == "uvu<v":
            assert mul_ir_in1.mul == mul_ir_in2.mul
            assert mul_ir_in1.mul * (mul_ir_in1.mul - 1) // 2 == mul_ir_out.mul
            # upper-triangular index selection
            i = jnp.triu_indices(mul_ir_in1.mul, k=1)
            xx = xx[:, i[0], i[1]]  # zuvij -> zwij
            if ins.has_weight:
                # TODO implement specialized code
                result = jnp.einsum(f"{z}w,ijk,zwij->zwk", w, w3j, xx)
            else:
                # TODO implement specialized code
                result = jnp.einsum("ijk,zwij->zwk", w3j, xx)

        elif ins.connection_mode == "u<vw":
            assert mul_ir_in1.mul == mul_ir_in2.mul
            assert ins.has_weight
            # upper-triangular index selection
            i = jnp.triu_indices(mul_ir_in1.mul, k=1)
            xx = xx[:, i[0], i[1]]  # zuvij -> zqij
            # TODO implement specialized code
            result = jnp.einsum(f"{z}qw,ijk,zqij->zwk", w, w3j, xx)

        # apply path weight and reshape
        result = ins.path_weight * result
        outputs += [result.reshape(batch_numel, mul_ir_out.dim)]

    # = Sum outputs per output irrep =
    outputs_summed = [
        _sum_tensors(
            [out for ins, out in zip(instructions, outputs) if ins.i_out == i_out],
            shape=(batch_numel, mul_ir_out.dim),
        )
        for i_out, mul_ir_out in enumerate(irreps_out)
        if mul_ir_out.mul > 0
    ]

    # Concatenate summed outputs along the output irrep dimension
    if len(outputs_summed) > 1:
        outputs_cat = jnp.concatenate(outputs_summed, axis=1)
    else:
        outputs_cat = outputs_summed[0]

    # Final output shape = batch dimensions + total irreps_out dimension
    final_output_shape = output_shape + (irreps_out.dim,)
    outputs_final = jnp.reshape(outputs_cat, final_output_shape)

    return outputs_final


def codegen_tensor_product_right(
    x2: jax.Array,
    weights: jax.Array,
    irreps_in1: Irreps,
    irreps_in2: Irreps,
    irreps_out: Irreps,
    instructions: List[Instruction],
    shared_weights: bool = False,
    specialized_code: bool = True,
) -> jax.Array:
    """
    JAX port of codegen_tensor_product_right (no FX graph, directly functional).
    Returns array with shape (..., irreps_in1.dim, irreps_out.dim).
    """

    # = Broadcast shapes =
    if shared_weights:
        output_shape = x2[..., :1].shape[:-1]
    else:
        output_shape = jnp.broadcast_shapes(x2[..., :1].shape, weights[..., :1].shape)[
            :-1
        ]

    # = Short-circuit for zero dimensional =
    instructions = [ins for ins in instructions if 0 not in ins.path_shape]
    if len(instructions) == 0:
        return jnp.zeros(
            output_shape + (irreps_in1.dim, irreps_out.dim), dtype=jnp.float32
        )

    # = Broadcast inputs if not shared =
    if not shared_weights:
        x2 = jnp.broadcast_to(x2, output_shape + (x2.shape[-1],))
        weights = jnp.broadcast_to(weights, output_shape + (weights.shape[-1],))

    output_shape = output_shape + (irreps_in1.dim, irreps_out.dim)

    # Flatten batch dims
    x2s = jnp.reshape(x2, (-1, irreps_in2.dim))
    batch_numel = x2s.shape[0]

    # = Determine number of weights and reshape =
    weight_numel = sum(prod(ins.path_shape) for ins in instructions if ins.has_weight)
    if weight_numel > 0:
        weights = jnp.reshape(weights, (-1, weight_numel))

    # = Extract individual input irreps =
    x2_list = []
    if len(irreps_in2) == 1:
        mul_ir = irreps_in2[0]
        x2_list.append(x2s.reshape(batch_numel, mul_ir.mul, mul_ir.ir.dim))
    else:
        for sl, mul_ir in zip(irreps_in2.slices(), irreps_in2):
            x2_list.append(x2s[:, sl].reshape(batch_numel, mul_ir.mul, mul_ir.ir.dim))

    # allowed connection modes (same as PyTorch)
    _valid_connection_modes = {
        "uvw",
        "uvu",
        "uvv",
        "uuw",
        "uuu",
        "uvuv",
        "uvu<v",
        "u<vw",
    }

    # Precompute identity matrices for all multiplicities and ir dims used in irreps_in1/2
    _eye_mul_cache: dict[int, jax.Array] = {}
    _eye_ir_dim_cache: dict[int, jax.Array] = {}

    def _get_eye_mul(m: int, dtype):
        # reuse if already created
        if m not in _eye_mul_cache:
            _eye_mul_cache[m] = jnp.eye(m, dtype=dtype)
        return _eye_mul_cache[m]

    def _get_eye_ir_dim(d: int, dtype):
        if d not in _eye_ir_dim_cache:
            _eye_ir_dim_cache[d] = jnp.eye(d, dtype=dtype)
        return _eye_ir_dim_cache[d]

    # = Setup book-keeping =
    z = "" if shared_weights else "z"
    flat_weight_index = 0
    outputs = []

    for ins in instructions:
        mul_ir_in1 = irreps_in1[ins.i_in1]
        mul_ir_in2 = irreps_in2[ins.i_in2]
        mul_ir_out = irreps_out[ins.i_out]

        # sanity checks (same as in PyTorch)
        assert mul_ir_in1.ir.p * mul_ir_in2.ir.p == mul_ir_out.ir.p
        assert (
            abs(mul_ir_in1.ir.l - mul_ir_in2.ir.l)
            <= mul_ir_out.ir.l
            <= mul_ir_in1.ir.l + mul_ir_in2.ir.l
        )

        # skip degenerate cases just like torch version
        if mul_ir_in1.dim == 0 or mul_ir_in2.dim == 0 or mul_ir_out.dim == 0:
            continue

        # select the x2 piece for this instruction
        x2 = x2_list[ins.i_in2]

        # create/reuse identity matrices; dtype chosen to match x2s
        dtype = x2s.dtype
        e1 = _get_eye_mul(mul_ir_in1.mul, dtype)  # shape (mul_in1.mul, mul_in1.mul)
        e2 = _get_eye_mul(mul_ir_in2.mul, dtype)  # shape (mul_in2.mul, mul_in2.mul)
        i1 = _get_eye_ir_dim(mul_ir_in1.ir.dim, dtype)  # shape (ir.dim, ir.dim)

        # validate connection mode
        assert ins.connection_mode in _valid_connection_modes

        # --- Handle weights ---
        if ins.has_weight:
            # Slice weights corresponding to this instruction
            size = prod(ins.path_shape)
            w = weights[:, flat_weight_index : flat_weight_index + size]
            new_shape = (() if shared_weights else (-1,)) + tuple(ins.path_shape)
            w = jnp.reshape(w, new_shape)
            flat_weight_index += size
        else:
            w = None

        # --- Wigner 3j symbols ---
        # Name would be "_w3j_l1_l2_lout" in the PyTorch codegen,
        # here we just compute them directly
        l1, l2, lout = mul_ir_in1.ir.l, mul_ir_in2.ir.l, mul_ir_out.ir.l
        w3j = jnp.array(wigner_3j(l1, l2, lout))  # shape (2*l1+1, 2*l2+1, 2*lout+1)

        if ins.connection_mode == "uvw":
            assert ins.has_weight
            if specialized_code and (
                mul_ir_in1.ir.l,
                mul_ir_in2.ir.l,
                mul_ir_out.ir.l,
            ) == (0, 0, 0):
                result = jnp.einsum(
                    f"{z}uvw,zv->zuw", w, x2.reshape(batch_numel, mul_ir_in2.dim)
                )
            elif specialized_code and mul_ir_in1.ir.l == 0:
                result = jnp.einsum(f"{z}uvw,zvi->zuwi", w, x2) / jnp.sqrt(
                    mul_ir_out.ir.dim
                )
            elif specialized_code and mul_ir_in2.ir.l == 0:
                result = jnp.einsum(
                    f"{z}uvw,ij,zv->zuiwj",
                    w,
                    i1,
                    x2.reshape(batch_numel, mul_ir_in2.dim),
                ) / jnp.sqrt(mul_ir_out.ir.dim)
            elif specialized_code and mul_ir_out.ir.l == 0:
                result = jnp.einsum(f"{z}uvw,zvi->zuiw", w, x2) / jnp.sqrt(
                    mul_ir_in1.ir.dim
                )
            else:
                result = jnp.einsum(f"{z}uvw,ijk,zvj->zuiwk", w, w3j, x2)

        elif ins.connection_mode == "uvu":
            assert mul_ir_in1.mul == mul_ir_out.mul
            if ins.has_weight:
                if specialized_code and (
                    mul_ir_in1.ir.l,
                    mul_ir_in2.ir.l,
                    mul_ir_out.ir.l,
                ) == (0, 0, 0):
                    result = jnp.einsum(
                        f"{z}uv,uw,zv->zuw",
                        w,
                        e1,
                        x2.reshape(batch_numel, mul_ir_in2.dim),
                    )
                elif specialized_code and mul_ir_in1.ir.l == 0:
                    result = jnp.einsum(f"{z}uv,uw,zvi->zuwi", w, e1, x2) / jnp.sqrt(
                        mul_ir_out.ir.dim
                    )
                elif specialized_code and mul_ir_in2.ir.l == 0:
                    result = jnp.einsum(
                        f"{z}uv,ij,uw,zv->zuiwj",
                        w,
                        i1,
                        e1,
                        x2.reshape(batch_numel, mul_ir_in2.dim),
                    ) / jnp.sqrt(mul_ir_out.ir.dim)
                elif specialized_code and mul_ir_out.ir.l == 0:
                    result = jnp.einsum(f"{z}uv,uw,zvi->zuiw", w, e1, x2) / jnp.sqrt(
                        mul_ir_in1.ir.dim
                    )
                else:
                    result = jnp.einsum(f"{z}uv,ijk,uw,zvj->zuiwk", w, w3j, e1, x2)

        elif ins.connection_mode == "uvv":
            assert mul_ir_in2.mul == mul_ir_out.mul
            if ins.has_weight:
                if specialized_code and (
                    mul_ir_in1.ir.l,
                    mul_ir_in2.ir.l,
                    mul_ir_out.ir.l,
                ) == (0, 0, 0):
                    result = jnp.einsum(
                        f"{z}uv,vw,zv->zuw",
                        w,
                        e2,
                        x2.reshape(batch_numel, mul_ir_in2.dim),
                    )
                elif specialized_code and mul_ir_in1.ir.l == 0:
                    result = jnp.einsum(f"{z}uv,vw,zvi->zuwi", w, e2, x2) / jnp.sqrt(
                        mul_ir_out.ir.dim
                    )
                elif specialized_code and mul_ir_in2.ir.l == 0:
                    result = jnp.einsum(
                        f"{z}uv,ij,vw,zv->zuiwj",
                        w,
                        i1,
                        e2,
                        x2.reshape(batch_numel, mul_ir_in2.dim),
                    ) / jnp.sqrt(mul_ir_out.ir.dim)
                elif specialized_code and mul_ir_out.ir.l == 0:
                    result = jnp.einsum(f"{z}uv,vw,zvi->zuiw", w, e2, x2) / jnp.sqrt(
                        mul_ir_in1.ir.dim
                    )
                else:
                    result = jnp.einsum(f"{z}uv,ijk,zvj->zuivk", w, w3j, x2)
            else:
                # not so useful operation because u is summed
                # only specialize out for this path
                s2ones = jnp.ones((mul_ir_in1.mul,), dtype=x2.dtype)
                result = jnp.einsum("u,ijk,zvj->zuivk", s2ones, w3j, x2)

        elif ins.connection_mode == "uuw":
            assert mul_ir_in1.mul == mul_ir_in2.mul
            if ins.has_weight:
                # TODO: specialize right()
                result = jnp.einsum(f"{z}uw,ijk,zuj->zuiwk", w, w3j, x2)
            else:
                # equivalent to tp(x, y, 'uuu').sum('u')
                assert mul_ir_out.mul == 1
                result = jnp.einsum("ijk,zuj->zuik", w3j, x2)

        elif ins.connection_mode == "uuu":
            assert mul_ir_in1.mul == mul_ir_in2.mul == mul_ir_out.mul
            if ins.has_weight:
                if specialized_code and (
                    mul_ir_in1.ir.l,
                    mul_ir_in2.ir.l,
                    mul_ir_out.ir.l,
                ) == (0, 0, 0):
                    result = jnp.einsum(
                        f"{z}u,uw,zu->zuw",
                        w,
                        e2,
                        x2.reshape(batch_numel, mul_ir_in2.dim),
                    )
                elif specialized_code and (
                    mul_ir_in1.ir.l,
                    mul_ir_in2.ir.l,
                    mul_ir_out.ir.l,
                ) == (1, 1, 1):
                    # For cross product, use the general case right()
                    result = jnp.einsum(f"{z}u,ijk,uw,zuj->zuiwk", w, w3j, e1, x2)
                elif specialized_code and mul_ir_in1.ir.l == 0:
                    result = jnp.einsum(f"{z}u,uw,zui->zuwi", w, e2, x2) / jnp.sqrt(
                        mul_ir_out.ir.dim
                    )
                elif specialized_code and mul_ir_in2.ir.l == 0:
                    result = jnp.einsum(
                        f"{z}u,ij,uw,zu->zuiwj",
                        w,
                        i1,
                        e2,
                        x2.reshape(batch_numel, mul_ir_in2.dim),
                    ) / jnp.sqrt(mul_ir_out.ir.dim)
                elif specialized_code and mul_ir_out.ir.l == 0:
                    result = jnp.einsum(f"{z}u,uw,zui->zuiw", w, e2, x2) / jnp.sqrt(
                        mul_ir_in1.ir.dim
                    )
                else:
                    result = jnp.einsum(f"{z}u,ijk,uw,zuj->zuiwk", w, w3j, e1, x2)
            else:
                if specialized_code and (
                    mul_ir_in1.ir.l,
                    mul_ir_in2.ir.l,
                    mul_ir_out.ir.l,
                ) == (0, 0, 0):
                    result = jnp.einsum(
                        "uw,zu->zuw", e2, x2.reshape(batch_numel, mul_ir_in2.dim)
                    )
                elif specialized_code and (
                    mul_ir_in1.ir.l,
                    mul_ir_in2.ir.l,
                    mul_ir_out.ir.l,
                ) == (1, 1, 1):
                    # For cross product, use the general case right()
                    result = jnp.einsum("ijk,uw,zuj->zuiwk", w3j, e1, x2)
                elif specialized_code and mul_ir_in1.ir.l == 0:
                    result = jnp.einsum("uw,zui->zuwi", e2, x2) / jnp.sqrt(
                        mul_ir_out.ir.dim
                    )
                elif specialized_code and mul_ir_in2.ir.l == 0:
                    result = jnp.einsum(
                        "ij,uw,zu->zuiwj",
                        i1,
                        e2,
                        x2.reshape(batch_numel, mul_ir_in2.dim),
                    ) / jnp.sqrt(mul_ir_out.ir.dim)
                elif specialized_code and mul_ir_out.ir.l == 0:
                    result = jnp.einsum("uw,zui->zuiw", e2, x2) / jnp.sqrt(
                        mul_ir_in1.ir.dim
                    )
                else:
                    result = jnp.einsum("ijk,uw,zuj->zuiwk", w3j, e1, x2)

        elif ins.connection_mode == "uvuv":
            assert mul_ir_in1.mul * mul_ir_in2.mul == mul_ir_out.mul
            if ins.has_weight:
                # TODO implement specialized code
                result = jnp.einsum(f"{z}uv,ijk,uw,zvj->zuiwvk", w, w3j, e1, x2)
            else:
                # TODO implement specialized code
                result = jnp.einsum("ijk,uw,zvj->zuiwvk", w3j, e1, x2)

        elif ins.connection_mode == "uvu<v":
            raise NotImplementedError(
                "Connection mode 'uvu<v' is not implemented in JAX version."
            )

        elif ins.connection_mode == "u<vw":
            raise NotImplementedError(
                "Connection mode 'u<vw' is not implemented in JAX version."
            )

        result = ins.path_weight * result
        outputs.append(result.reshape(batch_numel, mul_ir_in1.dim, mul_ir_out.dim))

    # Build summed outputs per (i_in1, i_out)
    outputs_grouped = [
        jnp.concatenate(
            [
                # For each output irrep i_out with mul>0, sum all instruction outputs matching (i_in1, i_out)
                _sum_tensors(
                    [
                        out
                        for ins, out in zip(instructions, outputs)
                        if (ins.i_in1, ins.i_out) == (i_in1, i_out)
                    ],
                    shape=(batch_numel, mul_ir_in1.dim, mul_ir_out.dim),
                )
                for i_out, mul_ir_out in enumerate(irreps_out)
                if mul_ir_out.mul > 0
            ],
            axis=2,  # concatenate along the output-irrep dimension
        )
        for i_in1, mul_ir_in1 in enumerate(irreps_in1)
        if mul_ir_in1.mul > 0
    ]

    # Concatenate the different i_in1 blocks along the input-irrep dimension
    if len(outputs_grouped) > 1:
        outputs_cat = jnp.concatenate(outputs_grouped, axis=1)
    else:
        outputs_cat = outputs_grouped[0]

    # Reshape to final output shape: the broadcasted batch dims + (irreps_in1.dim, irreps_out.dim)
    outputs_final = jnp.reshape(outputs_cat, output_shape)

    # Return final array (no FX graph to wrap)
    return outputs_final
