from math import prod
from typing import Iterator, List, NamedTuple, Optional, Tuple, Union

import haiku as hk
import jax.numpy as jnp
from e3nn_jax import Irreps

from ._tensor_product._codegen import _sum_tensors


class Instruction(NamedTuple):
    i_in: int
    i_out: int
    path_shape: tuple
    path_weight: float


class Linear(hk.Module):
    """Linear operation equivariant to O(3), JAX/Haiku version."""

    def __init__(
        self,
        irreps_in: Irreps,
        irreps_out: Irreps,
        *,
        f_in: Optional[int] = None,
        f_out: Optional[int] = None,
        shared_weights: Optional[bool] = None,
        instructions: Optional[List[Tuple[int, int]]] = None,
        biases: Union[bool, List[bool]] = False,
        path_normalization: str = "element",
        name: Optional[str] = None,
    ):
        super().__init__(name=name)

        irreps_in = Irreps(irreps_in)
        irreps_out = Irreps(irreps_out)

        # Default instructions: connect matching irreps
        if instructions is None:
            instructions = [
                (i_in, i_out)
                for i_in, (_, ir_in) in enumerate(irreps_in)
                for i_out, (_, ir_out) in enumerate(irreps_out)
                if ir_in == ir_out
            ]

        # Convert to full Instruction objects
        instructions = [
            Instruction(
                i_in=i_in,
                i_out=i_out,
                path_shape=(irreps_in[i_in].mul, irreps_out[i_out].mul),
                path_weight=1.0,
            )
            for i_in, i_out in instructions
        ]

        # Apply path normalization
        def alpha(ins):
            x = sum(
                irreps_in[i.i_in if path_normalization == "element" else ins.i_in].mul
                for i in instructions
                if i.i_out == ins.i_out
            )
            if f_in is not None:
                x *= f_in
            return 1.0 if x == 0 else x

        instructions = [
            Instruction(
                i_in=ins.i_in,
                i_out=ins.i_out,
                path_shape=ins.path_shape,
                path_weight=alpha(ins) ** (-0.5),
            )
            for ins in instructions
        ]

        # Add bias instructions
        if isinstance(biases, bool):
            biases = [biases and ir.is_scalar() for _, ir in irreps_out]
        assert len(biases) == len(irreps_out)

        instructions += [
            Instruction(i_in=-1, i_out=i_out, path_shape=(mul_ir.dim,), path_weight=1.0)
            for i_out, (bias, mul_ir) in enumerate(zip(biases, irreps_out))
            if bias
        ]

        if shared_weights is None:
            shared_weights = True
        self.shared_weights = shared_weights

        self.irreps_in = irreps_in
        self.irreps_out = irreps_out
        self.instructions = instructions
        self.f_in = f_in
        self.f_out = f_out

        # Compute weight_numel and bias_numel
        self.weight_numel = sum(
            prod(ins.path_shape) for ins in instructions if ins.i_in != -1
        )
        self.bias_numel = sum(
            prod(ins.path_shape) for ins in instructions if ins.i_in == -1
        )

    def __call__(
        self,
        x: jnp.ndarray,
        w: Optional[jnp.ndarray] = None,
        b: Optional[jnp.ndarray] = None,
    ):
        """Evaluate the linear layer.

        Parameters
        ----------
        x : jnp.ndarray
            Input tensor of shape (..., irreps_in.dim)
        w : jnp.ndarray, optional
            Weight tensor of shape (f_in?, f_out?, weight_numel)
        b : jnp.ndarray, optional
            Bias tensor of shape (f_out?, bias_numel)

        Returns
        -------
        jnp.ndarray
            Output tensor of shape (..., irreps_out.dim)
        """
        # Initialize weights if needed
        if w is None and self.weight_numel > 0:
            w_shape = ()
            if self.f_in is not None:
                w_shape += (self.f_in,)
            if self.f_out is not None:
                w_shape += (self.f_out,)
            w_shape += (self.weight_numel,)
            w = hk.get_parameter("weight", w_shape, init=hk.initializers.RandomNormal())

        if b is None and self.bias_numel > 0:
            b_shape = ()
            if self.f_out is not None:
                b_shape += (self.f_out,)
            b_shape += (self.bias_numel,)
            b = hk.get_parameter("bias", b_shape, init=jnp.zeros)

        return _codegen_linear(
            x,
            w,
            b,
            self.irreps_in,
            self.irreps_out,
            self.instructions,
            self.f_in,
            self.f_out,
            self.shared_weights,
        )

    def weight_view_for_instruction(
        self, instruction: int, weight: Optional[jnp.ndarray] = None
    ) -> jnp.ndarray:
        """
        View of weights corresponding to `instruction`.
        """
        if weight is None:
            weight = hk.get_parameter(
                "weight", (self.weight_numel,), init=hk.initializers.RandomNormal()
            )

        batchshape = weight.shape[:-1]
        offset = sum(prod(ins.path_shape) for ins in self.instructions[:instruction])
        ins = self.instructions[instruction]
        flatsize = prod(ins.path_shape)
        return weight[..., offset : offset + flatsize].reshape(
            batchshape + ins.path_shape
        )

    def weight_views(
        self, weight: Optional[jnp.ndarray] = None, yield_instruction: bool = False
    ) -> Union[Iterator[jnp.ndarray], Iterator[Tuple[int, Instruction, jnp.ndarray]]]:
        """
        Iterator over weight views for all instructions.
        """
        if weight is None:
            weight = hk.get_parameter(
                "weight", (self.weight_numel,), init=hk.initializers.RandomNormal()
            )

        batchshape = weight.shape[:-1]
        offset = 0
        for ins_i, ins in enumerate(self.instructions):
            flatsize = prod(ins.path_shape)
            this_weight = weight[..., offset : offset + flatsize].reshape(
                batchshape + ins.path_shape
            )
            offset += flatsize
            if yield_instruction:
                yield ins_i, ins, this_weight
            else:
                yield this_weight


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
