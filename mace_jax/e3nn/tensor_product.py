import haiku as hk
import jax.numpy as jnp
from typing import List, Optional
from functools import reduce
from operator import mul

from e3nn_jax import Irreps, IrrepsArray, tensor_product

from .instruction import Instruction


class TensorProduct(hk.Module):
    def __init__(
        self,
        irreps_in1: Irreps,
        irreps_in2: Irreps,
        irreps_out: Irreps,
        instructions: List[tuple],
        internal_weights: Optional[bool] = None,
        shared_weights: Optional[bool] = None,
        name=None,
    ):
        super().__init__(name=name)

        self.irreps_in1 = irreps_in1
        self.irreps_in2 = irreps_in2
        self.irreps_out = irreps_out

        # Convert tuples to NamedTuples with default path_weight=1.0
        self.instructions = [x if len(x) == 6 else x + (1.0,) for x in instructions]
        self.instructions = [
            Instruction(
                i_in1=i_in1,
                i_in2=i_in2,
                i_out=i_out,
                connection_mode=connection_mode,
                has_weight=has_weight,
                path_weight=path_weight,
                path_shape={
                    "uvw": (
                        self.irreps_in1[i_in1].mul,
                        self.irreps_in2[i_in2].mul,
                        self.irreps_out[i_out].mul,
                    ),
                    "uvu": (self.irreps_in1[i_in1].mul, self.irreps_in2[i_in2].mul),
                    "uvv": (self.irreps_in1[i_in1].mul, self.irreps_in2[i_in2].mul),
                    "uuw": (self.irreps_in1[i_in1].mul, self.irreps_out[i_out].mul),
                    "uuu": (self.irreps_in1[i_in1].mul,),
                    "uvuv": (self.irreps_in1[i_in1].mul, self.irreps_in2[i_in2].mul),
                    "uvu<v": (
                        self.irreps_in1[i_in1].mul
                        * (self.irreps_in2[i_in2].mul - 1)
                        // 2,
                    ),
                    "u<vw": (
                        self.irreps_in1[i_in1].mul
                        * (self.irreps_in2[i_in2].mul - 1)
                        // 2,
                        self.irreps_out[i_out].mul,
                    ),
                }[connection_mode],
            )
            for i_in1, i_in2, i_out, connection_mode, has_weight, path_weight in self.instructions
        ]

        if shared_weights is None:
            shared_weights = True
        self.shared_weights = shared_weights

        if internal_weights is None:
            internal_weights = shared_weights and any(
                ins.has_weight for ins in self.instructions
            )
        self.internal_weights = internal_weights

        # Allocate weight parameters if requested
        self.weight_numel = sum(
            reduce(mul, ins.path_shape) for ins in self.instructions if ins.has_weight
        )
        if self.internal_weights and self.weight_numel > 0:
            self.weight = hk.get_parameter(
                "weight",
                shape=(self.weight_numel,),
                dtype=jnp.float32,
                init=hk.initializers.RandomNormal(),
            )
        else:
            self.weight = None

    def _get_weights(self, external_weight: Optional[jnp.ndarray] = None):
        if self.weight_numel == 0:
            return None
        if external_weight is not None:
            return external_weight
        return self.weight

    def weight_view_for_instruction(
        self, instruction_idx: int, weight: Optional[jnp.ndarray] = None
    ) -> jnp.ndarray:
        """Return a view of the weights corresponding to a single instruction"""
        weight = self._get_weights(weight)
        ins = self.instructions[instruction_idx]
        if not ins.has_weight:
            raise ValueError(f"Instruction {instruction_idx} has no weights.")
        offset = sum(
            reduce(mul, self.instructions[i].path_shape)
            for i in range(instruction_idx)
            if self.instructions[i].has_weight
        )
        return weight[offset : offset + reduce(mul, ins.path_shape)].reshape(
            ins.path_shape
        )

    def __call__(
        self, x1: IrrepsArray, x2: IrrepsArray, weight: Optional[jnp.ndarray] = None
    ) -> IrrepsArray:
        w = self._get_weights(weight)
        y_out = jnp.zeros(x1.shape[:-1] + (self.irreps_out.dim,), dtype=x1.dtype)

        offset = 0
        for idx, ins in enumerate(self.instructions):
            # --- Slice inputs ---
            start1 = sum(r.mul for r in self.irreps_in1[: ins.i_in1])
            end1 = start1 + self.irreps_in1[ins.i_in1].mul
            x1_slice = x1[..., start1:end1]

            start2 = sum(r.mul for r in self.irreps_in2[: ins.i_in2])
            end2 = start2 + self.irreps_in2[ins.i_in2].mul
            x2_slice = x2[..., start2:end2]

            # --- Compute tensor product according to connection_mode ---
            tp_result = tensor_product(x1_slice, x2_slice)

            if ins.connection_mode == "uvw":
                # fully connected, nothing to reduce
                pass
            elif ins.connection_mode == "uvu":
                # sum over second multiplicity
                tp_result = IrrepsArray(
                    tp_result.irreps, jnp.sum(tp_result.array, axis=-2, keepdims=True)
                )
            elif ins.connection_mode == "uvv":
                tp_result = IrrepsArray(
                    tp_result.irreps, jnp.sum(tp_result.array, axis=-3, keepdims=True)
                )
            elif ins.connection_mode == "uuw":
                tp_result = IrrepsArray(
                    tp_result.irreps, jnp.sum(tp_result.array, axis=-2, keepdims=True)
                )
            elif ins.connection_mode == "uuu":
                tp_result = IrrepsArray(
                    tp_result.irreps,
                    jnp.sum(tp_result.array, axis=(-2, -3), keepdims=True),
                )
            elif ins.connection_mode == "uvuv":
                # keep both multiplicities separately
                pass
            elif ins.connection_mode == "uvu<v":
                # Only combine certain multiplicities: collapse second dimension partially
                x2_mul = x2_slice.shape[-1]
                tp_result = IrrepsArray(
                    tp_result.irreps,
                    jnp.sum(tp_result.array[..., : x2_mul - 1], axis=-2, keepdims=True),
                )
            elif ins.connection_mode == "u<vw":
                x1_mul = x1_slice.shape[-1]
                x2_mul = x2_slice.shape[-1]
                tp_result = IrrepsArray(
                    tp_result.irreps,
                    jnp.sum(
                        tp_result.array[..., : x1_mul * (x2_mul - 1) // 2],
                        axis=-2,
                        keepdims=True,
                    ),
                )
            else:
                raise NotImplementedError(
                    f"Connection mode {ins.connection_mode} not implemented"
                )

            # --- Apply weight if present ---
            if ins.has_weight:
                tp_weight = w[offset : offset + reduce(mul, ins.path_shape)].reshape(
                    ins.path_shape
                )
                tp_result = IrrepsArray(
                    tp_result.irreps, tp_result.array * jnp.sum(tp_weight)
                )
                offset += reduce(mul, ins.path_shape)

            # --- Add contribution to output slice ---
            out_start = sum(r.mul for r in self.irreps_out[: ins.i_out])
            out_end = out_start + self.irreps_out[ins.i_out].mul
            y_out = y_out.at[..., out_start:out_end].add(tp_result.array)

        return IrrepsArray(self.irreps_out, y_out)
