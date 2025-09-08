import haiku as hk
import jax.numpy as jnp
from typing import List, Optional
from functools import reduce
from operator import mul

from e3nn_jax import Irreps, IrrepsArray, tensor_product

from .instruction import Instruction


import haiku as hk
import jax.numpy as jnp
from typing import List, Optional, Union
from functools import reduce
from operator import mul

from e3nn_jax import Irreps, IrrepsArray, tensor_product

from .instruction import Instruction


def prod(iterable):
    return reduce(mul, iterable, 1)


def sqrt(x):
    return jnp.sqrt(x)


class TensorProduct(hk.Module):
    def __init__(
        self,
        irreps_in1: Irreps,
        irreps_in2: Irreps,
        irreps_out: Irreps,
        instructions: List[tuple],
        in1_var: Optional[Union[List[float], jnp.ndarray]] = None,
        in2_var: Optional[Union[List[float], jnp.ndarray]] = None,
        out_var: Optional[Union[List[float], jnp.ndarray]] = None,
        irrep_normalization: str = "component",
        path_normalization: str = "element",
        internal_weights: Optional[bool] = None,
        shared_weights: Optional[bool] = None,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self.irreps_in1 = Irreps(irreps_in1)
        self.irreps_in2 = Irreps(irreps_in2)
        self.irreps_out = Irreps(irreps_out)
        self.irrep_normalization = irrep_normalization
        self.path_normalization = path_normalization

        # Default variance values
        if in1_var is None:
            in1_var = [1.0 for _ in range(len(self.irreps_in1))]
        else:
            in1_var = [float(var) for var in in1_var]

        if in2_var is None:
            in2_var = [1.0 for _ in range(len(self.irreps_in2))]
        else:
            in2_var = [float(var) for var in in2_var]

        if out_var is None:
            out_var = [1.0 for _ in range(len(self.irreps_out))]
        else:
            out_var = [float(var) for var in out_var]

        # Normalize instruction tuples to length 6 (add default path_weight=1.0)
        instructions = [x if len(x) == 6 else x + (1.0,) for x in instructions]

        # Build Instructions with path shapes
        self.instructions = []
        for (
            i_in1,
            i_in2,
            i_out,
            connection_mode,
            has_weight,
            path_weight,
        ) in instructions:
            path_shape = {
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
                    self.irreps_in1[i_in1].mul * (self.irreps_in2[i_in2].mul - 1) // 2,
                ),
                "u<vw": (
                    self.irreps_in1[i_in1].mul * (self.irreps_in2[i_in2].mul - 1) // 2,
                    self.irreps_out[i_out].mul,
                ),
            }[connection_mode]

            self.instructions.append(
                Instruction(
                    i_in1=i_in1,
                    i_in2=i_in2,
                    i_out=i_out,
                    connection_mode=connection_mode,
                    has_weight=bool(has_weight),
                    path_weight=float(path_weight),
                    path_shape=path_shape,
                )
            )

        # Calculate normalization coefficients (similar to PyTorch version)
        def num_elements(ins):
            return {
                "uvw": (
                    self.irreps_in1[ins.i_in1].mul * self.irreps_in2[ins.i_in2].mul
                ),
                "uvu": self.irreps_in2[ins.i_in2].mul,
                "uvv": self.irreps_in1[ins.i_in1].mul,
                "uuw": self.irreps_in1[ins.i_in1].mul,
                "uuu": 1,
                "uvuv": 1,
                "uvu<v": 1,
                "u<vw": self.irreps_in1[ins.i_in1].mul
                * (self.irreps_in2[ins.i_in2].mul - 1)
                // 2,
            }[ins.connection_mode]

        normalization_coefficients = []
        for ins in self.instructions:
            mul_ir_in1 = self.irreps_in1[ins.i_in1]
            mul_ir_in2 = self.irreps_in2[ins.i_in2]
            mul_ir_out = self.irreps_out[ins.i_out]

            if irrep_normalization == "component":
                alpha = mul_ir_out.ir.dim
            elif irrep_normalization == "norm":
                alpha = mul_ir_in1.ir.dim * mul_ir_in2.ir.dim
            else:  # "none"
                alpha = 1

            if path_normalization == "element":
                x = sum(
                    in1_var[i.i_in1] * in2_var[i.i_in2] * num_elements(i)
                    for i in self.instructions
                    if i.i_out == ins.i_out
                )
            elif path_normalization == "path":
                x = in1_var[ins.i_in1] * in2_var[ins.i_in2] * num_elements(ins)
                x *= len([i for i in self.instructions if i.i_out == ins.i_out])
            else:  # "none"
                x = 1

            if x > 0.0:
                alpha /= x

            alpha *= out_var[ins.i_out]
            alpha *= ins.path_weight

            normalization_coefficients.append(sqrt(alpha))

        # Update instructions with normalization coefficients
        self.instructions = [
            Instruction(
                ins.i_in1,
                ins.i_in2,
                ins.i_out,
                ins.connection_mode,
                ins.has_weight,
                alpha,  # Store normalization coefficient as path_weight
                ins.path_shape,
            )
            for ins, alpha in zip(self.instructions, normalization_coefficients)
        ]
        # Weight handling
        if shared_weights is None:
            shared_weights = True
        self.shared_weights = shared_weights

        if internal_weights is None:
            internal_weights = self.shared_weights and any(
                ins.has_weight for ins in self.instructions
            )
        self.internal_weights = internal_weights

        self.weight_numel = sum(
            prod(ins.path_shape) for ins in self.instructions if ins.has_weight
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

    def _get_weights(
        self, external_weight: Optional[jnp.ndarray] = None
    ) -> Optional[jnp.ndarray]:
        if self.weight_numel == 0:
            return None
        if external_weight is not None:
            return external_weight
        return self.weight

    def weight_view_for_instruction(
        self, instruction_idx: int, weight: Optional[jnp.ndarray] = None
    ) -> jnp.ndarray:
        """Return a view of the weights corresponding to a single instruction."""
        weight = self._get_weights(weight)
        ins = self.instructions[instruction_idx]
        if not ins.has_weight:
            raise ValueError(f"Instruction {instruction_idx} has no weights.")

        # Compute offset
        offset = sum(
            prod(self.instructions[i].path_shape)
            for i in range(instruction_idx)
            if self.instructions[i].has_weight
        )

        flatsize = prod(ins.path_shape)
        if self.shared_weights:
            return weight[offset : offset + flatsize].reshape(ins.path_shape)
        else:
            return weight[..., offset : offset + flatsize].reshape(
                weight.shape[:-1] + ins.path_shape
            )

    def __call__(
        self, x1: jnp.ndarray, x2: jnp.ndarray, weight: Optional[jnp.ndarray] = None
    ) -> jnp.ndarray:
        """
        Apply tensor product operation.

        Args:
            x1: Input tensor with shape (..., irreps_in1.dim)
            x2: Input tensor with shape (..., irreps_in2.dim)
            weight: Optional weight tensor

        Returns:
            Output tensor with shape (..., irreps_out.dim)
        """
        # Handle IrrepsArray inputs
        if isinstance(x1, IrrepsArray):
            x1 = x1.array
        if isinstance(x2, IrrepsArray):
            x2 = x2.array

        w = self._get_weights(weight)

        # Initialize output
        batch_shape = jnp.broadcast_shapes(x1.shape[:-1], x2.shape[:-1])
        out_shape = batch_shape + (self.irreps_out.dim,)
        y_out = jnp.zeros(out_shape, dtype=x1.dtype)

        weight_offset = 0

        for ins in self.instructions:
            # Get input slices for this instruction
            mul1, ir1 = self.irreps_in1[ins.i_in1].mul, self.irreps_in1[ins.i_in1].ir
            start1 = sum(
                self.irreps_in1[i].mul * self.irreps_in1[i].ir.dim
                for i in range(ins.i_in1)
            )
            end1 = start1 + mul1 * ir1.dim
            x1_slice = x1[..., start1:end1]

            mul2, ir2 = self.irreps_in2[ins.i_in2].mul, self.irreps_in2[ins.i_in2].ir
            start2 = sum(
                self.irreps_in2[i].mul * self.irreps_in2[i].ir.dim
                for i in range(ins.i_in2)
            )
            end2 = start2 + mul2 * ir2.dim
            x2_slice = x2[..., start2:end2]

            mul_out, ir_out = (
                self.irreps_out[ins.i_out].mul,
                self.irreps_out[ins.i_out].ir,
            )
            start_out = sum(
                self.irreps_out[i].mul * self.irreps_out[i].ir.dim
                for i in range(ins.i_out)
            )
            end_out = start_out + mul_out * ir_out.dim

            # Check if this tensor product combination is valid
            possible_irs = ir1 * ir2
            if ir_out not in possible_irs:
                # Skip invalid combinations
                if ins.has_weight:
                    weight_offset += prod(ins.path_shape)
                continue

            # Reshape inputs to separate multiplicity and irrep dimensions
            x1_reshaped = x1_slice.reshape(x1_slice.shape[:-1] + (mul1, ir1.dim))
            x2_reshaped = x2_slice.reshape(x2_slice.shape[:-1] + (mul2, ir2.dim))

            # Compute tensor product for each combination
            # This is the key part - we need to handle different connection modes
            if ins.connection_mode == "uvw":
                # Fully connected: all combinations of multiplicities
                result = jnp.zeros(batch_shape + (mul1, mul2, mul_out, ir_out.dim))
                for u in range(mul1):
                    for v in range(mul2):
                        # Compute tensor product between irreps
                        tp_result = self._compute_irrep_tensor_product(
                            x1_reshaped[..., u, :],
                            x2_reshaped[..., v, :],
                            ir1,
                            ir2,
                            ir_out,
                        )
                        # Broadcast to all output multiplicities
                        for w in range(mul_out):
                            result = result.at[..., u, v, w, :].set(tp_result)

            elif ins.connection_mode == "uuu":
                # Diagonal: only u=v=w combinations
                min_mul = min(mul1, mul2, mul_out)
                result = jnp.zeros(batch_shape + (mul1, mul2, mul_out, ir_out.dim))
                for u in range(min_mul):
                    tp_result = self._compute_irrep_tensor_product(
                        x1_reshaped[..., u, :], x2_reshaped[..., u, :], ir1, ir2, ir_out
                    )
                    result = result.at[..., u, u, u, :].set(tp_result)

            elif ins.connection_mode == "uvu":
                # u=w, sum over v
                result = jnp.zeros(batch_shape + (mul1, 1, mul_out, ir_out.dim))
                min_mul = min(mul1, mul_out)
                for u in range(min_mul):
                    tp_sum = jnp.zeros(batch_shape + (ir_out.dim,))
                    for v in range(mul2):
                        tp_result = self._compute_irrep_tensor_product(
                            x1_reshaped[..., u, :],
                            x2_reshaped[..., v, :],
                            ir1,
                            ir2,
                            ir_out,
                        )
                        tp_sum = tp_sum + tp_result
                    result = result.at[..., u, 0, u, :].set(tp_sum)

            elif ins.connection_mode == "uvv":
                # v=w, sum over u
                result = jnp.zeros(batch_shape + (1, mul2, mul_out, ir_out.dim))
                min_mul = min(mul2, mul_out)
                for v in range(min_mul):
                    tp_sum = jnp.zeros(batch_shape + (ir_out.dim,))
                    for u in range(mul1):
                        tp_result = self._compute_irrep_tensor_product(
                            x1_reshaped[..., u, :],
                            x2_reshaped[..., v, :],
                            ir1,
                            ir2,
                            ir_out,
                        )
                        tp_sum = tp_sum + tp_result
                    result = result.at[..., 0, v, v, :].set(tp_sum)

            elif ins.connection_mode == "uuw":
                # u=v, w independent
                result = jnp.zeros(batch_shape + (mul1, 1, mul_out, ir_out.dim))
                min_mul = min(mul1, mul2)
                for u in range(min_mul):
                    tp_result = self._compute_irrep_tensor_product(
                        x1_reshaped[..., u, :], x2_reshaped[..., u, :], ir1, ir2, ir_out
                    )
                    for w in range(mul_out):
                        result = result.at[..., u, 0, w, :].set(tp_result)

            else:
                raise NotImplementedError(
                    f"Connection mode {ins.connection_mode} not implemented"
                )

            # Apply weights if present
            if ins.has_weight:
                flatsize = prod(ins.path_shape)
                if self.shared_weights:
                    weight_slice = w[weight_offset : weight_offset + flatsize].reshape(
                        ins.path_shape
                    )
                else:
                    weight_slice = w[
                        ..., weight_offset : weight_offset + flatsize
                    ].reshape(w.shape[:-1] + ins.path_shape)
                weight_offset += flatsize

                # Apply weights by broadcasting over the appropriate dimensions
                if ins.connection_mode == "uvw":
                    result = result * weight_slice[..., :, :, :, None]
                elif ins.connection_mode in ["uuu", "uvu", "uvv", "uuw"]:
                    # For these modes, we need to handle the weight application carefully
                    # This is a simplified version - you may need to adjust based on exact requirements
                    weight_broadcast = jnp.broadcast_to(
                        weight_slice.reshape(
                            weight_slice.shape
                            + (1,) * (result.ndim - weight_slice.ndim)
                        ),
                        result.shape,
                    )
                    result = result * weight_broadcast

            # Apply normalization coefficient (stored in path_weight)
            result = result * ins.path_weight

            # Sum over multiplicity dimensions and add to output
            result_summed = jnp.sum(
                result, axis=(-4, -3, -2)
            )  # Sum over mul1, mul2, mul_out

            # Add to the appropriate slice of the output
            y_out = y_out.at[..., start_out:end_out].add(
                result_summed.reshape(batch_shape + (-1,))
            )

        return y_out

    def _compute_irrep_tensor_product(self, x1, x2, ir1, ir2, ir_out):
        """Compute tensor product between individual irreps using e3nn_jax."""
        # Create single-irrep IrrepsArrays
        x1_irreps = IrrepsArray(
            Irreps([(1, ir1)]), x1.reshape(x1.shape[:-1] + (1, ir1.dim))
        )
        x2_irreps = IrrepsArray(
            Irreps([(1, ir2)]), x2.reshape(x2.shape[:-1] + (1, ir2.dim))
        )

        # Compute tensor product
        result = tensor_product(
            x1_irreps,
            x2_irreps,
            filter_ir_out=[ir_out],
            irrep_normalization=self.irrep_normalization,
            regroup_output=True,
        )

        # Extract the result and reshape
        return result.array.reshape(x1.shape[:-1] + (ir_out.dim,))

    def __repr__(self) -> str:
        npath = sum(prod(i.path_shape) for i in self.instructions)
        return (
            f"{self.__class__.__name__}"
            f"({self.irreps_in1} x {self.irreps_in2} "
            f"-> {self.irreps_out} | {npath} paths | {self.weight_numel} weights)"
        )
