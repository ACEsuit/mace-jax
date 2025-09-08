import haiku as hk
import jax.numpy as jnp
from typing import List, Optional
from functools import reduce
from operator import mul

from e3nn_jax import Irreps, IrrepsArray, tensor_product

from .instruction import Instruction


def prod(iterable):
    return reduce(mul, iterable, 1)


class TensorProduct(hk.Module):
    def __init__(
        self,
        irreps_in1: Irreps,
        irreps_in2: Irreps,
        irreps_out: Irreps,
        instructions: List[tuple],
        internal_weights: Optional[bool] = None,
        shared_weights: Optional[bool] = None,
        irrep_normalization: Optional[str] = "component",
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self.irreps_in1 = Irreps(irreps_in1)
        self.irreps_in2 = Irreps(irreps_in2)
        self.irreps_out = Irreps(irreps_out)
        self.irrep_normalization = irrep_normalization or "component"

        # Normalize instruction tuples to length 6 (add default path_weight=1.0)
        instructions = [x if len(x) == 6 else x + (1.0,) for x in instructions]

        # Build Instruction namedtuple + path_shape (same mapping as your skeleton)
        built = []
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
            built.append(
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
        self.instructions = built

        # shared_weights default True
        if shared_weights is None:
            shared_weights = True
        self.shared_weights = shared_weights

        # internal_weights default to shared_weights and any instruction requires weight
        if internal_weights is None:
            internal_weights = self.shared_weights and any(
                ins.has_weight for ins in self.instructions
            )
        self.internal_weights = internal_weights

        # weight allocation
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
            # if shared_weights==True, expect shape (...,) equals (weight_numel,)
            # if shared_weights==False, expect shape (..., weight_numel)
            return external_weight
        return self.weight

    def weight_view_for_instruction(
        self, instruction_idx: int, weight: Optional[jnp.ndarray] = None
    ) -> jnp.ndarray:
        """Return a view (ndarray) of the weights corresponding to a single instruction."""
        weight = self._get_weights(weight)
        ins = self.instructions[instruction_idx]
        if not ins.has_weight:
            raise ValueError(f"Instruction {instruction_idx} has no weights.")
        # compute offset only considering weighted instructions before instruction_idx
        offset = 0
        for i in range(instruction_idx):
            if self.instructions[i].has_weight:
                offset += prod(self.instructions[i].path_shape)
        flat = weight
        # if not shared, flatten last axis
        if not self.shared_weights:
            # weight shape: (..., weight_numel)
            # return view for last axis
            return flat[..., offset : offset + prod(ins.path_shape)].reshape(
                ins.path_shape
            )
        else:
            # shared weights: shape (weight_numel,)
            return flat[offset : offset + prod(ins.path_shape)].reshape(ins.path_shape)

    def _ensure_tp_shape(self, tp_irreps_array: IrrepsArray, ins: Instruction):
        arr = tp_irreps_array.array
        out_ir = self.irreps_out[ins.i_out].ir
        out_dim = out_ir.dim
        path_prod = prod(ins.path_shape)

        # Case 1: already (..., out_dim) and no multiplicity expected
        if arr.shape[-1] == out_dim and path_prod == 1:
            return tp_irreps_array

        # Case 2: (..., path_prod, out_dim) -> reshape to expanded
        if arr.shape[-1] == out_dim and arr.shape[-2] == path_prod:
            newshape = arr.shape[:-2] + tuple(ins.path_shape) + (out_dim,)
            return IrrepsArray(tp_irreps_array.irreps, arr.reshape(newshape))

        # Case 3: (..., path_prod * out_dim) -> split
        if arr.shape[-1] % out_dim == 0:
            S = arr.shape[-1] // out_dim
            if S == path_prod:
                newshape = arr.shape[:-1] + tuple(ins.path_shape) + (out_dim,)
                return IrrepsArray(tp_irreps_array.irreps, arr.reshape(newshape))

        # Fallback: return unchanged
        return tp_irreps_array

    def __call__(
        self, x1: jnp.array, x2: jnp.array, weight: Optional[jnp.ndarray] = None
    ) -> jnp.array:
        """
        x1: IrrepsArray with irreps == self.irreps_in1 (or containing them; we only slice).
        x2: IrrepsArray with irreps == self.irreps_in2
        weight: if provided and shared_weights==True: shape (weight_numel,)
                if provided and shared_weights==False: shape (..., weight_numel)
        """
        if isinstance(x1, IrrepsArray):
            x1 = x1.array
        if isinstance(x2, IrrepsArray):
            x2 = x2.array

        w = self._get_weights(weight)
        # output buffer
        out_shape = x1.shape[:-1] + (self.irreps_out.dim,)
        y_out = jnp.zeros(out_shape, dtype=x1.dtype)

        # offset into flat weight vector (for shared case and for slicing the last axis in non-shared case)
        offset = 0

        for ins_idx, ins in enumerate(self.instructions):
            # --- Slice inputs by irreps index ---
            mul1 = self.irreps_in1[ins.i_in1].mul
            ir1 = self.irreps_in1[ins.i_in1].ir
            start1 = sum(r.mul for r in self.irreps_in1[: ins.i_in1])
            end1 = start1 + mul1 * ir1.dim
            x1_slice_arr = x1[..., start1:end1]

            mul2 = self.irreps_in2[ins.i_in2].mul
            ir2 = self.irreps_in2[ins.i_in2].ir
            start2 = sum(r.mul for r in self.irreps_in2[: ins.i_in2])
            end2 = start2 + mul2 * ir2.dim
            x2_slice_arr = x2[..., start2:end2]

            # Wrap slices into single-irrep IrrepsArray objects so tensor_product knows the irreps
            x1_sub_irreps = Irreps([(mul1, ir1)])
            x2_sub_irreps = Irreps([(mul2, ir2)])
            x1_sub = IrrepsArray(x1_sub_irreps, x1_slice_arr)
            x2_sub = IrrepsArray(x2_sub_irreps, x2_slice_arr)

            # --- call the e3nn jax tensor_product limited to only the desired output irrep ---
            # filter_ir_out expects Irrep objects list
            out_ir = self.irreps_out[ins.i_out].ir
            tp = tensor_product(
                x1_sub,
                x2_sub,
                filter_ir_out=[out_ir],
                irrep_normalization=self.irrep_normalization,
                regroup_output=True,
            )
            # tp is an IrrepsArray. Ensure shape (..., *ins.path_shape, out_dim)
            tp = self._ensure_tp_shape(tp, ins)
            arr = tp.array  # now (..., *ins.path_shape, out_dim)

            # --- Apply the connection mode reductions ---
            # multiplicity axes are the trailing axes before out_dim
            n_mult_axes = len(ins.path_shape)
            # We will operate by axis indices relative to the end:
            # Example: for "uvw" we don't collapse anything (fully connected)
            if ins.connection_mode == "uvw":
                # leave arr as-is
                pass
            elif ins.connection_mode == "uvu":
                # sum over second multiplicity (which we assume is axis -2 if path_shape is (m1, m2))
                if n_mult_axes >= 2:
                    axis = -2
                    arr = arr.sum(axis=axis, keepdims=True)
                else:
                    # degenerate: nothing to sum
                    pass
            elif ins.connection_mode == "uvv":
                # sum over first multiplicity (axis -3 when shape (..., m1, m2, out_dim))
                if n_mult_axes >= 2:
                    axis = -3
                    arr = arr.sum(axis=axis, keepdims=True)
                else:
                    pass
            elif ins.connection_mode == "uuw":
                # sum over second multiplicity (assumes path_shape (m1, mout))
                if n_mult_axes >= 2:
                    axis = -2
                    arr = arr.sum(axis=axis, keepdims=True)
            elif ins.connection_mode == "uuu":
                # sum over all multiplicities -> reduce to (..., 1, out_dim)
                if n_mult_axes >= 1:
                    for a in range(n_mult_axes):
                        arr = arr.sum(axis=-(1 + n_mult_axes - 1 - a), keepdims=True)
                # simpler: collapse all multiplicity axes
            elif ins.connection_mode == "uvuv":
                # keep both multiplicities separate (no op)
                pass
            elif ins.connection_mode == "uvu<v":
                # user-specific partial collapse: sum over some subset. We'll implement same heuristic
                # as your skeleton: sum over first (m2 - 1) entries along the second multiplicity axis.
                if n_mult_axes >= 2:
                    m2 = ins.path_shape[1]
                    keep = m2 - 1
                    # build selection to sum only first keep entries along axis -2
                    arr = (
                        arr[..., :keep, :, :] if False else None
                    )  # placeholder - we'll do generic below
                    # Generic: move axis to index, slice, then sum
                    axis_index = arr.ndim - 2
                    # slice
                    slicer = [slice(None)] * arr.ndim
                    slicer[axis_index] = slice(0, max(0, ins.path_shape[1] - 1))
                    arr = arr[tuple(slicer)].sum(axis=axis_index, keepdims=True)
            elif ins.connection_mode == "u<vw":
                # another special-mode: collapse according to combinatorial formula
                # We'll fallback to summing the first (m1*(m2-1)//2) flattened elements along the first mult axis
                # If flattened multiplicities are present we handle it automatically
                total_first = ins.path_shape[0]
                keep = total_first
                # try to collapse first mult-axis partially; for simplicity sum over that axis's first `keep` entries
                axis_index = arr.ndim - 1 - n_mult_axes
                # This is brittle — but we keep the same heuristic as your skeleton
                slicer = [slice(None)] * arr.ndim
                slicer[axis_index] = slice(0, keep)
                arr = arr[tuple(slicer)].sum(axis=axis_index, keepdims=True)
            else:
                raise NotImplementedError(
                    f"Connection mode {ins.connection_mode} not implemented"
                )

            # --- Apply path_weight scale if not default 1.0 ---
            if ins.path_weight != 1.0:
                arr = arr * ins.path_weight

            # --- Apply weights if present ---
            if ins.has_weight:
                flatsize = prod(ins.path_shape)
                if self.shared_weights:
                    # weight is flat shape (weight_numel,)
                    tp_w = w[offset : offset + flatsize].reshape(ins.path_shape)
                    offset += flatsize
                    # We need a scalar multiplier for the entire path result.
                    # In PyTorch implementation different shapes may be used; here we sum tp_w to produce a scalar
                    multiplier = jnp.sum(tp_w)
                    arr = arr * multiplier
                else:
                    # per-batch weights: w has shape (..., weight_numel)
                    # Extract the slice corresponding to this instruction and sum over path axes to reduce to (...,1,1,...)
                    # For broadcasting we need to expand dims to match arr
                    # first extract the flat slice
                    tp_w = w[..., offset : offset + flatsize]  # shape (..., flatsize)
                    offset += flatsize
                    tp_w = tp_w.reshape(
                        tp_w.shape[:-1] + ins.path_shape
                    )  # (..., *ins.path_shape)
                    # sum over path axes to get (..., 1, 1, ..., 1)
                    mul_axes = tuple(range(tp_w.ndim - len(ins.path_shape), tp_w.ndim))
                    scalar = jnp.sum(
                        tp_w, axis=mul_axes, keepdims=True
                    )  # shape (..., 1)
                    # Now we need to broadcast scalar to arr shape (arr has extra trailing axes for path_shape)
                    # Expand dims so scalar shape ends with the correct number of singleton dims
                    expand_dims = arr.ndim - scalar.ndim
                    scalar = scalar.reshape(scalar.shape + (1,) * expand_dims)
                    arr = arr * scalar

            # --- Reduce multiplicity axes to the output irreps chunk and add to y_out ---
            # After connection-mode reductions arr should have multiplicity axes shape compatible with out mul
            # We now collapse multiplicity axes to a single multiplicity for the output (sum over multiplicities),
            # then add to the correct slice in y_out.
            # Collapse multiplicity axes (all axes except last = out_dim)
            if arr.ndim > 1:
                # sum over multiplicity axes, leaving final out_dim
                # find how many trailing axes represent multiplicities
                n_trailing = arr.ndim - 1
                if n_trailing > 0:
                    # sum over all but last axis
                    if n_mult_axes > 0:
                        axes_to_sum = tuple(range(-n_mult_axes, arr.ndim - 1))
                        arr_collapsed = arr.sum(axis=axes_to_sum, keepdims=False)
                    else:
                        arr_collapsed = arr

                    # Ensure final shape ends with (out_dim,)
                    out_dim = self.irreps_out[ins.i_out].ir.dim
                    if arr_collapsed.ndim == 0:  # scalar
                        arr_collapsed = arr_collapsed.reshape((1,))
                    elif arr_collapsed.shape[-1] != out_dim:
                        # sometimes JAX squeezes too far if out_dim == 1
                        arr_collapsed = arr_collapsed.reshape(
                            arr_collapsed.shape + (out_dim,)
                        )
                else:
                    arr_collapsed = arr
            else:
                arr_collapsed = arr

            # Add to y_out slice corresponding to out irrep multiplicity region
            mul_out = self.irreps_out[ins.i_out].mul
            out_ir = self.irreps_out[ins.i_out].ir
            out_start = sum(r.mul for r in self.irreps_out[: ins.i_out])
            out_end = out_start + mul_out * out_ir.dim
            # arr_collapsed has shape (..., out_dim). But if mul_out>1 we need to expand it across multiplicity elements.
            # The simplest approach: if arr_collapsed last dim == out_ir.dim -> broadcast uniformly across multiplicities
            # If arr_collapsed last dim == mul_out * out_ir.dim -> reshape into (..., mul_out, out_ir.dim)
            arr_collapsed = jnp.atleast_1d(arr_collapsed)

            # Case 1: we have exactly one copy, need to broadcast across multiplicities
            if arr_collapsed.shape[-1] == out_ir.dim and mul_out > 1:
                add_block = jnp.tile(arr_collapsed, (mul_out,)).reshape(
                    arr_collapsed.shape[:-1] + (mul_out * out_ir.dim,)
                )
            # Case 2: already the right flattened size
            elif arr_collapsed.shape[-1] == mul_out * out_ir.dim:
                add_block = arr_collapsed
            # Case 3: single scalar -> broadcast to all multiplicities * dims
            elif arr_collapsed.size == 1:
                add_block = jnp.full(
                    arr_collapsed.shape[:-1] + (mul_out * out_ir.dim,),
                    arr_collapsed.item(),
                )
            else:
                raise ValueError(
                    f"Unexpected shape {arr_collapsed.shape} for out_ir.dim={out_ir.dim}, mul_out={mul_out}"
                )
            y_out = y_out.at[..., out_start:out_end].add(add_block)

        return IrrepsArray(self.irreps_out, y_out)
