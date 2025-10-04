"""Cue-equivariant tensor product implemented via segmented polynomials."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Iterable, Sequence
from typing import Optional

import cuequivariance as cue
import cuequivariance_jax as cuex
import haiku as hk
import jax
import jax.numpy as jnp
from e3nn_jax import Irreps

from mace_jax.e3nn._tensor_product._instruction import Instruction as E3Instruction


def _to_cue_irreps(group: cue.Group, irreps: Irreps) -> cue.Irreps:
    """Convert an :mod:`e3nn_jax` Irreps specification to a cuequivariance Irreps."""
    return cue.Irreps(group, str(Irreps(irreps)))


@dataclasses.dataclass
class _PathSpec:
    mode: str
    mul_in1: int
    mul_in2: int
    mul_out: int
    weight_shape: tuple[int, ...]
    poly_shape: tuple[int, int, int]
    basis: Optional[jnp.ndarray]

    def expand(self, weight: jnp.ndarray) -> jnp.ndarray:
        batch_ndim = weight.ndim - len(self.weight_shape)
        batch_shape = weight.shape[:batch_ndim]
        weight = jnp.reshape(weight, batch_shape + self.weight_shape)

        if self.mode == 'uvw':
            return jnp.reshape(weight, batch_shape + self.poly_shape)

        basis = self._broadcast_basis(weight.dtype, batch_shape)

        if self.mode == 'uvu':
            expanded = jnp.expand_dims(weight, axis=-1)
            return expanded * basis
        if self.mode == 'uvv':
            expanded = jnp.expand_dims(weight, axis=-1)
            return expanded * basis
        if self.mode == 'uuw':
            expanded = jnp.expand_dims(weight, axis=-2)
            return expanded * basis
        if self.mode == 'uuu':
            expanded = weight[..., :, None, None]
            return expanded * basis
        if self.mode == 'uvuv':
            expanded = jnp.expand_dims(weight, axis=-1)
            return expanded * basis

        raise NotImplementedError(f'Unsupported connection mode {self.mode!r}.')

    def _broadcast_basis(
        self, dtype: jnp.dtype, batch_shape: tuple[int, ...]
    ) -> jnp.ndarray:
        assert self.basis is not None
        basis = self.basis.astype(dtype)
        return jnp.reshape(basis, (1,) * len(batch_shape) + basis.shape)


def _make_path_spec(
    mode: str,
    mul_in1: int,
    mul_in2: int,
    mul_out: int,
    weight_shape: tuple[int, ...],
) -> _PathSpec:
    if mode == 'uvw':
        poly_shape = (mul_in1, mul_in2, mul_out)
        basis = None
    elif mode == 'uvu':
        poly_shape = (mul_in1, mul_in2, mul_out)
        eye = jnp.eye(mul_out, dtype=jnp.float32)
        basis = jnp.broadcast_to(eye[:, None, :], poly_shape)
        basis = basis / jnp.sqrt(mul_out)
    elif mode == 'uvv':
        poly_shape = (mul_in1, mul_in2, mul_out)
        eye = jnp.eye(mul_out, dtype=jnp.float32)
        basis = jnp.broadcast_to(eye[None, :, :], poly_shape)
    elif mode == 'uuw':
        if mul_in1 != mul_in2:
            raise ValueError('uuw mode requires equal input multiplicities.')
        poly_shape = (mul_in1, mul_in2, mul_out)
        eye = jnp.eye(mul_in1, dtype=jnp.float32)
        basis = jnp.broadcast_to(eye[:, :, None], poly_shape)
        basis = basis / jnp.sqrt(mul_in1)
    elif mode == 'uuu':
        if not (mul_in1 == mul_in2 == mul_out):
            raise ValueError('uuu mode requires equal multiplicities.')
        poly_shape = (mul_in1, mul_in2, mul_out)
        basis = jnp.zeros(poly_shape, dtype=jnp.float32)
        idx = jnp.arange(mul_in1)
        basis = basis.at[idx, idx, idx].set(1.0)
    elif mode == 'uvuv':
        if mul_out != mul_in1 * mul_in2:
            raise ValueError('uvuv mode expects mul_out == mul_in1 * mul_in2.')
        poly_shape = (mul_in1, mul_in2, mul_out)
        basis = jnp.zeros(poly_shape, dtype=jnp.float32)
        u_idx = jnp.repeat(jnp.arange(mul_in1), mul_in2)
        v_idx = jnp.tile(jnp.arange(mul_in2), mul_in1)
        w_idx = jnp.arange(mul_out)
        basis = basis.at[u_idx, v_idx, w_idx].set(1.0)
    else:
        raise NotImplementedError(f'Cue TensorProduct does not support mode {mode!r}.')

    return _PathSpec(
        mode=mode,
        mul_in1=mul_in1,
        mul_in2=mul_in2,
        mul_out=mul_out,
        weight_shape=weight_shape,
        poly_shape=poly_shape,
        basis=basis,
    )


def _infer_path_shape(
    mode: str, mul_in1: int, mul_in2: int, mul_out: int
) -> tuple[int, ...]:
    if mode == 'uvw':
        return (mul_in1, mul_in2, mul_out)
    if mode == 'uvu':
        return (mul_in1, mul_in2)
    if mode == 'uvv':
        return (mul_in1, mul_in2)
    if mode == 'uuw':
        return (mul_in1, mul_out)
    if mode == 'uuu':
        return (mul_in1,)
    if mode == 'uvuv':
        return (mul_in1, mul_in2)
    raise NotImplementedError(f'Unsupported connection mode {mode!r}.')


class TensorProduct(hk.Module):
    """Tensor product evaluated with :func:`cuequivariance_jax.segmented_polynomial`.

    The class mirrors the public API of :class:`mace_jax.e3nn._tensor_product.TensorProduct`
    for the subset of connection modes required by the CuEquivariance backend.
    Currently only ``'uvw'`` (fully-connected) and ``'uvu'`` (channelwise) modes are
    supported. All other connection modes raise ``NotImplementedError``.
    """

    def __init__(
        self,
        irreps_in1: Irreps,
        irreps_in2: Irreps,
        irreps_out: Irreps,
        instructions: Optional[Sequence] = None,
        *,
        shared_weights: bool = False,
        internal_weights: bool = False,
        cueq_config=None,
        method: Optional[str] = None,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name=name)

        self.irreps_in1 = Irreps(irreps_in1)
        self.irreps_in2 = Irreps(irreps_in2)
        self.irreps_out = Irreps(irreps_out)

        if cueq_config is not None and getattr(cueq_config, 'group', None) is not None:
            self._group = cueq_config.group
        else:
            self._group = cue.O3

        self._layout = getattr(cueq_config, 'layout', cue.ir_mul)
        self._method = method or getattr(cueq_config, 'tp_method', 'naive')

        if self._layout not in (cue.ir_mul, cue.mul_ir):
            raise ValueError(f'Unsupported CuEquivariance layout {self._layout}.')

        self._cue_irreps_in1 = _to_cue_irreps(self._group, self.irreps_in1)
        self._cue_irreps_in2 = _to_cue_irreps(self._group, self.irreps_in2)
        self._cue_irreps_out = _to_cue_irreps(self._group, self.irreps_out)

        self._cue_layout_in1 = cue.IrrepsAndLayout(
            self._cue_irreps_in1, cue.ir_mul
        )
        self._cue_layout_in2 = cue.IrrepsAndLayout(
            self._cue_irreps_in2, cue.ir_mul
        )
        self._cue_layout_out = cue.IrrepsAndLayout(
            self._cue_irreps_out, cue.ir_mul
        )

        self.instructions = self._normalize_instructions(instructions)

        (
            self._equivariant_polynomial,
            self._weight_slice_list,
            self._weight_shapes,
        ) = self._build_equivariant_polynomial()

        self.weight_numel = sum(slc.stop - slc.start for slc in self._weight_slice_list)

        if shared_weights is False and internal_weights is None:
            internal_weights = False
        if shared_weights is None:
            shared_weights = True
        if internal_weights is None:
            internal_weights = shared_weights and self.weight_numel > 0

        assert shared_weights or not internal_weights
        self.shared_weights = shared_weights
        self.internal_weights = internal_weights

        self._weight_initializer = (
            hk.initializers.RandomNormal() if self.weight_numel > 0 else None
        )

    # ---------------------------------------------------------------------
    # Instruction utilities
    # ---------------------------------------------------------------------
    def _normalize_instructions(
        self, instrs: Optional[Sequence]
    ) -> list[E3Instruction]:
        if instrs is None:
            instrs = self._default_instructions()

        normalized: list[E3Instruction] = []
        for ins in instrs:
            if isinstance(ins, E3Instruction):
                instruction = ins
            else:
                i_in1, i_in2, i_out, mode, has_weight, *rest = ins
                path_weight = float(rest[0]) if rest else 1.0
                instruction = E3Instruction(
                    int(i_in1),
                    int(i_in2),
                    int(i_out),
                    str(mode),
                    bool(has_weight),
                    path_weight,
                    (0,),
                )

            if not instruction.has_weight:
                raise NotImplementedError('Cue TensorProduct requires weighted paths.')

            mul_in1 = self.irreps_in1[instruction.i_in1].mul
            mul_in2 = self.irreps_in2[instruction.i_in2].mul
            mul_out = self.irreps_out[instruction.i_out].mul

            expected_shape = _infer_path_shape(
                instruction.connection_mode, mul_in1, mul_in2, mul_out
            )
            instruction = E3Instruction(
                instruction.i_in1,
                instruction.i_in2,
                instruction.i_out,
                instruction.connection_mode,
                instruction.has_weight,
                instruction.path_weight,
                expected_shape,
            )

            normalized.append(instruction)

        return normalized

    def _default_instructions(self) -> list[E3Instruction]:
        instructions: list[E3Instruction] = []
        for i_in1, (_, _) in enumerate(self.irreps_in1):
            for i_in2, (_, _) in enumerate(self.irreps_in2):
                for i_out, (_, ir_out) in enumerate(self.irreps_out):
                    if ir_out in self.irreps_in1[i_in1].ir * self.irreps_in2[i_in2].ir:
                        mul_in1 = self.irreps_in1[i_in1].mul
                        mul_in2 = self.irreps_in2[i_in2].mul
                        mul_out = self.irreps_out[i_out].mul
                        instructions.append(
                            E3Instruction(
                                i_in1,
                                i_in2,
                                i_out,
                                'uvw',
                                True,
                                1.0,
                                (mul_in1, mul_in2, mul_out),
                            )
                        )
        return instructions

    # ---------------------------------------------------------------------
    # Polynomial construction
    # ---------------------------------------------------------------------
    def _build_equivariant_polynomial(self):
        d = cue.SegmentedTensorProduct.from_subscripts('uvw,iu,jv,kw+ijk')

        for mul_ir in self._cue_irreps_in1:
            d.add_segment(1, (mul_ir.ir.dim, mul_ir.mul))
        for mul_ir in self._cue_irreps_in2:
            d.add_segment(2, (mul_ir.ir.dim, mul_ir.mul))
        for mul_ir in self._cue_irreps_out:
            d.add_segment(3, (mul_ir.ir.dim, mul_ir.mul))

        weight_slices: list[slice] = []
        weight_shapes: list[tuple[int, ...]] = []
        offset = 0

        self._path_specs = []
        poly_weight_shapes: list[tuple[int, int, int]] = []

        for ins in self.instructions:
            mul_ir_in1 = self._cue_irreps_in1[ins.i_in1]
            mul_ir_in2 = self._cue_irreps_in2[ins.i_in2]
            mul_ir_out = self._cue_irreps_out[ins.i_out]

            cg = self._group.clebsch_gordan(mul_ir_in1.ir, mul_ir_in2.ir, mul_ir_out.ir)
            if cg.shape[0] != 1:
                raise NotImplementedError(
                    'Multiple Clebsch-Gordan solutions are not supported.'
                )
            coeff = cg[0] * ins.path_weight

            spec = _make_path_spec(
                ins.connection_mode,
                mul_ir_in1.mul,
                mul_ir_in2.mul,
                mul_ir_out.mul,
                tuple(ins.path_shape),
            )
            self._path_specs.append(spec)
            poly_weight_shapes.append(spec.poly_shape)

            d.add_path(
                spec.poly_shape,
                ins.i_in1,
                ins.i_in2,
                ins.i_out,
                c=coeff,
            )

            size = int(jnp.prod(jnp.array(spec.weight_shape)))
            weight_slices.append(slice(offset, offset + size))
            weight_shapes.append(spec.weight_shape)
            offset += size

        equivariant_poly = cue.EquivariantPolynomial(
            [
                cue.IrrepsAndLayout(
                    self._cue_irreps_in1.new_scalars(
                        sum(math.prod(shape) for shape in poly_weight_shapes)
                    ),
                    cue.ir_mul,
                ),
                self._cue_layout_in1,
                self._cue_layout_in2,
            ],
            [self._cue_layout_out],
            cue.SegmentedPolynomial.eval_last_operand(d),
        )

        equivariant_poly = equivariant_poly.flatten_coefficient_modes().squeeze_modes()

        self._poly_weight_shapes = poly_weight_shapes
        self._poly_weight_numel = sum(math.prod(shape) for shape in poly_weight_shapes)

        return equivariant_poly, weight_slices, weight_shapes

    # ---------------------------------------------------------------------
    # Weight helpers
    # ---------------------------------------------------------------------
    def _prep_weights(self, weight):
        if isinstance(weight, list):
            if len(weight) != len(self._weight_shapes):
                raise ValueError(
                    f'Expected {len(self._weight_shapes)} weight tensors, got {len(weight)}.'
                )
            if self.shared_weights:
                flats = [
                    jnp.reshape(w, (math.prod(shape),))
                    for w, shape in zip(weight, self._weight_shapes)
                ]
                return jnp.concatenate(flats, axis=-1) if flats else jnp.zeros((0,))

            flats = [
                jnp.reshape(
                    w,
                    w.shape[: -len(shape)] + (math.prod(shape),),
                )
                for w, shape in zip(weight, self._weight_shapes)
            ]
            return jnp.concatenate(flats, axis=-1) if flats else jnp.zeros((0,))
        return weight

    def _get_weights(self, weight):
        if self.weight_numel == 0:
            return None

        weight = self._prep_weights(weight)

        if weight is None:
            if not self.internal_weights:
                raise RuntimeError(
                    'Weights must be provided when internal_weights=False.'
                )
            initializer = self._weight_initializer or hk.initializers.RandomNormal()
            weight = hk.get_parameter('weight', (self.weight_numel,), init=initializer)
        else:
            weight = jnp.asarray(weight)
            if self.shared_weights:
                if weight.shape != (self.weight_numel,):
                    raise ValueError(
                        f'Expected weight shape {(self.weight_numel,)}, got {weight.shape}.'
                    )
            else:
                if weight.shape[-1] != self.weight_numel:
                    raise ValueError(
                        f'Expected last weight dimension {self.weight_numel}, got {weight.shape[-1]}.'
                    )
        return weight

    def _split_weights(self, weight: jnp.ndarray) -> list[jnp.ndarray]:
        if self.weight_numel == 0:
            return []
        if self.shared_weights:
            return [
                weight[slc].reshape(shape)
                for slc, shape in zip(self._weight_slice_list, self._weight_shapes)
            ]
        return [
            weight[..., slc].reshape(weight.shape[:-1] + shape)
            for slc, shape in zip(self._weight_slice_list, self._weight_shapes)
        ]

    def _expand_weights(self, weight: jnp.ndarray) -> jnp.ndarray:
        if self.weight_numel == 0:
            if self.shared_weights:
                return jnp.zeros((0,), dtype=weight.dtype)
            return jnp.zeros(weight.shape[:-1] + (0,), dtype=weight.dtype)

        splitted = self._split_weights(weight)
        expanded = [spec.expand(part) for spec, part in zip(self._path_specs, splitted)]

        if self.shared_weights:
            return jnp.concatenate([arr.reshape(-1) for arr in expanded], axis=-1)

        return jnp.concatenate(
            [
                arr.reshape(arr.shape[: -len(spec.poly_shape)] + (-1,))
                for arr, spec in zip(expanded, self._path_specs)
            ],
            axis=-1,
        )

    # ---------------------------------------------------------------------
    # Forward evaluation
    # ---------------------------------------------------------------------
    def __call__(
        self,
        x1: jnp.ndarray,
        x2: jnp.ndarray,
        weight: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        x1 = jnp.asarray(x1)
        x2 = jnp.asarray(x2)

        if x1.shape[-1] != self.irreps_in1.dim:
            raise ValueError(
                f'First input has incompatible last dimension {x1.shape[-1]} (expected {self.irreps_in1.dim}).'
            )
        if x2.shape[-1] != self.irreps_in2.dim:
            raise ValueError(
                f'Second input has incompatible last dimension {x2.shape[-1]} (expected {self.irreps_in2.dim}).'
            )

        leading_shape = jnp.broadcast_shapes(x1.shape[:-1], x2.shape[:-1])
        x1 = jnp.broadcast_to(x1, leading_shape + (self.irreps_in1.dim,))
        x2 = jnp.broadcast_to(x2, leading_shape + (self.irreps_in2.dim,))

        weight = self._get_weights(weight)
        if self.weight_numel == 0:
            raise ValueError('Cue TensorProduct expects learnable weights.')

        expanded_weight = self._expand_weights(weight)

        desired_shape = leading_shape + (self._poly_weight_numel,)
        if expanded_weight.shape != desired_shape:
            broadcast_shape = (1,) * len(leading_shape) + (self._poly_weight_numel,)
            expanded_weight = jnp.reshape(expanded_weight, broadcast_shape)
            expanded_weight = jnp.broadcast_to(expanded_weight, desired_shape)

        x1_flat = jnp.reshape(x1, (-1, self.irreps_in1.dim))
        x2_flat = jnp.reshape(x2, (-1, self.irreps_in2.dim))
        weight_flat = jnp.reshape(expanded_weight, (-1, self._poly_weight_numel))

        out_dtype = x1.dtype

        def _evaluate_single(weight_row, x1_row, x2_row):
            weight_row = weight_row.astype(out_dtype)
            rep_x1 = cuex.RepArray({0: self._cue_layout_in1}, x1_row)
            rep_x2 = cuex.RepArray({0: self._cue_layout_in2}, x2_row)
            outputs = cuex.equivariant_polynomial(
                self._equivariant_polynomial,
                [weight_row, rep_x1, rep_x2],
                jax.ShapeDtypeStruct((self.irreps_out.dim,), out_dtype),
                method=self._method,
            )
            return outputs.array

        batched_eval = jax.vmap(_evaluate_single, in_axes=(0, 0, 0))

        with cue.assume(self._group, cue.ir_mul):
            outputs_flat = batched_eval(weight_flat, x1_flat, x2_flat)

        return jnp.reshape(outputs_flat, desired_shape[:-1] + (self.irreps_out.dim,))

    # ---------------------------------------------------------------------
    # Weight inspection utilities
    # ---------------------------------------------------------------------
    def weight_view_for_instruction(
        self,
        instruction: int,
        weight: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        if self.weight_numel == 0:
            raise ValueError('TensorProduct has no weights.')
        weight_array = self._get_weights(weight)
        slc = self._weight_slice_list[instruction]
        shape = self._weight_shapes[instruction]
        if self.shared_weights:
            return weight_array[slc].reshape(shape)
        return weight_array[..., slc].reshape(weight_array.shape[:-1] + shape)

    def weight_views(
        self,
        weight: Optional[jnp.ndarray] = None,
        *,
        yield_instruction: bool = False,
    ) -> Iterable:
        if self.weight_numel == 0:
            return []
        weight_array = self._get_weights(weight)

        def iterator():
            for idx, shape in enumerate(self._weight_shapes):
                view = self.weight_view_for_instruction(idx, weight_array)
                if yield_instruction:
                    yield idx, self.instructions[idx], view
                else:
                    yield view

        return iterator()

    # ---------------------------------------------------------------------
    # Unsupported operations
    # ---------------------------------------------------------------------
    def right(self, *args, **kwargs):  # pragma: no cover - compatibility stub
        raise NotImplementedError('The cue TensorProduct does not implement right().')
