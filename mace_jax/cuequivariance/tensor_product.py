"""Cue-equivariant tensor product implemented via segmented polynomials."""

from __future__ import annotations

import math
from typing import Iterable, Optional, Sequence

import haiku as hk
import jax
import jax.numpy as jnp

import cuequivariance as cue
import cuequivariance_jax as cuex
from e3nn_jax import Irreps

from mace_jax.e3nn._tensor_product._instruction import Instruction as E3Instruction


def _to_cue_irreps(group: cue.Group, irreps: Irreps) -> cue.Irreps:
    """Convert an :mod:`e3nn_jax` Irreps specification to a cuequivariance Irreps."""
    return cue.Irreps(group, str(Irreps(irreps)))


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
            hk.initializers.RandomNormal()
            if self.weight_numel > 0
            else None
        )

    # ---------------------------------------------------------------------
    # Instruction utilities
    # ---------------------------------------------------------------------
    def _normalize_instructions(self, instrs: Optional[Sequence]) -> list[E3Instruction]:
        if instrs is None:
            instrs = self._default_instructions()

        normalized: list[E3Instruction] = []
        for ins in instrs:
            if isinstance(ins, E3Instruction):
                instruction = ins
            else:
                # tuple/list form
                i_in1, i_in2, i_out, mode, has_weight, *rest = ins
                path_weight = float(rest[0]) if rest else 1.0
                instruction = E3Instruction(
                    int(i_in1),
                    int(i_in2),
                    int(i_out),
                    str(mode),
                    bool(has_weight),
                    path_weight,
                    (0,),  # placeholder
                )

            if not instruction.has_weight:
                raise NotImplementedError('Cue TensorProduct requires weighted paths.')

            if instruction.connection_mode == 'uvu':
                mul_in1 = self.irreps_in1[instruction.i_in1].mul
                mul_in2 = self.irreps_in2[instruction.i_in2].mul
                mul_out = self.irreps_out[instruction.i_out].mul
                if mul_out != mul_in1:
                    raise ValueError(
                        'Expected output multiplicity to match first input for "uvu" paths.'
                    )
                instruction = E3Instruction(
                    instruction.i_in1,
                    instruction.i_in2,
                    instruction.i_out,
                    'uvw',
                    instruction.has_weight,
                    instruction.path_weight,
                    (mul_in1, mul_in2, mul_out),
                )
            elif instruction.connection_mode == 'uvw':
                mul_in1 = self.irreps_in1[instruction.i_in1].mul
                mul_in2 = self.irreps_in2[instruction.i_in2].mul
                mul_out = self.irreps_out[instruction.i_out].mul
                path_shape = (
                    mul_in1,
                    mul_in2,
                    mul_out,
                )
                instruction = E3Instruction(
                    instruction.i_in1,
                    instruction.i_in2,
                    instruction.i_out,
                    instruction.connection_mode,
                    instruction.has_weight,
                    instruction.path_weight,
                    path_shape,
                )
            else:
                raise NotImplementedError(
                    f'Unsupported connection mode {instruction.connection_mode!r} '
                    'for CuEquivariance tensor product.'
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

        for ins in self.instructions:
            mul_ir_in1 = self._cue_irreps_in1[ins.i_in1]
            mul_ir_in2 = self._cue_irreps_in2[ins.i_in2]
            mul_ir_out = self._cue_irreps_out[ins.i_out]

            cg = self._group.clebsch_gordan(
                mul_ir_in1.ir, mul_ir_in2.ir, mul_ir_out.ir
            )
            if cg.shape[0] != 1:
                raise NotImplementedError(
                    'Multiple Clebsch-Gordan solutions are not supported.'
                )
            coeff = cg[0] * ins.path_weight

            d.add_path(
                ins.path_shape,
                ins.i_in1,
                ins.i_in2,
                ins.i_out,
                c=coeff,
            )

            size = int(jnp.prod(jnp.array(ins.path_shape)))
            weight_slices.append(slice(offset, offset + size))
            weight_shapes.append(ins.path_shape)
            offset += size

        equivariant_poly = cue.EquivariantPolynomial(
            [
                cue.IrrepsAndLayout(
                    self._cue_irreps_in1.new_scalars(offset), cue.ir_mul
                ),
                cue.IrrepsAndLayout(self._cue_irreps_in1, cue.ir_mul),
                cue.IrrepsAndLayout(self._cue_irreps_in2, cue.ir_mul),
            ],
            [cue.IrrepsAndLayout(self._cue_irreps_out, cue.ir_mul)],
            cue.SegmentedPolynomial.eval_last_operand(d),
        )

        # Flatten coefficient modes for better interoperability with segmented_polynomial
        equivariant_poly = (
            equivariant_poly.flatten_coefficient_modes().squeeze_modes()
        )

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
                    w.shape[:-len(shape)]
                    + (math.prod(shape),),
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

        if not self.shared_weights:
            if weight.shape[:-1] != leading_shape:
                weight = jnp.broadcast_to(weight, leading_shape + (self.weight_numel,))
        outputs_shape = jax.ShapeDtypeStruct(
            leading_shape + (self.irreps_out.dim,), x1.dtype
        )

        with cue.assume(self._group, cue.ir_mul):
            rep_x1 = cuex.RepArray(
                {x1.ndim - 1: cue.IrrepsAndLayout(self._cue_irreps_in1, cue.ir_mul)},
                x1,
            )
            rep_x2 = cuex.RepArray(
                {x2.ndim - 1: cue.IrrepsAndLayout(self._cue_irreps_in2, cue.ir_mul)},
                x2,
            )
            outputs = cuex.equivariant_polynomial(
                self._equivariant_polynomial,
                [weight, rep_x1, rep_x2],
                outputs_shape,
                method=self._method,
            )

        return outputs.array

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
