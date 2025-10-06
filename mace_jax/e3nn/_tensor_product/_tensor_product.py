import math
from collections.abc import Iterable
from typing import Optional, Union

import e3nn_jax as e3nn
import haiku as hk
import jax
import jax.numpy as jnp
from e3nn_jax import Irreps
from e3nn_jax.legacy import FunctionalTensorProduct
from e3nn_jax.utils import vmap as e3nn_vmap

from mace_jax.haiku.torch import register_import

from ._instruction import Instruction


def _weight_list_to_flat(
    weights: list[jnp.ndarray], weight_shapes: list[tuple[int, ...]], shared: bool
) -> jnp.ndarray:
    if not weights:
        return jnp.zeros((0,), dtype=jnp.float32)

    if shared:
        flats = [
            jnp.reshape(w, (math.prod(tuple(int(x) for x in shape)),))
            for w, shape in zip(weights, weight_shapes)
        ]
        return jnp.concatenate(flats, axis=-1)

    leading_shape = None
    flats = []
    for w, shape in zip(weights, weight_shapes):
        w = jnp.asarray(w)
        shape = tuple(int(x) for x in shape)
        curr_leading = w.shape[: w.ndim - len(shape)]
        if leading_shape is None:
            leading_shape = curr_leading
        elif curr_leading != leading_shape:
            raise ValueError('All weight tensors must share the same leading shape.')
        flats.append(jnp.reshape(w, curr_leading + (math.prod(shape),)))

    return jnp.concatenate(flats, axis=-1)


def _as_functional_instruction(ins) -> tuple[int, int, int, str, bool, float]:
    """Convert any legacy instruction representation into the 6-tuple expected by
    :class:`e3nn_jax.legacy.FunctionalTensorProduct`.

    The original ``TensorProduct`` class accepted instructions in multiple
    formats (our ``Instruction`` NamedTuple, objects coming directly from the
    functional backend, or plain tuples/lists with optional ``path_weight``).
    The helper collapses all of these into the canonical functional form so we
    can delegate the math to ``FunctionalTensorProduct`` while keeping the old
    user-facing API intact.
    """
    if isinstance(ins, Instruction):
        return (
            ins.i_in1,
            ins.i_in2,
            ins.i_out,
            ins.connection_mode,
            bool(ins.has_weight),
            float(ins.path_weight),
        )

    attrs = ('i_in1', 'i_in2', 'i_out', 'connection_mode', 'has_weight', 'path_weight')
    if all(hasattr(ins, attr) for attr in attrs):
        return (
            int(getattr(ins, 'i_in1')),
            int(getattr(ins, 'i_in2')),
            int(getattr(ins, 'i_out')),
            getattr(ins, 'connection_mode'),
            bool(getattr(ins, 'has_weight')),
            float(getattr(ins, 'path_weight')),
        )

    seq = tuple(ins)
    if len(seq) == 5:
        seq = seq + (1.0,)
    elif len(seq) >= 6:
        seq = seq[:6]
    else:
        raise ValueError('Instruction must have at least 5 fields.')
    return (
        int(seq[0]),
        int(seq[1]),
        int(seq[2]),
        seq[3],
        bool(seq[4]),
        float(seq[5]),
    )


# @register_import('e3nn.o3._tensor_product._tensor_product.TensorProduct')
class TensorProduct(hk.Module):
    """Haiku wrapper around :class:`e3nn_jax.legacy.FunctionalTensorProduct`."""

    def __init__(
        self,
        irreps_in1: Irreps,
        irreps_in2: Irreps,
        irreps_out: Irreps,
        instructions: list[tuple],
        in1_var: Optional[Union[list[float], jnp.ndarray]] = None,
        in2_var: Optional[Union[list[float], jnp.ndarray]] = None,
        out_var: Optional[Union[list[float], jnp.ndarray]] = None,
        irrep_normalization: Optional[str] = None,
        path_normalization: Optional[str] = None,
        internal_weights: Optional[bool] = None,
        shared_weights: Optional[bool] = None,
        normalization=None,  # deprecated e3nn API compatibility
        _specialized_code: Optional[bool] = None,
        _optimize_einsums: Optional[bool] = None,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name=name)

        if normalization is not None:
            irrep_normalization = normalization

        raw_instructions = [] if instructions is None else list(instructions)
        functional_instructions_input = [
            _as_functional_instruction(ins) for ins in raw_instructions
        ]

        self._functional = FunctionalTensorProduct(
            irreps_in1=irreps_in1,
            irreps_in2=irreps_in2,
            irreps_out=irreps_out,
            instructions=functional_instructions_input,
            in1_var=in1_var,
            in2_var=in2_var,
            out_var=out_var,
            irrep_normalization=irrep_normalization,
            path_normalization=path_normalization,
        )

        functional_instructions = list(self._functional.instructions)
        self.instructions = [
            Instruction(
                ins.i_in1,
                ins.i_in2,
                ins.i_out,
                ins.connection_mode,
                bool(ins.has_weight),
                float(ins.path_weight),
                tuple(int(x) for x in ins.path_shape),
            )
            for ins in functional_instructions
        ]
        self.irreps_in1 = self._functional.irreps_in1
        self.irreps_in2 = self._functional.irreps_in2
        self.irreps_out = self._functional.irreps_out
        self.output_mask = self._functional.output_mask.astype(jnp.float32)

        self._weight_slice_list: list[slice] = []
        self._weight_shapes: list[tuple[int, ...]] = []
        self._instruction_slices: list[Optional[slice]] = []
        self._instruction_shapes: list[Optional[tuple[int, ...]]] = []

        offset = 0
        for ins in functional_instructions:
            shape = tuple(int(x) for x in ins.path_shape)
            if ins.has_weight:
                size = math.prod(shape)
                slc = slice(offset, offset + size)
                self._weight_slice_list.append(slc)
                self._weight_shapes.append(shape)
                self._instruction_slices.append(slc)
                self._instruction_shapes.append(shape)
                offset += size
            else:
                self._instruction_slices.append(None)
                self._instruction_shapes.append(None)

        self.weight_numel = offset

        if shared_weights is False and internal_weights is None:
            internal_weights = False
        if shared_weights is None:
            shared_weights = True
        if internal_weights is None:
            internal_weights = shared_weights and self.weight_numel > 0
        assert shared_weights or not internal_weights
        self.shared_weights = shared_weights
        self.internal_weights = internal_weights

        if self.weight_numel > 0:
            self._weight_initializer = hk.initializers.RandomNormal()
        else:
            self._weight_initializer = None

    def __repr__(self) -> str:
        npath = sum(math.prod(ins.path_shape) for ins in self.instructions)
        return (
            f'{self.__class__.__name__}'
            f'({self.irreps_in1.simplify()} x {self.irreps_in2.simplify()} '
            f'-> {self.irreps_out.simplify()} | {npath} paths | {self.weight_numel} weights)'
        )

    # ------------------------------------------------------------------
    # Weight helpers
    # ------------------------------------------------------------------
    def _prep_weights(
        self, weight: Optional[Union[jnp.ndarray, list[jnp.ndarray]]]
    ) -> Optional[jnp.ndarray]:
        if isinstance(weight, list):
            return _weight_list_to_flat(
                weight, self._weight_shapes, self.shared_weights
            )
        return weight

    def _get_weights(
        self, weight: Optional[Union[jnp.ndarray, list[jnp.ndarray]]]
    ) -> Optional[jnp.ndarray]:
        weight = self._prep_weights(weight)

        if self.weight_numel == 0:
            return None

        if weight is None:
            if not self.internal_weights:
                raise RuntimeError(
                    'Weights must be provided when the TensorProduct does not have internal_weights.'
                )
            init = self._weight_initializer or hk.initializers.RandomNormal()
            weight = hk.get_parameter('weight', (self.weight_numel,), init=init)
        else:
            weight = jnp.asarray(weight)
            if self.shared_weights:
                if weight.ndim != 1 or weight.shape[0] != self.weight_numel:
                    raise ValueError(
                        f'Invalid weight shape {weight.shape}; expected {(self.weight_numel,)}.'
                    )
            else:
                if weight.shape[-1] != self.weight_numel:
                    raise ValueError(
                        f'Invalid weight shape {weight.shape}; last dim must be {self.weight_numel}.'
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

    def _weight_view_from_array(
        self, instruction: int, weight_array: jnp.ndarray
    ) -> jnp.ndarray:
        slc = self._instruction_slices[instruction]
        shape = self._instruction_shapes[instruction]
        if slc is None or shape is None:
            raise ValueError(f'Instruction {instruction} has no weights.')
        if self.shared_weights:
            return weight_array[slc].reshape(shape)
        return weight_array[..., slc].reshape(weight_array.shape[:-1] + shape)

    # ------------------------------------------------------------------
    # Forward calls
    # ------------------------------------------------------------------
    def __call__(
        self,
        x1: jnp.ndarray,
        x2: jnp.ndarray,
        weight: Optional[Union[jnp.ndarray, list[jnp.ndarray]]] = None,
    ) -> jnp.ndarray:
        assert x1.shape[-1] == self.irreps_in1.dim
        assert x2.shape[-1] == self.irreps_in2.dim

        weight = self._get_weights(weight)

        x1_ir = e3nn.IrrepsArray(self.irreps_in1, jnp.asarray(x1))
        x2_ir = e3nn.IrrepsArray(self.irreps_in2, jnp.asarray(x2))

        leading_shape = jnp.broadcast_shapes(x1_ir.shape[:-1], x2_ir.shape[:-1])
        x1_ir = x1_ir.broadcast_to(leading_shape + (-1,))
        x2_ir = x2_ir.broadcast_to(leading_shape + (-1,))

        if self.weight_numel == 0:

            def core(a, b):
                return self._functional.left_right([], a, b)

            args = (x1_ir, x2_ir)
        elif self.shared_weights:

            def core(a, b):
                return self._functional.left_right(weight, a, b)

            args = (x1_ir, x2_ir)
        else:
            weight = jnp.asarray(weight)
            if weight.shape[:-1] != leading_shape:
                weight = jnp.broadcast_to(weight, leading_shape + (self.weight_numel,))

            def core(w, a, b):
                return self._functional.left_right(w, a, b)

            args = (weight, x1_ir, x2_ir)

        mapped = core
        for _ in range(len(leading_shape)):
            mapped = e3nn_vmap(mapped)

        output = mapped(*args)
        return output.array

    def right(
        self,
        y: jnp.ndarray,
        weight: Optional[Union[jnp.ndarray, list[jnp.ndarray]]] = None,
    ) -> jnp.ndarray:
        assert y.shape[-1] == self.irreps_in2.dim

        weight = self._get_weights(weight)

        def call_right(
            weight_vec: Optional[jnp.ndarray], y_vec: jnp.ndarray
        ) -> jnp.ndarray:
            weights_list = (
                [] if self.weight_numel == 0 else self._split_weights(weight_vec)
            )
            return self._functional.right(
                weights_list,
                e3nn.IrrepsArray(self.irreps_in2, y_vec),
            )

        leading_shape = y.shape[:-1]

        if len(leading_shape) == 0:
            weights_arg = None if self.weight_numel == 0 else weight
            return call_right(weights_arg, jnp.asarray(y))

        if self.weight_numel == 0:

            def fun(y_elem):
                return call_right(None, y_elem)

            mapped = fun
            for _ in range(len(leading_shape)):
                mapped = jax.vmap(mapped)
            return mapped(jnp.asarray(y))

        if self.shared_weights:

            def fun(y_elem):
                return call_right(weight, y_elem)

            mapped = fun
            for _ in range(len(leading_shape)):
                mapped = jax.vmap(mapped)
            return mapped(jnp.asarray(y))

        weight_array = jnp.asarray(weight)
        if weight_array.shape[:-1] != leading_shape:
            weight_array = jnp.broadcast_to(
                weight_array, leading_shape + (self.weight_numel,)
            )

        mapped = call_right
        for _ in range(len(leading_shape)):
            mapped = jax.vmap(mapped)
        return mapped(weight_array, jnp.asarray(y))

    # ------------------------------------------------------------------
    # Weight utilities
    # ------------------------------------------------------------------
    @classmethod
    def import_from_torch(cls, torch_module, hk_params, scope):
        hk_params = hk.data_structures.to_mutable_dict(hk_params)
        if torch_module.weight_numel > 0 and torch_module.internal_weights:
            hk_params[scope]['weight'] = jnp.array(
                torch_module.weight.detach().cpu().numpy()
            )
        return hk.data_structures.to_immutable_dict(hk_params)

    def weight_view_for_instruction(
        self,
        instruction: int,
        weight: Optional[Union[jnp.ndarray, list[jnp.ndarray]]] = None,
    ) -> jnp.ndarray:
        weight_array = self._get_weights(weight)
        if weight_array is None:
            raise ValueError('No weights available for this tensor product.')
        return self._weight_view_from_array(instruction, weight_array)

    def weight_views(
        self,
        weight: Optional[Union[jnp.ndarray, list[jnp.ndarray]]] = None,
        *,
        yield_instruction: bool = False,
    ) -> Iterable[Union[jnp.ndarray, tuple[int, Instruction, jnp.ndarray]]]:
        if self.weight_numel == 0:
            return ()

        weight_array = self._get_weights(weight)
        if weight_array is None:
            raise ValueError('No weights available for this tensor product.')

        def iterator():
            for idx, ins in enumerate(self.instructions):
                if not ins.has_weight:
                    continue
                view = self._weight_view_from_array(idx, weight_array)
                if yield_instruction:
                    yield idx, ins, view
                else:
                    yield view

        return iterator()
