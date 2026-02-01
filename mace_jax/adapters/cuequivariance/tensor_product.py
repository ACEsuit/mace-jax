"""Cue-equivariant tensor product implemented via segmented polynomials (Flax)."""

from __future__ import annotations

from collections import OrderedDict
from math import prod as _prod

import cuequivariance_jax as cuex
import jax
import jax.numpy as jnp
import numpy as np
from e3nn_jax import Irreps  # type: ignore
from flax import nnx

import cuequivariance as cue
from mace_jax.adapters.nnx.torch import nxx_auto_import_from_torch
from mace_jax.nnx_config import ConfigVar
from mace_jax.tools.cg import O3_e3nn
from mace_jax.tools.dtype import default_dtype
from mace_jax.tools.scatter import scatter_sum

from .ir_dict import IR_DICT, is_ir_dict, mul_ir_to_ir_dict
from .utility import collapse_ir_mul_segments, ir_mul_to_mul_ir, mul_ir_to_ir_mul


def _expected_channelwise_instructions(
    irreps_in1: Irreps, irreps_in2: Irreps, target_irreps: Irreps
) -> tuple[Irreps, list[tuple[int, int, int, str, bool, float]]]:
    """Return the irreps and instructions for channel-wise tensor products.

    The channel-wise tensor product considered here mimics the ``uvu`` path
    returned by e3nn: each entry pairs a multiplicity channel from ``irreps_in1``
    with the compatible irreps from ``irreps_in2`` under Clebsch–Gordan fusion.

    Args:
        irreps_in1: Irreps carried by the first argument of the tensor product.
        irreps_in2: Irreps carried by the second argument.
        target_irreps: Expected output irreps; used as a filter.

    Returns:
        A tuple containing the sorted output irreps and a list of instruction
        tuples ``(i_in1, i_in2, i_out, mode, has_weight, path_weight)`` matching
        the conventions of cue/e3nn tensor products.
    """
    collected: list[tuple[int, Irreps]] = []
    instructions: list[tuple[int, int, int, str, bool, float]] = []
    for i_in1, (mul_in1, ir_in1) in enumerate(irreps_in1):
        for i_in2, (_, ir_in2) in enumerate(irreps_in2):
            for ir_out in ir_in1 * ir_in2:
                if ir_out in target_irreps:
                    idx = len(collected)
                    collected.append((mul_in1, ir_out))
                    instructions.append((i_in1, i_in2, idx, 'uvu', True, 1.0))

    irreps_out = Irreps(collected)
    irreps_out_sorted, perm, _ = irreps_out.sort()
    remapped_instructions = [
        (i_in1, i_in2, perm[i_out], mode, has_weight, path_weight)
        for i_in1, i_in2, i_out, mode, has_weight, path_weight in instructions
    ]
    remapped_instructions.sort(key=lambda item: item[2])
    return irreps_out_sorted, remapped_instructions


def _normalise_instruction(inst) -> tuple[int, int, int, str, bool, float]:
    """Ensure the instruction tuple conforms to the canonical six-field format.

    Accepts both the five-element format used by e3nn (omitting ``path_weight``)
    and the expanded representation.
    """
    if len(inst) == 5:
        i1, i2, i_out, mode, has_weight = inst
        path_weight = 1.0
    elif len(inst) == 6:
        i1, i2, i_out, mode, has_weight, path_weight = inst
    else:
        raise ValueError(
            'TensorProduct instructions must have length 5 or 6, '
            f'got length {len(inst)}'
        )
    return (
        int(i1),
        int(i2),
        int(i_out),
        str(mode),
        bool(has_weight),
        float(path_weight),
    )


@nxx_auto_import_from_torch(allow_missing_mapper=True)
class TensorProduct(nnx.Module):
    """Channel-wise tensor product evaluated with cuequivariance-jax.

    This module wraps the cue channel-wise tensor product descriptor, taking two
    inputs each organised in mul_ir order and returning an output in the same
    convention.  The contraction proceeds per irrep block, mirroring the
    ``uvu`` instructions produced by e3nn.  Weight handling supports both
    internal parameters and external arrays with optional sharing across the
    batch dimension.
    """

    irreps_in1: Irreps
    irreps_in2: Irreps
    irreps_out: Irreps
    shared_weights: bool = False
    internal_weights: bool = False
    instructions: list[tuple[int, int, int, str, bool, float]] | None = None
    conv_fusion: bool = False
    group: object = O3_e3nn
    layout: object = cue.mul_ir

    def __init__(
        self,
        irreps_in1: Irreps,
        irreps_in2: Irreps,
        irreps_out: Irreps,
        shared_weights: bool = False,
        internal_weights: bool = False,
        instructions: list[tuple[int, int, int, str, bool, float]] | None = None,
        conv_fusion: bool = False,
        group: object = O3_e3nn,
        layout: object = cue.mul_ir,
        *,
        rngs: nnx.Rngs | None = None,
    ) -> None:
        self.irreps_in1 = irreps_in1
        self.irreps_in2 = irreps_in2
        self.irreps_out = irreps_out
        self.shared_weights = shared_weights
        self.internal_weights = internal_weights
        self.instructions = instructions
        self.conv_fusion = conv_fusion
        self.group = group
        self.layout = layout
        # Initialise cue descriptors and validate the instruction template.
        if self.internal_weights and not self.shared_weights:
            raise ValueError(
                'TensorProduct requires shared_weights=True when internal_weights=True'
            )
        self._shared_weights = self.shared_weights
        self._internal_weights = self.internal_weights

        self.irreps_in1_o3 = Irreps(self.irreps_in1)
        self.irreps_in2_o3 = Irreps(self.irreps_in2)
        self.irreps_out_o3 = Irreps(self.irreps_out)
        self._api_layout, self._layout_str = self._resolve_layout(self.layout)

        self.irreps_in1_cue = cue.Irreps(self.group, self.irreps_in1_o3)
        self.irreps_in2_cue = cue.Irreps(self.group, self.irreps_in2_o3)
        self.irreps_out_cue = cue.Irreps(self.group, self.irreps_out_o3)

        descriptor = cue.descriptors.channelwise_tensor_product(
            self.irreps_in1_cue, self.irreps_in2_cue, self.irreps_out_cue
        )
        self.descriptor = descriptor
        self._ir_dict_poly = (
            descriptor.split_operand_by_irrep(2)
            .split_operand_by_irrep(1)
            .split_operand_by_irrep(-1)
            .polynomial
        )
        self.weight_irreps = descriptor.inputs[0].irreps
        self.weight_numel = descriptor.polynomial.operands[0].size
        self.descriptor_out_irreps_str = str(descriptor.outputs[0].irreps)
        self.output_segment_shapes = tuple(descriptor.polynomial.operands[-1].segments)
        self.descriptor_out_dim = Irreps(self.descriptor_out_irreps_str).dim
        layout_code = 0 if self._layout_str == 'mul_ir' else 1
        self.layout_config = ConfigVar(
            jnp.asarray(layout_code, dtype=jnp.int32),
            is_mutable=False,
        )

        expected_irreps, expected_instructions = _expected_channelwise_instructions(
            self.irreps_in1_o3, self.irreps_in2_o3, self.irreps_out_o3
        )
        if expected_irreps != self.irreps_out_o3:
            raise ValueError(
                'TensorProduct irreps_out is incompatible with channel-wise descriptor'
            )

        if self.instructions is not None:
            normalised = [_normalise_instruction(inst) for inst in self.instructions]
            if normalised != expected_instructions:
                raise ValueError(
                    'TensorProduct only supports channel-wise "uvu" instructions '
                    'matching those returned by e3nn; received '
                    f'{self.instructions!r}'
                )

        self._conv_fusion = bool(self.conv_fusion)
        self._conv_method = 'naive'

        conv_polynomial = None
        if self._conv_fusion:
            try:
                conv_descriptor = (
                    cue.descriptors.channelwise_tensor_product(
                        self.irreps_in1_cue,
                        self.irreps_in2_cue,
                        self.irreps_out_cue,
                    )
                    .flatten_coefficient_modes()
                    .squeeze_modes()
                )
                conv_polynomial = conv_descriptor.polynomial
            except ValueError:
                conv_polynomial = None
        self._conv_polynomial = conv_polynomial
        if self._internal_weights:
            if rngs is None:
                raise ValueError('rngs is required when internal_weights=True')
            self.weight = nnx.Param(
                jax.random.normal(
                    rngs(),
                    (1, self.weight_numel),
                    dtype=default_dtype(),
                )
            )
        else:
            self.weight = None

    def _weight_param(self) -> jnp.ndarray:
        """Create the shared/internal weight parameter."""
        if self.weight is None:
            raise ValueError('Internal weights are not initialized for TensorProduct.')
        return self.weight

    def _as_rep(
        self,
        array: jnp.ndarray,
        irreps_o3: Irreps,
        irreps_cue: cue.Irreps,
    ) -> cuex.RepArray:
        """Convert array in configured layout to cue RepArray with matching metadata."""
        if self._api_layout == cue.mul_ir:
            payload = mul_ir_to_ir_mul(array, irreps_o3)
        elif self._api_layout == cue.ir_mul:
            payload = array
        else:
            raise ValueError(
                f'TensorProduct does not support layout {self._api_layout!r}.'
            )
        return cuex.RepArray(irreps_cue, jnp.asarray(payload), cue.ir_mul)

    @staticmethod
    def _resolve_layout(layout_obj: object) -> tuple[cue.IrrepsLayout, str]:
        if isinstance(layout_obj, str):
            if layout_obj not in {'mul_ir', 'ir_mul'}:
                raise ValueError(
                    f"TensorProduct received unsupported layout string '{layout_obj}'."
                )
            return getattr(cue, layout_obj), layout_obj
        if layout_obj == cue.mul_ir:
            return layout_obj, 'mul_ir'
        if layout_obj == cue.ir_mul:
            return layout_obj, 'ir_mul'
        raise ValueError(
            'TensorProduct received an unknown layout object; expected cue.mul_ir '
            'or cue.ir_mul.'
        )

    def _resolve_weight_tensor(
        self,
        weights: jnp.ndarray | None,
        *,
        dtype: jnp.dtype,
        batch_size: int,
    ) -> jnp.ndarray:
        """Return a validated weight tensor with shape ``(batch, weight_numel)``."""
        if self._internal_weights:
            if weights is not None:
                raise ValueError(
                    'TensorProduct uses internal weights; weights argument must be None'
                )
            tensor = self._weight_param().astype(dtype)
        else:
            if weights is None:
                raise ValueError(
                    'TensorProduct requires explicit weights when internal_weights=False'
                )
            tensor = jnp.asarray(weights, dtype=dtype)

        if tensor.ndim == 1:
            tensor = tensor[jnp.newaxis, :]
        elif tensor.ndim != 2:
            raise ValueError(f'Weights must have rank 1 or 2, got rank {tensor.ndim}')

        if tensor.shape[-1] != self.weight_numel:
            raise ValueError(
                f'Expected weights last dimension {self.weight_numel}, got {tensor.shape[-1]}'
            )

        leading = tensor.shape[0]
        if self._shared_weights:
            if leading not in (1, batch_size):
                raise ValueError(
                    'Shared weights require leading dimension 1 or equal to the batch size'
                )
            if leading == 1 and batch_size != 1:
                tensor = jnp.broadcast_to(tensor, (batch_size, self.weight_numel))
        else:
            if leading != batch_size:
                raise ValueError(
                    'Unshared weights require leading dimension equal to the batch size'
                )

        return tensor

    def _reshape_weights_for_ir_dict(self, weights: jnp.ndarray) -> jnp.ndarray:
        """Reshape weights into segmented form expected by ir_dict polynomials."""
        desc = self._ir_dict_poly.inputs[0]
        seg_shape = tuple(desc.segment_shape)
        expected = desc.num_segments * (int(_prod(seg_shape)) if seg_shape else 1)
        if weights.shape[-1] != expected:
            raise ValueError(
                'TensorProduct ir_dict weights mismatch: '
                f'expected {expected}, got {weights.shape[-1]}'
            )
        if not seg_shape:
            return weights.reshape(weights.shape[0], desc.num_segments)
        return weights.reshape(weights.shape[0], desc.num_segments, *seg_shape)

    def _prepare_ir_dict_inputs(
        self,
        x1: jnp.ndarray | dict,
        x2: dict,
    ) -> tuple[dict, dict]:
        """Convert mul_ir arrays to ir_dict inputs with per-irrep segmentation."""
        irreps_in1 = Irreps(self.irreps_in1_o3)
        irreps_in2 = Irreps(self.irreps_in2_o3)
        if is_ir_dict(x1):
            x1_dict = x1
        else:
            x1_dict = mul_ir_to_ir_dict(
                irreps_in1,
                x1,
                group=self.group,
                layout_str=self._layout_str,
            )
        if not is_ir_dict(x2):
            x2 = mul_ir_to_ir_dict(
                irreps_in2,
                x2,
                group=self.group,
                layout_str=self._layout_str,
            )
        input_descs = list(self._ir_dict_poly.inputs)
        if len(input_descs) != 1 + len(x1_dict) + len(x2):
            raise ValueError(
                'TensorProduct ir_dict input descriptors do not match expected inputs.'
            )
        x1_descs = input_descs[1 : 1 + len(x1_dict)]
        x2_descs = input_descs[1 + len(x1_dict) :]

        x1_order = [ir for _, ir in self.irreps_in1_cue]
        x2_order = [ir for _, ir in self.irreps_in2_cue]
        if set(x1_dict.keys()) != set(x1_order):
            raise ValueError('TensorProduct ir_dict inputs missing irreps for x1.')
        if set(x2.keys()) != set(x2_order):
            raise ValueError('TensorProduct ir_dict inputs missing irreps for x2.')

        def _reshape_leaf(value: jnp.ndarray, desc) -> jnp.ndarray:
            if value.ndim < 2:
                raise ValueError(
                    'TensorProduct ir_dict inputs must be at least rank-2.'
                )
            leading = value.shape[:-2]
            mul = int(value.shape[-2])
            ir_dim = int(value.shape[-1])
            total = mul * ir_dim
            seg_shape = tuple(desc.segment_shape)
            expected = desc.num_segments * (int(_prod(seg_shape)) if seg_shape else 1)
            if expected != total:
                raise ValueError(
                    'TensorProduct ir_dict input size mismatch: '
                    f'expected {expected}, got {total}.'
                )
            ir_mul = jnp.swapaxes(value, -1, -2)
            flat = ir_mul.reshape(*leading, total)
            if not seg_shape:
                return flat.reshape(*leading, desc.num_segments)
            return flat.reshape(*leading, desc.num_segments, *seg_shape)

        x1_dict = OrderedDict(
            (ir, _reshape_leaf(x1_dict[ir], desc))
            for ir, desc in zip(x1_order, x1_descs)
        )
        x2_dict = OrderedDict(
            (ir, _reshape_leaf(x2[ir], desc)) for ir, desc in zip(x2_order, x2_descs)
        )
        return x1_dict, x2_dict

    def _ir_dict_outputs_to_ir_mul(self, outputs) -> jnp.ndarray:
        """Convert ir_dict polynomial outputs into descriptor ir_mul layout."""
        if isinstance(outputs, dict):
            outputs_list = [outputs[ir] for (_, ir) in self.irreps_out_cue]
        else:
            outputs_list = list(outputs)

        if len(outputs_list) != len(self._ir_dict_poly.outputs):
            raise ValueError(
                'TensorProduct ir_dict outputs length mismatch: '
                f'expected {len(self._ir_dict_poly.outputs)}, got {len(outputs_list)}.'
            )

        flats: list[jnp.ndarray] = []
        for out, desc in zip(outputs_list, self._ir_dict_poly.outputs):
            if out.ndim < 1 + desc.ndim:
                raise ValueError(
                    'TensorProduct ir_dict output has too few dimensions for descriptor.'
                )
            leading = out.shape[: -(1 + desc.ndim)]
            flat = out.reshape(*leading, desc.size)
            flats.append(flat)

        if not flats:
            return jnp.zeros((0, 0), dtype=default_dtype())

        return jnp.concatenate(flats, axis=-1)

    def __call__(
        self,
        x1: jnp.ndarray | dict,
        x2: jnp.ndarray | dict,
        weights: jnp.ndarray | None = None,
        edge_index: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """Evaluate the tensor product on two mul_ir inputs.

        Under the hood each input is re-expressed in cue's ``ir_mul`` layout,
        multiplied via the segmented polynomial corresponding to the ``uvu``
        contraction, and the result is collapsed back into ``mul_ir`` order.

        Args:
            x1: First input batch in mul_ir ordering.
            x2: Second input batch in mul_ir ordering.
            weights: Optional external weights; required when
                ``internal_weights`` is ``False``.

        Returns:
            ``jax.numpy`` array carrying irreps ``self.irreps_out`` in mul_ir order.

        Raises:
            ValueError: On weight shape mismatches or invalid sharing policy.
        """
        if is_ir_dict(x1):
            sample = next(iter(x1.values()))
            dtype = sample.dtype
            batch_size = sample.shape[0]
        else:
            dtype = x1.dtype
            batch_size = x1.shape[0]

        if not self._conv_fusion:
            weight_tensor = self._resolve_weight_tensor(
                weights, dtype=dtype, batch_size=batch_size
            )
            if is_ir_dict(x1) and not is_ir_dict(x2):
                raise ValueError('TensorProduct expects ir_dict x2 when x1 is ir_dict.')
            if is_ir_dict(x2):
                out = self._channelwise_apply_ir_dict(
                    x1,
                    x2,
                    weight_tensor,
                    dtype=dtype,
                )
            else:
                out = self._channelwise_apply(
                    x1,
                    x2,
                    weight_tensor,
                    dtype=dtype,
                )
            object.__setattr__(self, '_conv_method', 'naive')
            return out

        if edge_index is None:
            raise ValueError('TensorProduct conv_fusion requires edge_index')
        edge_index = jnp.asarray(edge_index)
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError(
                'edge_index must have shape (2, num_edges); '
                f'received {edge_index.shape}'
            )

        sender = jnp.asarray(edge_index[0])
        receiver = jnp.asarray(edge_index[1])
        num_nodes = batch_size
        edge_batch = sender.shape[0]

        if is_ir_dict(x1):
            edge_x1 = None
        else:
            edge_x1 = x1[sender]
        weight_tensor = self._resolve_weight_tensor(
            weights, dtype=dtype, batch_size=edge_batch
        )

        if is_ir_dict(x1) and not is_ir_dict(x2):
            raise ValueError('TensorProduct expects ir_dict x2 when x1 is ir_dict.')
        if is_ir_dict(x2):
            fused_out = self._conv_fused_apply_ir_dict(
                node_feats=x1,
                edge_attrs=x2,
                weights=weight_tensor,
                sender=sender,
                receiver=receiver,
                num_nodes=num_nodes,
                dtype=dtype,
            )
            object.__setattr__(self, '_conv_method', 'uniform_1d')
            return fused_out
        fused_out: jnp.ndarray | None = None
        if self._conv_polynomial is not None:
            try:
                fused_out = self._conv_fused_apply(
                    node_feats=x1,
                    edge_attrs=x2,
                    weights=weight_tensor,
                    sender=sender,
                    receiver=receiver,
                    num_nodes=num_nodes,
                    dtype=dtype,
                )
            except (RuntimeError, ValueError, NotImplementedError):
                fused_out = None
        if fused_out is not None:
            object.__setattr__(self, '_conv_method', 'uniform_1d')
            return fused_out

        if is_ir_dict(x2):
            per_edge = self._channelwise_apply_ir_dict(
                edge_x1,
                x2,
                weight_tensor,
                dtype=dtype,
            )
        else:
            if edge_x1 is None:
                raise ValueError(
                    'TensorProduct array path requires array x1 when conv_fusion is disabled.'
                )
            per_edge = self._channelwise_apply(
                edge_x1,
                x2,
                weight_tensor,
                dtype=dtype,
            )
        aggregated = scatter_sum(per_edge, receiver, dim=0, dim_size=num_nodes)
        object.__setattr__(self, '_conv_method', 'naive')
        return aggregated

    def _channelwise_apply(
        self,
        x1: jnp.ndarray,
        x2: jnp.ndarray,
        weight_tensor: jnp.ndarray,
        *,
        dtype: jnp.dtype,
    ) -> jnp.ndarray:
        irreps_in1 = Irreps(self.irreps_in1_o3)
        irreps_in2 = Irreps(self.irreps_in2_o3)
        x1_rep = self._as_rep(x1, irreps_in1, self.irreps_in1_cue)
        x2_rep = self._as_rep(x2, irreps_in2, self.irreps_in2_cue)
        weight_rep = cuex.RepArray(self.weight_irreps, weight_tensor, cue.ir_mul)

        output_rep = cuex.equivariant_polynomial(
            self.descriptor,
            [weight_rep, x1_rep, x2_rep],
            math_dtype=dtype,
            method='naive',
        )
        out_ir_mul = collapse_ir_mul_segments(
            output_rep.array,
            Irreps(self.descriptor_out_irreps_str),
            Irreps(self.irreps_out_o3),
            self.output_segment_shapes,
        )
        if self._api_layout == cue.ir_mul:
            return out_ir_mul
        return ir_mul_to_mul_ir(out_ir_mul, Irreps(self.irreps_out_o3))

    def _channelwise_apply_ir_dict(
        self,
        x1: jnp.ndarray | dict,
        x2: dict,
        weight_tensor: jnp.ndarray,
        *,
        dtype: jnp.dtype,
    ) -> jnp.ndarray:
        """Evaluate tensor product using ir_dict inputs for edge attributes."""
        x1_dict, x2_dict = self._prepare_ir_dict_inputs(x1, x2)
        weights = self._reshape_weights_for_ir_dict(weight_tensor)
        batch_size = next(iter(x1_dict.values())).shape[0]
        outputs = [
            jax.ShapeDtypeStruct(
                (batch_size, desc.num_segments) + tuple(desc.segment_shape),
                dtype,
            )
            for desc in self._ir_dict_poly.outputs
        ]
        y = IR_DICT.segmented_polynomial_uniform_1d(
            self._ir_dict_poly,
            [weights, x1_dict, x2_dict],
            outputs,
            math_dtype=dtype,
        )
        out_descriptor = self._ir_dict_outputs_to_ir_mul(y)
        out_ir_mul = collapse_ir_mul_segments(
            out_descriptor,
            Irreps(self.descriptor_out_irreps_str),
            Irreps(self.irreps_out_o3),
            self.output_segment_shapes,
        )
        if self._api_layout == cue.ir_mul:
            return out_ir_mul
        return ir_mul_to_mul_ir(out_ir_mul, Irreps(self.irreps_out_o3))

    def _conv_fused_apply_ir_dict(
        self,
        *,
        node_feats: jnp.ndarray,
        edge_attrs: dict,
        weights: jnp.ndarray,
        sender: jnp.ndarray,
        receiver: jnp.ndarray,
        num_nodes: int,
        dtype: jnp.dtype,
    ) -> jnp.ndarray | None:
        x1_dict, x2_dict = self._prepare_ir_dict_inputs(node_feats, edge_attrs)
        w = self._reshape_weights_for_ir_dict(weights)
        outputs = [
            jax.ShapeDtypeStruct(
                (num_nodes, desc.num_segments) + tuple(desc.segment_shape),
                dtype,
            )
            for desc in self._ir_dict_poly.outputs
        ]
        y = IR_DICT.segmented_polynomial_uniform_1d(
            self._ir_dict_poly,
            [w, x1_dict, x2_dict],
            outputs,
            input_indices=[None, sender, None],
            output_indices=receiver,
            math_dtype=dtype,
        )
        out_descriptor = self._ir_dict_outputs_to_ir_mul(y)
        out_ir_mul = collapse_ir_mul_segments(
            out_descriptor,
            Irreps(self.descriptor_out_irreps_str),
            Irreps(self.irreps_out_o3),
            self.output_segment_shapes,
        )
        if self._api_layout == cue.ir_mul:
            return out_ir_mul
        return ir_mul_to_mul_ir(out_ir_mul, Irreps(self.irreps_out_o3))

    def _conv_fused_apply(
        self,
        *,
        node_feats: jnp.ndarray,
        edge_attrs: jnp.ndarray,
        weights: jnp.ndarray,
        sender: jnp.ndarray,
        receiver: jnp.ndarray,
        num_nodes: int,
        dtype: jnp.dtype,
    ) -> jnp.ndarray | None:
        if self._conv_polynomial is None:
            return None

        irreps_in1 = Irreps(self.irreps_in1_o3)
        irreps_in2 = Irreps(self.irreps_in2_o3)
        irreps_out = Irreps(self.irreps_out_o3)

        if self._api_layout == cue.ir_mul:
            node_ir_mul = node_feats
            edge_ir_mul = edge_attrs
        else:
            node_ir_mul = mul_ir_to_ir_mul(node_feats, irreps_in1)
            edge_ir_mul = mul_ir_to_ir_mul(edge_attrs, irreps_in2)

        outputs_shape_dtype = [
            jax.ShapeDtypeStruct((num_nodes, self.descriptor_out_dim), dtype)
        ]
        indices = [
            None,
            (sender,),
            None,
            (receiver,),
        ]
        math_dtype = jnp.dtype(dtype).name
        [out_ir_mul] = cuex.segmented_polynomial(
            self._conv_polynomial,
            [weights, node_ir_mul, edge_ir_mul],
            outputs_shape_dtype,
            indices=indices,
            method='uniform_1d',
            math_dtype=math_dtype,
        )
        out_ir_mul = collapse_ir_mul_segments(
            out_ir_mul,
            Irreps(self.descriptor_out_irreps_str),
            irreps_out,
            self.output_segment_shapes,
        )
        if self._api_layout == cue.ir_mul:
            return out_ir_mul
        return ir_mul_to_mul_ir(out_ir_mul, irreps_out)


def _tensor_product_import_from_torch(cls, torch_module, variables):
    """Copy Torch tensor product weights into the NNX parameter dict."""
    params = variables
    expected_layout = params.get('layout_config', None)

    def _decode_layout(val):
        if isinstance(val, jnp.ndarray):
            try:
                val_int = int(val)
            except Exception:
                return None
            return 'mul_ir' if val_int == 0 else 'ir_mul'
        if isinstance(val, (int, np.integer)):
            return 'mul_ir' if int(val) == 0 else 'ir_mul'
        return val

    def _layout_str_from_obj(layout_obj) -> str | None:
        if layout_obj is None:
            return None
        if isinstance(layout_obj, str):
            return layout_obj
        for attr in ('layout_str', 'name', '__name__'):
            val = getattr(layout_obj, attr, None)
            if val is not None:
                return str(val)
        return str(layout_obj)

    expected_layout = _decode_layout(expected_layout)
    torch_layout_str = _layout_str_from_obj(getattr(torch_module, 'layout', None))
    if torch_layout_str is None:
        descriptor = getattr(torch_module, 'descriptor', None) or getattr(
            torch_module, '_descriptor', None
        )
        if descriptor is not None:
            try:
                torch_layout_str = _layout_str_from_obj(descriptor.inputs[1].layout)
            except Exception:
                torch_layout_str = None
    if torch_layout_str is None:
        torch_layout_str = 'mul_ir'

    if expected_layout is not None and str(expected_layout) != str(torch_layout_str):
        raise ValueError(
            f'JAX TensorProduct expected layout {expected_layout!r} but Torch module '
            f'uses layout {torch_layout_str!r}.'
        )

    if (
        getattr(torch_module, 'internal_weights', False)
        and getattr(torch_module, 'weight_numel', 0) > 0
    ):
        weight_np = torch_module.weight.detach().cpu().numpy()
        if weight_np.ndim == 1:
            weight_np = weight_np.reshape(1, -1)
        existing = params.get('weight')
        dtype = existing.dtype if existing is not None else weight_np.dtype
        params['weight'] = jnp.asarray(weight_np, dtype=dtype)

    return params


TensorProduct.import_from_torch = classmethod(_tensor_product_import_from_torch)
