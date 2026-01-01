"""Cue-equivariant tensor product implemented via segmented polynomials (Flax)."""

from __future__ import annotations

import cuequivariance as cue
import cuequivariance_jax as cuex
import jax
import jax.numpy as jnp
from e3nn_jax import Irreps  # type: ignore
from flax import linen as fnn
from flax.core import freeze, unfreeze

from mace_jax.adapters.flax.torch import auto_import_from_torch_flax
from mace_jax.tools.cg import O3_e3nn
from mace_jax.tools.scatter import scatter_sum

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


@auto_import_from_torch_flax(allow_missing_mapper=True)
class TensorProduct(fnn.Module):
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

    def setup(self) -> None:
        """Initialise cue descriptors and validate the instruction template.

        Raises:
            ValueError: If the requested output irreps cannot be produced by the
                channel-wise descriptor, or if user-specified instructions are
                incompatible with the e3nn-generated pattern.
        """
        if self.internal_weights and not self.shared_weights:
            raise ValueError(
                'TensorProduct requires shared_weights=True when internal_weights=True'
            )
        self._shared_weights = self.shared_weights
        self._internal_weights = self.internal_weights

        self.irreps_in1_o3 = Irreps(self.irreps_in1)
        self.irreps_in2_o3 = Irreps(self.irreps_in2)
        self.irreps_out_o3 = Irreps(self.irreps_out)

        self.irreps_in1_cue = cue.Irreps(O3_e3nn, self.irreps_in1_o3)
        self.irreps_in2_cue = cue.Irreps(O3_e3nn, self.irreps_in2_o3)
        self.irreps_out_cue = cue.Irreps(O3_e3nn, self.irreps_out_o3)

        descriptor = cue.descriptors.channelwise_tensor_product(
            self.irreps_in1_cue, self.irreps_in2_cue, self.irreps_out_cue
        )
        self.descriptor = descriptor
        self.weight_irreps = descriptor.inputs[0].irreps
        self.weight_numel = descriptor.polynomial.operands[0].size
        self.descriptor_out_irreps_str = str(descriptor.outputs[0].irreps)
        self.output_segment_shapes = tuple(descriptor.polynomial.operands[-1].segments)
        self.descriptor_out_dim = Irreps(self.descriptor_out_irreps_str).dim

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

        object.__setattr__(self, '_conv_fusion', bool(self.conv_fusion))
        object.__setattr__(self, '_conv_method', 'naive')

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
        object.__setattr__(self, '_conv_polynomial', conv_polynomial)

    def _weight_param(self) -> jnp.ndarray:
        """Create the shared/internal weight parameter."""
        init = lambda rng: jax.random.normal(rng, (1, self.weight_numel))
        return self.param('weight', init)

    def _as_rep(
        self,
        array: jnp.ndarray,
        irreps_o3: Irreps,
        irreps_cue: cue.Irreps,
    ) -> cuex.RepArray:
        """Convert mul_ir array to cue RepArray with matching metadata."""
        ir_mul = mul_ir_to_ir_mul(array, irreps_o3)
        return cuex.RepArray(irreps_cue, jnp.asarray(ir_mul), cue.ir_mul)

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

    @fnn.compact
    def __call__(
        self,
        x1: jnp.ndarray,
        x2: jnp.ndarray,
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
        dtype = x1.dtype

        if not self._conv_fusion:
            batch_size = x1.shape[0]
            weight_tensor = self._resolve_weight_tensor(
                weights, dtype=dtype, batch_size=batch_size
            )
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
        num_nodes = x1.shape[0]
        edge_batch = sender.shape[0]

        edge_x1 = x1[sender]
        weight_tensor = self._resolve_weight_tensor(
            weights, dtype=dtype, batch_size=edge_batch
        )

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

        per_edge = self._channelwise_apply(
            edge_x1,
            x2,
            weight_tensor,
            dtype=dtype,
        )
        aggregated = scatter_sum(per_edge, receiver, dim_size=num_nodes)
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
        return ir_mul_to_mul_ir(out_ir_mul, irreps_out)


def _tensor_product_import_from_torch(cls, torch_module, flax_variables):
    """Copy Torch tensor product weights into the Flax parameter tree."""
    variables = unfreeze(flax_variables)
    params = variables.setdefault('params', {})

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

    variables['params'] = params
    return freeze(variables)


TensorProduct.import_from_torch = classmethod(_tensor_product_import_from_torch)
