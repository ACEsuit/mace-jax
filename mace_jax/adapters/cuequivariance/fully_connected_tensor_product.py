"""Cue-equivariant fully connected tensor product implemented with Flax."""

from __future__ import annotations

import logging

import cuequivariance_jax as cuex
import jax
import jax.numpy as jnp
import numpy as np
from e3nn_jax import Irreps  # type: ignore
from flax import nnx

import cuequivariance as cue
from mace_jax.adapters.nnx.torch import (
    _resolve_scope,
    nxx_auto_import_from_torch,
    nxx_register_import_mapper,
)
from mace_jax.nnx_config import ConfigVar
from mace_jax.tools.dtype import default_dtype

from .utility import ir_mul_to_mul_ir, mul_ir_to_ir_mul


@nxx_auto_import_from_torch(allow_missing_mapper=True)
class FullyConnectedTensorProduct(nnx.Module):
    """Cue-equivariant fully connected tensor product implemented in Flax.

    This adapter mirrors the behaviour of both the e3nn and cue Torch tensor
    products while executing through cuequivariance-jax.  Weight handling
    matches the Torch semantics: ``internal_weights`` keeps parameters inside
    the module state; otherwise weights are expected as inputs and can be shared
    or batch-specific depending on ``shared_weights``.
    """

    irreps_in1: Irreps
    irreps_in2: Irreps
    irreps_out: Irreps
    shared_weights: bool = True
    internal_weights: bool = True
    group: object = cue.O3
    layout: object = cue.mul_ir

    def __init__(
        self,
        irreps_in1: Irreps,
        irreps_in2: Irreps,
        irreps_out: Irreps,
        shared_weights: bool = True,
        internal_weights: bool = True,
        group: object = cue.O3,
        layout: object = cue.mul_ir,
        *,
        rngs: nnx.Rngs | None = None,
    ) -> None:
        self.irreps_in1 = irreps_in1
        self.irreps_in2 = irreps_in2
        self.irreps_out = irreps_out
        self.shared_weights = shared_weights
        self.internal_weights = internal_weights
        self.group = group
        self.layout = layout
        # Prepare cue descriptors and cache Irreps metadata for evaluation.
        if self.internal_weights and not self.shared_weights:
            raise ValueError(
                'FullyConnectedTensorProduct requires shared_weights=True when internal_weights=True'
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

        descriptor = cue.descriptors.fully_connected_tensor_product(
            self.irreps_in1_cue,
            self.irreps_in2_cue,
            self.irreps_out_cue,
        )
        self.descriptor = descriptor
        self.weight_irreps = descriptor.inputs[0].irreps
        self.weight_numel = descriptor.polynomial.operands[0].size
        descriptor_out_irreps = Irreps(str(descriptor.outputs[0].irreps))
        self.descriptor_out_dim = descriptor_out_irreps.dim
        layout_code = 0 if self._layout_str == 'mul_ir' else 1
        self.layout_config = ConfigVar(
            jnp.asarray(layout_code, dtype=jnp.int32),
            is_mutable=False,
        )
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
        """Return the learnable weight parameter initialised with Gaussian noise.

        Returns:
            ``(1, weight_numel)`` array stored in the Flax parameter tree.
        """
        if self.weight is None:
            raise ValueError('Internal weights are not initialized for FCTP.')
        return self.weight

    def _input_rep(
        self,
        array: jnp.ndarray,
        irreps_o3: Irreps,
        irreps_cue: cue.Irreps,
    ) -> cuex.RepArray:
        """Convert an input in configured layout to a cue RepArray.

        Args:
            array: Input batch whose trailing dimension follows mul_ir ordering.
            irreps_o3: e3nn Irreps describing the input representation.
            irreps_cue: cue Irreps that share the same content as ``irreps_o3``.

        Returns:
            ``cuequivariance_jax.RepArray`` suitable for segmented polynomial evaluation.
        """
        if self._api_layout == cue.mul_ir:
            data = mul_ir_to_ir_mul(array, irreps_o3)
        elif self._api_layout == cue.ir_mul:
            data = array
        else:
            raise ValueError(
                'FullyConnectedTensorProduct does not support layout '
                f'{self._api_layout!r}.'
            )
        return cuex.RepArray(irreps_cue, data, cue.ir_mul)

    @staticmethod
    def _resolve_layout(layout_obj: object) -> tuple[cue.IrrepsLayout, str]:
        if isinstance(layout_obj, str):
            if layout_obj not in {'mul_ir', 'ir_mul'}:
                raise ValueError(
                    'FullyConnectedTensorProduct received unsupported layout string '
                    f"'{layout_obj}'."
                )
            return getattr(cue, layout_obj), layout_obj
        if layout_obj == cue.mul_ir:
            return layout_obj, 'mul_ir'
        if layout_obj == cue.ir_mul:
            return layout_obj, 'ir_mul'
        raise ValueError(
            'FullyConnectedTensorProduct received an unknown layout object; expected '
            'cue.mul_ir or cue.ir_mul.'
        )

    def _resolve_weight_rep(
        self,
        weights: jnp.ndarray | None,
        *,
        dtype: jnp.dtype,
        batch_size: int,
    ) -> cuex.RepArray:
        """Produce a cue RepArray for weights, validating shapes and semantics.

        Args:
            weights: External weight tensor supplied to ``__call__`` or ``None``.
            dtype: Numeric type to cast the resulting weights to.
            batch_size: Size of the leading batch dimension for the current call.

        Returns:
            ``cuequivariance_jax.RepArray`` carrying weights in cue layout.

        Raises:
            ValueError: If weight semantics violate the shared/unshared contract.
        """
        if self._internal_weights:
            if weights is not None:
                raise ValueError(
                    'Weights must be None when internal weights are used in FullyConnectedTensorProduct'
                )
            weight_array = self._weight_param().astype(dtype)
        else:
            if weights is None:
                raise ValueError(
                    'Weights must be provided when internal weights are not used'
                )
            weight_array = self._normalise_external_weights(
                weights, dtype=dtype, batch_size=batch_size
            )

        return cuex.RepArray(self.weight_irreps, weight_array, cue.ir_mul)

    def _normalise_external_weights(
        self,
        weights: jnp.ndarray,
        *,
        dtype: jnp.dtype,
        batch_size: int,
    ) -> jnp.ndarray:
        """Validate and reshape external weights to the canonical form.

        Args:
            weights: Candidate external weight tensor.
            dtype: Target dtype for downstream operations.
            batch_size: Batch size of the current forward call.

        Returns:
            Weight array with shape ``(batch, weight_numel)`` or ``(1, weight_numel)``
            depending on the sharing policy.

        Raises:
            ValueError: If the weight tensor rank or leading dimension is invalid.
        """
        weights = jnp.asarray(weights, dtype=dtype)
        if weights.ndim == 1:
            if weights.shape[0] != self.weight_numel:
                raise ValueError(
                    f'Expected weights last dimension {self.weight_numel}, got {weights.shape[-1]}'
                )
            weights = weights[jnp.newaxis, :]
        elif weights.ndim == 2:
            if weights.shape[-1] != self.weight_numel:
                raise ValueError(
                    f'Expected weights last dimension {self.weight_numel}, got {weights.shape[-1]}'
                )
        else:
            raise ValueError(
                'Weights must have rank 1 or 2 when internal weights are not used'
            )

        leading = weights.shape[0]
        if self._shared_weights:
            if leading not in (1, batch_size):
                raise ValueError(
                    'Shared weights require leading dimension 1 or equal to the batch size'
                )
            if leading == 1 and batch_size != 1:
                weights = jnp.broadcast_to(weights, (batch_size, self.weight_numel))
        else:
            if leading != batch_size:
                raise ValueError(
                    'Unshared weights require leading dimension equal to the batch size'
                )

        return weights

    def __call__(
        self,
        x1: jnp.ndarray,
        x2: jnp.ndarray,
        weights: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """Evaluate the tensor product returning outputs in mul_ir layout.

        Args:
            x1: First input batch organised in mul_ir order.
            x2: Second input batch organised in mul_ir order.
            weights: Optional external weights matching the chosen policy.

        Returns:
            ``jax.numpy`` array in mul_ir layout with representation ``irreps_out``.

        Raises:
            ValueError: If the supplied inputs or weights violate shape policies.
        """
        batch_size = x1.shape[0]
        dtype = x1.dtype

        irreps_in1 = Irreps(self.irreps_in1_o3)
        irreps_in2 = Irreps(self.irreps_in2_o3)
        irreps_out = Irreps(self.irreps_out_o3)

        x1_rep = self._input_rep(x1, irreps_in1, self.irreps_in1_cue)
        x2_rep = self._input_rep(x2, irreps_in2, self.irreps_in2_cue)
        weight_rep = self._resolve_weight_rep(
            weights, dtype=dtype, batch_size=batch_size
        )

        [out_ir_mul] = cuex.segmented_polynomial(
            self.descriptor.polynomial,
            [weight_rep.array, x1_rep.array, x2_rep.array],
            [
                jax.ShapeDtypeStruct(
                    (*x1.shape[:-1], self.descriptor_out_dim),
                    dtype,
                )
            ],
            method='naive',
            math_dtype=dtype,
        )

        if self._api_layout == cue.ir_mul:
            return out_ir_mul
        return ir_mul_to_mul_ir(out_ir_mul, irreps_out)


def _fctp_import_from_torch(cls, torch_module, variables):
    """Copy Torch fully connected tensor product weights into NNX variables."""
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
        logging.warning(
            'JAX FullyConnectedTensorProduct layout %r differs from Torch layout %r; importing weights without conversion.',
            expected_layout,
            torch_layout_str,
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


FullyConnectedTensorProduct.import_from_torch = classmethod(_fctp_import_from_torch)


@nxx_register_import_mapper('e3nn.o3._tensor_product._sub.FullyConnectedTensorProduct')
@nxx_register_import_mapper(
    'cuequivariance_torch.operations.fully_connected_tensor_product.FullyConnectedTensorProduct'
)
def _map_fctp(module, variables, scope) -> None:
    """Import mapper invoked by the generic Torch→Flax importer.

    Args:
        module: Torch module providing parameters.
        variables: Mutable view of the Flax parameter tree.
        scope: Sequence describing where within the tree to copy parameters.
    """
    target = _resolve_scope(variables, scope)
    updated = FullyConnectedTensorProduct.import_from_torch(module, target)
    if updated is not None and updated is not target:
        target.clear()
        target.update(updated)
