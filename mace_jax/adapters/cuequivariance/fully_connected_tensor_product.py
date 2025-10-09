"""Cue-equivariant fully connected tensor product implemented with Flax."""

from __future__ import annotations

import cuequivariance_jax as cuex
import jax
import jax.numpy as jnp
from e3nn_jax import Irreps  # type: ignore
from flax import linen as fnn
from flax.core import freeze, unfreeze

import cuequivariance as cue
from mace_jax.adapters.flax.torch import (
    _resolve_scope,
    auto_import_from_torch_flax,
    register_import_mapper,
)

from .utility import ir_mul_to_mul_ir, mul_ir_to_ir_mul


@auto_import_from_torch_flax(allow_missing_mapper=True)
class FullyConnectedTensorProduct(fnn.Module):
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

    def setup(self) -> None:
        """Prepare cue descriptors and cache Irreps metadata for evaluation.

        Raises:
            ValueError: If ``internal_weights`` is requested without enabling
                ``shared_weights`` (which mirrors the Torch constraint).
        """
        if self.internal_weights and not self.shared_weights:
            raise ValueError(
                'FullyConnectedTensorProduct requires shared_weights=True when internal_weights=True'
            )
        self._shared_weights = self.shared_weights
        self._internal_weights = self.internal_weights

        self.irreps_in1_o3 = Irreps(self.irreps_in1)
        self.irreps_in2_o3 = Irreps(self.irreps_in2)
        self.irreps_out_o3 = Irreps(self.irreps_out)

        self.irreps_in1_cue = cue.Irreps(cue.O3, self.irreps_in1_o3)
        self.irreps_in2_cue = cue.Irreps(cue.O3, self.irreps_in2_o3)
        self.irreps_out_cue = cue.Irreps(cue.O3, self.irreps_out_o3)

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

    def _weight_param(self) -> jnp.ndarray:
        """Return the learnable weight parameter initialised with Gaussian noise.

        Returns:
            ``(1, weight_numel)`` array stored in the Flax parameter tree.
        """
        init = lambda rng: jax.random.normal(rng, (1, self.weight_numel))
        return self.param('weight', init)

    def _input_rep(
        self,
        array: jnp.ndarray,
        irreps_o3: Irreps,
        irreps_cue: cue.Irreps,
    ) -> cuex.RepArray:
        """Convert an input in mul_ir layout to a cue RepArray.

        Args:
            array: Input batch whose trailing dimension follows mul_ir ordering.
            irreps_o3: e3nn Irreps describing the input representation.
            irreps_cue: cue Irreps that share the same content as ``irreps_o3``.

        Returns:
            ``cuequivariance_jax.RepArray`` suitable for segmented polynomial evaluation.
        """
        data = mul_ir_to_ir_mul(array, irreps_o3)
        return cuex.RepArray(irreps_cue, data, cue.ir_mul)

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

    @fnn.compact
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

        out_mul_ir = ir_mul_to_mul_ir(out_ir_mul, irreps_out)
        return out_mul_ir


def _fctp_import_from_torch(cls, torch_module, flax_variables):
    """Copy Torch fully connected tensor product weights into Flax variables.

    Args:
        cls: The Flax module class (ignored, present for ``classmethod``).
        torch_module: Source Torch module.
        flax_variables: Destination FrozenDict produced by ``Module.init``.

    Returns:
        FrozenDict mirroring ``flax_variables`` with weight parameters imported.
    """
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


FullyConnectedTensorProduct.import_from_torch = classmethod(_fctp_import_from_torch)


@register_import_mapper('e3nn.o3._tensor_product._sub.FullyConnectedTensorProduct')
@register_import_mapper(
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
    wrapped = freeze({'params': target})
    updated = FullyConnectedTensorProduct.import_from_torch(module, wrapped)
    updated_params = unfreeze(updated).get('params', {})
    target.clear()
    target.update(updated_params)
