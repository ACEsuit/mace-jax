"""Cue-equivariant linear layer implemented in Flax."""

from __future__ import annotations

import cuequivariance_jax as cuex
import jax
import jax.numpy as jnp
from e3nn_jax import Irreps, IrrepsArray  # type: ignore
from flax import linen as fnn

import cuequivariance as cue
from mace_jax.adapters.flax.torch import auto_import_from_torch_flax

from .utility import ir_mul_to_mul_ir, mul_ir_to_ir_mul


@auto_import_from_torch_flax(allow_missing_mapper=True)
class Linear(fnn.Module):
    """Cue-equivariant linear map evaluated with cuequivariance-jax.

    The adapter mirrors the public API of the Torch module while delegating the
    heavy lifting to cue's linear descriptor.  Inputs are accepted either as raw
    ``jnp.ndarray`` batches in mul_ir order or as ``IrrepsArray`` instances.
    Depending on ``internal_weights`` and ``shared_weights`` the layer keeps
    parameters internally or expects callers to supply them explicitly.
    """

    irreps_in: Irreps
    irreps_out: Irreps
    shared_weights: bool | None = None
    internal_weights: bool | None = None

    def setup(self) -> None:
        """Resolve configuration flags and construct the cue descriptor."""
        shared_weights = True if self.shared_weights is None else self.shared_weights
        internal_weights = (
            True if self.internal_weights is None else self.internal_weights
        )
        if shared_weights is False and self.internal_weights is None:
            internal_weights = False
        self._shared_weights = shared_weights
        self._internal_weights = internal_weights

        self.irreps_in_o3 = Irreps(self.irreps_in)
        self.irreps_out_o3 = Irreps(self.irreps_out)
        self.irreps_in_cue = cue.Irreps(cue.O3, self.irreps_in_o3)
        self.irreps_out_cue = cue.Irreps(cue.O3, self.irreps_out_o3)

        descriptor = cue.descriptors.linear(
            self.irreps_in_cue,
            self.irreps_out_cue,
        )
        self.descriptor = descriptor
        self.weight_irreps = descriptor.inputs[0].irreps
        self.weight_numel = descriptor.polynomial.operands[0].size

    def _weight_param(self) -> jnp.ndarray:
        """Initialise or fetch the internal weight parameter."""
        init = lambda rng: jax.random.normal(rng, (1, self.weight_numel))
        return self.param('weight', init)

    def _extract_array(self, x: jnp.ndarray | IrrepsArray) -> tuple[jnp.ndarray, bool]:
        """Return the raw array for ``x`` and whether it carried Irreps metadata."""
        if isinstance(x, IrrepsArray):
            if x.irreps != self.irreps_in_o3:
                raise ValueError(
                    f'Linear expects input irreps {self.irreps_in_o3}, got {x.irreps}'
                )
            return jnp.asarray(x.array), True
        return jnp.asarray(x), False

    def _as_rep(self, array: jnp.ndarray) -> cuex.RepArray:
        """Convert a mul_ir array to a cue RepArray with cached metadata."""
        ir_mul = mul_ir_to_ir_mul(array, Irreps(self.irreps_in_o3))
        return cuex.RepArray(
            self.irreps_in_cue,
            jnp.asarray(ir_mul),
            cue.ir_mul,
        )

    def _resolve_weight_operand(
        self,
        weights: jnp.ndarray | None,
        *,
        dtype: jnp.dtype,
    ) -> cuex.RepArray | None:
        """Prepare the weight operand according to the configured policy."""
        if self._internal_weights:
            if weights is not None:
                raise ValueError(
                    'Weights must be None when internal_weights=True in Linear'
                )
            return cuex.RepArray(
                self.weight_irreps,
                self._weight_param().astype(dtype),
                cue.ir_mul,
            )

        if self._shared_weights:
            if weights is None:
                raise ValueError(
                    'Weights must be provided when internal_weights=False and shared_weights=True in Linear'
                )
            array = jnp.asarray(weights, dtype=dtype)
            if array.ndim == 1:
                if array.shape[0] != self.weight_numel:
                    raise ValueError(
                        f'Expected weights last dimension {self.weight_numel}, got {array.shape[-1]}'
                    )
                array = array[jnp.newaxis, :]
            elif array.ndim == 2:
                if array.shape[-1] != self.weight_numel:
                    raise ValueError(
                        f'Expected weights last dimension {self.weight_numel}, got {array.shape[-1]}'
                    )
            else:
                raise ValueError(
                    'Weights must have rank 1 or 2 when shared external weights are used'
                )
            return cuex.RepArray(self.weight_irreps, array, cue.ir_mul)

        # Unshared external weights are passed directly through the descriptor;
        # cue expects them as part of the evaluation input, so we return None.
        return None

    @fnn.compact
    def __call__(
        self,
        x: jnp.ndarray | IrrepsArray,
        weights: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """Apply the linear map and return an array in mul_ir layout."""
        array, had_irreps = self._extract_array(x)
        dtype = array.dtype

        x_rep = self._as_rep(array)
        weight_rep = self._resolve_weight_operand(weights, dtype=dtype)

        output_rep = cuex.equivariant_polynomial(
            self.descriptor,
            [weight_rep, x_rep],
            math_dtype=dtype,
            method='naive',
        )
        out_ir_mul = output_rep.array
        irreps_out = Irreps(self.irreps_out_o3)
        out_mul_ir = ir_mul_to_mul_ir(out_ir_mul, irreps_out)

        if had_irreps:
            return IrrepsArray(irreps_out, out_mul_ir)
        return out_mul_ir
