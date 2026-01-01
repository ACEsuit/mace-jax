"""Cue-equivariant linear layer implemented in Flax."""

from __future__ import annotations

import cuequivariance as cue
import cuequivariance_jax as cuex
import jax
import jax.numpy as jnp
from e3nn_jax import Irreps, IrrepsArray  # type: ignore
from flax import linen as fnn
from flax.errors import ScopeCollectionNotFound

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
    layout: object = 'mul_ir'

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
        self._api_layout, self._layout_str = self._resolve_layout(self.layout)

        descriptor = cue.descriptors.linear(
            self.irreps_in_cue,
            self.irreps_out_cue,
        )
        self.descriptor = descriptor
        self.weight_irreps = descriptor.inputs[0].irreps
        self.weight_numel = descriptor.polynomial.operands[0].size
        self._weight_layout = descriptor.inputs[0].layout
        self._descriptor_input_layout = descriptor.inputs[1].layout
        self._descriptor_output_layout = descriptor.outputs[0].layout
        # Stash chosen layout for later validation (e.g., during Torch import).
        # Store as int code (0=mul_ir, 1=ir_mul) to keep the variables tree JIT-safe.
        layout_code = 0 if self._layout_str == 'mul_ir' else 1
        self.variable(
            'config', 'layout', lambda: jnp.asarray(layout_code, dtype=jnp.int32)
        )

    @staticmethod
    def _resolve_layout(layout_obj: object) -> tuple[cue.IrrepsLayout, str]:
        """Return the cue layout object and a readable identifier."""
        if isinstance(layout_obj, str):
            if layout_obj not in {'mul_ir', 'ir_mul'}:
                raise ValueError(
                    f"Linear received unsupported layout string '{layout_obj}'."
                )
            return getattr(cue, layout_obj), layout_obj

        if layout_obj == cue.mul_ir:
            return layout_obj, 'mul_ir'
        if layout_obj == cue.ir_mul:
            return layout_obj, 'ir_mul'

        raise ValueError(
            'Linear received an unknown layout object; expected cue.mul_ir or '
            'cue.ir_mul.'
        )

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
        """Convert the public array to the descriptor layout."""
        if self._api_layout == self._descriptor_input_layout:
            payload = array
        elif (
            self._api_layout == cue.mul_ir
            and self._descriptor_input_layout == cue.ir_mul
        ):
            payload = mul_ir_to_ir_mul(array, Irreps(self.irreps_in_o3))
        elif (
            self._api_layout == cue.ir_mul
            and self._descriptor_input_layout == cue.mul_ir
        ):
            payload = ir_mul_to_mul_ir(array, Irreps(self.irreps_in_o3))
        else:
            raise ValueError(
                'Linear does not support conversion from '
                f'{self._api_layout!r} to {self._descriptor_input_layout!r}.'
            )

        return cuex.RepArray(
            self.irreps_in_cue,
            jnp.asarray(payload),
            self._descriptor_input_layout,
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
                self._weight_layout,
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
            return cuex.RepArray(self.weight_irreps, array, self._weight_layout)

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
        raw_output = output_rep.array
        irreps_out = Irreps(self.irreps_out_o3)
        if self._descriptor_output_layout == self._api_layout:
            out_mul_ir = raw_output
        elif (
            self._descriptor_output_layout == cue.ir_mul
            and self._api_layout == cue.mul_ir
        ):
            out_mul_ir = ir_mul_to_mul_ir(raw_output, irreps_out)
        elif (
            self._descriptor_output_layout == cue.mul_ir
            and self._api_layout == cue.ir_mul
        ):
            out_mul_ir = mul_ir_to_ir_mul(raw_output, irreps_out)
        else:
            raise ValueError(
                'Linear does not support conversion from '
                f'{self._descriptor_output_layout!r} to {self._api_layout!r}.'
            )

        if had_irreps:
            return IrrepsArray(irreps_out, out_mul_ir)
        return out_mul_ir


def _linear_import_from_torch_with_layout(cls, torch_module, flax_variables):
    """Wrapper around the auto-generated import that enforces layout parity."""
    cfg = flax_variables.get('config', {}) or flax_variables.get('meta', {})
    expected_layout = None
    if isinstance(cfg, dict):
        expected_layout = cfg.get('layout', None)
    elif hasattr(cfg, 'get'):
        expected_layout = cfg.get('layout', None)

    def _decode_layout(val):
        # Meta layout is stored as int code (0=mul_ir, 1=ir_mul) for JIT safety.
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
            f'JAX Linear expected layout {expected_layout!r} but Torch module uses '
            f'layout {torch_layout_str!r}.'
        )

    return cls._import_from_torch_impl(
        torch_module,
        flax_variables,
        skip_root=False,
    )


# Override the auto-generated import_from_torch with layout validation.
Linear.import_from_torch = classmethod(_linear_import_from_torch_with_layout)
