"""Cue-equivariant symmetric contraction implemented with Flax."""

from __future__ import annotations

import cuequivariance_jax as cuex
import jax
import jax.numpy as jnp
import cuequivariance as cue
from e3nn_jax import Irreps  # type: ignore
from flax import linen as fnn
from cuequivariance.group_theory.experimental.mace.symmetric_contractions import (
    symmetric_contraction as cue_mace_symmetric_contraction,
)
from mace_jax.adapters.flax.torch import (
    _resolve_scope,
    auto_import_from_torch_flax,
    register_import_mapper,
)

from .utility import ir_mul_to_mul_ir

_NATIVE_SC_ERROR = (
    'Importing parameters from the native Torch SymmetricContraction is not supported; '
    "run with '--only_cueq=True' to enable the cuequivariance-backed implementation."
)


def _raise_native_sym_contraction_not_supported() -> None:
    raise NotImplementedError(_NATIVE_SC_ERROR)


@auto_import_from_torch_flax(allow_missing_mapper=True)
class SymmetricContraction(fnn.Module):
    """Symmetric contraction layer evaluated with cuequivariance-jax.

    The module contracts higher-order correlations of MACE-like features using
    the symmetric contraction descriptor provided by cue.  It supports either
    reduced or full Clebsch–Gordan bases and offers runtime selection of the
    weight slice through ``indices``.
    """

    irreps_in: Irreps
    irreps_out: Irreps
    correlation: int
    num_elements: int
    use_reduced_cg: bool = True
    input_layout: str = 'mul_ir'

    def setup(self) -> None:
        """Validate configuration and prepare the cue descriptor."""
        if self.correlation <= 0:
            raise ValueError('correlation must be a positive integer')
        if self.num_elements <= 0:
            raise ValueError('num_elements must be positive')
        if self.input_layout not in {'mul_ir', 'ir_mul'}:
            raise ValueError(
                "input_layout must be either 'mul_ir' or 'ir_mul'; "
                f'got {self.input_layout!r}'
            )

        irreps_in_o3 = Irreps(self.irreps_in)
        irreps_out_o3 = Irreps(self.irreps_out)
        self.irreps_in_o3_str = str(irreps_in_o3)
        self.irreps_out_o3_str = str(irreps_out_o3)

        muls_in = {mul for mul, _ in irreps_in_o3}
        muls_out = {mul for mul, _ in irreps_out_o3}
        if len(muls_in) != 1 or len(muls_out) != 1 or muls_in != muls_out:
            raise ValueError(
                'SymmetricContraction requires all input/output irreps to share the same multiplicity'
            )
        self.mul = next(iter(muls_in))

        self.irreps_in_cue = cue.Irreps(cue.O3, irreps_in_o3)
        self.irreps_out_cue = cue.Irreps(cue.O3, irreps_out_o3)
        self.feature_dim = sum(ir.dim for _, ir in irreps_in_o3)
        self.irreps_in_cue_base = self.irreps_in_cue.set_mul(1)
        self.irreps_out_dim = irreps_out_o3.dim

        degrees = tuple(range(1, self.correlation + 1))
        descriptor, projection = cue_mace_symmetric_contraction(
            self.irreps_in_cue,
            self.irreps_out_cue,
            degrees,
        )
        self.descriptor = descriptor
        self.weight_irreps = descriptor.inputs[0].irreps
        self.weight_numel = self.weight_irreps.dim

        if self.use_reduced_cg:
            self.projection = None
            self.weight_basis_dim = self.weight_numel // self.mul
        else:
            self.projection = jnp.asarray(projection)
            self.weight_basis_dim = self.projection.shape[0]

        self.weight_param_shape = (self.num_elements, self.weight_basis_dim, self.mul)

    def _weight_param(self) -> jnp.ndarray:
        """Return learnable basis weights initialised with Gaussian noise."""
        init = lambda rng: jax.random.normal(rng, self.weight_param_shape)
        return self.param('weight', init)

    def _ensure_mul_ir_layout(self, x: jnp.ndarray) -> jnp.ndarray:
        """Convert inputs to mul_ir layout if needed."""
        if self.input_layout == 'ir_mul':
            return self._convert_ir_mul_to_mul_ir(x)
        return x

    def _project_basis_weights(
        self, basis_weights: jnp.ndarray, dtype: jnp.dtype
    ) -> jnp.ndarray:
        """Project basis weights to the full Clebsch–Gordan space."""
        if self.projection is None:
            return basis_weights.astype(dtype)
        projection = jnp.asarray(self.projection, dtype=dtype)
        return jnp.einsum('zau,ab->zbu', basis_weights.astype(dtype), projection)

    def _weight_rep_from_indices(
        self,
        indices: jnp.ndarray,
        dtype: jnp.dtype,
    ) -> cuex.RepArray:
        """Select element weights and wrap them as a cue RepArray."""
        basis_weights = self._weight_param()
        projected = self._project_basis_weights(basis_weights, dtype)
        weight_flat = projected.reshape(self.num_elements, self.weight_numel)
        selected = _select_weights(
            weight_flat,
            indices,
            dtype=dtype,
            num_elements=self.num_elements,
        )
        return cuex.RepArray(self.weight_irreps, selected, cue.ir_mul)

    @fnn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        indices: jnp.ndarray,
    ) -> jnp.ndarray:
        """Apply the symmetric contraction and return mul_ir-ordered features."""
        array = jnp.asarray(x)
        dtype = array.dtype

        array = self._ensure_mul_ir_layout(array)
        _validate_features(array, self.mul, self.feature_dim)

        weight_rep = self._weight_rep_from_indices(indices, dtype)
        x_rep = self._features_to_rep(array, dtype)

        [out_ir_mul] = cuex.segmented_polynomial(
            self.descriptor.polynomial,
            [weight_rep.array, x_rep.array],
            [
                jax.ShapeDtypeStruct(
                    (array.shape[0], self.irreps_out_dim),
                    dtype,
                )
            ],
            method='naive',
            math_dtype=dtype,
        )

        out_mul_ir = ir_mul_to_mul_ir(out_ir_mul, Irreps(self.irreps_out_o3_str))
        return out_mul_ir

    def _features_to_rep(self, x: jnp.ndarray, dtype: jnp.dtype) -> cuex.RepArray:
        """Pack mul_ir features into cue RepArray segments."""
        segments: list[jnp.ndarray] = []
        offset = 0
        for mul_ir in self.irreps_in_cue_base:
            width = mul_ir.ir.dim
            seg = x[:, :, offset : offset + width]
            if seg.shape[-1] != width:
                raise ValueError('Input feature dimension mismatch with irreps.')
            segments.append(jnp.swapaxes(seg, -2, -1))
            offset += width

        return cuex.from_segments(
            self.irreps_in_cue,
            segments,
            (x.shape[0], self.mul),
            cue.ir_mul,
            dtype=dtype,
        )

    def _convert_ir_mul_to_mul_ir(self, x: jnp.ndarray) -> jnp.ndarray:
        """Reorder ir_mul input layout to mul_ir."""
        segments: list[jnp.ndarray] = []
        offset = 0
        for mul_ir in self.irreps_in_cue_base:
            width = mul_ir.ir.dim
            seg = x[:, offset : offset + width, :]
            if seg.shape[1] != width:
                raise ValueError('Input feature dimension mismatch with irreps.')
            segments.append(jnp.swapaxes(seg, -1, -2))
            offset += width

        if offset != x.shape[1]:
            raise ValueError('Input feature dimension mismatch with irreps.')

        if not segments:
            return jnp.swapaxes(x, -1, -2)

        return jnp.concatenate(segments, axis=-1)


@register_import_mapper(
    'cuequivariance_torch.operations.symmetric_contraction.SymmetricContraction'
)
def _import_cue_symmetric_contraction(module, variables, scope) -> None:
    """Import mapper that transfers Torch symmetric-contraction weights."""
    target = _resolve_scope(variables, scope)
    weight = module.weight.detach().cpu().numpy()
    existing = target.get('weight')
    dtype = existing.dtype if existing is not None else weight.dtype
    target['weight'] = jnp.asarray(weight, dtype=dtype)


@register_import_mapper('mace.modules.symmetric_contraction.SymmetricContraction')
def _import_native_symmetric_contraction(module, variables, scope) -> None:
    """Explicitly reject attempts to pull parameters from the native Torch module."""
    _raise_native_sym_contraction_not_supported()


def _validate_features(x: jnp.ndarray, mul: int, feature_dim: int) -> None:
    """Ensure the feature tensor shape matches the expected mul_ir layout."""
    if x.ndim != 3 or x.shape[1] != mul or x.shape[2] != feature_dim:
        raise ValueError(
            'SymmetricContraction expects input with shape '
            f'(batch, {mul}, {feature_dim}); got {tuple(x.shape)}'
        )


def _select_weights(
    weight_flat: jnp.ndarray,
    selector: jnp.ndarray,
    *,
    dtype: jnp.dtype,
    num_elements: int,
) -> jnp.ndarray:
    """Select element weights by index or mixing matrix."""
    selector = jnp.asarray(selector)
    if selector.ndim == 1:
        idx = selector.astype(jnp.int32)
        if jnp.any(idx < 0) or jnp.any(idx >= num_elements):
            raise ValueError('indices out of range for the available elements')
        return weight_flat[idx]

    if selector.ndim == 2:
        if selector.shape[1] != num_elements:
            raise ValueError('Mixing matrix must have second dimension num_elements')
        mix = jnp.asarray(selector, dtype=dtype)
        return mix @ weight_flat

    raise ValueError('indices must be rank-1 (element ids) or rank-2 (mixing matrix)')
