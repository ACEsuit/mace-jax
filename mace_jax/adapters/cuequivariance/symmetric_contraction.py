"""Cue-equivariant symmetric contraction implemented with Flax."""

from __future__ import annotations

from functools import cache, lru_cache
from typing import TYPE_CHECKING

import cuequivariance_jax as cuex
import jax
import jax.numpy as jnp
from e3nn import o3
from e3nn_jax import Irreps  # type: ignore
from flax import linen as fnn

import cuequivariance as cue
from cuequivariance.group_theory.experimental.mace.symmetric_contractions import (
    symmetric_contraction as cue_mace_symmetric_contraction,
)
from mace_jax.adapters.flax.torch import (
    _resolve_scope,
    auto_import_from_torch_flax,
    register_import_mapper,
)

from .utility import ir_mul_to_mul_ir

if TYPE_CHECKING:
    import cuequivariance_torch as cuet
    import torch


def _torch_modules():
    import torch  # noqa: PLC0415
    import torch.nn.functional as F  # noqa: PLC0415

    return torch, F


def _cue_torch_module():
    import cuequivariance_torch as cuet  # noqa: PLC0415

    return cuet


def _native_weight_blocks(module: torch.nn.Module) -> list[torch.Tensor]:
    """Collect native MACE weight tensors reshaped to (elements, dim, mul)."""
    blocks: list[torch.Tensor] = []
    mul: int | None = None

    for contraction in getattr(module, 'contractions', []):
        weight_max = contraction.weights_max
        if mul is None:
            mul = int(weight_max.shape[2])
        blocks.append(weight_max.reshape(weight_max.shape[0], -1, mul))

        for weight in contraction.weights:
            if mul is None:
                mul = int(weight.shape[2])
            blocks.append(weight.reshape(weight.shape[0], -1, mul))

    return blocks


def _extract_native_weights(module: torch.nn.Module) -> torch.Tensor:
    """Stack native weights into (num_elements, native_dim, mul)."""
    torch, _ = _torch_modules()
    blocks = _native_weight_blocks(module)
    if not blocks:
        return torch.zeros((0, 0, 1), dtype=torch.get_default_dtype())

    return torch.cat(blocks, dim=1)


def _set_mul_one(irreps: o3.Irreps) -> o3.Irreps:
    """Return irreps with multiplicity one for each irrep."""
    return o3.Irreps([(1, ir.ir) for ir in irreps])


@cache
def _native_to_cue_transform(
    module_cls: type,
    irreps_in_str: str,
    irreps_out_str: str,
    correlation: int,
    use_reduced_cg: bool,
) -> torch.Tensor:
    """Return the basis transform mapping native weights to cue ordering.

    The native MACE module and the cue implementation both determine their
    weight layouts dynamically: zeroed Clebsch–Gordan paths are dropped, reduced
    CG bases shrink dimensions, and the product bases are stacked in different
    orders depending on the correlation.  Re-deriving the exact permutation by
    hand would require re-implementing that logic and would be brittle against
    upstream changes.  Instead we instantiate tiny ``mul = 1`` helper modules
    and *probe* each implementation once to observe its canonical basis.  The
    resulting linear map is cached per (irreps, correlation, reduced-CG) tuple,
    so subsequent imports reuse the transform without any further probing.
    """
    torch, F = _torch_modules()
    dtype = torch.float64
    device = torch.device('cpu')

    irreps_in = o3.Irreps(irreps_in_str)
    irreps_out = o3.Irreps(irreps_out_str)
    base_in = _set_mul_one(irreps_in)
    base_out = _set_mul_one(irreps_out)

    native_base = module_cls(
        irreps_in=base_in,
        irreps_out=base_out,
        correlation=correlation,
        num_elements=1,
        use_reduced_cg=use_reduced_cg,
    ).to(device=device, dtype=dtype)
    native_base.eval()

    cue_irreps_in = cue.Irreps(cue.O3, base_in)
    cue_irreps_out = cue.Irreps(cue.O3, base_out)
    cue_module = (
        _cue_torch_module()
        .SymmetricContraction(
            cue_irreps_in,
            cue_irreps_out,
            contraction_degree=correlation,
            num_elements=1,
            layout_in=cue.ir_mul,
            layout_out=cue.mul_ir,
            original_mace=False,
            dtype=dtype,
            math_dtype=dtype,
        )
        .to(device=device)
    )
    cue_module.eval()

    native_weights = _extract_native_weights(native_base)
    native_dim = native_weights.shape[1]
    cue_dim = cue_module.weight.shape[1]

    if native_dim == 0 or cue_dim == 0:
        return torch.zeros((cue_dim, native_dim), dtype=dtype, device=device)

    feature_dim = base_in.dim
    output_dim = base_out.dim
    target_dim = max(native_dim, cue_dim)
    batch = max(1, (target_dim + output_dim - 1) // output_dim)

    generator = torch.Generator(device=device).manual_seed(0)
    features_native = torch.randn(
        batch,
        1,
        feature_dim,
        generator=generator,
        dtype=dtype,
        device=device,
    )
    features_ir_mul = features_native.transpose(1, 2).reshape(batch, -1)
    indices = torch.zeros(batch, dtype=torch.long, device=device)
    selector = F.one_hot(indices, num_classes=1).to(dtype=dtype)

    basis_columns: list[torch.Tensor] = []
    with torch.no_grad():
        cue_module.weight.zero_()
        for basis_idx in range(cue_dim):
            cue_module.weight.zero_()
            cue_module.weight[0, basis_idx, 0] = 1.0
            column = cue_module(features_ir_mul, indices).reshape(-1)
            basis_columns.append(column)
    coeffs = torch.stack(basis_columns, dim=1)

    params: list[torch.Tensor] = []
    for contraction in native_base.contractions:
        params.append(contraction.weights_max)
        params.extend(contraction.weights)

    native_columns: list[torch.Tensor] = []
    with torch.no_grad():
        saved = [param.detach().clone() for param in params]
        for param in params:
            param.zero_()

        for param in params:
            num_params = param.shape[1]
            num_feats = param.shape[2]
            for idx_param in range(num_params):
                for idx_feat in range(num_feats):
                    param.zero_()
                    param[0, idx_param, idx_feat] = 1.0
                    column = native_base(features_native, selector).reshape(-1)
                    native_columns.append(column)
            param.zero_()

        for param, data in zip(params, saved):
            param.copy_(data)

    native_matrix = torch.stack(native_columns, dim=1)
    transform = torch.linalg.lstsq(coeffs, native_matrix).solution
    return transform.detach()


def _map_native_weights_to_cue(
    module: torch.nn.Module,
    *,
    dtype: torch.dtype,
    target_dim: int,
) -> torch.Tensor:
    """Convert native MACE weights into cue ordering for all elements.

    This lifts the cached single-element transform produced by
    :func:`_native_to_cue_transform` to the full multiplicity of the imported
    module.  Because the expensive probing happens only on cache misses, the
    typical import path reduces to a reshape followed by a matrix multiply.
    Degenerate cases (e.g. empty bases) fall back to the identity transform so
    that shape checks remain straightforward.
    """
    torch, _ = _torch_modules()
    if not getattr(module, 'contractions', None):
        return torch.zeros((0, 0, 1), dtype=dtype)

    irreps_in = str(module.irreps_in)
    irreps_out = str(module.irreps_out)
    correlation = int(module.contractions[0].correlation)
    use_reduced_cg = bool(getattr(module, 'use_reduced_cg', False))

    native_weights = _extract_native_weights(module).to(dtype=dtype)
    native_dim = native_weights.shape[1]
    if native_weights.numel() == 0:
        return torch.zeros((native_weights.shape[0], target_dim, 1), dtype=dtype)

    try:
        transform = _native_to_cue_transform(
            module.__class__,
            irreps_in,
            irreps_out,
            correlation,
            use_reduced_cg,
        ).to(dtype=dtype)
    except Exception:
        if target_dim != native_dim:
            raise
        transform = torch.eye(native_dim, dtype=dtype)

    if transform.shape[0] != target_dim or transform.shape[1] != native_dim:
        raise ValueError(
            'Cached transform shape mismatch: '
            f'expected ({target_dim}, {native_dim}), got {tuple(transform.shape)}.'
        )

    cue_weights = torch.einsum('ab,zbc->zac', transform, native_weights)
    return cue_weights


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
    """Import mapper that mirrors the native Torch module weight layout."""
    target = _resolve_scope(variables, scope)
    existing = target.get('weight')
    if existing is None:
        raise KeyError(
            'Target variables missing SymmetricContraction weight parameter.'
        )
    torch_dtype = module.contractions[0].weights_max.dtype
    cue_weights = _map_native_weights_to_cue(
        module,
        dtype=torch_dtype,
        target_dim=existing.shape[1],
    )

    if cue_weights.shape != existing.shape:
        raise ValueError(
            'Converted symmetric contraction weights have unexpected shape '
            f'{tuple(cue_weights.shape)}; expected {tuple(existing.shape)}.'
        )

    cue_weights_cpu = cue_weights.detach().cpu()
    module.weight = cue_weights_cpu.to(torch_dtype)
    target['weight'] = jnp.asarray(cue_weights_cpu.numpy(), dtype=existing.dtype)


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
