"""Cue-equivariant symmetric contraction implemented with segmented polynomials."""

from __future__ import annotations

import cuequivariance_jax as cuex
import haiku as hk
import jax
import jax.numpy as jnp
from e3nn_jax import Irreps  # type: ignore

import cuequivariance as cue
from cuequivariance.group_theory.experimental.mace.symmetric_contractions import (
    symmetric_contraction as cue_mace_symmetric_contraction,
)
from mace_jax.haiku.torch import register_import

from .utility import ir_mul_to_mul_ir


@register_import('mace.modules.symmetric_contraction.SymmetricContraction')
class SymmetricContraction(hk.Module):
    r"""Symmetric contraction evaluated with cue-equivariant segmented polynomials.

    Given an input feature vector ``x`` whose irreps are described by
    :attr:`irreps_in`, an integer ``index`` selecting one of ``num_elements``
    learned weight vectors, and a contraction degree ``correlation``, this
    module computes

    .. math::

        z_{w,k} = \sum_{d=1}^{C} \sum_{u_1,\ldots,u_d}
            w^{(d)}_{w,u_1,\ldots,u_d}
            \left\langle x_{u_1} \otimes \cdots \otimes x_{u_d},
            \mathrm{CG}_{u_1,\ldots,u_d \to k} \right\rangle,

    where ``C`` is ``correlation`` and the Clebsch–Gordan (CG) coefficients are
    supplied by the cue descriptor.  The element index ``w`` selects which row
    of the weight tensor is used for each batch item.  Inputs and outputs are in
    the familiar e3nn ``mul_ir`` layout; cue handles the segmented-polynomial
    evaluation in ``ir_mul`` order internally.  Depending on ``use_reduced_cg``
    the weights are either parameterised in the reduced CG basis or projected to
    the original MACE basis using the matrices returned by
    :mod:`cuequivariance`.
    """

    def __init__(
        self,
        irreps_in: Irreps,
        irreps_out: Irreps,
        *,
        correlation: int,
        num_elements: int,
        use_reduced_cg: bool = True,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)

        if correlation <= 0:
            raise ValueError('correlation must be a positive integer')
        if num_elements <= 0:
            raise ValueError('num_elements must be positive')

        self.correlation = correlation
        self.num_elements = num_elements
        self.use_reduced_cg = use_reduced_cg

        self.irreps_in_o3 = Irreps(irreps_in)
        self.irreps_out_o3 = Irreps(irreps_out)

        muls_in = {mul for mul, _ in self.irreps_in_o3}
        muls_out = {mul for mul, _ in self.irreps_out_o3}
        if len(muls_in) != 1 or len(muls_out) != 1 or muls_in != muls_out:
            raise ValueError(
                'SymmetricContraction requires all input/output irreps to share the same multiplicity'
            )
        self.mul = next(iter(muls_in))

        self.irreps_in_cue = cue.Irreps(cue.O3, irreps_in)
        self.irreps_out_cue = cue.Irreps(cue.O3, irreps_out)
        self.feature_dim = sum(ir.dim for _, ir in self.irreps_in_o3)
        self.irreps_in_cue_base = self.irreps_in_cue.set_mul(1)

        degrees = tuple(range(1, correlation + 1))
        descriptor, projection = cue_mace_symmetric_contraction(
            self.irreps_in_cue,
            self.irreps_out_cue,
            degrees,
        )
        self.descriptor = descriptor
        self.weight_irreps = descriptor.inputs[0].irreps
        self.weight_numel = self.weight_irreps.dim

        if use_reduced_cg:
            self.projection = None
            self.weight_basis_dim = self.weight_numel // self.mul
        else:
            self.projection = jnp.asarray(projection)
            self.weight_basis_dim = self.projection.shape[0]

        self.weight_param_shape = (self.num_elements, self.weight_basis_dim, self.mul)

    def __call__(
        self,
        x: jnp.ndarray,
        indices: jnp.ndarray,
    ) -> jnp.ndarray:
        """Apply the symmetric contraction.

        Parameters
        ----------
        x:
            Input features shaped ``(batch, multiplicity, ell_dim_sum)`` where
            ``ell_dim_sum`` is the total representation dimension of
            :attr:`irreps_in`.
        indices:
            Either a rank-1 tensor with element indices or a rank-2 mixing
            matrix ``(batch, num_elements)`` specifying per-node weights.
        """
        x = jnp.asarray(x)
        dtype = x.dtype

        _validate_features(x, self.mul, self.feature_dim)

        basis_weights = hk.get_parameter(
            'weight',
            shape=self.weight_param_shape,
            dtype=dtype,
            init=hk.initializers.RandomNormal(),
        )

        if self.projection is not None:
            projection = jnp.asarray(self.projection, dtype=dtype)
            weight_flat = jnp.einsum('zau,ab->zbu', basis_weights, projection)
        else:
            weight_flat = basis_weights

        weight_flat = weight_flat.reshape(self.num_elements, self.weight_numel)

        selected_weights = _select_weights(
            weight_flat,
            indices,
            dtype=dtype,
            num_elements=self.num_elements,
        )

        weight_rep = cuex.RepArray(self.weight_irreps, selected_weights, cue.ir_mul)

        x_rep = self._features_to_rep(x, dtype)

        [out_ir_mul] = cuex.segmented_polynomial(
            self.descriptor.polynomial,
            [weight_rep.array, x_rep.array],
            [
                jax.ShapeDtypeStruct(
                    (x.shape[0], self.irreps_out_o3.dim),
                    dtype,
                )
            ],
            method='naive',
            math_dtype=dtype,
        )

        out_mul_ir = ir_mul_to_mul_ir(out_ir_mul, self.irreps_out_o3)
        return out_mul_ir

    # ------------------------------------------------------------------
    def _features_to_rep(self, x: jnp.ndarray, dtype: jnp.dtype) -> cuex.RepArray:
        """Convert ``(batch, mul, ell_sum)`` features to a cue ``RepArray``."""
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

    @classmethod
    def import_from_torch(cls, torch_module, hk_params, scope):
        """Flatten per-block MACE weights into adapter layout.

        MACE splits the learnable parameters across one ``Contraction`` object per
        output irrep. Each contraction stores ``weights_max`` (highest-order term)
        and additional tensors in ``contraction.weights`` for the lower degrees.
        For parity we reshape each of those tensors to
        ``(num_elements, params_per_degree, num_features)`` and concatenate along
        the parameter axis so the combined array matches the single Haiku parameter
        ``(num_elements, total_params, num_features)`` used by the adapter.
        """
        hk_params = hk.data_structures.to_mutable_dict(hk_params)

        parts: list[jnp.ndarray] = []
        for contraction in torch_module.contractions:
            tensors = [contraction.weights_max, *contraction.weights]
            for tensor in tensors:
                if tensor.numel() == 0:
                    continue
                parts.append(jnp.array(tensor.detach().cpu().numpy()))

        hk_params[scope]['weight'] = jnp.concatenate(parts, axis=1)

        return hk.data_structures.to_immutable_dict(hk_params)


def _validate_features(x: jnp.ndarray, mul: int, feature_dim: int) -> None:
    """Ensure features are provided in the expected MACE layout."""
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
    """Select or mix weight vectors according to ``selector``."""
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
