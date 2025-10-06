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
from cuequivariance_torch.operations.symmetric_contraction import (
    SymmetricContraction as CueSymmetricContractionTorch,
)
from mace_jax.haiku.torch import register_import

from .utility import ir_mul_to_mul_ir


@register_import('cuequivariance_torch.operations.symmetric_contraction.SymmetricContraction')
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
        input_layout: str = 'mul_ir',
        name: str | None = None,
    ) -> None:
        """Construct a cue-backed symmetric contraction layer.

        Parameters
        ----------
        irreps_in, irreps_out
            Input and output irreducible representations in e3nn format.  The
            cue backend ultimately receives the corresponding cue irreps, but we
            validate here that all multiplicities match because the segmented
            polynomial assumes a single shared multiplicity across all blocks.
        correlation
            Highest polynomial degree used in the contraction (``nu`` in the
            MACE paper).  Degrees ``1..correlation`` are bundled into the single
            segmented polynomial descriptor supplied by
            :func:`cue_mace_symmetric_contraction`.
        num_elements
            Cardinality of the per-element weight table.  The module indexes
            into this table using the ``indices`` argument supplied to
            :meth:`__call__`.
        use_reduced_cg
            Mirrors the torch API but only controls whether we pre-multiply by
            the projection matrix returned by cue.  ``True`` stores weights in
            the reduced CG basis; ``False`` keeps the expanded MACE basis and
            projects on the fly.
        input_layout
            Layout of the incoming feature tensor.  ``'mul_ir'`` (default)
            expects the usual e3nn ordering ``(batch, mul, ell_dim_sum)``.
            ``'ir_mul'`` accepts the transposed cue layout and is converted back
            internally before the descriptor is evaluated.  The wrapper in
            :mod:`mace_jax.modules.wrapper_ops` toggles this when the runtime is
            already operating in ``ir_mul`` order.
        name
            Optional Haiku scope name.
        """
        super().__init__(name=name)

        if correlation <= 0:
            raise ValueError('correlation must be a positive integer')
        if num_elements <= 0:
            raise ValueError('num_elements must be positive')

        if input_layout not in {'mul_ir', 'ir_mul'}:
            raise ValueError(
                "input_layout must be either 'mul_ir' or 'ir_mul'; "
                f'got {input_layout!r}'
            )

        self.correlation = correlation
        self.num_elements = num_elements
        self.use_reduced_cg = use_reduced_cg
        self.input_layout = input_layout

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
        """Apply the symmetric contraction to the provided features.

        Parameters
        ----------
        x
            Feature tensor whose trailing axes follow ``input_layout``.  For the
            default ``'mul_ir'`` the shape is ``(batch, mul, ell_dim_sum)`` where
            ``ell_dim_sum`` equals ``irreps_in.dim / mul``.
        indices
            Either integer indices ``(batch,)`` selecting a single element per
            batch item or a mixing matrix ``(batch, num_elements)`` describing a
            linear combination of element-specific weights.

        Returns
        -------
        jnp.ndarray
            Output features in ``mul_ir`` layout matching :attr:`irreps_out`.
        """
        x = jnp.asarray(x)
        dtype = x.dtype

        if self.input_layout == 'ir_mul':
            x = self._convert_ir_mul_to_mul_ir(x)

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
        """Convert ``mul_ir`` features to a cue ``RepArray``.

        cue expects a sequence of ``(ir_dim, mul)`` segments for each distinct
        irrep.  This helper slices the incoming ``(batch, mul, ell_dim_sum)``
        tensor into those segments, performs the required transpose to obtain
        ``(batch, ir_dim, mul)``, and finally wraps the concatenated data in a
        :class:`cuequivariance_jax.RepArray` carrying the corresponding cue
        metadata.
        """
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
        """Reorder features from ``ir_mul`` to ``mul_ir`` layout.

        The wrapper switches the symmetric contraction into this branch when the
        upstream activations already reside in cue’s ``ir_mul`` order (for
        example when the entire graph operates in that layout).  Each irrep
        segment is transposed back to ``(mul, ir_dim)`` so that the regular
        :meth:`_features_to_rep` conversion can run unchanged.
        """

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

    @classmethod
    def import_from_torch(cls, torch_module, hk_params, scope):
        """Import weights from ``cuequivariance_torch`` symmetric contraction.

        Parameters
        ----------
        torch_module
            Instance of :class:`cuequivariance_torch.operations.symmetric_contraction.SymmetricContraction`
            whose parameters should be mirrored into the Haiku state.  Native
            ``mace`` symmetric contractions are rejected, since their parameter
            layout does not map cleanly onto the cue tables used here.
        hk_params
            Haiku parameter dictionary to update.  The routine operates on a
            mutable copy so callers can chain multiple imports.
        scope
            Haiku module scope under which the symmetric contraction stores its
            weights.

        Returns
        -------
        hk.Params
            Immutable Haiku parameter dictionary containing the mirrored cue
            weights.
        """

        hk_params = hk.data_structures.to_mutable_dict(hk_params)
        module_qualname = (
            f'{torch_module.__class__.__module__}.{torch_module.__class__.__name__}'
        )

        if not isinstance(torch_module, CueSymmetricContractionTorch):
            if module_qualname == 'mace.modules.symmetric_contraction.SymmetricContraction':
                raise TypeError(
                    'Importing native MACE SymmetricContraction is not supported; '
                    'enable cuequivariance in the Torch model so that '
                    'cuequivariance_torch.operations.symmetric_contraction.SymmetricContraction '
                    'is used instead.'
                )
            raise TypeError(
                'SymmetricContraction.import_from_torch expects an instance of '
                'cuequivariance_torch.operations.symmetric_contraction.SymmetricContraction; '
                f'received {module_qualname!r}.'
            )

        hk_params.setdefault(scope, {})
        hk_params[scope]['weight'] = jnp.array(
            torch_module.weight.detach().cpu().numpy()
        )

        return hk.data_structures.to_immutable_dict(hk_params)


def _validate_features(x: jnp.ndarray, mul: int, feature_dim: int) -> None:
    """Ensure features follow ``(batch, mul, feature_dim)`` layout.

    A descriptive error is raised when either the rank or the trailing
    dimensions do not match the expected configuration.  Validation happens
    after any optional layout conversion so downstream code can assume the
    canonical ordering.
    """
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
    """Select or mix weight vectors according to ``selector``.

    When ``selector`` is rank-1 it is interpreted as element indices.  Rank-2
    selectors are treated as mixing matrices (typically one-hot or softmax
    outputs) whose rightmost dimension must match ``num_elements``.
    """
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
