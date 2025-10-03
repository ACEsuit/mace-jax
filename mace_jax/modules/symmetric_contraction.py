from dataclasses import dataclass
from typing import Optional, Union

import cuequivariance as cue
import cuequivariance_jax as cuex
import haiku as hk
import jax.numpy as jnp
import numpy as np
from cuequivariance.group_theory.experimental.mace.symmetric_contractions import (
    _symmetric_contraction,
    symmetric_contraction,
)
from e3nn_jax import Irrep, Irreps

from mace_jax.haiku.torch import register_import
from mace_jax.tools.dtype import default_dtype


def _convert_to_cue_irreps(e3_irreps: Irreps) -> cue.Irreps:
    """Convert an :class:`e3nn_jax.Irreps` to :class:`cue.Irreps`."""
    mul_irreps: list[cue.MulIrrep] = []
    for mul, ir in e3_irreps:
        cue_ir = cue.O3(ir.l, 1 if ir.p == 1 else -1)
        mul_irreps.append(cue.MulIrrep(mul=mul, ir=cue_ir))
    return cue.Irreps(cue.O3, mul_irreps)


@dataclass(frozen=True)
class _BlockConfig:
    """Per-output contraction data."""

    poly: Optional[cue.EquivariantPolynomial]
    projection: Optional[jnp.ndarray]
    total_params: int  # number of pyro parameters per feature (sum over degrees)
    output_dim: int


@register_import('mace.modules.symmetric_contraction.SymmetricContraction')
class SymmetricContraction(hk.Module):
    """Symmetric contraction using ``cuequivariance-jax`` polynomials."""

    def __init__(
        self,
        irreps_in: Union[str, Irreps],
        irreps_out: Union[str, Irreps],
        correlation: Union[int, dict[Irrep, int]],
        irrep_normalization: str = 'component',
        path_normalization: str = 'element',
        use_reduced_cg: bool = False,
        internal_weights: Optional[bool] = None,
        shared_weights: Optional[bool] = None,
        num_elements: Optional[int] = None,
        method: str = 'naive',
        name: Optional[str] = None,
    ):
        super().__init__(name=name)

        if irrep_normalization is None:
            irrep_normalization = 'component'
        if path_normalization is None:
            path_normalization = 'element'

        assert irrep_normalization in {'component', 'norm', 'none'}
        assert path_normalization in {'element', 'path', 'none'}

        if internal_weights is None:
            internal_weights = True
        if not internal_weights:
            raise NotImplementedError(
                'External weights are not supported in cue backend yet.'
            )
        if shared_weights:
            raise NotImplementedError(
                'Shared weights are not supported in cue backend yet.'
            )
        if use_reduced_cg:
            raise NotImplementedError('Reduced CGs are not yet wired for cue backend.')

        self.method = method

        self.e3_irreps_in = Irreps(irreps_in)
        self.e3_irreps_out = Irreps(irreps_out)
        self.num_elements = int(num_elements or 1)

        num_features = self.e3_irreps_in.count((0, 1))
        if num_features <= 0:
            raise ValueError('Input irreps must contain at least one scalar channel.')
        self.num_features = num_features

        # Drop multiplicities when constructing coupling irreps (matches Torch behaviour)
        coupling_irreps = Irreps([ir for _, ir in self.e3_irreps_in])
        self.cue_irreps_in_base = _convert_to_cue_irreps(coupling_irreps)
        self.cue_irreps_in = self.cue_irreps_in_base.set_mul(self.num_features)

        # Normalise correlation argument into a map keyed by Irrep objects
        if isinstance(correlation, dict):
            corr_map: dict[Irrep, int] = {}
            for key, value in correlation.items():
                key_ir = key if isinstance(key, Irrep) else Irrep(key)
                corr_map[key_ir] = int(value)
        else:
            corr_map = {ir: int(correlation) for _, ir in self.e3_irreps_out}
        self.correlation_map = corr_map

        self.blocks: list[_BlockConfig] = []

        for _, ir in self.e3_irreps_out:
            corr = self.correlation_map[ir]
            degrees = tuple(range(1, corr + 1))
            degrees_desc = tuple(range(corr, 0, -1))

            cue_irrep_out_base = _convert_to_cue_irreps(Irreps([(1, ir)]))
            cue_irrep_out = cue_irrep_out_base.set_mul(self.num_features)

            try:
                poly, projection = symmetric_contraction(
                    self.cue_irreps_in,
                    cue_irrep_out,
                    degrees,
                )
                projection = jnp.asarray(projection, dtype=default_dtype())
            except ValueError:
                poly, projection = None, None

            # Determine number of parameters per degree following Torch ordering (descending)
            params_per_degree: list[int] = []
            for deg in degrees_desc:
                poly_deg = _symmetric_contraction(
                    self.cue_irreps_in,
                    cue_irrep_out,
                    deg,
                )
                dim = poly_deg.inputs[0].irreps.dim
                params_per_degree.append(dim // self.num_features)

            total_params = int(sum(params_per_degree))
            self.blocks.append(
                _BlockConfig(
                    poly=poly,
                    projection=projection,
                    total_params=total_params,
                    output_dim=cue_irrep_out.dim,
                )
            )

    # ---------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _features_to_rep(self, x: jnp.ndarray) -> cuex.RepArray:
        if x.ndim != 3:
            raise ValueError(
                f'x must have shape (batch, num_features, num_ell); got {x.shape}'
            )
        if x.shape[1] != self.num_features:
            raise ValueError(
                f'x.shape[1] ({x.shape[1]}) != expected num_features {self.num_features}'
            )
        segments: list[jnp.ndarray] = []
        start = 0
        for mul_ir in self.cue_irreps_in_base:
            dim = mul_ir.ir.dim
            seg = x[:, :, start : start + dim]
            if seg.shape[-1] != dim:
                raise ValueError('Input feature dimension mismatch with irreps.')
            segments.append(jnp.swapaxes(seg, -2, -1))  # -> (batch, dim, num_features)
            start += dim
        return cuex.from_segments(
            self.cue_irreps_in,
            segments,
            (x.shape[0], self.num_features),
            cue.ir_mul,
            dtype=x.dtype,
        )

    # ------------------------------------------------------------------
    def __call__(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        x = jnp.asarray(x)
        y = jnp.asarray(y)
        if y.ndim != 2:
            raise ValueError(f'y must have shape (batch, num_elements); got {y.shape}')
        if y.shape[1] != self.num_elements:
            raise ValueError(
                f'y.shape[1] ({y.shape[1]}) != expected num_elements {self.num_elements}'
            )

        x_rep = self._features_to_rep(x)
        outputs: list[jnp.ndarray] = []

        batch_size = x.shape[0]

        for idx, block in enumerate(self.blocks):
            if (
                block.total_params <= 0
                or block.poly is None
                or block.projection is None
            ):
                outputs.append(jnp.zeros((batch_size, block.output_dim), dtype=x.dtype))
                continue

            weights = hk.get_parameter(
                f'weights_{idx}',
                shape=(self.num_elements, block.total_params, self.num_features),
                init=hk.initializers.RandomNormal(
                    stddev=1.0 / max(block.total_params, 1)
                ),
                dtype=default_dtype(),
            )

            weights_combined = jnp.einsum('be,epf->bpf', y, weights)
            weights_projected = jnp.einsum(
                'bpf,pa->baf', weights_combined, block.projection
            )
            weights_flat = weights_projected.reshape(weights_projected.shape[0], -1)

            out_rep = cuex.equivariant_polynomial(
                block.poly,
                [weights_flat, x_rep],
                method=self.method,
            )
            outputs.append(out_rep.change_layout(cue.mul_ir).array)

        return jnp.concatenate(outputs, axis=-1)

    # ------------------------------------------------------------------
    @classmethod
    def import_from_torch(cls, torch_module, hk_params, scope):
        hk_params = hk.data_structures.to_mutable_dict(hk_params)

        for idx, contraction in enumerate(torch_module.contractions):
            param_key = f'weights_{idx}'
            if param_key not in hk_params[scope]:
                continue  # Block has no learnable parameters (zero projection)

            weights = [
                contraction.weights_max.detach().cpu().numpy(),
                *[w.detach().cpu().numpy() for w in contraction.weights],
            ]
            expected = hk_params[scope][param_key].shape[1]
            if expected == 0:
                continue

            collected = []
            remaining = expected
            for w in weights:
                w_flat = w.reshape(w.shape[0], -1, w.shape[-1])
                if remaining <= 0:
                    break
                if w_flat.shape[1] <= remaining:
                    collected.append(w_flat)
                    remaining -= w_flat.shape[1]
                else:
                    collected.append(w_flat[:, :remaining])
                    remaining = 0

            if remaining > 0:
                # pad with zeros if Torch had fewer active params than expected
                zeros = np.zeros(
                    (weights[0].shape[0], remaining, weights[0].shape[-1]),
                    dtype=weights[0].dtype,
                )
                collected.append(zeros)

            concat = np.concatenate(collected, axis=1)
            hk_params[scope][param_key] = jnp.asarray(concat, dtype=default_dtype())

        return hk.data_structures.to_immutable_dict(hk_params)
