"""Spherical harmonics layers matching the Torch ``e3nn.o3`` API."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import jax.numpy as jnp
from e3nn_jax import Irreps, spherical_harmonics
from flax import nnx

from mace_jax.adapters.nnx.torch import nxx_auto_import_from_torch


@nxx_auto_import_from_torch(allow_missing_mapper=True)
class SphericalHarmonics(nnx.Module):
    """Evaluate spherical harmonics with cues that mirror the Torch layer API."""

    irreps_out: int | Sequence[int] | str | Irreps
    normalize: bool
    normalization: str = 'integral'
    irreps_in: Any = None

    def __init__(
        self,
        irreps_out: int | Sequence[int] | str | Irreps,
        normalize: bool,
        normalization: str = 'integral',
        irreps_in: Any = None,
        *,
        layout_str: str = 'mul_ir',
    ) -> None:
        self.irreps_out = irreps_out
        self.normalize = normalize
        self.normalization = normalization
        if layout_str not in {'mul_ir', 'ir_mul'}:
            raise ValueError(
                f"layout_str must be either 'mul_ir' or 'ir_mul'; got {layout_str!r}."
            )
        self.layout_str = layout_str
        self.irreps_in = irreps_in
        irreps_out_input = self.irreps_out
        if isinstance(irreps_out_input, str):
            irreps_out = Irreps(irreps_out_input)
        elif isinstance(irreps_out_input, Irreps):
            irreps_out = irreps_out_input
        else:
            irreps_out = Irreps(irreps_out_input)

        irreps_in = self.irreps_in
        if irreps_in is None:
            irreps_in = Irreps('1o')
        else:
            irreps_in = Irreps(irreps_in)

        if irreps_in not in (Irreps('1x1o'), Irreps('1x1e')):
            raise ValueError(
                f'irreps_in must be either `1x1o` or `1x1e`; received {irreps_in!s}'
            )
        self._irreps_in = irreps_in

        input_parity = irreps_in[0].ir.p

        ls = []
        for mul, ir in irreps_out:
            if ir.p != input_parity**ir.l:
                raise ValueError(
                    'Output parity mismatch in SphericalHarmonics: '
                    f'l={ir.l}, p={ir.p}, expected {input_parity**ir.l}'
                )
            ls.extend([ir.l] * mul)

        irreps_out = Irreps([(1, (lv, input_parity**lv)) for lv in ls]).simplify()
        self._ls_list = ls
        self._irreps_out = irreps_out
        self._lmax = max(ls) if ls else 0
        self._is_range_lmax = ls == list(range(self._lmax + 1))

        if self._lmax > 11:
            raise NotImplementedError(
                f'spherical_harmonics maximum l implemented is 11, got {self._lmax}'
            )

        if self.normalization not in {'integral', 'component', 'norm'}:
            raise ValueError(
                "normalization must be 'integral', 'component', or 'norm'; "
                f'got {self.normalization!r}'
            )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray | dict:
        """Compute spherical harmonics for input vectors x [*, 3]."""
        if self.normalize:
            x = x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)

        sh = spherical_harmonics(
            self._irreps_out, x, normalize=False, normalization='component'
        )

        if not self._is_range_lmax:
            pieces = []
            for l_value in self._ls_list:
                start, end = l_value * l_value, (l_value + 1) * (l_value + 1)
                pieces.append(sh[..., start:end])
            sh = jnp.concatenate(pieces, axis=-1)

        if self.normalization == 'integral':
            sh = sh / math.sqrt(4 * math.pi)
        elif self.normalization == 'norm':
            factors = []
            for l_value in self._ls_list:
                factors.extend([math.sqrt(2 * l_value + 1)] * (2 * l_value + 1))
            sh = sh / jnp.array(factors, dtype=sh.dtype)

        array = sh.array
        if self.layout_str == 'ir_mul':
            from mace_jax.adapters.cuequivariance.utility import (  # noqa: PLC0415
                mul_ir_to_ir_mul,
            )

            array = mul_ir_to_ir_mul(array, self._irreps_out)
            return array
        return array
