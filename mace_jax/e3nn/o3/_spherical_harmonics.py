import math
from typing import Any, Union

import haiku as hk
import jax.numpy as jnp
from e3nn_jax import Irreps, spherical_harmonics

from mace_jax.haiku.torch import (
    auto_import_from_torch,
    register_import,
)


@register_import('e3nn.o3._spherical_harmonics.SphericalHarmonics')
@auto_import_from_torch(separator='~')
class SphericalHarmonics(hk.Module):
    """Haiku module version of :func:`e3nn_jax.spherical_harmonics`.

    Mirrors the PyTorch `e3nn.o3.SphericalHarmonics` behavior.
    """

    def __init__(
        self,
        irreps_out: Union[int, list[int], str, Irreps],
        normalize: bool,
        normalization: str = 'integral',
        irreps_in: Any = None,
        name: str = 'spherical_harmonics',
    ):
        super().__init__(name=name)

        self.normalize = normalize
        self.normalization = normalization
        assert normalization in ['integral', 'component', 'norm']

        # Parse irreps_out
        if isinstance(irreps_out, str):
            irreps_out = Irreps(irreps_out)

        if isinstance(irreps_out, Irreps) and irreps_in is None:
            for mul, (l, p) in irreps_out:
                if l % 2 == 1 and p == 1:
                    irreps_in = Irreps('1e')

        if irreps_in is None:
            irreps_in = Irreps('1o')

        irreps_in = Irreps(irreps_in)
        if irreps_in not in (Irreps('1x1o'), Irreps('1x1e')):
            raise ValueError(
                f'irreps_in for SphericalHarmonics must be either `1x1o` or `1x1e`, not `{irreps_in}`'
            )
        self.irreps_in = irreps_in
        input_p = irreps_in[0].ir.p

        if isinstance(irreps_out, Irreps):
            ls = []
            for mul, (l, p) in irreps_out:
                if p != input_p**l:
                    raise ValueError(
                        f'Output parity mismatch: got l={l}, p={p}, expected {input_p**l}'
                    )
                ls.extend([l] * mul)
        elif isinstance(irreps_out, int):
            ls = [irreps_out]
        else:
            ls = list(irreps_out)

        irreps_out = Irreps([(1, (l, input_p**l)) for l in ls]).simplify()
        self.irreps_out = irreps_out
        self._ls_list = ls
        self._lmax = max(ls)
        self._is_range_lmax = ls == list(range(max(ls) + 1))
        self._prof_str = f'spherical_harmonics({ls})'

        if self._lmax > 11:
            raise NotImplementedError(
                f'spherical_harmonics maximum l implemented is 11, got {self._lmax}'
            )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Compute spherical harmonics for input vectors x [*,3]."""
        if self.normalize:
            # Normalize to unit vectors, but keep zero-radius stable
            x = x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)

        sh = spherical_harmonics(
            self.irreps_out, x, normalize=False, normalization='component'
        )

        if not self._is_range_lmax:
            # Select only the irreps requested
            pieces = []
            for ls in self._ls_list:
                start, end = ls * ls, (ls + 1) * (ls + 1)
                pieces.append(sh[..., start:end])
            sh = jnp.concatenate(pieces, axis=-1)

        # Apply normalization mode
        if self.normalization == 'integral':
            sh = sh / math.sqrt(4 * math.pi)
        elif self.normalization == 'norm':
            factors = []
            for ls in self._ls_list:
                factors.extend([math.sqrt(2 * ls + 1)] * (2 * ls + 1))
            factors = jnp.array(factors, dtype=sh.dtype)
            sh = sh / factors

        return sh.array
