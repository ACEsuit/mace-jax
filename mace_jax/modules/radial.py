###########################################################################################
# Radial basis and cutoff
# Authors: Ilyes Batatia, Gregor Simm
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

import logging
from collections.abc import Sequence

import ase
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as fnn

from mace_jax.adapters.flax.torch import (
    auto_import_from_torch_flax,
)
from mace_jax.tools.dtype import default_dtype
from mace_jax.tools.scatter import scatter_sum

from .special import chebyshev_polynomial_t


@auto_import_from_torch_flax(allow_missing_mapper=True)
class BesselBasis(fnn.Module):
    """Flax implementation of the Bessel basis from MACE."""

    r_max: float
    num_basis: int = 8
    trainable: bool = False

    @fnn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = jnp.asarray(x)
        dtype = x.dtype
        init_bessel = (
            np.pi
            / float(self.r_max)
            * jnp.linspace(1.0, self.num_basis, self.num_basis, dtype=default_dtype())
        )

        if self.trainable:
            bessel_weights = self.param(
                'bessel_weights', lambda rng: init_bessel
            ).astype(dtype)
        else:
            bessel_weights = init_bessel.astype(dtype)

        prefactor = jnp.sqrt(2.0 / jnp.asarray(self.r_max, dtype=dtype))

        eps = jnp.asarray(jnp.finfo(dtype).eps, dtype=dtype)
        near_zero = jnp.abs(x) < eps
        safe_denominator = jnp.where(near_zero, 1.0, x)

        numerator = jnp.sin(bessel_weights * x)
        safe_denominator = jnp.broadcast_to(safe_denominator, numerator.shape)
        ratio = numerator / safe_denominator
        near_zero_broadcast = jnp.broadcast_to(near_zero, ratio.shape)
        weights_broadcast = jnp.broadcast_to(bessel_weights, ratio.shape)
        ratio = jnp.where(near_zero_broadcast, weights_broadcast, ratio)
        return prefactor * ratio

    def __repr__(self):
        return (
            f'{self.__class__.__name__}(r_max={self.r_max}, '
            f'num_basis={self.num_basis}, trainable={self.trainable})'
        )


@auto_import_from_torch_flax(allow_missing_mapper=True)
class ChebychevBasis(fnn.Module):
    """Flax implementation of the Chebyshev polynomial basis."""

    r_max: float
    num_basis: int = 8

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        dtype = x.dtype
        n = jnp.arange(1, self.num_basis + 1, dtype=dtype)
        x_broadcast = jnp.broadcast_to(x, x.shape[:-1] + (self.num_basis,))
        return chebyshev_polynomial_t(x_broadcast, n)

    def __repr__(self):
        return (
            f'{self.__class__.__name__}(r_max={self.r_max}, num_basis={self.num_basis})'
        )


@auto_import_from_torch_flax(allow_missing_mapper=True)
class GaussianBasis(fnn.Module):
    """Gaussian radial basis functions."""

    r_max: float
    num_basis: int = 128
    trainable: bool = False

    @fnn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        dtype = x.dtype
        init_gaussians = jnp.linspace(
            0.0, float(self.r_max), self.num_basis, dtype=default_dtype()
        )

        if self.trainable:
            gaussian_weights = self.param(
                'gaussian_weights', lambda rng: init_gaussians
            ).astype(dtype)
        else:
            gaussian_weights = init_gaussians.astype(dtype)

        spacing = float(self.r_max) / float(self.num_basis - 1)
        coeff = jnp.asarray(-0.5 / (spacing**2), dtype=dtype)
        shifted = x[..., None] - gaussian_weights
        return jnp.exp(coeff * jnp.square(shifted))


class PolynomialCutoff(fnn.Module):
    """Polynomial cutoff function that goes from 1 to 0 as r approaches r_max."""

    r_max: float
    p: int = 6

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        r_max = jnp.asarray(self.r_max, dtype=x.dtype)
        p = jnp.asarray(float(self.p), dtype=x.dtype)
        return self.calculate_envelope(x, r_max, p)

    @staticmethod
    def calculate_envelope(
        x: jnp.ndarray,
        r_max: jnp.ndarray,
        p: jnp.ndarray,
    ) -> jnp.ndarray:
        r_over_r_max = x / r_max
        envelope = (
            1.0
            - ((p + 1.0) * (p + 2.0) / 2.0) * jnp.power(r_over_r_max, p)
            + p * (p + 2.0) * jnp.power(r_over_r_max, p + 1.0)
            - (p * (p + 1.0) / 2.0) * jnp.power(r_over_r_max, p + 2.0)
        )
        return envelope * (x < r_max)

    def __repr__(self):
        return f'{self.__class__.__name__}(p={self.p}, r_max={self.r_max})'


@auto_import_from_torch_flax(allow_missing_mapper=True)
class ZBLBasis(fnn.Module):
    """Ziegler-Biersack-Littmark (ZBL) potential with polynomial cutoff."""

    p: int = 6
    trainable: bool = False
    r_max: float | None = None  # kept for backward compatibility

    def setup(self):
        if self.r_max is not None:
            logging.warning(
                'r_max is deprecated. r_max is determined from the covalent radii.'
            )

        self._c = jnp.array(
            [0.1818, 0.5099, 0.2802, 0.02817],
            dtype=default_dtype(),
        )
        self._covalent_radii = jnp.array(
            ase.data.covalent_radii,
            dtype=default_dtype(),
        )

    @fnn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        node_attrs: jnp.ndarray,
        edge_index: jnp.ndarray,
        atomic_numbers: jnp.ndarray,
    ) -> jnp.ndarray:
        sender, receiver = edge_index

        node_atomic_numbers = atomic_numbers[jnp.argmax(node_attrs, axis=1)][..., None]
        Z_u = node_atomic_numbers[sender].astype(jnp.int32)
        Z_v = node_atomic_numbers[receiver].astype(jnp.int32)

        if self.trainable:
            a_exp = self.param(
                'a_exp',
                lambda rng: jnp.array(0.300, dtype=default_dtype()),
            )
            a_prefactor = self.param(
                'a_prefactor',
                lambda rng: jnp.array(0.4543, dtype=default_dtype()),
            )
        else:
            a_exp = jnp.array(0.300, dtype=x.dtype)
            a_prefactor = jnp.array(0.4543, dtype=x.dtype)

        a_exp = a_exp.astype(x.dtype)
        a_prefactor = a_prefactor.astype(x.dtype)

        a = (
            a_prefactor
            * jnp.asarray(0.529, dtype=x.dtype)
            / (jnp.power(Z_u, a_exp) + jnp.power(Z_v, a_exp))
        )
        r_over_a = x / a

        phi = (
            self._c[0] * jnp.exp(-3.2 * r_over_a)
            + self._c[1] * jnp.exp(-0.9423 * r_over_a)
            + self._c[2] * jnp.exp(-0.4028 * r_over_a)
            + self._c[3] * jnp.exp(-0.2016 * r_over_a)
        )

        v_edges = (14.3996 * Z_u * Z_v) / x * phi

        r_max = self._covalent_radii[Z_u] + self._covalent_radii[Z_v]
        envelope = PolynomialCutoff.calculate_envelope(
            x, r_max.astype(x.dtype), jnp.array(float(self.p), dtype=x.dtype)
        )
        v_edges = 0.5 * v_edges * envelope

        V_ZBL = scatter_sum(v_edges, receiver, dim=0, dim_size=node_attrs.shape[0])
        return jnp.squeeze(V_ZBL, axis=-1)

    def __repr__(self):
        return f'{self.__class__.__name__}(c={self._c})'


@auto_import_from_torch_flax(allow_missing_mapper=True)
class AgnesiTransform(fnn.Module):
    """Agnesi transform used for radial scaling."""

    q: float = 0.9183
    p: float = 4.5791
    a: float = 1.0805
    trainable: bool = False

    def setup(self):
        self._covalent_radii = jnp.array(
            ase.data.covalent_radii,
            dtype=default_dtype(),
        )

    @fnn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        node_attrs: jnp.ndarray,
        edge_index: jnp.ndarray,
        atomic_numbers: jnp.ndarray,
    ) -> jnp.ndarray:
        sender, receiver = edge_index

        node_atomic_numbers = atomic_numbers[jnp.argmax(node_attrs, axis=1)][..., None]
        Z_u = node_atomic_numbers[sender].astype(jnp.int32)
        Z_v = node_atomic_numbers[receiver].astype(jnp.int32)

        if self.trainable:
            a = self.param('a', lambda rng: jnp.array(self.a, dtype=default_dtype()))
            q = self.param('q', lambda rng: jnp.array(self.q, dtype=default_dtype()))
            p = self.param('p', lambda rng: jnp.array(self.p, dtype=default_dtype()))
        else:
            dtype = x.dtype
            a = jnp.array(self.a, dtype=dtype)
            q = jnp.array(self.q, dtype=dtype)
            p = jnp.array(self.p, dtype=dtype)

        a = a.astype(x.dtype)
        q = q.astype(x.dtype)
        p = p.astype(x.dtype)

        r_0 = 0.5 * (self._covalent_radii[Z_u] + self._covalent_radii[Z_v])
        r_over_r_0 = x / r_0

        numerator = a * jnp.power(r_over_r_0, q)
        denominator = 1.0 + jnp.power(r_over_r_0, q - p)
        return 1.0 / (1.0 + numerator / denominator)

    def __repr__(self):
        return (
            f'{self.__class__.__name__}(a={float(self.a):.4f}, '
            f'q={float(self.q):.4f}, p={float(self.p):.4f})'
        )


@auto_import_from_torch_flax(allow_missing_mapper=True)
class SoftTransform(fnn.Module):
    """Soft transform with a learnable alpha parameter."""

    alpha: float = 4.0
    trainable: bool = False

    def setup(self):
        self._covalent_radii = jnp.array(ase.data.covalent_radii, dtype=default_dtype())

    def compute_r_0(
        self,
        node_attrs: jnp.ndarray,
        edge_index: jnp.ndarray,
        atomic_numbers: jnp.ndarray,
    ) -> jnp.ndarray:
        sender, receiver = edge_index
        node_atomic_numbers = atomic_numbers[jnp.argmax(node_attrs, axis=1)].reshape(
            -1, 1
        )
        Z_u = node_atomic_numbers[sender].astype(jnp.int32)
        Z_v = node_atomic_numbers[receiver].astype(jnp.int32)
        return self._covalent_radii[Z_u] + self._covalent_radii[Z_v]

    @fnn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        node_attrs: jnp.ndarray,
        edge_index: jnp.ndarray,
        atomic_numbers: jnp.ndarray,
    ) -> jnp.ndarray:
        dtype = x.dtype
        r_0 = self.compute_r_0(node_attrs, edge_index, atomic_numbers).astype(dtype)
        p_0 = (3.0 / 4.0) * r_0
        p_1 = (4.0 / 3.0) * r_0
        m = 0.5 * (p_0 + p_1)
        if self.trainable:
            alpha_param = self.param(
                'alpha', lambda rng: jnp.array(self.alpha, dtype=default_dtype())
            )
        else:
            alpha_param = jnp.array(self.alpha, dtype=dtype)
        alpha = alpha_param.astype(dtype) / (p_1 - p_0)
        s_x = 0.5 * (1.0 + jnp.tanh(alpha * (x - m)))
        return p_0 + (x - p_0) * s_x

    def __repr__(self):
        return f'{self.__class__.__name__}(alpha={self.alpha:.4f}, trainable={self.trainable})'


@auto_import_from_torch_flax(allow_missing_mapper=True)
class _RadialSequential(fnn.Module):
    channels: Sequence[int]

    @fnn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        if len(self.channels) < 2:
            raise ValueError('channels must have length >= 2 for RadialMLP')

        x = inputs
        last_idx = len(self.channels) - 1
        layer_idx = 0
        for idx, out_channels in enumerate(self.channels[1:], start=1):
            x = fnn.Dense(out_channels, use_bias=True, name=str(layer_idx))(x)
            layer_idx += 1
            if idx != last_idx:
                x = fnn.LayerNorm(
                    use_bias=True,
                    use_scale=True,
                    reduction_axes=-1,
                    feature_axes=-1,
                    epsilon=1e-5,
                    name=str(layer_idx),
                )(x)
                layer_idx += 1
                x = jax.nn.silu(x)
                layer_idx += 1
        return x


@auto_import_from_torch_flax(allow_missing_mapper=True)
class RadialMLP(fnn.Module):
    """Wrapper that aligns with the Torch RadialMLP parameter layout."""

    channels: Sequence[int]

    def setup(self) -> None:
        self.net = _RadialSequential(tuple(self.channels), name='net')

    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        return self.net(inputs)
