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
from flax import nnx

from mace_jax.adapters.nnx.torch import nxx_auto_import_from_torch
from mace_jax.tools.dtype import default_dtype
from mace_jax.tools.scatter import scatter_sum

from .special import chebyshev_polynomial_t


@nxx_auto_import_from_torch(allow_missing_mapper=True)
class BesselBasis(nnx.Module):
    """Flax implementation of the Bessel basis from MACE."""

    r_max: float
    num_basis: int = 8
    trainable: bool = False

    def __init__(
        self,
        r_max: float,
        num_basis: int = 8,
        trainable: bool = False,
        *,
        rngs: nnx.Rngs | None = None,
    ) -> None:
        self.r_max = r_max
        self.num_basis = num_basis
        self.trainable = trainable
        init_bessel = (
            np.pi
            / float(self.r_max)
            * jnp.linspace(1.0, self.num_basis, self.num_basis, dtype=default_dtype())
        )
        self._bessel_weights = init_bessel
        if self.trainable:
            if rngs is None:
                raise ValueError('rngs is required for trainable BesselBasis')
            self.bessel_weights = nnx.Param(init_bessel)
        else:
            self.bessel_weights = None

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = jnp.asarray(x)
        dtype = x.dtype
        if self.trainable and self.bessel_weights is not None:
            bessel_weights = jnp.asarray(self.bessel_weights, dtype=dtype)
        else:
            bessel_weights = jnp.asarray(self._bessel_weights, dtype=dtype)

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


@nxx_auto_import_from_torch(allow_missing_mapper=True)
class ChebychevBasis(nnx.Module):
    """Flax implementation of the Chebyshev polynomial basis."""

    r_max: float
    num_basis: int = 8

    def __init__(self, r_max: float, num_basis: int = 8) -> None:
        self.r_max = r_max
        self.num_basis = num_basis

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        dtype = x.dtype
        n = jnp.arange(1, self.num_basis + 1, dtype=dtype)
        x_broadcast = jnp.broadcast_to(x, x.shape[:-1] + (self.num_basis,))
        return chebyshev_polynomial_t(x_broadcast, n)

    def __repr__(self):
        return (
            f'{self.__class__.__name__}(r_max={self.r_max}, num_basis={self.num_basis})'
        )


@nxx_auto_import_from_torch(allow_missing_mapper=True)
class GaussianBasis(nnx.Module):
    """Gaussian radial basis functions."""

    r_max: float
    num_basis: int = 128
    trainable: bool = False

    def __init__(
        self,
        r_max: float,
        num_basis: int = 128,
        trainable: bool = False,
        *,
        rngs: nnx.Rngs | None = None,
    ) -> None:
        self.r_max = r_max
        self.num_basis = num_basis
        self.trainable = trainable
        init_gaussians = jnp.linspace(
            0.0, float(self.r_max), self.num_basis, dtype=default_dtype()
        )
        self._gaussian_weights = init_gaussians
        if self.trainable:
            if rngs is None:
                raise ValueError('rngs is required for trainable GaussianBasis')
            self.gaussian_weights = nnx.Param(init_gaussians)
        else:
            self.gaussian_weights = None

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        dtype = x.dtype
        if self.trainable and self.gaussian_weights is not None:
            gaussian_weights = jnp.asarray(self.gaussian_weights, dtype=dtype)
        else:
            gaussian_weights = jnp.asarray(self._gaussian_weights, dtype=dtype)

        spacing = float(self.r_max) / float(self.num_basis - 1)
        coeff = jnp.asarray(-0.5 / (spacing**2), dtype=dtype)
        shifted = x[..., None] - gaussian_weights
        return jnp.exp(coeff * jnp.square(shifted))


class PolynomialCutoff(nnx.Module):
    """Polynomial cutoff function that goes from 1 to 0 as r approaches r_max."""

    r_max: float
    p: int = 6

    def __init__(self, r_max: float, p: int = 6) -> None:
        self.r_max = r_max
        self.p = p

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


@nxx_auto_import_from_torch(allow_missing_mapper=True)
class ZBLBasis(nnx.Module):
    """Ziegler-Biersack-Littmark (ZBL) potential with polynomial cutoff."""

    p: int = 6
    trainable: bool = False
    r_max: float | None = None  # kept for backward compatibility

    def __init__(
        self,
        p: int = 6,
        trainable: bool = False,
        r_max: float | None = None,
        *,
        rngs: nnx.Rngs | None = None,
    ) -> None:
        self.p = p
        self.trainable = trainable
        self.r_max = r_max
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
        if self.trainable:
            if rngs is None:
                raise ValueError('rngs is required for trainable ZBLBasis')
            self.a_exp = nnx.Param(jnp.array(0.300, dtype=default_dtype()))
            self.a_prefactor = nnx.Param(jnp.array(0.4543, dtype=default_dtype()))
        else:
            self.a_exp = None
            self.a_prefactor = None

    def __call__(
        self,
        x: jnp.ndarray,
        node_attrs: jnp.ndarray,
        edge_index: jnp.ndarray,
        atomic_numbers: jnp.ndarray,
        node_attrs_index: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        sender, receiver = edge_index

        if node_attrs_index is None:
            node_attrs_index = jnp.argmax(node_attrs, axis=1)
        node_atomic_numbers = atomic_numbers[
            jnp.asarray(node_attrs_index, dtype=jnp.int32).reshape(-1)
        ][..., None]
        Z_u = node_atomic_numbers[sender].astype(jnp.int32)
        Z_v = node_atomic_numbers[receiver].astype(jnp.int32)

        if self.trainable and self.a_exp is not None and self.a_prefactor is not None:
            a_exp = jnp.asarray(self.a_exp, dtype=x.dtype)
            a_prefactor = jnp.asarray(self.a_prefactor, dtype=x.dtype)
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


@nxx_auto_import_from_torch(allow_missing_mapper=True)
class AgnesiTransform(nnx.Module):
    """Agnesi transform used for radial scaling."""

    q: float = 0.9183
    p: float = 4.5791
    a: float = 1.0805
    trainable: bool = False

    def __init__(
        self,
        q: float = 0.9183,
        p: float = 4.5791,
        a: float = 1.0805,
        trainable: bool = False,
        *,
        rngs: nnx.Rngs | None = None,
    ) -> None:
        self.q = q
        self.p = p
        self.a = a
        self.trainable = trainable
        self._covalent_radii = jnp.array(
            ase.data.covalent_radii,
            dtype=default_dtype(),
        )
        if self.trainable:
            if rngs is None:
                raise ValueError('rngs is required for trainable AgnesiTransform')
            self.a_param = nnx.Param(jnp.array(self.a, dtype=default_dtype()))
            self.q_param = nnx.Param(jnp.array(self.q, dtype=default_dtype()))
            self.p_param = nnx.Param(jnp.array(self.p, dtype=default_dtype()))
        else:
            self.a_param = None
            self.q_param = None
            self.p_param = None

    def __call__(
        self,
        x: jnp.ndarray,
        node_attrs: jnp.ndarray,
        edge_index: jnp.ndarray,
        atomic_numbers: jnp.ndarray,
        node_attrs_index: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        sender, receiver = edge_index

        if node_attrs_index is None:
            node_attrs_index = jnp.argmax(node_attrs, axis=1)
        node_atomic_numbers = atomic_numbers[
            jnp.asarray(node_attrs_index, dtype=jnp.int32).reshape(-1)
        ][..., None]
        Z_u = node_atomic_numbers[sender].astype(jnp.int32)
        Z_v = node_atomic_numbers[receiver].astype(jnp.int32)

        if self.trainable and self.a_param is not None:
            a = jnp.asarray(self.a_param, dtype=x.dtype)
            q = jnp.asarray(self.q_param, dtype=x.dtype)
            p = jnp.asarray(self.p_param, dtype=x.dtype)
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


@nxx_auto_import_from_torch(allow_missing_mapper=True)
class SoftTransform(nnx.Module):
    """Soft transform with a learnable alpha parameter."""

    alpha: float = 4.0
    trainable: bool = False

    def __init__(
        self,
        alpha: float = 4.0,
        trainable: bool = False,
        *,
        rngs: nnx.Rngs | None = None,
    ) -> None:
        self.alpha = alpha
        self.trainable = trainable
        self._covalent_radii = jnp.array(ase.data.covalent_radii, dtype=default_dtype())
        if self.trainable:
            if rngs is None:
                raise ValueError('rngs is required for trainable SoftTransform')
            self.alpha_param = nnx.Param(jnp.array(self.alpha, dtype=default_dtype()))
        else:
            self.alpha_param = None

    def compute_r_0(
        self,
        node_attrs: jnp.ndarray,
        edge_index: jnp.ndarray,
        atomic_numbers: jnp.ndarray,
        node_attrs_index: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        sender, receiver = edge_index
        if node_attrs_index is None:
            node_attrs_index = jnp.argmax(node_attrs, axis=1)
        node_atomic_numbers = atomic_numbers[
            jnp.asarray(node_attrs_index, dtype=jnp.int32).reshape(-1)
        ].reshape(-1, 1)
        Z_u = node_atomic_numbers[sender].astype(jnp.int32)
        Z_v = node_atomic_numbers[receiver].astype(jnp.int32)
        return self._covalent_radii[Z_u] + self._covalent_radii[Z_v]

    def __call__(
        self,
        x: jnp.ndarray,
        node_attrs: jnp.ndarray,
        edge_index: jnp.ndarray,
        atomic_numbers: jnp.ndarray,
        node_attrs_index: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        dtype = x.dtype
        r_0 = self.compute_r_0(
            node_attrs, edge_index, atomic_numbers, node_attrs_index=node_attrs_index
        ).astype(dtype)
        p_0 = (3.0 / 4.0) * r_0
        p_1 = (4.0 / 3.0) * r_0
        m = 0.5 * (p_0 + p_1)
        if self.trainable and self.alpha_param is not None:
            alpha_param = jnp.asarray(self.alpha_param, dtype=dtype)
        else:
            alpha_param = jnp.array(self.alpha, dtype=dtype)
        alpha = alpha_param.astype(dtype) / (p_1 - p_0)
        s_x = 0.5 * (1.0 + jnp.tanh(alpha * (x - m)))
        return p_0 + (x - p_0) * s_x

    def __repr__(self):
        return f'{self.__class__.__name__}(alpha={self.alpha:.4f}, trainable={self.trainable})'


@nxx_auto_import_from_torch(allow_missing_mapper=True)
class _RadialSequential(nnx.Module):
    channels: Sequence[int]

    def __init__(self, channels: Sequence[int], *, rngs: nnx.Rngs) -> None:
        self.channels = tuple(channels)
        if len(self.channels) < 2:
            raise ValueError('channels must have length >= 2 for RadialMLP')

        self.layers = nnx.Dict()
        self._layer_order: list[tuple[str, str | None]] = []
        last_idx = len(self.channels) - 1
        layer_idx = 0
        for idx, out_channels in enumerate(self.channels[1:], start=1):
            key = str(layer_idx)
            self.layers[key] = nnx.Linear(
                self.channels[idx - 1],
                out_channels,
                use_bias=True,
                rngs=rngs,
            )
            self._layer_order.append(('linear', key))
            layer_idx += 1
            if idx != last_idx:
                key = str(layer_idx)
                self.layers[key] = nnx.LayerNorm(
                    num_features=out_channels,
                    use_bias=True,
                    use_scale=True,
                    reduction_axes=-1,
                    feature_axes=-1,
                    epsilon=1e-5,
                    rngs=rngs,
                )
                self._layer_order.append(('norm', key))
                layer_idx += 1
                self._layer_order.append(('act', None))
                layer_idx += 1

    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        x = inputs
        for kind, key in self._layer_order:
            if kind == 'act':
                x = jax.nn.silu(x)
                continue
            if key is None:
                raise ValueError('Missing layer key for radial sequential block')
            x = self.layers[key](x)
        return x


@nxx_auto_import_from_torch(allow_missing_mapper=True)
class RadialMLP(nnx.Module):
    """Wrapper that aligns with the Torch RadialMLP parameter layout."""

    channels: Sequence[int]

    def __init__(self, channels: Sequence[int], *, rngs: nnx.Rngs) -> None:
        self.channels = tuple(channels)
        self.net = _RadialSequential(self.channels, rngs=rngs)

    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        return self.net(inputs)
