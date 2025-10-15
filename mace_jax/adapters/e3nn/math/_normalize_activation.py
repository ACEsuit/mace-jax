"""Statistical utilities for matching ``e3nn.math`` activation behaviour."""

from collections.abc import Callable
from typing import Optional

import jax
import jax.numpy as jnp


def moment(
    f: Callable, n: int, key: jax.random.PRNGKey, dtype=jnp.float32
) -> jnp.ndarray:
    """Estimate the n-th raw moment ``E[f(z)^n]`` where ``z ~ Normal(0, 1)``.

    Args:
        f: Scalar activation or callable applied element-wise.
        n: Order of the moment to compute.
        key: PRNG key used to draw the Monte-Carlo samples.
        dtype: Target dtype for the returned moment.

    Returns:
        The Monte-Carlo estimate of ``E[f(z)^n]`` as a scalar array.
    """
    z = jax.random.normal(key, shape=(1_000_000,), dtype=dtype)
    return jnp.mean(jnp.power(f(z), n))


def normalize2mom(
    f: Callable,
    key: jax.random.PRNGKey = None,
    dtype=jnp.float32,
) -> Callable:
    """Scale an activation so its output variance under ``N(0, 1)`` equals one.

    The helper mirrors the normalisation used in the original Torch
    implementation so that activations plugged into Flax modules produce
    feature statistics consistent with ``e3nn`` defaults.

    Args:
        f: Activation function to normalise.
        key: Optional PRNG key; when omitted a deterministic default is used.
        dtype: Floating-point dtype for the normalisation constant.

    Returns:
        A callable that rescales ``f`` while preserving its input signature.
    """

    if key is None:
        key = jax.random.PRNGKey(0)

    @jax.jit
    def compute_const(prng_key):
        return moment(f, 2, prng_key, dtype=dtype) ** -0.5

    const = jnp.asarray(compute_const(key), dtype=dtype)

    # When ``normalize2mom`` is invoked while tracing (e.g. inside a ``jax.jit``),
    # ``const`` will be a tracer and therefore cannot be converted to a Python
    # ``float``. We retain the scalar as a JAX array in that situation so the
    # caller can safely stage it out as part of the larger computation.
    def _normalized(x: jnp.ndarray) -> jnp.ndarray:
        scale = const.astype(x.dtype)
        return f(x) * scale

    _normalized._normalize2mom_const = const  # type: ignore[attr-defined]
    _normalized._normalize2mom_original = f  # type: ignore[attr-defined]
    return _normalized
