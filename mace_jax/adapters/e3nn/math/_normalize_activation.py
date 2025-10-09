"""Statistical utilities for matching ``e3nn.math`` activation behaviour."""

from typing import Callable, Optional

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
    key: Optional[jax.random.PRNGKey] = None,
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

    cst = float(moment(f, 2, key, dtype=dtype) ** -0.5)
    if abs(cst - 1.0) < 1e-4:
        return f

    def _normalized(x: jnp.ndarray) -> jnp.ndarray:
        return f(x) * jnp.asarray(cst, dtype=x.dtype)

    return _normalized
