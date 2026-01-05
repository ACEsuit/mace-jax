# ###Functions taken from the PR of @merajhashemi d5d72605a5 ####

from typing import Any, NamedTuple

import chex
import jax
import jax.numpy as jnp
import optax
from optax._src import base, numerics, utils

_abs_sq = numerics.abs_sq


def update_moment(updates, moments, decay, order):
    """Compute the exponential moving average of the `order`-th moment."""
    return jax.tree_util.tree_map(
        lambda g, t: (1 - decay) * (g**order) + decay * t, updates, moments
    )


def update_moment_per_elem_norm(updates, moments, decay, order):
    """Compute the EMA of the `order`-th moment of the element-wise norm."""

    def orderth_norm(g):
        if jnp.isrealobj(g):
            return g**order
        else:
            half_order = order / 2
            # JAX generates different HLO for int and float `order`
            if half_order.is_integer():
                half_order = int(half_order)
            return _abs_sq(g) ** half_order

    return jax.tree_util.tree_map(
        lambda g, t: (1 - decay) * orderth_norm(g) + decay * t, updates, moments
    )


def bias_correction(moment, decay, count):
    """Perform bias correction. This becomes a no-op as count goes to infinity."""
    bias_correction_ = 1 - decay**count
    return jax.tree_util.tree_map(
        lambda t: t / bias_correction_.astype(t.dtype), moment
    )


class ScaleByAmsgradState(NamedTuple):
    """State for the AMSGrad algorithm."""

    count: chex.Array  # shape=(), dtype=jnp.int32.
    mu: base.Updates
    nu: base.Updates
    nu_max: base.Updates


def scale_by_amsgrad(
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    mu_dtype: Any | None = None,
) -> optax.GradientTransformation:
    """Rescale updates according to the AMSGrad algorithm.
    References:
        [Reddi et al, 2018](https://openreview.net/forum?id=ryQu7f-RZ)
    Args:
        b1: decay rate for the exponentially weighted average of grads.
        b2: decay rate for the exponentially weighted average of squared grads.
        eps: term added to the denominator to improve numerical stability.
        eps_root: term added to the denominator inside the square-root to improve
          numerical stability when backpropagating gradients through the rescaling.
        mu_dtype: optional `dtype` to be used for the first order accumulator; if
          `None` then the `dtype is inferred from `params` and `updates`.
    Returns:
        An (init_fn, update_fn) tuple.
    """

    mu_dtype = utils.canonicalize_dtype(mu_dtype)

    def init_fn(params):
        mu = jax.tree_util.tree_map(  # First moment
            lambda t: jnp.zeros_like(t, dtype=mu_dtype), params
        )
        nu = jax.tree_util.tree_map(jnp.zeros_like, params)  # Second moment
        nu_max = jax.tree_util.tree_map(jnp.zeros_like, params)
        return ScaleByAmsgradState(
            count=jnp.zeros([], jnp.int32), mu=mu, nu=nu, nu_max=nu_max
        )

    def update_fn(updates, state, params=None):
        del params
        mu = update_moment(updates, state.mu, b1, 1)
        nu = update_moment_per_elem_norm(updates, state.nu, b2, 2)
        count_inc = numerics.safe_int32_increment(state.count)
        mu_hat = bias_correction(mu, b1, count_inc)
        nu_hat = bias_correction(nu, b2, count_inc)
        nu_max = jax.tree_util.tree_map(jnp.maximum, state.nu_max, nu_hat)
        updates = jax.tree_util.tree_map(
            lambda m, v: m / (jnp.sqrt(v + eps_root) + eps), mu_hat, nu_max
        )
        mu = utils.cast_tree(mu, mu_dtype)
        return updates, ScaleByAmsgradState(
            count=count_inc, mu=mu, nu=nu, nu_max=nu_max
        )

    return base.GradientTransformation(init_fn, update_fn)
