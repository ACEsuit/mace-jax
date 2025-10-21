"""Statistical utilities for matching ``e3nn.math`` activation behaviour."""

from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp

# Mirrors the Torch ``normalize2mom.cst`` cache keyed by activation identity.
_CONST_OVERRIDES: dict[str, float] = {}
# Tracks JAX wrappers created before Torch weights are imported so they can be
# retroactively updated once the moment constant becomes available.
_REGISTERED_FUNCTIONS: dict[str, list[Callable]] = {}


def _activation_key(f: Callable | None) -> str | None:
    """Return a backend-agnostic identifier for an activation callable.

    The Torch implementation stores ``normalize2mom`` instances directly on
    ``torch.nn.Module`` objects.  On the JAX side we instead key the registry by
    human-readable names so plain callables (e.g. ``jax.nn.silu``) can be
    matched against their Torch module equivalents.
    """

    if f is None:
        return None

    key = getattr(f, '_normalize2mom_key', None)
    if isinstance(key, str):
        return key

    if isinstance(f, partial):
        return _activation_key(f.func)

    name = getattr(f, '__name__', None)
    if name is None:
        cls = getattr(f, '__class__', None)
        name = getattr(cls, '__name__', None)

    if not name:
        return None

    identifier = name.replace('<lambda>', 'lambda').lower()
    return identifier


def register_normalize2mom_const(
    identifier: str | Callable,
    value: float,
) -> None:
    """Record a Torch-derived normalisation constant for a given activation.

    Args:
        identifier: Either the activation callable itself or an explicit key
            understood by :func:`_activation_key`.
        value: The constant stored on the Torch ``normalize2mom`` wrapper.

    When a JAX ``normalize2mom`` wrapper already exists, this function updates
    its cached constant in-place so subsequent forward passes use the imported
    statistics rather than a locally re-estimated value.
    """

    key = _activation_key(identifier) if callable(identifier) else identifier
    if key is None:
        return

    float_value = float(value)
    _CONST_OVERRIDES[key] = float_value

    for fn in _REGISTERED_FUNCTIONS.get(key, []):
        holder = getattr(fn, '_normalize2mom_const_holder', None)
        if holder is None:
            continue
        current = holder['value']
        holder['value'] = jnp.asarray(float_value, dtype=current.dtype)
        fn._normalize2mom_const = holder['value']  # type: ignore[attr-defined]


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

    identifier = _activation_key(f)
    override = _CONST_OVERRIDES.get(identifier, None) if identifier else None

    @jax.jit
    def compute_const(prng_key):
        return moment(f, 2, prng_key, dtype=dtype) ** -0.5

    initial_const = (
        jnp.asarray(override, dtype=dtype)
        if override is not None
        else jnp.asarray(compute_const(key), dtype=dtype)
    )

    const_holder: dict[str, jnp.ndarray] = {'value': initial_const}

    # When ``normalize2mom`` is invoked while tracing (e.g. inside a ``jax.jit``),
    # ``const`` will be a tracer and therefore cannot be converted to a Python
    # ``float``. We retain the scalar as a JAX array in that situation so the
    # caller can safely stage it out as part of the larger computation.
    def _normalized(x: jnp.ndarray) -> jnp.ndarray:
        scale = const_holder['value'].astype(x.dtype)
        return f(x) * scale

    _normalized._normalize2mom_const_holder = const_holder  # type: ignore[attr-defined]
    _normalized._normalize2mom_const = const_holder['value']  # type: ignore[attr-defined]
    _normalized._normalize2mom_original = f  # type: ignore[attr-defined]

    if identifier is not None:
        _normalized._normalize2mom_key = identifier  # type: ignore[attr-defined]
        _REGISTERED_FUNCTIONS.setdefault(identifier, []).append(_normalized)

    return _normalized
