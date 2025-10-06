from typing import Callable, Optional

import haiku as hk
import jax
import jax.numpy as jnp


def moment(
    f: Callable, n: int, key: jax.random.PRNGKey, dtype=jnp.float32
) -> jnp.ndarray:
    """Compute n-th moment <f(z)^n> for z ~ Normal(0,1)."""
    z = jax.random.normal(key, shape=(1_000_000,), dtype=jnp.float64).astype(dtype)
    return jnp.mean(jnp.power(f(z), n))


class normalize2mom(hk.Module):
    """Normalize activation so that its 2nd moment under N(0,1) is 1."""

    def __init__(
        self,
        f: Callable,
        key: Optional[jax.random.PRNGKey] = None,
        dtype=jnp.float32,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self.f = f
        if key is None:
            key = jax.random.PRNGKey(0)

        # Compute normalization constant
        cst = moment(f, 2, key, dtype=jnp.float64) ** -0.5
        cst = float(cst)  # convert DeviceArray → float

        if abs(cst - 1.0) < 1e-4:
            self._is_id = True
        else:
            self._is_id = False
        self.cst = cst

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        if self._is_id:
            return self.f(x)
        else:
            return self.f(x) * self.cst

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(f={self.f}, cst={self.cst:.4f}, is_id={self._is_id})'
