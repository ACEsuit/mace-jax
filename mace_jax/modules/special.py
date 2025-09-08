import jax
import jax.numpy as jnp

def chebyshev_polynomial_t(x: jnp.ndarray, n: int) -> jnp.ndarray:
    if n == 0:
        return jnp.ones_like(x)
    elif n == 1:
        return x
    else:
        T0, T1 = jnp.ones_like(x), x
        def body(i, val):
            Tnm1, Tn = val
            Tnp1 = 2 * x * Tn - Tnm1
            return (Tn, Tnp1)
        _, Tn = jax.lax.fori_loop(1, n, body, (T0, T1))
        return Tn
