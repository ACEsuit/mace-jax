import jax
import jax.numpy as jnp


def chebyshev_polynomial_t(x: jnp.ndarray, n: jnp.ndarray) -> jnp.ndarray:
    """
    Vectorized Chebyshev polynomial of the first kind.
    x: [batch, num_basis] or [batch, 1]
    n: [num_basis] or [batch, num_basis] (integers)
    Returns: [batch, num_basis]
    """
    if isinstance(n, jnp.ndarray):
        n = n.astype(int)
    else:
        n = jnp.array(n)

    # Ensure n has same batch shape as x
    while n.ndim < x.ndim:
        n = jnp.expand_dims(n, 0)  # from (num_basis,) → (1, num_basis)
    n = jnp.broadcast_to(n, x.shape)  # [batch, num_basis]

    # T0(x) = 1, T1(x) = x
    t0 = jnp.ones_like(x)  # [batch, num_basis]
    t1 = x

    def body(carry, _):
        t_km1, t_k = carry
        t_kp1 = 2 * x * t_k - t_km1
        return (t_k, t_kp1), t_kp1

    max_n = jnp.max(n)
    (_, _), t_seq = jax.lax.scan(body, (t0, t1), jnp.arange(1, max_n))

    # [max_n-1, batch, num_basis] → stack with T0 and T1
    all_terms = jnp.concatenate(
        [t0[None, ...], t1[None, ...], t_seq], axis=0
    )  # [max_n+1, batch, num_basis]

    # Reorder to [batch, num_basis, max_n+1]
    all_terms = jnp.moveaxis(all_terms, 0, -1)

    # Gather along polynomial order
    return jnp.take_along_axis(all_terms, n[..., None], axis=-1).squeeze(-1)
