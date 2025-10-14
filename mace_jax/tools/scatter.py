from typing import Optional

import jax.numpy as jnp


def _broadcast(src: jnp.ndarray, other: jnp.ndarray, dim: int) -> jnp.ndarray:
    """Mimic PyTorch _broadcast more permissively in JAX."""
    if dim < 0:
        dim = other.ndim + dim

    # If src is 1D, unsqueeze until dim
    if src.ndim == 1:
        for _ in range(dim):
            src = jnp.expand_dims(src, 0)

    # Add trailing singleton dims until ranks match
    while src.ndim < other.ndim:
        src = jnp.expand_dims(src, -1)

    # Now expand each dimension individually
    shape = []
    for s_dim, o_dim in zip(src.shape, other.shape):
        if s_dim == o_dim:
            shape.append(s_dim)
        elif s_dim == 1:
            shape.append(o_dim)
        else:
            # PyTorch would raise here if dimension mismatch
            raise ValueError(
                f'Incompatible shapes for broadcasting: {src.shape} vs {other.shape}'
            )

    # Use jnp.broadcast_to with the adjusted shape
    return jnp.broadcast_to(src, shape)


def scatter_sum(
    src: jnp.ndarray,
    index: jnp.ndarray,
    dim: int = -1,
    out: Optional[jnp.ndarray] = None,
    dim_size: Optional[int] = None,
    reduce: str = 'sum',
) -> jnp.ndarray:
    assert reduce == 'sum'
    index = _broadcast(index, src, dim)

    if dim < 0:
        dim = src.ndim + dim

    if out is None:
        size = list(src.shape)
        if dim_size is not None:
            size[dim] = dim_size
        elif index.size == 0:
            size[dim] = 0
        else:
            size[dim] = int(index.max()) + 1
        out = jnp.zeros(size, dtype=src.dtype)

    # Build indices for scatter
    idx_grids = jnp.meshgrid(*[jnp.arange(s) for s in src.shape], indexing='ij')
    idx_grids[dim] = index  # replace the dimension with scatter indices
    scatter_indices = tuple(idx_grids)

    return out.at[scatter_indices].add(src)


def scatter_std(
    src: jnp.ndarray,
    index: jnp.ndarray,
    dim: int = -1,
    out: Optional[jnp.ndarray] = None,
    dim_size: Optional[int] = None,
    unbiased: bool = True,
) -> jnp.ndarray:
    """
    JAX version of PyTorch scatter_std that supports arbitrary dim and higher-rank tensors.
    """
    if dim < 0:
        dim = src.ndim + dim

    # Ensure index is broadcastable to src
    index = _broadcast(index, src, dim)

    # Compute count per index
    ones = jnp.ones_like(src)
    count = scatter_sum(ones, index, dim=dim, dim_size=dim_size)

    # Compute sum per index → mean
    sum_per_index = scatter_sum(src, index, dim=dim, dim_size=dim_size)
    count_safe = jnp.maximum(_broadcast(count, sum_per_index, dim), 1)
    mean = sum_per_index / count_safe

    # Compute squared deviations
    # src - mean[index] along dim
    # Use gather-like indexing via take_along_axis
    gather_idx = index
    mean_gathered = jnp.take_along_axis(mean, gather_idx, axis=dim)
    sq_diff = (src - mean_gathered) ** 2

    # Sum squared deviations per index
    var_sum = scatter_sum(sq_diff, index, dim=dim, dim_size=dim_size)

    # Adjust for unbiased if needed
    if unbiased:
        count_safe = jnp.maximum(count_safe - 1, 1)

    out = jnp.sqrt(var_sum / count_safe)

    return out


def scatter_mean(
    src: jnp.ndarray,
    index: jnp.ndarray,
    dim: int = -1,
    out: jnp.ndarray = None,
    dim_size: int = None,
) -> jnp.ndarray:
    """
    JAX version of PyTorch scatter_mean along arbitrary dimension.
    """
    # Step 1: sum per index
    out = scatter_sum(src, index, dim=dim, out=out, dim_size=dim_size)

    # Determine dim_size
    if dim_size is None:
        dim_size = out.shape[dim]

    # Step 2: compute count per index
    index_dim = dim
    if index_dim < 0:
        index_dim += src.ndim
    if index.ndim <= index_dim:
        index_dim = index.ndim - 1

    ones = jnp.ones_like(index, dtype=src.dtype)
    count = scatter_sum(ones, index, dim=index_dim, dim_size=dim_size)
    count = jnp.maximum(count, 1)

    # Step 3: broadcast count to out shape
    count = _broadcast(count, out, dim)

    # Step 4: divide sum by count
    mean = out / count
    return mean
