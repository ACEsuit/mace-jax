"""Differentiable Gaussian electrostatics with static shapes for JAX.

The LES convention follows ChengUCB/les at fadfb0ca3b89ed371e5de6b383f4425defabd4a8.
All sums exclude padding and interactions between different configurations.
"""

import functools
import itertools
import math

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import erf

from mace_jax.tools.scatter import scatter_sum


@functools.lru_cache(maxsize=16)
def reciprocal_indices(max_index):
    if not isinstance(max_index, int) or max_index < 1:
        raise ValueError('reciprocal_max_index must be a positive integer.')
    return np.asarray(
        list(itertools.product(range(-max_index, max_index + 1), repeat=3))
    )


def _raise_grid_error():
    raise ValueError(
        'Reciprocal grid is too small for this cell; increase reciprocal_max_index.'
    )


def check_reciprocal_grid(required, max_index):
    # Fail also inside jit; silently truncating reciprocal sums changes the model.
    jax.lax.cond(
        jnp.any(required > max_index),
        lambda: jax.debug.callback(_raise_grid_error),
        lambda: None,
    )


def gaussian_coulomb(vector, sigma):
    """erf(r/(sqrt(2)*sigma))/r, including its smooth value at zero."""
    r2 = jnp.sum(vector * vector, axis=-1)
    x = r2 / (2 * sigma**2)
    # The series avoids cancellation in multipole derivatives, especially in
    # float32. Clamp its unused branch to prevent overflow at large separations.
    series_x = jnp.minimum(x, 1.0)
    coefficients = [(-1.0) ** k / (math.factorial(k) * (2 * k + 1)) for k in range(18)]
    series = coefficients[-1]
    for coefficient in coefficients[-2::-1]:
        series = series * series_x + coefficient
    safe_r = jnp.sqrt(jnp.maximum(r2, 1e-12 * sigma**2))
    direct = erf(safe_r / (math.sqrt(2) * sigma)) / safe_r
    return jnp.where(x < 1.0, math.sqrt(2 / math.pi) / sigma * series, direct)


def multipole_kernels(positions, sigma, order=2):
    vector = positions[None, :, :] - positions[:, None, :]
    function = lambda r: gaussian_coulomb(r, sigma)
    kernels = []
    for degree in range(order + 1):
        kernels.append(jax.vmap(jax.vmap(function))(vector))
        if degree < order:
            function = jax.jacfwd(function)
    return kernels


def les_electrostatics(
    data,
    q,
    u=None,
    quad=None,
    kappa=None,
    alpha=None,
    *,
    sigma=1.0,
    dl=2.0,
    remove_self_interaction=True,
    reciprocal_max_index=8,
):
    """LES energy and induced moments for open molecules and periodic cells.

    The reciprocal integer grid is static; a runtime check prevents truncation.
    The finite grid and spherical cutoff match the pinned Torch LES implementation.
    """
    positions, batch = data['positions'], data['batch']
    n_graphs = data['ptr'].shape[0] - 1
    mask = jnp.asarray(data.get('node_mask', jnp.ones(q.shape)), dtype=q.dtype)
    cell = data['cell'].reshape(n_graphs, 3, 3)
    pbc = data.get('pbc', jnp.ones((n_graphs, 3), dtype=bool)).reshape(n_graphs, 3)
    populated = scatter_sum(mask, batch, dim=0, dim_size=n_graphs) > 0
    periodic = jnp.any(pbc, axis=-1) & (jnp.linalg.det(cell) >= 1e-6) & populated
    q = q * mask
    if u is not None:
        u = u * mask[:, None]
    if quad is not None:
        quad = quad * mask[:, None, None]
    norm = 90.4756
    norm_real = norm / (2 * math.pi)
    self_q = norm / (sigma * (2 * math.pi) ** 1.5)
    self_u = norm / (3 * sigma**3 * (2 * math.pi) ** 1.5)
    self_quad = norm / (10 * sigma**5 * (2 * math.pi) ** 1.5)
    self_energy = self_q * q**2
    if u is not None:
        self_energy += self_u * jnp.sum(u**2, axis=-1)
    if quad is not None:
        self_energy += self_quad * jnp.sum(quad**2, axis=(-2, -1))

    def open_quantities():
        degree = (
            4 if quad is not None else (2 if u is not None or alpha is not None else 0)
        )
        kernels = multipole_kernels(positions, sigma, degree)
        pair_mask = (batch[:, None] == batch[None, :]) * mask[:, None] * mask[None, :]
        pair_mask *= 1 - jnp.eye(len(q), dtype=q.dtype)
        kernels = [
            value * (norm_real * pair_mask).reshape(pair_mask.shape + (1,) * i)
            for i, value in enumerate(kernels)
        ]
        phi = jnp.einsum('i,ij->j', q, kernels[0])
        field = jnp.zeros_like(positions)
        node_energy = 0.5 * q * phi
        if degree >= 1:
            field = -jnp.einsum('i,ijc->jc', q, kernels[1])
        if u is not None:
            phi_u = -jnp.einsum('ic,ijc->j', u, kernels[1])
            field_u = jnp.einsum('ic,ijcd->jd', u, kernels[2])
            phi += phi_u
            field += field_u
            node_energy += q * phi_u - 0.5 * jnp.sum(u * field_u, axis=-1)
        if quad is not None:
            phi_q = 0.5 * jnp.einsum('iab,ijab->j', quad, kernels[2])
            field_q = -0.5 * jnp.einsum('iab,ijabc->jc', quad, kernels[3])
            phi += phi_q
            field += field_q
            node_energy += q * phi_q + 0.125 * jnp.einsum(
                'iab,ijabcd,jcd->j', quad, kernels[4], quad
            )
            if u is not None:
                node_energy -= jnp.sum(u * field_q, axis=-1)
        if not remove_self_interaction:
            node_energy += self_energy
            phi += 2 * self_q * q
            if u is not None:
                field -= 2 * self_u * u
        open_energy = scatter_sum(node_energy, batch, dim=0, dim_size=n_graphs)

        return open_energy, phi, field

    def periodic_quantities():
        safe_cell = jnp.where(
            periodic[:, None, None], cell, jnp.eye(3, dtype=cell.dtype)
        )
        volume = jnp.linalg.det(safe_cell)
        indices = jnp.asarray(
            reciprocal_indices(reciprocal_max_index), dtype=cell.dtype
        )
        required = jnp.maximum(1, jnp.floor(jnp.linalg.norm(safe_cell, axis=-1) / dl))
        check_reciprocal_grid(
            jnp.where(periodic[:, None], required, 0), reciprocal_max_index
        )
        k = jnp.einsum(
            'ki,gij->gkj',
            indices,
            2 * math.pi * jnp.swapaxes(jnp.linalg.inv(safe_cell), -1, -2),
        )
        k2 = jnp.sum(k * k, axis=-1)
        keep = (k2 > 0) & (k2 <= (2 * math.pi / dl) ** 2)
        keep &= jnp.all(jnp.abs(indices[None]) <= required[:, None], axis=-1)
        kfac = jnp.where(
            keep, jnp.exp(-(sigma**2) * k2 / 2) / jnp.maximum(k2, 1e-20), 0
        )
        node_k = k[batch]
        phase = jnp.einsum('ni,nki->nk', positions, node_k)
        cosine, sine = (
            jnp.cos(phase) * mask[:, None],
            jnp.sin(phase) * mask[:, None],
        )
        amp = jnp.broadcast_to(q[:, None], phase.shape)
        if quad is not None:
            amp = amp - 0.5 * jnp.einsum('nki,nij,nkj->nk', node_k, quad, node_k)
        pool = lambda x: scatter_sum(x, batch, dim=0, dim_size=n_graphs)
        real = pool(amp * cosine)
        imag = pool(amp * sine)
        if u is not None:
            uk = jnp.einsum('ni,nki->nk', u, node_k)
            real -= pool(uk * sine)
            imag += pool(uk * cosine)
        periodic_energy = norm * jnp.sum(kfac * (real**2 + imag**2), axis=-1) / volume
        prefactor = 2 * norm * kfac / volume[:, None]
        phi_pbc = jnp.sum(
            prefactor[batch] * (real[batch] * cosine + imag[batch] * sine),
            axis=-1,
        )
        field_pbc = jnp.einsum(
            'nk,nk,nkc->nc',
            prefactor[batch],
            real[batch] * sine - imag[batch] * cosine,
            node_k,
        )
        if remove_self_interaction:
            periodic_energy -= scatter_sum(self_energy, batch, dim=0, dim_size=n_graphs)
            phi_pbc -= 2 * self_q * q
            if u is not None:
                field_pbc += 2 * self_u * u
        return periodic_energy, phi_pbc, field_pbc

    empty = lambda: (
        jnp.zeros(n_graphs, dtype=q.dtype),
        jnp.zeros_like(q),
        jnp.zeros_like(positions),
    )
    open_energy, phi, field = jax.lax.cond(
        jnp.any(~periodic & populated), open_quantities, empty
    )
    periodic_energy, phi_pbc, field_pbc = jax.lax.cond(
        jnp.any(periodic & populated), periodic_quantities, empty
    )
    energy = jnp.where(periodic, periodic_energy, open_energy)
    phi = jnp.where(periodic[batch], phi_pbc, phi)
    field = jnp.where(periodic[batch, None], field_pbc, field)
    induced_q, induced_u = jnp.zeros_like(q), jnp.zeros_like(positions)
    correction = jnp.zeros_like(q)
    if kappa is not None:
        induced_q = -kappa * phi * mask
        correction += 0.5 * phi * induced_q
    if alpha is not None:
        external = jnp.asarray(data.get('external_field', jnp.zeros((n_graphs, 3))))
        external = jnp.broadcast_to(external, (n_graphs, 3))[batch]
        field_total = field + external
        induced_u = (
            alpha[:, None] * field_total
            if alpha.ndim == 1
            else jnp.einsum('ni,nij->nj', field_total, alpha)
        ) * mask[:, None]
        correction -= 0.5 * jnp.sum(field * induced_u, axis=-1)
    energy += scatter_sum(correction, batch, dim=0, dim_size=n_graphs)
    return energy, q + induced_q, (u if u is not None else 0) + induced_u
