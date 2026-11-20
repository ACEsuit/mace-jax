"""Gaussian orbital electrostatics for PolarMACE.

Conventions, finite-difference offsets, and boundary corrections match
WillBaldwin0/graph_electrostatics (graph-longrange 0.4.4). All runtime
calculations use JAX; reciprocal sums have a checked static size.
"""

import functools
import math

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import erf
from scipy.integrate import trapezoid
from scipy.special import gamma, gammainc

from mace_jax.modules.electrostatics import check_reciprocal_grid, reciprocal_indices
from mace_jax.tools.scatter import scatter_sum

FIELD_CONSTANT = 1 / (5.526349406 * 1e-3)


def normalization(degree, sigma, receiver=False):
    denominator = (
        2 ** ((degree + 1) / 2) * gamma((degree + 3) / 2) * sigma ** (degree + 3)
        if receiver
        else np.sqrt(4 * np.pi / (2 * degree + 1))
        * 2 ** ((2 * degree + 1) / 2)
        * gamma((2 * degree + 3) / 2)
        * sigma ** (2 * degree + 3)
    )
    return 1 / denominator


@functools.lru_cache(maxsize=32)
def self_matrix(source_l, source_sigma, target_l, widths, receiver=True):
    # Match the reference's tabulated quadrature, including its finite grid.
    matrix = np.zeros((len(widths) * (target_l + 1) ** 2, (source_l + 1) ** 2))
    for degree in range(min(source_l, target_l) + 1):
        for i, sigma in enumerate(widths):
            grid = np.linspace(0.0001, 10 * max(source_sigma, sigma), 10000)
            f1 = (
                2 ** (degree + 0.5)
                * sigma ** (2 * degree + 3)
                * gamma(degree + 1.5)
                * gammainc(degree + 1.5, 0.5 * grid**2 / sigma**2)
                / grid ** (degree + 1)
            )
            f2 = sigma**2 * grid**degree * np.exp(-0.5 * grid**2 / sigma**2)
            value = trapezoid(
                grid ** (degree + 2)
                * np.exp(-0.5 * grid**2 / source_sigma**2)
                * (f1 + f2),
                grid,
            )
            value *= (
                FIELD_CONSTANT
                / (2 * degree + 1)
                * normalization(degree, source_sigma)
                * normalization(degree, sigma, receiver)
            )
            for m in range(2 * degree + 1):
                matrix[
                    len(widths) * degree * degree + i * (2 * degree + 1) + m,
                    degree * degree + m,
                ] = value
    return matrix


def field_features(potential, field, widths, max_l):
    scalars = jnp.stack(
        [potential * normalization(0, s, True) / normalization(0, s) for s in widths],
        axis=-1,
    )
    if max_l == 0:
        return scalars
    vectors = jnp.stack(
        [
            field[..., [1, 2, 0]] * (normalization(1, s, True) / normalization(1, s))
            for s in widths
        ],
        axis=-2,
    )
    return jnp.concatenate((scalars, vectors.reshape(*potential.shape, -1)), axis=-1)


def _expanded_sources(positions, coefficients, offset, expand):
    if not expand:
        return positions, coefficients[:, :1]
    offsets = jnp.concatenate((jnp.zeros((1, 3)), jnp.eye(3) * offset), axis=0)
    extended = (positions[:, None, :] + offsets).reshape(-1, 3)
    dipoles = (
        coefficients[:, [3, 1, 2]] / offset
        if coefficients.shape[-1] > 1
        else jnp.zeros_like(positions)
    )
    charges = jnp.concatenate(
        (coefficients[:, :1] - dipoles.sum(-1, keepdims=True), dipoles), axis=-1
    )
    return extended, charges


def _real_potential(data, coefficients, sigma, widths, offset, expand):
    positions, charges = _expanded_sources(
        data['positions'], coefficients, offset, expand
    )
    repeats = charges.shape[-1]
    batch = jnp.repeat(data['batch'], repeats)
    original = jnp.repeat(jnp.arange(len(coefficients)), repeats)
    mask = jnp.repeat(data.get('node_mask', jnp.ones(len(coefficients))), repeats)
    difference = positions[:, None] - positions[None, :]
    r2 = jnp.sum(difference**2, axis=-1)
    distance = jnp.where(r2 > 0, jnp.sqrt(jnp.where(r2 > 0, r2, 1)), 0)
    keep = (batch[:, None] == batch[None, :]) & (original[:, None] != original[None, :])
    keep = keep * mask[:, None] * mask[None, :]
    total_widths = jnp.sqrt((sigma**2 + jnp.asarray(widths) ** 2) / 2)
    kernel = erf(0.5 * distance[:, :, None] / total_widths) / (
        distance[:, :, None] + 1e-6
    )
    potential = (
        jnp.einsum('i,ij,ijw->jw', charges.reshape(-1), keep, kernel)
        * FIELD_CONSTANT
        / (4 * math.pi)
    )
    return potential.reshape(len(coefficients), repeats, len(widths)), charges


def real_features(data, coefficients, sigma, widths, max_l, include_self=False):
    expand = max_l > 0 or coefficients.shape[-1] > 1
    values, _ = _real_potential(data, coefficients, sigma, widths, 0.1, expand)
    factors0 = jnp.asarray(
        [normalization(0, s, True) / normalization(0, s) for s in widths]
    )
    result = values[:, 0] * factors0
    if max_l == 1:
        factors1 = jnp.asarray(
            [
                math.sqrt(3)
                * s**2
                * normalization(1, s, True)
                / normalization(0, s)
                / 0.1
                for s in widths
            ]
        )
        vectors = (values[:, [2, 3, 1]] - values[:, :1]) * factors1
        result = jnp.concatenate(
            (result, vectors.swapaxes(1, 2).reshape(len(coefficients), -1)), axis=-1
        )
    if include_self:
        result += (
            coefficients
            @ jnp.asarray(
                self_matrix(
                    int(coefficients.shape[-1] > 1), sigma, max_l, tuple(widths)
                )
            ).T
        )
    return result


def real_energy(data, coefficients, sigma, include_self=False):
    potential, charges = _real_potential(
        data, coefficients, sigma, (sigma,), 0.02, coefficients.shape[-1] > 1
    )
    energy = 0.5 * jnp.sum(charges * potential[..., 0], axis=-1)
    if include_self:
        matrix = jnp.asarray(
            self_matrix(
                int(coefficients.shape[-1] > 1),
                sigma,
                int(coefficients.shape[-1] > 1),
                (sigma,),
                False,
            )
        )
        energy += 0.5 * jnp.einsum('ni,ij,nj->n', coefficients, matrix, coefficients)
    return scatter_sum(energy, data['batch'], dim=0, dim_size=len(data['ptr']) - 1)


def _raise_cell_error():
    raise ValueError('Periodic electrostatics requires a finite, nonsingular cell.')


def reciprocal_geometry(data, cutoff, max_index, active=True):
    n_graphs = len(data['ptr']) - 1
    cell = data['cell'].reshape(n_graphs, 3, 3)
    mask = jnp.asarray(
        data.get('node_mask', jnp.ones(len(data['batch']))), dtype=cell.dtype
    )
    counts = scatter_sum(mask, data['batch'], dim=0, dim_size=n_graphs)
    valid_cell = jnp.all(jnp.isfinite(cell), axis=(-2, -1)) & (
        jnp.abs(jnp.linalg.det(cell)) > 1e-12
    )
    jax.lax.cond(
        jnp.any(active & (counts > 0) & ~valid_cell),
        lambda: jax.debug.callback(_raise_cell_error),
        lambda: None,
    )
    valid = valid_cell & (counts > 0) & active
    cell = jnp.where(valid[:, None, None], cell, jnp.eye(3))
    volume = jnp.abs(jnp.linalg.det(cell))
    reciprocal = 2 * math.pi * jnp.linalg.inv(cell).swapaxes(-1, -2)
    required = jnp.ceil(cutoff * jnp.linalg.norm(cell, axis=-1) / (2 * math.pi))
    check_reciprocal_grid(jnp.where(valid[:, None], required, 0), max_index)
    bounds = jnp.max(jnp.where(valid[:, None], required, 1), axis=0)
    indices = jnp.asarray(reciprocal_indices(max_index), dtype=cell.dtype)
    a, b, c = indices.T
    half = (
        (a > 0)
        & (a < bounds[0])
        & (b >= -bounds[1])
        & (b < bounds[1])
        & (c >= -bounds[2])
        & (c < bounds[2])
    )
    half |= (a == 0) & (b > 0) & (b < bounds[1]) & (c >= -bounds[2]) & (c < bounds[2])
    half |= (a == 0) & (b == 0) & (c > 0) & (c < bounds[2])
    k = jnp.einsum('ki,gij->gkj', indices, reciprocal)
    k2 = jnp.sum(k * k, axis=-1)
    keep = half[None] & (k2 <= cutoff**2) & (k2 > 0) & valid[:, None]
    factor = jnp.where(
        keep, FIELD_CONSTANT / jnp.maximum(k2, 1e-20) / volume[:, None], 0
    )
    phase = jnp.einsum('ni,nki->nk', data['positions'], k[data['batch']])
    return (
        k,
        k2,
        factor,
        jnp.cos(phase) * mask[:, None],
        jnp.sin(phase) * mask[:, None],
        data['batch'],
        volume,
    )


def _structure(coefficients, sigma, geometry):
    k, k2, _, cosine, sine, batch, volume = geometry
    pool = lambda x: scatter_sum(x, batch, dim=0, dim_size=len(volume))
    q = coefficients[:, 0]
    real = pool(q[:, None] * cosine)
    imag = -pool(q[:, None] * sine)
    if coefficients.shape[-1] > 1:
        uk = jnp.einsum('ni,nki->nk', coefficients[:, [3, 1, 2]], k[batch])
        real -= pool(uk * sine)
        imag -= pool(uk * cosine)
    gaussian = jnp.exp(-0.5 * sigma**2 * k2)
    return real * gaussian, imag * gaussian


def periodic_features(coefficients, sigma, widths, max_l, geometry, include_self=False):
    k, k2, factor, cosine, sine, batch, _ = geometry
    real, imag = _structure(coefficients, sigma, geometry)
    scalars, vectors = [], []
    for width in widths:
        gaussian = jnp.exp(-0.5 * width**2 * k2)
        scalar = 2 * jnp.sum(
            (factor * gaussian)[batch] * (real[batch] * cosine - imag[batch] * sine),
            axis=-1,
        )
        scalars.append(scalar * normalization(0, width, True) / normalization(0, width))
        if max_l:
            vector = -2 * jnp.einsum(
                'nk,nk,nkc->nc',
                (factor * gaussian)[batch],
                imag[batch] * cosine + real[batch] * sine,
                k[batch],
            )
            vectors.append(
                vector[:, [1, 2, 0]]
                * normalization(1, width, True)
                / normalization(1, width)
            )
    result = jnp.stack(scalars, axis=-1)
    if max_l:
        result = jnp.concatenate((result, jnp.concatenate(vectors, axis=-1)), axis=-1)
    if not include_self:
        result -= (
            coefficients
            @ jnp.asarray(
                self_matrix(
                    int(coefficients.shape[-1] > 1), sigma, max_l, tuple(widths)
                )
            ).T
        )
    return result


def periodic_energy(data, coefficients, sigma, geometry, include_self=False):
    real, imag = _structure(coefficients, sigma, geometry)
    energy = jnp.sum(geometry[2] * (real**2 + imag**2), axis=-1)
    if not include_self:
        matrix = jnp.asarray(
            self_matrix(
                int(coefficients.shape[-1] > 1),
                sigma,
                int(coefficients.shape[-1] > 1),
                (sigma,),
                False,
            )
        )
        self_energy = 0.5 * jnp.einsum(
            'ni,ij,nj->n', coefficients, matrix, coefficients
        )
        energy -= scatter_sum(
            self_energy, data['batch'], dim=0, dim_size=len(data['ptr']) - 1
        )
    return energy


def boundary_corrections(data, coefficients, volume, mode):
    batch, pos = data['batch'], data['positions']
    ng = len(data['ptr']) - 1
    pool = lambda x: scatter_sum(x, batch, dim=0, dim_size=ng)
    q = coefficients[:, 0]
    local = (
        coefficients[:, [3, 1, 2]]
        if coefficients.shape[-1] > 1
        else jnp.zeros_like(pos)
    )
    charge, dipole = pool(q), pool(q[:, None] * pos + local)
    quadrupole = pool(q * jnp.sum(pos**2, axis=-1) + 2 * jnp.sum(pos * local, axis=-1))
    const = FIELD_CONSTANT / (4 * math.pi)
    pbc = data['pbc'].reshape(ng, 3)
    molecule = jnp.full(ng, mode == 'molecule_in_box')
    slab = jnp.full(ng, mode == 'slab')
    if mode in ('mixed_periodic', 'auto'):
        molecule = ~jnp.any(pbc, axis=-1)
        slab = pbc[:, 0] & pbc[:, 1] & ~pbc[:, 2]
    energy = (
        0.5 * 2.837297 * const * charge**2 / volume**0.3333
        + FIELD_CONSTANT
        / (6 * volume)
        * (jnp.sum(dipole**2, axis=-1) - charge * quadrupole)
    ) * molecule
    energy += FIELD_CONSTANT * dipole[:, 2] ** 2 / (2 * volume) * slab
    spreadq, spreadp, spreadv = charge[batch], dipole[batch], volume[batch]
    potential = (
        2.837297 * const * spreadq / spreadv**0.333333
        - FIELD_CONSTANT
        / (6 * spreadv)
        * (spreadq * jnp.sum(pos**2, axis=-1) + quadrupole[batch])
        + FIELD_CONSTANT / (3 * spreadv) * jnp.sum(spreadp * pos, axis=-1)
    ) * molecule[batch]
    field = (
        FIELD_CONSTANT
        / (3 * spreadv[:, None])
        * (spreadp - spreadq[:, None] * pos)
        * molecule[batch, None]
    )
    zfield = FIELD_CONSTANT * spreadp[:, 2] / spreadv * slab[batch]
    field = field.at[:, 2].add(zfield)
    potential += zfield * pos[:, 2]
    return energy, potential, field
