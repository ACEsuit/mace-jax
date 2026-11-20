"""Polar electrostatic kernels against graph-longrange."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from mace_jax.modules import gto

pytestmark = pytest.mark.usefixtures('runtime_devices')


@pytest.mark.parametrize('max_l', [0, 1])
@pytest.mark.parametrize('mode', ['realspace', 'pbc', 'slab', 'molecule_in_box'])
def test_gto_kernels(max_l, mode):
    Energy = pytest.importorskip('graph_longrange.energy').GTOElectrostaticEnergy
    Features = pytest.importorskip('graph_longrange.features').GTOElectrostaticFeatures
    from graph_longrange.kspace import compute_k_vectors_flat  # noqa: PLC0415

    pos = np.array([[0.1, 0.2, 0.3], [1.0, 0.4, 0.2], [0.3, 1.1, 0.8]])
    coeff = np.random.default_rng(8).normal(size=(3, (max_l + 1) ** 2)) * 0.2
    cell = torch.eye(3)[None] * 5.1
    tpos = torch.tensor(pos, requires_grad=True)
    batch = torch.zeros(3, dtype=torch.long)
    pbc = torch.tensor([[mode != 'realspace'] * 3])
    cutoff = 3.6
    kvals = compute_k_vectors_flat(
        cutoff, cell, 2 * np.pi * torch.linalg.inv(cell).transpose(-1, -2)
    )
    kwargs = dict(zip(('k_vectors', 'k_norm2', 'k_vector_batch', 'k0_mask'), kvals))
    kwargs.update(
        source_feats=torch.tensor(coeff),
        node_positions=tpos,
        batch=batch,
        volume=torch.linalg.det(cell),
        pbc=pbc,
    )
    source_energy = Energy(max_l, 0.9, cutoff, pbc_handling=mode)
    source_features = Features(
        density_max_l=max_l,
        density_smearing_width=0.9,
        feature_max_l=max_l,
        feature_smearing_widths=[1.0, 1.3],
        kspace_cutoff=cutoff,
        include_self_interaction=False,
        integral_normalization='receiver',
        pbc_handling=mode,
    )
    expected_e = source_energy(**kwargs)
    expected_f = source_features.forward_dynamic(
        source_features.precompute_geometry(
            **{k: v for k, v in kwargs.items() if k != 'source_feats'}
        ),
        torch.tensor(coeff),
    )
    data = dict(
        positions=jnp.asarray(pos),
        batch=jnp.zeros(3, dtype=int),
        ptr=jnp.array([0, 3]),
        cell=jnp.asarray(cell.numpy()),
        pbc=jnp.asarray(pbc.numpy()),
    )

    def evaluate(positions):
        d = dict(data, positions=positions)
        if mode == 'realspace':
            return gto.real_energy(d, jnp.asarray(coeff), 0.9), gto.real_features(
                d, jnp.asarray(coeff), 0.9, (1.0, 1.3), max_l
            )
        geometry = gto.reciprocal_geometry(d, cutoff, 4)
        e = gto.periodic_energy(d, jnp.asarray(coeff), 0.9, geometry)
        f = gto.periodic_features(jnp.asarray(coeff), 0.9, (1.0, 1.3), max_l, geometry)
        ec, pc, fc = gto.boundary_corrections(d, jnp.asarray(coeff), geometry[-1], mode)
        return e + ec, f + gto.field_features(pc, fc, (1.0, 1.3), max_l)

    actual = jax.jit(evaluate)(data['positions'])
    np.testing.assert_allclose(actual[0], expected_e.detach(), atol=1e-9, rtol=1e-7)
    np.testing.assert_allclose(actual[1], expected_f.detach(), atol=1e-9, rtol=1e-7)
    force = -jax.jit(jax.grad(lambda p: evaluate(p)[0].sum()))(data['positions'])
    expected_force = -torch.autograd.grad(expected_e.sum(), tpos)[0]
    np.testing.assert_allclose(force, expected_force.detach(), atol=1e-9, rtol=1e-7)


def test_reciprocal_grid_validation():
    data = dict(
        positions=jnp.zeros((1, 3)),
        batch=jnp.zeros(1, dtype=int),
        ptr=jnp.array([0, 1]),
        cell=jnp.eye(3)[None] * 40.0,
    )
    geometry = jax.jit(
        lambda d, active: gto.reciprocal_geometry(d, 3.6, 2, active=active)
    )
    # Large cells are irrelevant for real-space calculations and padding graphs.
    assert np.isfinite(geometry(data, False)[2]).all()
    assert np.isfinite(geometry(dict(data, node_mask=jnp.zeros(1)), True)[2]).all()
    with pytest.raises(Exception, match='Reciprocal grid is too small'):
        jax.block_until_ready(geometry(data, True))
    with pytest.raises(Exception, match='nonsingular cell'):
        jax.block_until_ready(geometry(dict(data, cell=jnp.zeros((1, 3, 3))), True))


def test_reciprocal_batch_independence():
    positions = jnp.array([[0.2, 0.1, 0.3], [0.9, 0.7, 0.4]])
    coeff = jnp.array([[0.1, 0.02, 0.03, 0.04], [-0.2, -0.03, 0.02, 0.01]])
    single = dict(
        positions=positions, batch=jnp.zeros(2, dtype=int), ptr=jnp.array([0, 2])
    )

    def evaluate(data, coefficients):
        geometry = gto.reciprocal_geometry(data, 3.6, 4)
        return (
            gto.periodic_energy(data, coefficients, 0.9, geometry),
            gto.periodic_features(coefficients, 0.9, (1.0, 1.3), 1, geometry),
        )

    cells = jnp.stack((jnp.eye(3) * 5.1, jnp.eye(3) * 6.2))
    expected = [
        evaluate(dict(single, cell=cell[None]), coeff * scale)
        for cell, scale in zip(cells, (1.0, 2.0))
    ]
    batched = dict(
        positions=jnp.concatenate((positions, positions + 1.0)),
        batch=jnp.array([0, 0, 1, 1]),
        ptr=jnp.array([0, 2, 4]),
        cell=cells,
    )
    actual = jax.jit(evaluate)(batched, jnp.concatenate((coeff, 2 * coeff)))
    for i in range(2):
        np.testing.assert_allclose(
            actual[i], jnp.concatenate([x[i] for x in expected]), atol=1e-10
        )
