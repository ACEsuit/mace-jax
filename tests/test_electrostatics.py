"""Native Gaussian electrostatics against MACE's pinned LES backend."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from mace_jax.modules.electrostatics import gaussian_coulomb, les_electrostatics

pytestmark = pytest.mark.usefixtures('runtime_devices')


@pytest.mark.parametrize('periodic', [False, True])
@pytest.mark.parametrize('multipoles', [False, True])
@pytest.mark.parametrize('remove_self', [False, True])
def test_les_energy_moments_and_forces(periodic, multipoles, remove_self):
    Ewald = pytest.importorskip('les.module.ewald').Ewald
    rng = np.random.default_rng(23)
    pos = rng.normal(size=(3, 3))
    cell = np.array([[5.1, 0.2, 0.1], [0.0, 4.9, 0.3], [0.1, 0.0, 5.3]])
    data = dict(
        positions=jnp.asarray(pos),
        cell=jnp.asarray(cell)[None],
        pbc=jnp.full((1, 3), periodic),
        batch=jnp.zeros(3, dtype=int),
        ptr=jnp.array([0, 3]),
    )
    moments = dict(
        q=rng.normal(size=3), kappa=rng.random(3) * 0.01, alpha=rng.random(3) * 0.01
    )
    if multipoles:
        quad = rng.normal(size=(3, 3, 3)) * 0.1
        quad += quad.transpose(0, 2, 1)
        quad -= np.trace(quad, axis1=1, axis2=2)[:, None, None] * np.eye(3) / 3
        moments.update(u=rng.normal(size=(3, 3)) * 0.1, quad=quad)
    tq = {k: torch.tensor(v) for k, v in moments.items()}
    tr = torch.tensor(pos, requires_grad=True)
    expected = Ewald(remove_self_interaction=remove_self)(
        r=tr, cell=torch.tensor(cell if periodic else np.zeros((3, 3)))[None], **tq
    )
    jf = lambda r: les_electrostatics(
        dict(data, positions=r),
        **{k: jnp.asarray(v) for k, v in moments.items()},
        remove_self_interaction=remove_self,
        reciprocal_max_index=3,
    )
    actual = jax.jit(jf)(data['positions'])
    np.testing.assert_allclose(actual[0], expected[0].detach(), atol=1e-10)
    np.testing.assert_allclose(
        actual[1], moments['q'] + expected[1].detach().numpy(), atol=1e-10
    )
    np.testing.assert_allclose(
        actual[2], moments.get('u', 0) + expected[2].detach().numpy(), atol=1e-10
    )
    expected_force = -torch.autograd.grad(expected[0].sum(), tr)[0]
    actual_force = -jax.jit(jax.grad(lambda r: jf(r)[0].sum()))(data['positions'])
    np.testing.assert_allclose(actual_force, expected_force.detach(), atol=2e-9)


def test_les_mixed_boundaries_and_padding():
    # Two separated graphs must equal independent evaluation, even when their
    # periodicity differs. Padding must not alter induced moments or gradients.
    pos = jnp.array([[0.1, 0.2, 0.3], [1.0, 0.4, 0.2]])
    cell = jnp.eye(3)[None] * 5.0
    single = dict(
        positions=pos, batch=jnp.zeros(2, dtype=int), ptr=jnp.array([0, 2]), cell=cell
    )
    q = jnp.array([0.2, -0.3])
    kwargs = dict(
        alpha=jnp.array([0.01, 0.02]),
        kappa=jnp.array([0.01, 0.03]),
        reciprocal_max_index=3,
    )
    expected = [
        les_electrostatics(dict(single, pbc=jnp.full((1, 3), periodic)), q, **kwargs)
        for periodic in (True, False)
    ]
    mixed = dict(
        positions=jnp.concatenate((pos, pos + 2.0, jnp.zeros((1, 3)))),
        batch=jnp.array([0, 0, 1, 1, 2]),
        ptr=jnp.array([0, 2, 4, 5]),
        cell=jnp.concatenate((cell, cell * 100, cell * 100)),
        pbc=jnp.array([[True] * 3, [False] * 3, [True] * 3]),
        node_mask=jnp.array([1.0, 1.0, 1.0, 1.0, 0.0]),
    )
    out = jax.jit(
        lambda d: les_electrostatics(
            d,
            jnp.concatenate((q, q, jnp.ones(1))),
            alpha=jnp.array([0.01, 0.02, 0.01, 0.02, 0.3]),
            kappa=jnp.array([0.01, 0.03, 0.01, 0.03, 0.4]),
            reciprocal_max_index=3,
        )
    )(mixed)
    for i, value in enumerate(out):
        target = jnp.concatenate([x[i] for x in expected])
        np.testing.assert_allclose(value[: len(target)], target, atol=1e-10)
        np.testing.assert_allclose(value[len(target) :], 0.0, atol=1e-12)


def test_gaussian_kernel_float32_derivatives():
    positions = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [1e-8, 0.0, 0.0],
            [0.01, 0.01, 0.01],
            [0.1, 0.1, 0.1],
            [0.5, 0.5, 0.5],
            [1e4, 0.0, 0.0],
        ],
        dtype=jnp.float64,
    )
    kernel = lambda r: gaussian_coulomb(r, 1.0)
    for _ in range(5):
        evaluate = jax.jit(jax.vmap(kernel))
        actual = evaluate(positions.astype(jnp.float32))
        assert np.isfinite(actual).all()
        np.testing.assert_allclose(actual, evaluate(positions), atol=2e-7)
        kernel = jax.jacfwd(kernel)
