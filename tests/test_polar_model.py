"""PolarMACE import and constrained field response against Torch."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from mace.modules.extensions import PolarMACE

from mace_jax.tools.import_from_torch import extract_config_mace_model
from tests.test_model_extensions import converted, inputs, source_model, torch_inputs

pytestmark = pytest.mark.usefixtures('runtime_devices')


def polar_source(**kwargs):
    config = extract_config_mace_model(source_model())
    config.pop('interaction_avg_num_neighbors', None)
    config.update(kwargs)
    return PolarMACE(**config)


@pytest.mark.parametrize('mode', ['realspace', 'pbc'])
@pytest.mark.parametrize('max_l', [0, 1])
def test_polar_conversion(mode, max_l):
    pytest.importorskip('graph_longrange')
    source = polar_source(
        atomic_multipoles_max_l=max_l,
        field_feature_max_l=max_l,
        field_feature_widths=[1.0, 1.3],
        pbc_handling=mode,
        add_local_electron_energy=True,
    )
    model, config = converted(source)
    model.reciprocal_max_index = 4
    data = inputs()
    data.update(
        total_spin=jnp.array([2.0]),
        fermi_level=jnp.array([0.0]),
        external_field=jnp.array([[0.02, -0.01, 0.03]]),
    )
    data.update(
        rcell=2 * jnp.pi * jnp.linalg.inv(data['cell']).swapaxes(-1, -2),
        volume=jnp.linalg.det(data['cell']),
    )
    expected = source(torch_inputs(data), compute_stress=True)
    actual = model(data, compute_stress=True)
    for key in (
        'energy',
        'node_energy',
        'forces',
        'stress',
        'density_coefficients',
        'spin_density',
        'dipole',
        'fukui_functions',
        'electron_energy',
        'electrostatic_energy',
    ):
        np.testing.assert_allclose(
            actual[key], expected[key].detach(), atol=2e-8, rtol=3e-7, err_msg=key
        )
    np.testing.assert_allclose(
        actual['charges'].sum(), data['total_charge'][0], atol=1e-12
    )
    np.testing.assert_allclose(
        actual['spins'].sum(), data['total_spin'][0] - 1, atol=1e-12
    )


@pytest.mark.parametrize('method', ['dipole', 'energy'])
def test_polar_field_response(method):
    pytest.importorskip('graph_longrange')
    source = polar_source(pbc_handling='realspace', add_local_electron_energy=True)
    model, _ = converted(source)
    data = dict(
        inputs(),
        external_field=jnp.array([[0.02, -0.01, 0.03]]),
        total_spin=jnp.array([2.0]),
        fermi_level=jnp.zeros(1),
    )
    data.update(
        rcell=2 * jnp.pi * jnp.linalg.inv(data['cell']).swapaxes(-1, -2),
        volume=jnp.linalg.det(data['cell']),
    )
    expected = source(
        torch_inputs(data),
        compute_force=False,
        compute_polarizability=True,
        polarizability_method=method,
    )
    actual = jax.jit(
        lambda d: model(
            d,
            compute_force=False,
            compute_polarizability=True,
            polarizability_method=method,
        )
    )(data)
    np.testing.assert_allclose(
        actual['polarizability'], expected['polarizability'].detach(), atol=1e-9
    )
    if method == 'energy':
        np.testing.assert_allclose(
            actual['dipole_from_energy'],
            expected['dipole_from_energy'].detach(),
            atol=1e-9,
        )
    else:
        eps = 1e-4
        plus = model(
            dict(data, external_field=data['external_field'].at[0, 1].add(eps)),
            compute_force=False,
        )
        minus = model(
            dict(data, external_field=data['external_field'].at[0, 1].add(-eps)),
            compute_force=False,
        )
        np.testing.assert_allclose(
            actual['polarizability'][0, :, 1],
            -(plus['dipole'] - minus['dipole'])[0] / (2 * eps),
            atol=1e-8,
        )
        response = model(
            data,
            compute_force=False,
            compute_polarizability=True,
            compute_dielectric_derivatives=True,
        )
        assert response['dalpha_dr'].shape == (9, 3, 3)
        assert np.isfinite(response['dalpha_dr']).all()


def test_isolated_atom_charge_spin_constraint():
    pytest.importorskip('graph_longrange')
    model, _ = converted(polar_source(pbc_handling='realspace'))
    values = dict(
        inputs(),
        positions=jnp.array([[0.1, 0.2, 0.3]]),
        node_attrs=jnp.array([[1.0, 0.0]]),
        batch=jnp.zeros(1, dtype=int),
        ptr=jnp.array([0, 1]),
        edge_index=jnp.zeros((2, 0), dtype=int),
        shifts=jnp.zeros((0, 3)),
        unit_shifts=jnp.zeros((0, 3)),
        total_spin=jnp.array([2.0]),
        node_mask=jnp.ones(1, dtype=bool),
    )
    result = jax.jit(model)(values)
    np.testing.assert_allclose(result['charges'], [1.0])
    np.testing.assert_allclose(result['spins'], [1.0])
    assert np.isfinite(result['forces']).all()
