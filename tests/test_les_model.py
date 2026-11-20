"""MACELES checkpoint and observable parity."""

import jax.numpy as jnp
import numpy as np
import pytest
from mace.modules.extensions import MACELES

from tests.test_model_extensions import converted, inputs, source_model, torch_inputs

pytestmark = pytest.mark.usefixtures('runtime_devices')


def les_source(model_options=None, **les_arguments):
    from mace_jax.tools.import_from_torch import extract_config_mace_model  # noqa: PLC0415

    config = extract_config_mace_model(source_model(**(model_options or {})))
    config.pop('interaction_avg_num_neighbors', None)
    return MACELES(les_arguments=les_arguments, **config)


@pytest.mark.parametrize('periodic', [False, True])
@pytest.mark.parametrize(
    'options',
    [
        {},
        dict(
            use_dipole=True,
            use_induced_charge=True,
            use_induced_dipole=True,
            make_alpha_positive=True,
        ),
        dict(
            use_dipole=True,
            use_quad=True,
            use_induced_dipole=True,
            use_anisotropic_polarizability=True,
        ),
        dict(
            use_quad=True,
            use_induced_dipole=True,
            use_anisotropic_polarizability=True,
            alpha_1o_nonlinear_readout=True,
            make_alpha_positive=True,
        ),
    ],
)
def test_les_conversion(periodic, options):
    pytest.importorskip('les')
    source = les_source(**options)
    model, config = converted(source)
    data = inputs()
    data['cell'] = data['cell'].reshape(-1, 3)
    data.update(atomic_numbers=jnp.array([1, 8, 1]), pbc=jnp.full((1, 3), periodic))
    # A smaller static grid is enough for this test cell.
    model.les_arguments['reciprocal_max_index'] = 3
    expected = source(torch_inputs(data), compute_force=True, compute_stress=True)
    actual = model(data, compute_stress=True)
    for key in (
        'energy',
        'node_energy',
        'forces',
        'stress',
        'les_energy',
        'latent_charges',
        'latent_dipoles',
        'latent_alphas',
        'latent_kappas',
        'latent_quads',
    ):
        if expected[key] is None:
            assert actual[key] is None
        else:
            np.testing.assert_allclose(
                actual[key], expected[key].detach(), atol=2e-9, rtol=2e-7, err_msg=key
            )


@pytest.mark.parametrize('periodic', [False, True])
def test_les_born_charges_and_baselines(periodic):
    pytest.importorskip('les')
    source = les_source(
        use_dipole=True,
        use_induced_dipole=True,
        use_fixed_atomic_charges=True,
        use_atomic_alpha=True,
        compute_bec=True,
    )
    model, _ = converted(source)
    model.les_arguments['reciprocal_max_index'] = 3
    data = inputs()
    data.update(
        atomic_numbers=jnp.array([1, 8, 1]),
        cell=data['cell'].reshape(-1, 3),
        pbc=jnp.full((1, 3), periodic),
    )
    expected = source(torch_inputs(data), compute_force=False)
    actual = model(data, compute_force=False)
    for key in ('energy', 'BEC', 'latent_charges', 'latent_alphas', 'latent_dipoles'):
        np.testing.assert_allclose(
            actual[key], expected[key].detach(), atol=1e-8, rtol=1e-7, err_msg=key
        )
