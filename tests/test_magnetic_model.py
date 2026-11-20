"""Magnetic model import and positional/magnetic derivative parity."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from mace.modules import blocks
from mace.modules.extensions import MagneticScaleShiftMACE

from mace_jax.tools.import_from_torch import extract_config_mace_model
from tests.test_model_extensions import converted, inputs, source_model, torch_inputs

pytestmark = pytest.mark.usefixtures('runtime_devices')


def magnetic_source(**kwargs):
    pytest.importorskip('sphericart.torch')
    config = extract_config_mace_model(source_model())
    config.pop('interaction_avg_num_neighbors', None)
    config.pop('keep_last_layer_irreps', None)
    config.update(
        m_max=[2.0, 3.0],
        num_mag_radial_basis=3,
        max_m_ell=1,
        interaction_cls_first=blocks.MagneticRealAgnosticSpinOrbitCoupledDensityInteractionBlock,
        interaction_cls=blocks.MagneticRealAgnosticResidueSpinOrbitCoupledDensityInteractionBlock,
    )
    config.update(kwargs)
    return MagneticScaleShiftMACE(**config)


@pytest.mark.parametrize('onebody', [False, True])
@pytest.mark.parametrize('layers', [1, 3])
def test_magnetic_conversion(onebody, layers):
    source = magnetic_source(
        use_magmom_one_body=onebody,
        num_mag_radial_basis_one_body=3,
        num_interactions=layers,
    )
    model, config = converted(source)
    data = dict(
        inputs(),
        magmom=jnp.array([[0.2, 0.1, -0.3], [0.8, -0.2, 0.4], [0.1, 0.3, 0.2]]),
    )
    expected = source(torch_inputs(data), compute_force=True, compute_stress=True)
    actual = model(data, compute_stress=True)
    for key in ('energy', 'forces', 'stress', 'magforces', 'node_feats'):
        np.testing.assert_allclose(
            actual[key], expected[key].detach(), atol=1e-9, rtol=1e-7, err_msg=key
        )
    zero = model(dict(data, magmom=jnp.zeros((3, 3))))
    assert np.isfinite(zero['magforces']).all()
    assert np.isfinite(zero['forces']).all()


def test_magnetic_scf_import_and_relaxation():
    from mace.modules.extensions import MagneticSCFMACE  # noqa: PLC0415

    source = MagneticSCFMACE(magnetic_source(), use_scf=False)
    model, _ = converted(source)
    data = dict(
        inputs(),
        magmom=jnp.array([[0.2, 0.1, -0.3], [0.8, -0.2, 0.4], [0.1, 0.3, 0.2]]),
    )
    expected = source(torch_inputs(data), compute_force=False)
    actual = model(data, compute_force=False)
    np.testing.assert_allclose(actual['energy'], expected['energy'].detach(), atol=1e-9)
    model.use_scf = True
    model.n_scf_step = 2
    model.use_collinear = True
    result = jax.jit(lambda d: model(d, compute_force=False))(data)
    assert np.isfinite(result['energy']).all()
    assert result['energy'][0] <= actual['energy'][0] + 1e-9
    np.testing.assert_allclose(
        result['equilibrated_magmom'][:, :2], data['magmom'][:, :2]
    )
    assert int(result['scf_steps']) <= 2
