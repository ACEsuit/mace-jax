"""End-to-end checks for the long-range and magnetic model families."""

import json

import ase.io
import h5py
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from e3nn import o3
from e3nn_jax import Irreps
from flax import nnx, serialization
from mace.modules.blocks import NonLinearBiasReadoutBlock as TorchBiasReadout

from mace_jax import data, modules
from mace_jax.adapters.nnx.torch import init_from_torch
from mace_jax.calculators.mace import MACEJAXCalculator
from mace_jax.cli.mace_jax_from_torch import _serialize_for_json
from mace_jax.cli.mace_jax_train import _build_predictor
from mace_jax.data.utils import save_configurations_as_HDF5
from mace_jax.modules.blocks import NonLinearBiasReadoutBlock
from mace_jax.nnx_utils import state_to_serializable_dict
from mace_jax.tools import bundle
from mace_jax.tools.gin_model import _export_model_config, _graph_to_data
from mace_jax.tools.gin_model import model as gin_model
from mace_jax.tools.model_builder import _build_jax_model
from mace_jax.tools.train import _checked_loss
from tests.test_les_model import les_source
from tests.test_magnetic_model import magnetic_source
from tests.test_model_extensions import (
    atoms_for_properties,
    converted,
    graph_for_atoms,
    inputs,
)
from tests.test_polar_model import polar_source


@pytest.mark.parametrize('family', ['les', 'polar', 'magnetic', 'magnetic_scf'])
def test_family_bundle_padding_calculator_and_gradients(family, tmp_path):
    if family == 'les':
        pytest.importorskip('les')
        source = les_source(use_dipole=True)
    elif family == 'polar':
        pytest.importorskip('graph_longrange')
        source = polar_source(
            pbc_handling='pbc', atomic_multipoles_max_l=1, field_feature_max_l=1
        )
    else:
        source = magnetic_source()
        if family == 'magnetic_scf':
            from mace.modules.extensions import MagneticSCFMACE  # noqa: PLC0415

            source = MagneticSCFMACE(source, use_scf=False)
    model, _ = converted(source)
    if family == 'les':
        model.les_arguments['reciprocal_max_index'] = 4
    elif family == 'polar':
        model.reciprocal_max_index = 4
    config = _export_model_config(model)
    (tmp_path / 'config.json').write_text(json.dumps(_serialize_for_json(config)))
    (tmp_path / 'params.msgpack').write_bytes(
        serialization.to_bytes(state_to_serializable_dict(nnx.state(model)))
    )
    saved = bundle.load_model_bundle(str(tmp_path), dtype='float64')
    atoms = atoms_for_properties()
    atoms.info.update(total_spin=2.0, external_field=np.array([0.01, -0.02, 0.03]))
    atoms.set_initial_magnetic_moments(
        [[0.2, 0.1, 0.3], [-0.1, 0.3, 0.4], [0.2, -0.1, 0.5]]
    )
    graph = graph_for_atoms(atoms)
    padded = _graph_to_data(graph, num_species=2)
    plain = dict(
        inputs(),
        total_spin=jnp.array([2.0]),
        external_field=jnp.asarray(atoms.info['external_field'])[None],
        magmom=jnp.asarray(atoms.get_initial_magnetic_moments()),
    )
    expected = model(plain)

    # NNX's CallableProxy cannot itself be passed to jax.jit (wrapper loop).
    def saved_apply(values):
        return saved.graphdef.apply(saved.params)(values)

    actual, _ = jax.jit(saved_apply)(padded)
    np.testing.assert_allclose(actual['energy'][0], expected['energy'][0], atol=1e-9)
    np.testing.assert_allclose(actual['forces'][:3], expected['forces'], atol=1e-9)
    np.testing.assert_array_equal(actual['forces'][3:], 0.0)

    def apply(params, g, **flags):
        out, _ = saved.graphdef.apply(params)(_graph_to_data(g, num_species=2), **flags)
        return out

    apply.model_config = config
    atoms.calc = MACEJAXCalculator(
        apply, saved.params, r_max=3.0, atomic_numbers=[1, 8]
    )
    np.testing.assert_allclose(
        atoms.get_potential_energy(), expected['energy'][0], atol=1e-9
    )
    if family == 'polar':
        assert atoms.calc.check_state(atoms) == []
        atoms.info['external_field'][0] += 0.02
        assert 'charge_inputs' in atoms.calc.check_state(atoms)
    if family.startswith('magnetic'):
        np.testing.assert_allclose(
            atoms.calc.get_property('magforces', atoms),
            expected['magforces'],
            atol=1e-9,
        )
        if family == 'magnetic_scf':
            np.testing.assert_allclose(
                atoms.calc.get_property('equilibrated_magmom', atoms), plain['magmom']
            )

    # Force-loss gradients exercise mixed parameter/geometry derivatives.
    def loss(m):
        return jnp.sum(m(plain)['forces'] ** 2)

    gradient = nnx.grad(loss)(model)
    leaves = jax.tree_util.tree_leaves(gradient)
    assert all(np.isfinite(np.asarray(x)).all() for x in leaves)
    assert any(np.any(np.asarray(x) != 0) for x in leaves)


def test_bias_readout_nonzero_bias_and_heads():
    source = TorchBiasReadout(
        o3.Irreps('3x0e+2x1o'),
        o3.Irreps('4x0e'),
        torch.nn.functional.silu,
        o3.Irreps('2x0e'),
        num_heads=2,
    )
    with torch.no_grad():
        source.linear_mid.bias.copy_(torch.tensor([0.4, -0.2, 0.7, 0.9]))
        source.linear_2.bias.copy_(torch.tensor([0.8, -0.6]))
    model = NonLinearBiasReadoutBlock(
        Irreps('3x0e+2x1o'),
        Irreps('4x0e'),
        jax.nn.silu,
        Irreps('2x0e'),
        num_heads=2,
        rngs=nnx.Rngs(3),
    )
    model, _ = init_from_torch(model, source)
    x = np.random.default_rng(7).normal(size=(4, 9))
    heads = np.array([0, 1, 1, 0])
    actual = model(jnp.asarray(x), jnp.asarray(heads))
    np.testing.assert_allclose(
        getattr(actual, 'array', actual),
        source(torch.tensor(x), torch.tensor(heads)).detach(),
        atol=1e-10,
    )


def test_model_inputs_survive_xyz_and_hdf5(tmp_path):
    atoms = atoms_for_properties()
    atoms.info.update(total_spin=2.0, external_field=np.array([0.1, 0.2, 0.3]))
    atoms.set_initial_magnetic_moments([0.2, -0.3, 0.4])
    xyz = tmp_path / 'inputs.xyz'
    ase.io.write(xyz, atoms)
    _, configurations = data.load_from_xyz(xyz)
    path = tmp_path / 'inputs.h5'
    with h5py.File(path, 'w') as handle:
        save_configurations_as_HDF5(configurations, 0, handle)
    loader = data.get_hdf5_dataloader(
        data_file=path,
        atomic_numbers=data.AtomicNumberTable([1, 8]),
        r_max=3.0,
        max_nodes=8,
        max_edges=18,
        max_graphs=3,
        num_workers=0,
        prefetch_batches=0,
    )
    try:
        graph = next(iter(loader))
    finally:
        loader.close()
    values = _graph_to_data(graph, num_species=2)
    np.testing.assert_allclose(values['total_spin'][0], 2.0)
    np.testing.assert_allclose(values['external_field'][0], [0.1, 0.2, 0.3])
    np.testing.assert_allclose(
        values['magmom'][:3], [[0.0, 0.0, 0.2], [0.0, 0.0, -0.3], [0.0, 0.0, 0.4]]
    )
    np.testing.assert_array_equal(values['magmom'][3:], 0.0)


@pytest.mark.parametrize('family', ['les', 'les_last', 'polar', 'magnetic'])
def test_family_multihead_and_correlation(family):
    options = dict(
        correlation=2,
        heads=['A', 'B'],
        atomic_energies=np.array([[0.1, 0.2], [0.3, 0.4]]),
        atomic_inter_scale=[1.2, 1.4],
        atomic_inter_shift=[0.3, 0.5],
    )
    if family.startswith('les'):
        pytest.importorskip('les')
        source = les_source(
            model_options=dict(options, use_last_readout_only=family == 'les_last'),
            use_dipole=True,
            use_quad=family != 'les_last',
        )
    elif family == 'polar':
        pytest.importorskip('graph_longrange')
        source = polar_source(**options, pbc_handling='realspace')
    else:
        source = magnetic_source(**options, max_m_ell=2)
    model, _ = converted(source)
    if family.startswith('les'):
        model.les_arguments['reciprocal_max_index'] = 3
    from tests.test_model_extensions import torch_inputs  # noqa: PLC0415

    values = dict(
        inputs(),
        head=jnp.array([1]),
        magmom=jnp.array([[0.1, 0.2, 0.3], [0.2, 0.3, 0.4], [0.3, 0.4, 0.5]]),
        total_spin=jnp.ones(1),
        external_field=jnp.zeros((1, 3)),
        fermi_level=jnp.zeros(1),
    )
    values.update(
        rcell=2 * jnp.pi * jnp.linalg.inv(values['cell']).swapaxes(-1, -2),
        volume=jnp.linalg.det(values['cell']),
        atomic_numbers=jnp.array([1, 8, 1]),
    )
    if family.startswith('les'):
        values['cell'] = values['cell'].reshape(-1, 3)
    expected = source(torch_inputs(values), compute_force=True)
    actual = model(values)
    for key in ('energy', 'forces'):
        np.testing.assert_allclose(
            actual[key], expected[key].detach(), atol=1e-8, err_msg=key
        )


def test_scf_config_roundtrip():
    from mace_jax.modules.magnetic import MagneticSCFMACE  # noqa: PLC0415

    model, _ = converted(magnetic_source())
    wrapper = MagneticSCFMACE(model, n_scf_step=2, use_collinear=True)
    restored = _build_jax_model(_export_model_config(wrapper))
    assert isinstance(restored, MagneticSCFMACE)
    assert restored.n_scf_step == 2
    assert restored.use_collinear


def native_family(kind):
    options = {}
    if kind == 'MACELES':
        options['les_arguments'] = {'reciprocal_max_index': 3}
    elif kind == 'PolarMACE':
        options.update(reciprocal_max_index=4, pbc_handling='realspace')
    else:
        options.update(m_max=[2.0, 3.0], num_mag_radial_basis=2, max_m_ell=1)
        if kind == 'MagneticSCFMACE':
            options.update(use_scf=False)
    return gin_model(
        model_class=kind,
        r_max=3.0,
        num_interactions=2,
        avg_num_neighbors=2.0,
        atomic_numbers=[1, 8],
        atomic_energies='zero',
        num_bessel=2,
        num_polynomial_cutoff=5,
        max_ell=2,
        hidden_irreps='2x0e+2x1o+2x2e',
        MLP_irreps='2x0e',
        correlation=1,
        radial_MLP=[3],
        **options,
    )


@pytest.mark.parametrize(
    'kind', ['MACELES', 'PolarMACE', 'MagneticScaleShiftMACE', 'MagneticSCFMACE']
)
def test_native_model_factory(kind):
    apply, state, _ = native_family(kind)
    output = jax.jit(lambda p, g: apply(p, g, compute_force=False))(
        state, graph_for_atoms(atoms_for_properties())
    )
    assert np.isfinite(output['energy']).all()
    assert apply.model_config['torch_model_class'] == kind
    if kind.startswith('Magnetic'):
        assert np.isfinite(output['magforces']).all()


def test_polar_training_requests_field_response():
    apply, state, _ = native_family('PolarMACE')
    graph = graph_for_atoms(atoms_for_properties())
    predictor = _build_predictor(
        apply, modules.WeightedEnergyForcesLoss(energy_weight=1.0, forces_weight=0.0)
    )
    assert predictor(state, graph)['polarizability'] is None
    loss = modules.DipolePolarLoss(dipole_weight=0.0, polarizability_weight=1.0)
    predictor = predictor.for_loss(loss)
    output = predictor(state, graph)
    assert output['polarizability'] is not None
    expected = apply(state, graph, compute_force=False, compute_polarizability=True)
    np.testing.assert_allclose(output['polarizability'], expected['polarizability'])
    gradient = jax.grad(
        lambda p: _checked_loss(
            loss, graph, predictor(dict(state, params=p), graph)
        ).sum()
    )(state['params'])
    leaves = jax.tree_util.tree_leaves(gradient)
    assert all(np.isfinite(np.asarray(x)).all() for x in leaves)
    assert any(np.any(np.asarray(x) != 0) for x in leaves)
