"""End-to-end parity for architecture metadata and electric observables."""

import json

import ase.io
import gin
import h5py
import jax
import jax.numpy as jnp
import jraph
import numpy as np
import optax
import pytest
import torch
from ase import Atoms
from e3nn import o3
from flax import nnx, serialization
from mace.modules import blocks, models

from mace_jax import data, modules
from mace_jax import data as data_tools
from mace_jax.calculators.mace import MACEJAXCalculator
from mace_jax.cli import mace_jax_train
from mace_jax.cli.mace_jax_from_torch import _serialize_for_json, convert_model
from mace_jax.cli.mace_jax_predict import _build_predictor
from mace_jax.data.utils import save_configurations_as_HDF5
from mace_jax.nnx_utils import state_to_serializable_dict
from mace_jax.tools import bundle, gin_datasets, gin_functions, gin_model
from mace_jax.tools.gin_model import _graph_to_data
from mace_jax.tools.import_from_torch import extract_config_mace_model
from mace_jax.tools.model_builder import _build_jax_model
from mace_jax.tools.train import _split_prediction_outputs
from tests.test_evaluate import _Logger


def source_model(kind='ScaleShiftMACE', **kwargs):
    config = dict(
        r_max=3.0,
        num_bessel=2,
        num_polynomial_cutoff=5,
        max_ell=2,
        num_interactions=2,
        num_elements=2,
        hidden_irreps=o3.Irreps('2x0e+2x1o+2x2e'),
        MLP_irreps=o3.Irreps('2x0e'),
        atomic_numbers=[1, 8],
        atomic_energies=np.zeros(2),
        interaction_cls=blocks.RealAgnosticResidualInteractionBlock,
        interaction_cls_first=blocks.RealAgnosticResidualInteractionBlock,
        avg_num_neighbors=2.0,
        correlation=1,
        gate=torch.nn.functional.silu,
        radial_MLP=[3],
    )
    if kind == 'ScaleShiftMACE':
        config.update(atomic_inter_scale=1.2, atomic_inter_shift=0.3)
    elif kind == 'AtomicDipolesMACE':
        config.update(atomic_energies=None, MLP_irreps=o3.Irreps('2x1o'))
    elif kind == 'EnergyDipolesMACE':
        config.update(MLP_irreps=o3.Irreps('2x0e+2x1o'))
    elif kind == 'AtomicDielectricMACE':
        config.update(atomic_energies=None, MLP_irreps=o3.Irreps('2x0e+2x1o+2x2e'))
    config.update(kwargs)
    return getattr(models, kind)(**config)


def inputs():
    return dict(
        positions=jnp.array([[0.1, 0.2, 0.3], [1.0, 0.4, 0.2], [0.3, 1.1, 0.8]]),
        node_attrs=jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]),
        batch=jnp.zeros(3, dtype=jnp.int32),
        ptr=jnp.array([0, 3]),
        head=jnp.zeros(1, dtype=jnp.int32),
        cell=jnp.eye(3)[None] * 5,
        pbc=jnp.ones((1, 3), dtype=bool),
        edge_index=jnp.array([[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]]),
        shifts=jnp.zeros((6, 3)),
        unit_shifts=jnp.zeros((6, 3)),
        charges=jnp.array([0.1, -0.2, 0.1]),
        total_charge=jnp.array([1.0]),
    )


def torch_inputs(data, device=None):
    return {k: torch.as_tensor(np.array(v), device=device) for k, v in data.items()}


def converted(source):
    config = extract_config_mace_model(source)
    graphdef, state, _ = convert_model(source, config)
    return nnx.merge(graphdef, state), config


@pytest.mark.parametrize(
    'kwargs',
    [
        dict(correlation=[1, 2]),
        dict(keep_last_layer_irreps=True),
        dict(num_interactions=1, keep_last_layer_irreps=True),
    ],
)
def test_architecture_conversion(kwargs, runtime_devices, tmp_path):
    source = source_model(**kwargs).to(runtime_devices['torch_device'])
    model, config = converted(source)
    data = inputs()
    expected = source(
        torch_inputs(data, next(source.parameters()).device), compute_force=True
    )
    actual = model(data)
    for key in ('energy', 'forces'):
        np.testing.assert_allclose(
            actual[key], expected[key].detach().cpu().numpy(), atol=1e-9, rtol=1e-7
        )
    assert config['correlation'] == kwargs.get('correlation', 1)
    assert config['keep_last_layer_irreps'] == kwargs.get(
        'keep_last_layer_irreps', False
    )

    (tmp_path / 'config.json').write_text(json.dumps(_serialize_for_json(config)))
    (tmp_path / 'params.msgpack').write_bytes(
        serialization.to_bytes(state_to_serializable_dict(nnx.state(model)))
    )
    saved = bundle.load_model_bundle(str(tmp_path), dtype='float64')
    reloaded, _ = saved.graphdef.apply(saved.params)(data)
    np.testing.assert_allclose(reloaded['forces'], actual['forces'], atol=1e-10)


@pytest.mark.parametrize(
    'kind', ['AtomicDipolesMACE', 'EnergyDipolesMACE', 'AtomicDielectricMACE']
)
def test_tensor_conversion(kind, runtime_devices):
    source = source_model(kind).to(runtime_devices['torch_device'])
    model, _ = converted(source)
    data = inputs()
    flags = dict(compute_force=kind == 'EnergyDipolesMACE')
    if kind == 'AtomicDielectricMACE':
        flags['compute_dielectric_derivatives'] = True
    expected = source(torch_inputs(data, next(source.parameters()).device), **flags)
    actual = model(data, **flags)
    if kind == 'EnergyDipolesMACE':
        np.testing.assert_allclose(
            actual['node_energy'].sum(), actual['energy'][0], atol=1e-10
        )
    for key in (
        'energy',
        'forces',
        'dipole',
        'atomic_dipoles',
        'charges',
        'polarizability',
        'polarizability_sh',
        'dmu_dr',
        'dalpha_dr',
    ):
        if expected.get(key) is not None:
            np.testing.assert_allclose(
                actual[key],
                expected[key].detach().cpu().numpy(),
                atol=1e-9,
                rtol=1e-7,
                err_msg=key,
            )


def test_energy_derivatives(runtime_devices):
    source = source_model().to(runtime_devices['torch_device'])
    model, _ = converted(source)
    data = inputs()
    flags = dict(
        compute_force=True,
        compute_stress=True,
        compute_hessian=True,
        compute_edge_forces=True,
        compute_atomic_stresses=True,
    )
    expected = source(torch_inputs(data, next(source.parameters()).device), **flags)
    actual = model(data, **flags)
    for key in ('hessian', 'edge_forces', 'atomic_virials', 'atomic_stresses'):
        np.testing.assert_allclose(
            actual[key],
            expected[key].detach().cpu().numpy(),
            atol=1e-9,
            rtol=1e-7,
            err_msg=key,
        )
    np.testing.assert_allclose(
        actual['atomic_virials'].sum(0), actual['virials'][0], atol=1e-10
    )
    np.testing.assert_allclose(
        actual['atomic_stresses'].sum(0), actual['stress'][0], atol=1e-10
    )
    h = np.asarray(actual['hessian']).reshape(9, 9)
    np.testing.assert_allclose(h, h.T, atol=1e-10)


def graph_for_atoms(atoms):

    return jraph.pad_with_graphs(
        data.graph_from_configuration(
            data.config_from_atoms(atoms),
            cutoff=3.0,
            z_table=data.AtomicNumberTable([1, 8]),
        ),
        len(atoms) + 1,
        20,
        2,
    )


def atoms_for_properties():
    atoms = Atoms(
        'HOH', positions=np.asarray(inputs()['positions']), cell=[5, 5, 5], pbc=True
    )
    atoms.set_initial_charges([0.1, -0.2, 0.1])
    atoms.info.update(
        total_charge=1.0,
        energy=0.0,
        dipole=np.array([0.2, 0.1, -0.3]),
        polarizability=np.eye(3),
    )
    atoms.set_array('forces', np.zeros((3, 3)))
    return atoms


@pytest.mark.parametrize(
    'kind', ['AtomicDipolesMACE', 'EnergyDipolesMACE', 'AtomicDielectricMACE']
)
def test_native_training_bundle_and_ase(tmp_path, kind):

    model, params, _ = gin_model.model(
        model_class=kind,
        r_max=3.0,
        atomic_numbers=(1, 8),
        atomic_energies='zero',
        avg_num_neighbors=2.0,
        num_interactions=2,
        num_bessel=2,
        num_polynomial_cutoff=5,
        max_ell=2,
        hidden_irreps='2x0e+2x1o+2x2e',
        MLP_irreps='2x1o'
        if kind == 'AtomicDipolesMACE'
        else ('2x0e+2x1o' if kind == 'EnergyDipolesMACE' else '2x0e+2x1o+2x2e'),
        correlation=1,
        gate=jax.nn.silu,
        radial_MLP=[3],
    )
    atoms = atoms_for_properties()
    graph = graph_for_atoms(atoms)
    output = jax.jit(model)(params, graph)
    assert np.isfinite(output['dipole']).all()
    np.testing.assert_allclose(output['dipole'][1], 0.0, atol=1e-12)
    loss = {
        'AtomicDipolesMACE': modules.DipoleSingleLoss,
        'EnergyDipolesMACE': modules.WeightedEnergyForcesDipoleLoss,
        'AtomicDielectricMACE': modules.DipolePolarLoss,
    }[kind]()
    value, grad = jax.value_and_grad(
        lambda p: loss(graph, model({**params, 'params': p}, graph)).sum()
    )(params['params'])
    assert np.isfinite(value)
    assert all(np.isfinite(x).all() for x in jax.tree.leaves(grad))
    assert sum(float(jnp.sum(x * x)) for x in jax.tree.leaves(grad)) > 1e-12
    config = model.model_config
    native = _build_jax_model(config)
    graphdef, state = nnx.split(native)
    nnx.replace_by_pure_dict(
        state, gin_model._merge_state_dicts(params.get('config'), params['params'])
    )
    (tmp_path / 'config.json').write_text(json.dumps(config))
    (tmp_path / 'params.msgpack').write_bytes(
        serialization.to_bytes(state_to_serializable_dict(state))
    )
    saved = bundle.load_model_bundle(str(tmp_path), dtype='float64')
    restored, _ = saved.graphdef.apply(saved.params)(
        gin_model._graph_to_data(graph, num_species=2)
    )
    np.testing.assert_allclose(restored['dipole'][0], output['dipole'][0], atol=1e-10)
    atoms.calc = MACEJAXCalculator(model, params, 3.0, atomic_numbers=[1, 8])
    factor = (
        1.0
        if kind == 'AtomicDielectricMACE'
        else 1 / (299792458.0 * 1.602176634e-19 / 1e-11)
    )
    np.testing.assert_allclose(
        atoms.get_dipole_moment(), output['dipole'][0] * factor, atol=1e-10
    )
    if kind == 'AtomicDielectricMACE':
        assert atoms.get_charges().sum() == pytest.approx(1.0)
        np.testing.assert_allclose(
            atoms.calc.get_property('polarizability', atoms),
            output['polarizability'][0],
        )
        atoms.info['total_charge'] = 2.0
        assert atoms.get_charges().sum() == pytest.approx(2.0)
        # The implicit total uses the explicit charge array, which may differ
        # from ASE initial charges. Adding/removing an override must invalidate
        # the cache even when that override equals the initial-charge sum.
        atoms.info.pop('total_charge')
        atoms.set_array('charges', np.array([0.2, 0.3, 0.5]))
        atoms.set_initial_charges(np.zeros(3))
        assert atoms.get_charges().sum() == pytest.approx(1.0)
        atoms.info['total_charge'] = 0.0
        assert atoms.get_charges().sum() == pytest.approx(0.0, abs=1e-12)
        atoms.info.pop('total_charge')
        assert atoms.get_charges().sum() == pytest.approx(1.0)
    else:
        first = atoms.get_dipole_moment().copy()
        atoms.arrays['initial_charges'][0] += 0.2
        np.testing.assert_allclose(
            atoms.get_dipole_moment() - first, 0.2 * atoms.positions[0], atol=1e-10
        )


def test_dielectric_padding_batch_derivatives_and_symmetry():

    model, _ = converted(source_model('AtomicDielectricMACE'))
    atoms = atoms_for_properties()
    other = atoms.copy()
    other.positions += np.array([0.2, -0.3, 0.4])
    other.info['total_charge'] = -1.0
    graphs = [
        data_tools.graph_from_configuration(
            data_tools.config_from_atoms(a),
            cutoff=3.0,
            z_table=data_tools.AtomicNumberTable([1, 8]),
        )
        for a in (atoms, other)
    ]
    batched = jraph.pad_with_graphs(jraph.batch(graphs), 8, 18, 4)
    data = _graph_to_data(batched, num_species=2)
    output = jax.jit(lambda d: model(d, compute_dielectric_derivatives=True))(data)
    for index, a in enumerate((atoms, other)):
        single = _graph_to_data(graph_for_atoms(a), num_species=2)
        reference = model(single)
        for key in ('dipole', 'polarizability'):
            np.testing.assert_allclose(
                output[key][index], reference[key][0], atol=1e-10
            )
        assert output['charges'][index * 3 : (index + 1) * 3].sum() == pytest.approx(
            a.info['total_charge']
        )
    np.testing.assert_allclose(output['charges'][6:], 0.0, atol=1e-12)
    assert all(np.isfinite(x).all() for x in jax.tree.leaves(output))
    eps = 1e-5
    for key, derivative in [('dipole', 'dmu_dr'), ('polarizability', 'dalpha_dr')]:
        plus = model(dict(data, positions=data['positions'].at[1, 2].add(eps)))[key][
            0
        ].reshape(-1)
        minus = model(dict(data, positions=data['positions'].at[1, 2].add(-eps)))[key][
            0
        ].reshape(-1)
        np.testing.assert_allclose(
            output[derivative][:, 1, 2],
            (plus - minus) / (2 * eps),
            atol=1e-7,
            rtol=1e-5,
        )
    # Full inversion is a reflection: dipoles are odd, polarizability is even.
    reflected = model(dict(data, positions=-data['positions'], shifts=-data['shifts']))
    np.testing.assert_allclose(
        reflected['dipole'][:2], -output['dipole'][:2], atol=1e-10
    )
    np.testing.assert_allclose(
        reflected['polarizability'][:2], output['polarizability'][:2], atol=1e-10
    )

    rotation = jnp.array([[0.8, -0.6, 0.0], [0.6, 0.8, 0.0], [0.0, 0.0, -1.0]])
    rotated = model(
        dict(
            data,
            positions=data['positions'] @ rotation.T,
            shifts=data['shifts'] @ rotation.T,
            cell=data['cell'] @ rotation.T,
        )
    )
    np.testing.assert_allclose(
        rotated['dipole'][:2], output['dipole'][:2] @ rotation.T, atol=1e-10
    )
    np.testing.assert_allclose(
        rotated['polarizability'][:2],
        rotation @ output['polarizability'][:2] @ rotation.T,
        atol=1e-10,
    )


def test_hessian_finite_difference_pbc_mask_and_padding():
    model, _ = converted(source_model())
    atoms = atoms_for_properties()
    atoms.pbc = False
    data = _graph_to_data(graph_for_atoms(atoms), num_species=2)
    output = jax.jit(
        lambda d: model(
            d,
            compute_force=False,
            compute_hessian=True,
            compute_atomic_stresses=True,
            compute_virials=True,
        )
    )(data)
    # A virial request also returns forces, matching the existing Torch interface.
    assert output['forces'] is not None
    assert np.isfinite(output['hessian']).all()
    np.testing.assert_array_equal(output['atomic_stresses'], 0.0)
    np.testing.assert_allclose(
        output['atomic_virials'][:3].sum(0), output['virials'][0], atol=1e-10
    )
    eps = 1e-5
    plus = model(dict(data, positions=data['positions'].at[1, 1].add(eps)))['forces']
    minus = model(dict(data, positions=data['positions'].at[1, 1].add(-eps)))['forces']
    np.testing.assert_allclose(
        output['hessian'][4], -(plus - minus) / (2 * eps), atol=1e-7, rtol=1e-5
    )
    np.testing.assert_allclose(output['hessian'][9:], 0.0, atol=1e-10)


@pytest.mark.parametrize('charges_key', ['q_ref', 'charges'])
def test_charge_inputs_survive_xyz_hdf5_and_streaming(tmp_path, charges_key):
    atoms = atoms_for_properties()
    # Exercise the preprocessing charge-key override, independent of ASE's
    # initial-charge convention.
    atoms.set_array(charges_key, np.array([0.2, -0.1, 0.3]))
    xyz = tmp_path / 'charged.xyz'
    ase.io.write(xyz, [atoms, atoms.copy()])
    _, configs = data.load_from_xyz(xyz, charges_key=charges_key)
    for config in configs:
        np.testing.assert_allclose(config.charges, [0.2, -0.1, 0.3])
        np.testing.assert_allclose(config.dipole, atoms.info['dipole'])
        assert config.dipole_weight == 1.0
    path = tmp_path / 'charged.h5'
    with h5py.File(path, 'w') as handle:
        save_configurations_as_HDF5(configs, 0, handle)
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
    inputs = _graph_to_data(graph, num_species=2)
    np.testing.assert_allclose(inputs['charges'][:6], np.tile([0.2, -0.1, 0.3], 2))
    np.testing.assert_allclose(inputs['total_charge'][:2], [1.0, 1.0])
    np.testing.assert_array_equal(inputs['charges'][6:], 0.0)


@pytest.mark.parametrize(
    'kind', ['AtomicDipolesMACE', 'EnergyDipolesMACE', 'AtomicDielectricMACE']
)
def test_one_layer_tensor_conversion(kind):
    source = source_model(kind, num_interactions=1)
    model, config = converted(source)
    expected = source(torch_inputs(inputs()), compute_force=False)
    actual = model(inputs(), compute_force=False)
    np.testing.assert_allclose(actual['dipole'], expected['dipole'].detach(), atol=1e-9)
    restored = _build_jax_model(config)
    assert type(restored).__name__ == kind


def test_mace_config_extraction_and_nondefault_tensor_gate():
    for kind in ('MACE', 'AtomicDielectricMACE'):
        source = source_model(kind, gate=torch.tanh, correlation=2)
        model, _ = converted(source)
        expected = source(torch_inputs(inputs()), compute_force=False)
        actual = model(inputs(), compute_force=False)
        key = 'energy' if kind == 'MACE' else 'polarizability'
        # Correlation > 1 uses independently constructed contraction bases;
        # allow the same backend rounding as the component parity tests.
        np.testing.assert_allclose(
            actual[key], expected[key].detach(), atol=1e-8, rtol=1e-6
        )


def test_periodic_atomic_virials_include_image_shifts():
    model, _ = converted(source_model())
    atoms = atoms_for_properties()
    atoms.cell = [2.0, 2.5, 2.7]
    conf = data.config_from_atoms(atoms)
    graph = data.graph_from_configuration(
        conf, cutoff=3.0, z_table=data.AtomicNumberTable([1, 8])
    )
    graph = jraph.pad_with_graphs(graph, 4, int(graph.n_edge[0]) + 2, 2)
    inputs = _graph_to_data(graph, num_species=2)
    assert np.any(np.asarray(inputs['unit_shifts']))
    output = model(inputs, compute_stress=True, compute_atomic_stresses=True)
    np.testing.assert_allclose(
        output['atomic_virials'][:3].sum(0), output['virials'][0], atol=1e-9
    )
    np.testing.assert_allclose(
        output['atomic_stresses'][:3].sum(0), output['stress'][0], atol=1e-9
    )


def test_streaming_output_partition_including_hessians():
    graphs = []
    for index, count in enumerate((2, 3)):
        atoms = atoms_for_properties()[:count]
        graph = data.graph_from_configuration(
            data.config_from_atoms(atoms),
            cutoff=3.0,
            z_table=data.AtomicNumberTable([1, 8]),
        )
        graphs.append(
            graph._replace(globals=graph.globals._replace(graph_id=np.array([index])))
        )
    graph = jraph.pad_with_graphs(jraph.batch(graphs), 6, 10, 3)
    values = dict(
        hessian=np.arange(18 * 6 * 3).reshape(18, 6, 3),
        dmu_dr=np.arange(3 * 6 * 3).reshape(3, 6, 3),
        edge_forces=np.arange(30).reshape(10, 3),
        polarizability_sh=np.arange(18).reshape(3, 6),
    )
    ids, outputs = _split_prediction_outputs(values, graph)
    assert ids == [0, 1]
    np.testing.assert_array_equal(outputs['hessian'][1], values['hessian'][6:15, 2:5])
    np.testing.assert_array_equal(outputs['dmu_dr'][1], values['dmu_dr'][:, 2:5])
    np.testing.assert_array_equal(outputs['edge_forces'][1], values['edge_forces'][2:8])
    np.testing.assert_array_equal(
        outputs['polarizability_sh'][1], values['polarizability_sh'][1]
    )


def test_dielectric_training_without_energy_labels(tmp_path, monkeypatch):

    atoms = atoms_for_properties()
    atoms.info.pop('energy')
    del atoms.arrays['forces']
    path = tmp_path / 'dielectric.h5'
    configs = [data.config_from_atoms(atoms) for _ in range(3)]
    with h5py.File(path, 'w') as handle:
        save_configurations_as_HDF5(configs, 0, handle)
    gin.clear_config()
    loaders = []
    try:
        gin.bind_parameter(
            'mace_jax.tools.gin_model.model.model_class', 'AtomicDielectricMACE'
        )
        train, valid, test, e0s, _ = gin_datasets.datasets(
            train_path=str(path),
            valid_path=str(path),
            r_max=3.0,
            n_edge=20,
            num_workers=0,
            stream_train_stats_workers=0,
            stream_train_prefetch=0,
        )
        loaders = [train, valid, test]
        assert all(x == 0.0 for x in e0s.values())
        model, params, _ = gin_model.model(
            model_class='AtomicDielectricMACE',
            r_max=3.0,
            atomic_numbers=(1, 8),
            avg_num_neighbors=2.0,
            num_interactions=1,
            num_bessel=2,
            num_polynomial_cutoff=5,
            max_ell=2,
            hidden_irreps='2x0e+2x1o+2x2e',
            MLP_irreps='2x0e+2x1o+2x2e',
            correlation=1,
            gate=jax.nn.silu,
            radial_MLP=[3],
        )
        loss = modules.DipolePolarLoss()
        monkeypatch.setattr(gin_functions, 'loss', lambda: loss)
        optimizer = optax.sgd(0.001)
        logger = _Logger()
        _, final = gin_functions.train(
            predictor=mace_jax_train._build_predictor(model, loss),
            params=params,
            optimizer_state=optimizer.init(params['params']),
            train_loader=train,
            valid_loader=valid,
            test_loader=None,
            gradient_transform=optimizer,
            max_epochs=1,
            logger=logger,
            directory=str(tmp_path),
            tag='dielectric',
            eval_train=False,
            progress_bar=False,
            log_errors='DipolePolarRMSE',
        )
        assert logger.rows
        assert all(np.isfinite(x).all() for x in jax.tree.leaves(final))
    finally:
        for loader in loaders:
            if loader is not None:
                loader.close()
        gin.clear_config()


def test_ase_hessian_and_atomic_stress_with_energy_scaling():
    model, params, _ = gin_model.model(
        r_max=3.0,
        atomic_numbers=(1, 8),
        atomic_energies='zero',
        avg_num_neighbors=2.0,
        num_interactions=1,
        num_bessel=2,
        num_polynomial_cutoff=5,
        max_ell=1,
        hidden_irreps='2x0e',
        MLP_irreps='2x0e',
        correlation=1,
        gate=jax.nn.silu,
        radial_MLP=[3],
        scaling=lambda graphs, energies: (0.2, 1.7),
    )
    atoms = atoms_for_properties()
    graph = graph_for_atoms(atoms)
    expected = model(
        params,
        graph,
        compute_stress=True,
        compute_hessian=True,
        compute_atomic_stresses=True,
    )
    atoms.calc = MACEJAXCalculator(
        model,
        params,
        3.0,
        atomic_numbers=[1, 8],
        energy_units_to_eV=2.0,
        length_units_to_A=3.0,
    )
    np.testing.assert_allclose(
        atoms.calc.get_hessian(atoms),
        expected['hessian'][:9, :3] * 2 / 9,
        rtol=1e-8,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        atoms.calc.get_atomic_virials(atoms),
        expected['atomic_virials'][:3] * 2,
        rtol=1e-8,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        atoms.get_stresses().sum(axis=0), atoms.get_stress(), rtol=1e-8, atol=1e-10
    )


def test_prediction_cli_returns_electric_properties_and_derivatives():
    model, config = converted(source_model('AtomicDielectricMACE'))
    graphdef, state = nnx.split(model)
    predict = _build_predictor(
        graphdef,
        model_config=config,
        compute_forces=None,
        compute_stress=False,
        full_outputs=False,
        compute_dielectric_derivatives=True,
    )
    graph = graph_for_atoms(atoms_for_properties())
    out = jax.jit(predict)(state, graph)
    assert out.get('energy') is None
    assert out['charges'][:3].sum() == pytest.approx(1.0)
    assert out['polarizability'].shape == (2, 3, 3)
    assert out['dmu_dr'].shape == (3, 4, 3)
    assert out['dalpha_dr'].shape == (9, 4, 3)
    assert np.isfinite(out['dalpha_dr']).all()


@pytest.mark.parametrize(
    'kind,loss_name,loss_cls,metrics',
    [
        ('AtomicDipolesMACE', 'dipole_single', modules.DipoleSingleLoss, 'DipoleRMSE'),
        (
            'AtomicDielectricMACE',
            'dipole_polar',
            modules.DipolePolarLoss,
            'DipolePolarRMSE',
        ),
        (
            'EnergyDipolesMACE',
            'dipole',
            modules.WeightedEnergyForcesDipoleLoss,
            'EnergyDipoleRMSE',
        ),
    ],
)
def test_electric_training_cli_and_weight_overrides(kind, loss_name, loss_cls, metrics):
    gin.clear_config()
    try:
        args, _ = mace_jax_train.parse_args(
            [
                '--model',
                kind,
                '--loss',
                loss_name,
                '--swa-loss',
                loss_name,
                '--dipole-key',
                'REF_dipole',
                '--polarizability-key',
                'REF_alpha',
            ]
        )
        mace_jax_train.apply_cli_overrides(args)
        assert gin.query_parameter('model.model_class') == kind
        assert gin.query_parameter('datasets.dipole_key') == 'REF_dipole'
        assert gin.query_parameter('datasets.polarizability_key') == 'REF_alpha'
        assert gin.query_parameter('train.log_errors') == metrics
        assert isinstance(gin_functions.loss(), loss_cls)
        assert mace_jax_train._resolve_swa_loss_factory(args, None) is loss_cls
        # A later weight-only override must retain the configured electric loss.
        weights, _ = mace_jax_train.parse_args(
            ['--dipole-weight', '2.5', '--swa-dipole-weight', '1.5']
        )
        mace_jax_train.apply_cli_overrides(weights)
        loss = gin_functions.loss()
        assert isinstance(loss, loss_cls)
        assert float(loss.dipole_weight) == pytest.approx(2.5)
        assert gin.query_parameter('train.log_errors') == metrics
        assert (
            mace_jax_train._resolve_swa_loss_factory(
                weights, mace_jax_train._collect_swa_loss_kwargs(weights)
            )
            is loss_cls
        )
    finally:
        gin.clear_config()
