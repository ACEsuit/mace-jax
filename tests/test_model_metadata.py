"""Model metadata must survive the same loaders and padding used in training."""

import json
import pickle

import ase
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
from e3nn import o3
from mace.modules import blocks, models

from mace_jax import data, tools
from mace_jax.calculators.mace import MACEJAXCalculator
from mace_jax.cli import mace_jax_predict, mace_jax_preprocess, mace_jax_train
from mace_jax.data.utils import save_configurations_as_HDF5
from mace_jax.modules.loss import (
    WeightedEnergyForcesLoss,
    WeightedEnergyForcesVirialsLoss,
)
from mace_jax.tools import gin_datasets, gin_functions, gin_model, model_builder
from mace_jax.tools.train import SWAConfig
from tests.test_bundle_roundtrip import _small_config
from tests.test_evaluate import _Loader, _Logger

SPECS = {
    'charge': dict(
        type='continuous', per='graph', in_dim=1, emb_dim=2, key='REF_charge'
    ),
    'spin': dict(type='categorical', per='graph', num_classes=3, offset=-1, emb_dim=2),
    'field': dict(type='continuous', per='graph', in_dim=3, emb_dim=2),
    'site': dict(type='categorical', per='node', num_classes=3, emb_dim=2),
    'descriptor': dict(type='continuous', per='node', in_dim=2, emb_dim=2),
}


def _atoms(charge=0.0):
    atoms = ase.Atoms(
        'H2', positions=[[0.0, 0.0, 0.0], [1.0, 0.2, 0.1]], cell=[6, 6, 6]
    )
    atoms.info.update(
        REF_charge=charge,
        spin=2,
        field=np.array([0.1, 0.2, 0.3]),
        energy=0.0,
        virials=np.zeros((3, 3)),
    )
    atoms.set_array('site', np.array([0, 2]))
    atoms.set_array('descriptor', np.array([[0.2, 0.4], [0.3, 0.1]]))
    return atoms


def _graph(atoms, specs=None):
    config = data.config_from_atoms(atoms, embedding_specs=specs)
    return data.graph_from_configuration(
        config, cutoff=3.0, z_table=data.AtomicNumberTable([1])
    )


def _model(specs=None):
    return gin_model.model(
        r_max=3.0,
        num_species=1,
        atomic_numbers=(1,),
        atomic_energies='zero',
        avg_num_neighbors=1.0,
        num_interactions=1,
        num_bessel=2,
        num_polynomial_cutoff=5,
        max_ell=1,
        correlation=1,
        gate=jax.nn.silu,
        hidden_irreps='2x0e',
        MLP_irreps='2x0e',
        embedding_specs=specs,
    )


def test_features_survive_xyz_hdf5_streaming_and_native_prediction(tmp_path):
    atoms = [_atoms(0.0), _atoms(1.0)]
    xyz = tmp_path / 'inputs.xyz'
    ase.io.write(xyz, atoms)
    _, configs = data.load_from_xyz(xyz, embedding_specs=SPECS)
    path = tmp_path / 'inputs.h5'
    with h5py.File(path, 'w') as handle:
        save_configurations_as_HDF5(configs, 0, handle)
    loader = data.get_hdf5_dataloader(
        data_file=path,
        atomic_numbers=data.AtomicNumberTable([1]),
        r_max=3.0,
        max_nodes=5,
        max_edges=8,
        max_graphs=3,
        embedding_specs=SPECS,
        num_workers=0,
        prefetch_batches=0,
    )
    try:
        graph = next(iter(loader))
    finally:
        loader.close()
    model_data = gin_model._graph_to_data(graph, num_species=1)
    np.testing.assert_array_equal(model_data['charge'][:2], [[0.0], [1.0]])
    np.testing.assert_array_equal(
        model_data['spin'], [2, 2, 2]
    )  # valid padding category
    np.testing.assert_array_equal(model_data['site'][:4], [0, 2, 0, 2])
    assert model_data['field'].shape == (3, 3)
    assert model_data['descriptor'].shape == (5, 2)
    # Device splitting must recurse into feature dictionaries when padding graphs.
    shards = data.split_graphs_for_devices(graph, 2)
    np.testing.assert_array_equal(shards[0].globals.features['charge'], [[0.0], [1.0]])
    assert shards[1].globals.features['field'].shape == (2, 3)
    predictor, params, _ = _model(SPECS)
    batched = jax.jit(predictor)(params, graph)
    individual = []
    for frame in atoms:
        single = jraph.pad_with_graphs(_graph(frame, SPECS), 3, 8, 2)
        individual.append(predictor(params, single)['energy'][0])
    np.testing.assert_allclose(
        batched['energy'][:2], individual, rtol=1e-10, atol=1e-10
    )
    assert abs(float(individual[0] - individual[1])) > 1e-8
    assert np.isfinite(batched['forces']).all()
    kwargs = dict(
        predictor=predictor,
        params=params,
        model_config=predictor.model_config,
        head_name='Default',
        head_to_index={'Default': 0},
        batch_max_edges=8,
        batch_max_nodes=5,
        batch_max_graphs=3,
        prefetch_batches=0,
        progress_bar=False,
    )
    for source, predict, extra in (
        (xyz, mace_jax_predict._predict_xyz, {}),
        (path, mace_jax_predict._predict_hdf5, {'num_workers': 0}),
    ):
        ids, predictions = predict(source, **kwargs, **extra)
        assert ids == [0, 1]
        np.testing.assert_allclose(
            np.asarray(predictions['energy']).reshape(-1), individual
        )
    # Inference through ASE discovers the same specification from the predictor.
    atoms[0].calc = MACEJAXCalculator(predictor, params, 3.0, atomic_numbers=[1])
    np.testing.assert_allclose(atoms[0].get_potential_energy(), individual[0])
    atoms[0].info['REF_charge'] = 1.0
    np.testing.assert_allclose(atoms[0].get_potential_energy(), individual[1])
    atoms[0].arrays['descriptor'][0, 0] += 1.0
    expected = predictor(
        params, jraph.pad_with_graphs(_graph(atoms[0].copy(), SPECS), 3, 8, 2)
    )
    np.testing.assert_allclose(atoms[0].get_potential_energy(), expected['energy'][0])
    atoms[0].info['field'][0] += 1.0
    expected = predictor(
        params, jraph.pad_with_graphs(_graph(atoms[0].copy(), SPECS), 3, 8, 2)
    )
    np.testing.assert_allclose(atoms[0].get_potential_energy(), expected['energy'][0])


def test_preprocessing_cli_preserves_configured_inputs(tmp_path):
    xyz = tmp_path / 'inputs.xyz'
    ase.io.write(xyz, [_atoms(float(i)) for i in range(3)])
    args = tools.build_preprocess_arg_parser().parse_args(
        [
            '--train_file',
            str(xyz),
            '--h5_prefix',
            str(tmp_path / 'processed'),
            '--num_process',
            '1',
            '--embedding-specs',
            json.dumps(SPECS),
        ]
    )
    mace_jax_preprocess.run(args)
    paths = list((tmp_path / 'processed').glob('*/*.h5'))
    assert paths
    for path in paths:
        with data.HDF5Dataset(path) as dataset:
            config = data.config_from_atoms(dataset[0], embedding_specs=SPECS)
            assert set(config.graph_features) == {'charge', 'spin', 'field'}
            assert set(config.node_features) == {'site', 'descriptor'}


@pytest.mark.parametrize(
    'change,match',
    [
        ('missing', 'Missing graph embedding'),
        ('range', 'class range'),
        ('shape', 'must have shape'),
        ('nan', 'finite numeric'),
        ('reserved', 'conflicts'),
    ],
)
def test_invalid_features_fail_before_model_execution(change, match):
    atoms = _atoms()
    specs = dict(SPECS)
    if change == 'missing':
        del atoms.info['REF_charge']
    if change == 'range':
        atoms.info['spin'] = 0
    if change == 'shape':
        atoms.info['field'] = np.ones(2)
    if change == 'nan':
        atoms.info['REF_charge'] = np.nan
    if change == 'reserved':
        specs['positions'] = specs['charge']
    with pytest.raises(ValueError, match=match):
        _graph(atoms, specs)


def test_no_embedding_spec_ignores_unrelated_metadata():
    graph = _graph(_atoms())
    assert graph.nodes.features is None
    assert graph.globals.features is None


@pytest.mark.parametrize('value', [np.uint8(2), np.array(2, dtype=np.uint16)])
def test_unsigned_categorical_features_with_negative_offset(value):
    atoms = _atoms()
    atoms.info['spin'] = value
    graph = _graph(atoms, SPECS)
    np.testing.assert_array_equal(graph.globals.features['spin'], [2])


def test_categorical_inputs_cannot_overflow_the_model_index_dtype():
    atoms = _atoms()
    atoms.info['spin'] = np.uint64(2**32 + 1)
    specs = {
        'spin': dict(
            type='categorical', per='graph', emb_dim=2, num_classes=3, offset=-(2**32)
        )
    }
    with pytest.raises(ValueError, match='must fit in int32'):
        _graph(atoms, specs)


def test_native_mace_hdf5_properties_supply_embeddings(tmp_path):
    config = data.config_from_atoms(_atoms(), embedding_specs=SPECS)
    path = tmp_path / 'native.h5'
    with h5py.File(path, 'w') as handle:
        save_configurations_as_HDF5([config], 0, handle)
        entry = handle['config_batch_0/config_0']
        for scope in ('node_features', 'graph_features'):
            for name in list(entry[scope]):
                entry.move(f'{scope}/{name}', f'properties/{name}')
            del entry[scope]
    with data.HDF5Dataset(path) as dataset:
        actual = data.config_from_atoms(dataset[0], embedding_specs=SPECS)
    for scope in ('node_features', 'graph_features'):
        for name, value in getattr(config, scope).items():
            np.testing.assert_array_equal(getattr(actual, scope)[name], value)


@pytest.mark.parametrize('checkpoint', [False, True])
def test_gin_datasets_infers_model_features(tmp_path, checkpoint):
    path = tmp_path / 'inputs.h5'
    configs = [
        data.config_from_atoms(_atoms(float(i)), embedding_specs=SPECS)
        for i in range(3)
    ]
    with h5py.File(path, 'w') as handle:
        save_configurations_as_HDF5(configs, 0, handle)
    gin.clear_config()
    loaders = []
    try:
        if checkpoint:
            model = torch.nn.Module()
            model.embedding_specs = SPECS
            model_path = tmp_path / 'model.pt'
            torch.save(model, model_path)
            gin.bind_parameter(
                'mace_jax.tools.gin_model.model.torch_checkpoint', str(model_path)
            )
        else:
            gin.bind_parameter('mace_jax.tools.gin_model.model.embedding_specs', SPECS)
        train, valid, test, _, _ = gin_datasets.datasets(
            train_path=str(path),
            r_max=3.0,
            n_edge=8,
            num_workers=0,
            stream_train_stats_workers=0,
            atomic_energies_override={1: 0.0},
            stream_train_prefetch=0,
        )
        loaders = [train, valid, test]
        assert set(train.graphs[0].globals.features) == {'charge', 'spin', 'field'}
    finally:
        for loader in loaders:
            if loader is not None:
                loader.close()
        gin.clear_config()


def test_converted_conditioned_model_matches_torch(tmp_path):
    specs = {name: SPECS[name] for name in ('charge', 'spin', 'site', 'descriptor')}
    source_specs = {
        name: {key: value for key, value in spec.items() if key != 'key'}
        for name, spec in specs.items()
    }
    torch_model = models.ScaleShiftMACE(
        r_max=3.0,
        num_bessel=2,
        num_polynomial_cutoff=5,
        max_ell=1,
        num_interactions=1,
        num_elements=1,
        hidden_irreps=o3.Irreps('2x0e'),
        MLP_irreps=o3.Irreps('2x0e'),
        atomic_numbers=[1],
        atomic_energies=np.zeros(1),
        interaction_cls=blocks.RealAgnosticResidualInteractionBlock,
        interaction_cls_first=blocks.RealAgnosticResidualInteractionBlock,
        avg_num_neighbors=1.0,
        correlation=1,
        gate=torch.nn.functional.silu,
        embedding_specs=source_specs,
        atomic_inter_scale=1.2,
        atomic_inter_shift=0.3,
    )
    checkpoint = tmp_path / 'conditioned.model'
    torch.save(torch_model, checkpoint)
    model, params, _ = gin_model.model(
        r_max=3.0,
        torch_checkpoint=str(checkpoint),
        embedding_specs=dict(reversed(list(specs.items()))),
    )
    assert list(model.model_config['embedding_specs']) == list(source_specs)
    assert model.model_config['embedding_specs']['charge']['key'] == 'REF_charge'
    with pytest.raises(ValueError, match='must match the Torch checkpoint'):
        gin_model.model(r_max=3.0, torch_checkpoint=str(checkpoint), embedding_specs={})
    graph = jraph.pad_with_graphs(
        jraph.batch([_graph(_atoms(q), specs) for q in (0.0, 1.0)]), 5, 8, 3
    )
    inputs = gin_model._graph_to_data(graph, num_species=1)
    torch_inputs = {
        key: torch.as_tensor(np.array(value)) for key, value in inputs.items()
    }
    # Torch does not support Jraph's zero-length dummy edges in force gradients.
    edge_mask = np.asarray(jraph.get_edge_padding_mask(graph))
    torch_inputs['edge_index'] = torch_inputs['edge_index'][:, edge_mask]
    for key in ('shifts', 'unit_shifts'):
        torch_inputs[key] = torch_inputs[key][edge_mask]
    # Torch accepts graph scalars as [B]; loaders keep an explicit feature axis.
    torch_inputs['charge'] = torch_inputs['charge'].reshape(-1)
    expected = torch_model(torch_inputs, compute_force=True, compute_virials=True)
    actual = jax.jit(lambda p, g: model(p, g, compute_virials=True))(params, graph)
    for key, count in (('energy', 2), ('forces', 4), ('virials', 2)):
        np.testing.assert_allclose(
            actual[key][:count],
            expected[key].detach().numpy()[:count],
            rtol=1e-6,
            atol=1e-8,
        )


def test_virial_only_training_has_nonzero_finite_gradients():
    graph = jraph.pad_with_graphs(_graph(_atoms()), 3, 3, 2)
    model, params, _ = _model()
    loss = WeightedEnergyForcesVirialsLoss(
        energy_weight=0.0, forces_weight=0.0, virials_weight=1.0
    )
    # Exercise switching from an energy-only predictor to a virial loss.
    predictor = mace_jax_train._build_predictor(
        model, WeightedEnergyForcesLoss(energy_weight=1.0, forces_weight=0.0)
    )
    assert predictor(params, graph)['virials'] is None
    predictor = predictor.for_loss(loss)
    output = predictor(params, graph)
    assert output['stress'] is None
    assert np.linalg.norm(output['virials'][0]) > 0
    value, gradient = jax.value_and_grad(
        lambda p: loss(graph, predictor({**params, 'params': p}, graph)).sum()
    )(params['params'])
    assert float(value) > 0
    leaves = jax.tree.leaves(gradient)
    assert all(np.isfinite(x).all() for x in leaves)
    assert sum(float(jnp.sum(x**2)) for x in leaves) > 0
    with pytest.raises(ValueError, match='requires virial predictions'):
        loss(graph, {'energy': output['energy']})


@pytest.mark.parametrize('resume', [False, True])
def test_swa_enables_virial_predictions(tmp_path, monkeypatch, resume):
    loader = _Loader(energies=(2.0,) * jax.local_device_count())
    loader.graphs = [
        g._replace(globals=g.globals._replace(virials=jnp.zeros_like(g.globals.stress)))
        for g in loader.graphs
    ]
    initial_loss = WeightedEnergyForcesLoss(energy_weight=1.0, forces_weight=0.0)
    monkeypatch.setattr(gin_functions, 'loss', lambda: initial_loss)

    def model(params, graph, *, compute_force, compute_stress, compute_virials=False):
        return {
            'energy': jnp.full_like(graph.globals.energy, params['w']),
            'forces': None,
            'stress': None,
            'virials': jnp.full_like(graph.globals.virials, params['w'])
            if compute_virials
            else None,
        }

    params = {'w': jnp.array(0.4)}
    optimizer = optax.sgd(0.2)
    checkpoint = None
    if resume:
        checkpoint = tmp_path / 'resume.ckpt'
        checkpoint.write_bytes(
            pickle.dumps(
                dict(epoch=1, params=params, optimizer_state=optimizer.init(params))
            )
        )
    logger = _Logger()
    _, final = gin_functions.train(
        predictor=mace_jax_train._build_predictor(model, initial_loss),
        params=params,
        optimizer_state=optimizer.init(params),
        train_loader=loader,
        valid_loader=loader,
        test_loader=None,
        gradient_transform=optimizer,
        max_epochs=3,
        logger=logger,
        directory=str(tmp_path),
        tag='virials',
        eval_train=False,
        progress_bar=False,
        resume_from=str(checkpoint) if resume else None,
        swa_config=SWAConfig(
            start_interval=1,
            prefer_swa_params=False,
            stage_loss_factory=WeightedEnergyForcesVirialsLoss,
            stage_loss_kwargs=dict(
                energy_weight=0.0, forces_weight=0.0, virials_weight=1.0
            ),
        ),
    )
    weight = 0.4
    expected = [weight**2 / 4 if resume else (weight - 2) ** 2 / 4]
    if not resume:
        weight -= 0.2 * (weight - 2) / 2
        expected.append((weight - 2) ** 2 / 4)
    for _ in range(2):
        weight -= 0.2 * weight / 2
        expected.append(weight**2 / 4)
    np.testing.assert_allclose(
        [row['loss'] for row in logger.rows if row['mode'] == 'valid'], expected
    )
    assert float(final['w']) == pytest.approx(weight)


@pytest.mark.parametrize('values', [[1.0, 2.0], [0.0], [-1.0], [np.nan], [np.inf]])
def test_invalid_neighbor_normalizations_fail(values):
    config = _small_config()
    config['interaction_avg_num_neighbors'] = values
    with pytest.raises(ValueError, match='normalizations|one value per interaction'):
        model_builder._build_jax_model(config)
