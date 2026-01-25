import pickle
from pathlib import Path

import gin
import h5py
import jax
import jax.numpy as jnp
import jraph
import numpy as np
import optax
import pytest

import mace_jax.data as data_pkg
import mace_jax.data.utils as data_utils
from mace_jax.data import neighborhood
from mace_jax.data.utils import Configuration, graph_from_configuration
from mace_jax.tools import gin_datasets, gin_functions, gin_model

GIN_CONFIG = """
import mace_jax.tools.gin_model
import mace_jax.tools.gin_datasets
import mace_jax.tools.gin_functions

mace_jax.tools.gin_model.model.r_max = 2.5
mace_jax.tools.gin_model.model.num_species = 3
mace_jax.tools.gin_model.model.num_bessel = 2
mace_jax.tools.gin_model.model.num_polynomial_cutoff = 2
mace_jax.tools.gin_model.model.max_ell = 1
mace_jax.tools.gin_model.model.hidden_irreps = '1x0e'
mace_jax.tools.gin_model.model.MLP_irreps = '1x0e'
mace_jax.tools.gin_model.model.num_interactions = 1
mace_jax.tools.gin_datasets.datasets.n_edge = 16
mace_jax.tools.gin_datasets.datasets.r_max = 2.5

mace_jax.tools.gin_functions.flags.seed = 0
mace_jax.tools.gin_functions.optimizer.max_epochs = 1
mace_jax.tools.gin_functions.train.progress_bar = False
"""


def _write_native_hdf5(
    path: Path,
    *,
    num_structures: int,
    seed: int = 0,
    atoms_per_structure: int | None = None,
) -> None:
    rng = np.random.default_rng(seed)
    with h5py.File(path, 'w') as handle:
        batch = handle.create_group('config_batch_0')
        for idx in range(num_structures):
            if atoms_per_structure is None:
                n_atoms = 2 + (idx % 3)
            else:
                n_atoms = int(atoms_per_structure)
            subgroup = batch.create_group(f'config_{idx}')
            numbers = np.asarray(
                [1 + (idx + j) % 3 for j in range(n_atoms)], dtype=np.int32
            )
            positions = rng.normal(scale=0.1, size=(n_atoms, 3)) + idx * 0.01
            subgroup.create_dataset('atomic_numbers', data=numbers)
            subgroup.create_dataset('positions', data=positions)
            subgroup.create_dataset('cell', data=np.eye(3) * 5.0)
            subgroup.create_dataset(
                'pbc', data=np.array([False, False, False], dtype=np.bool_)
            )
            subgroup.create_dataset('weight', data=np.array(1.0, dtype=np.float64))
            subgroup.create_dataset('config_type', data=np.array('Default', dtype='S'))
            subgroup.create_dataset('head', data=np.array('Default', dtype='S'))
            properties = subgroup.create_group('properties')
            properties.create_dataset(
                'energy', data=np.array(float(idx), dtype=np.float64)
            )
            properties.create_dataset('forces', data=np.zeros((n_atoms, 3)))
            properties.create_dataset('stress', data=np.zeros((3, 3)))
            prop_weights = subgroup.create_group('property_weights')
            for key in ('energy', 'forces', 'stress'):
                prop_weights.create_dataset(key, data=np.array(1.0, dtype=np.float64))


@pytest.fixture
def dataset_paths(tmp_path):
    data_dir = tmp_path / 'datasets'
    data_dir.mkdir()
    train = data_dir / 'train.h5'
    valid = data_dir / 'valid.h5'
    test = data_dir / 'test.h5'
    _write_native_hdf5(train, num_structures=6, seed=0)
    _write_native_hdf5(valid, num_structures=2, seed=10)
    _write_native_hdf5(test, num_structures=2, seed=20)
    return {'train': train, 'valid': valid, 'test': test}


@pytest.fixture(autouse=True)
def reset_gin(dataset_paths):
    gin.clear_config()
    gin.parse_config(GIN_CONFIG)
    gin.bind_parameter(
        'mace_jax.tools.gin_datasets.datasets.train_path',
        str(dataset_paths['train']),
    )
    gin.bind_parameter(
        'mace_jax.tools.gin_datasets.datasets.valid_path',
        str(dataset_paths['valid']),
    )
    gin.bind_parameter('mace_jax.tools.gin_datasets.datasets.test_path', None)
    # Keep toy dataset graphs by setting generous size limits.
    gin.bind_parameter('mace_jax.tools.gin_datasets.datasets.n_edge', 128)
    yield
    gin.clear_config()


@pytest.fixture(autouse=True)
def patch_get_neighborhood(monkeypatch):
    def _wrapper(*args, **kwargs):
        return neighborhood.get_neighborhood(*args, **kwargs)

    monkeypatch.setattr(data_utils, 'get_neighborhood', _wrapper)
    monkeypatch.setattr(data_pkg, 'get_neighborhood', _wrapper)
    yield


@pytest.fixture(autouse=True)
def patch_mace_module(monkeypatch):
    class _DummyMACE:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            for key, value in kwargs.items():
                setattr(self, key, value)
            atomic_numbers = tuple(kwargs.get('atomic_numbers', (1,)))
            atomic_energies = kwargs.get('atomic_energies')
            if atomic_energies is None:
                atomic_energies = np.zeros(len(atomic_numbers), dtype=float)
            self.r_max = float(kwargs.get('r_max', 0.0))
            self.num_bessel = int(kwargs.get('num_bessel', 1))
            self.num_polynomial_cutoff = int(kwargs.get('num_polynomial_cutoff', 1))
            self.max_ell = int(kwargs.get('max_ell', 0))
            self.interaction_cls = kwargs.get('interaction_cls')
            self.interaction_cls_first = kwargs.get('interaction_cls_first')
            self.num_interactions = int(kwargs.get('num_interactions', 1))
            self.hidden_irreps = kwargs.get('hidden_irreps', '1x0e')
            self.MLP_irreps = kwargs.get('MLP_irreps', '1x0e')
            self.atomic_numbers = atomic_numbers
            self.atomic_energies = np.asarray(atomic_energies, dtype=float)
            self.avg_num_neighbors = float(kwargs.get('avg_num_neighbors', 0.0))
            self.correlation = kwargs.get('correlation')
            self.radial_type = kwargs.get('radial_type', 'bessel')
            self.pair_repulsion = bool(kwargs.get('pair_repulsion', False))
            self.distance_transform = kwargs.get('distance_transform')
            self.embedding_specs = kwargs.get('embedding_specs')
            self.use_so3 = bool(kwargs.get('use_so3', False))
            self.use_reduced_cg = bool(kwargs.get('use_reduced_cg', True))
            self.use_agnostic_product = bool(kwargs.get('use_agnostic_product', False))
            self.use_last_readout_only = bool(
                kwargs.get('use_last_readout_only', False)
            )
            self.use_embedding_readout = bool(
                kwargs.get('use_embedding_readout', False)
            )
            self.collapse_hidden_irreps = bool(
                kwargs.get('collapse_hidden_irreps', True)
            )
            self.readout_cls = kwargs.get('readout_cls')
            self.gate = kwargs.get('gate')
            self.apply_cutoff = bool(kwargs.get('apply_cutoff', True))
            self.radial_MLP = kwargs.get('radial_MLP')
            self.edge_irreps = kwargs.get('edge_irreps')
            self.heads = kwargs.get('heads')
            self.cueq_config = kwargs.get('cueq_config')

        def init(self, rng, data):
            del rng, data
            return {'dummy': jnp.array(0.0)}

        def apply(
            self,
            params,
            data,
            compute_force: bool = True,
            compute_stress: bool = False,
        ):
            del params
            num_nodes = data['positions'].shape[0]
            energy = jnp.zeros((1,), dtype=data['positions'].dtype)
            forces = jnp.zeros((num_nodes, 3), dtype=data['positions'].dtype)
            stress = (
                jnp.zeros((1, 3, 3), dtype=data['positions'].dtype)
                if compute_stress
                else None
            )
            virials = jnp.zeros((1, 3, 3), dtype=data['positions'].dtype)
            dipole = jnp.zeros((1, 3), dtype=data['positions'].dtype)
            polarizability = jnp.zeros((1, 3, 3), dtype=data['positions'].dtype)
            outputs = {
                'energy': energy,
                'forces': forces,
                'stress': stress,
                'virials': virials,
                'dipole': dipole,
                'polarizability': polarizability,
            }
            return outputs

    monkeypatch.setattr(gin_model.modules, 'MACE', _DummyMACE)
    yield


@pytest.fixture(autouse=True)
def patch_optimizer(monkeypatch):
    def _optimizer(*args, **kwargs):
        return optax.identity(), 1

    monkeypatch.setattr(gin_functions, 'optimizer', _optimizer)
    yield


@pytest.fixture(autouse=True)
def patch_tools_train(monkeypatch):
    def _train(*, params, optimizer_state, **kwargs):
        del kwargs
        yield 0, params, optimizer_state, params

    monkeypatch.setattr(gin_functions.tools, 'train', _train)
    yield


def test_gin_training_smoke(tmp_path):
    _run_gin_training(tmp_path)


def _run_gin_training(tmp_path, **train_kwargs):
    train_loader, valid_loader, test_loader, atomic_energies_dict, r_max = (
        gin_datasets.datasets()
    )

    model_fn, params, num_message_passing = gin_model.model(
        r_max=r_max,
        atomic_energies_dict=atomic_energies_dict,
        train_graphs=train_loader.graphs,
        initialize_seed=0,
    )
    assert num_message_passing == 1

    params = gin_functions.reload(params)

    predictor = jax.jit(
        lambda w, g: model_fn(w, g, compute_force=True, compute_stress=True)
    )

    if gin_functions.checks(predictor, params, train_loader):
        pytest.skip('Sanity checks failed')

    approx_batches = max(1, int(train_loader.approx_length()))
    gradient_transform, max_epochs = gin_functions.optimizer(
        interval_length=approx_batches
    )

    params_for_opt, _ = gin_functions._split_config(params)
    optimizer_state = gradient_transform.init(params_for_opt)

    directory, tag, logger = gin_functions.logs(
        name='test-gin', directory=str(tmp_path)
    )

    interval, ema_params = gin_functions.train(
        predictor=predictor,
        params=params,
        optimizer_state=optimizer_state,
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        gradient_transform=gradient_transform,
        max_epochs=max_epochs,
        logger=logger,
        directory=directory,
        tag=tag,
        patience=None,
        eval_train=False,
        eval_test=False,
        log_errors='PerAtomRMSE',
        **train_kwargs,
    )

    assert ema_params is not None
    return ema_params


def test_multihead_training(tmp_path):
    head_default = tmp_path / 'head_default.h5'
    head_surface = tmp_path / 'head_surface.h5'
    _write_native_hdf5(head_default, num_structures=4, seed=3)
    _write_native_hdf5(head_surface, num_structures=4, seed=7)

    head_cfgs = {
        'Default': {'train_path': str(head_default)},
        'Surface': {'train_path': str(head_surface)},
    }
    gin.bind_parameter('mace_jax.tools.gin_datasets.datasets.head_configs', head_cfgs)
    gin.bind_parameter(
        'mace_jax.tools.gin_datasets.datasets.heads', ['Default', 'Surface']
    )
    gin.bind_parameter('mace_jax.tools.gin_model.model.heads', ['Default', 'Surface'])

    ema_params = _run_gin_training(tmp_path)
    assert ema_params is not None


def test_checkpoint_writes_file(tmp_path):
    ckpt_dir = tmp_path / 'checkpoints'
    _run_gin_training(
        tmp_path,
        checkpoint_dir=str(ckpt_dir),
        checkpoint_every=1,
        checkpoint_keep=2,
    )
    saved = list(ckpt_dir.glob('*.ckpt'))
    assert saved, 'Expected at least one checkpoint to be created.'


def test_resume_from_checkpoint_passes_state(tmp_path, monkeypatch):
    resume_path = tmp_path / 'resume.ckpt'
    expected_params = {'resume': jnp.array(1.0)}
    expected_opt = {'state': 42}
    state = {
        'epoch': 2,
        'params': expected_params,
        'optimizer_state': expected_opt,
        'eval_params': expected_params,
        'lowest_loss': 0.5,
        'patience_counter': 1,
        'checkpoint_format': 2,
    }
    with resume_path.open('wb') as f:
        pickle.dump(state, f)

    captured: dict[str, object] = {}

    def _fake_train(
        *,
        params,
        optimizer_state,
        start_interval=0,
        **kwargs,
    ):
        del kwargs
        captured['params'] = params
        captured['optimizer_state'] = optimizer_state
        captured['start_interval'] = start_interval
        yield start_interval, params, optimizer_state, params

    def _fake_evaluate(**kwargs):
        del kwargs
        return 0.0, {}

    monkeypatch.setattr(gin_functions.tools, 'train', _fake_train)
    monkeypatch.setattr(gin_functions.tools, 'evaluate', _fake_evaluate)

    _run_gin_training(
        tmp_path,
        checkpoint_dir=str(tmp_path / 'resume_out'),
        checkpoint_every=1,
        resume_from=str(resume_path),
    )

    assert captured['params'] == expected_params
    assert captured['optimizer_state'] == expected_opt
    assert captured['start_interval'] == state['epoch']


def test_graph_dataloader_split_by_heads(tmp_path):
    head_A_train = tmp_path / 'head_A_train.h5'
    head_B_train = tmp_path / 'head_B_train.h5'
    head_A_valid = tmp_path / 'head_A_valid.h5'
    head_B_valid = tmp_path / 'head_B_valid.h5'
    _write_native_hdf5(head_A_train, num_structures=4, seed=11)
    _write_native_hdf5(head_B_train, num_structures=4, seed=12)
    _write_native_hdf5(head_A_valid, num_structures=2, seed=21)
    _write_native_hdf5(head_B_valid, num_structures=2, seed=22)

    head_cfgs = {
        'Default': {
            'train_path': str(head_A_train),
            'valid_path': str(head_A_valid),
        },
        'Surface': {
            'train_path': str(head_B_train),
            'valid_path': str(head_B_valid),
        },
    }
    gin.bind_parameter('mace_jax.tools.gin_datasets.datasets.head_configs', head_cfgs)
    gin.bind_parameter(
        'mace_jax.tools.gin_datasets.datasets.heads', ['Default', 'Surface']
    )
    gin.bind_parameter('mace_jax.tools.gin_model.model.heads', ['Default', 'Surface'])
    _, valid_loader, _, _, _ = gin_datasets.datasets()

    per_head = valid_loader.split_by_heads()
    assert set(per_head) == {'Default', 'Surface'}
    for head_name, loader in per_head.items():
        assert loader.heads == (head_name,)
        assert loader.graphs, f'Expected graphs for head {head_name}'
        expected_idx = ['Default', 'Surface'].index(head_name)
        for graph in loader.graphs:
            head_attr = np.asarray(graph.globals.head).reshape(-1)[0]
            assert int(head_attr) == expected_idx


def test_streaming_loader_shard_splits_graphs(tmp_path):
    dataset_path = tmp_path / 'shard_stream.h5'
    _write_native_hdf5(dataset_path, num_structures=6, seed=0, atoms_per_structure=2)
    spec = data_pkg.StreamingDatasetSpec(path=dataset_path)
    atomic_numbers = gin_datasets._unique_atomic_numbers_from_hdf5([dataset_path])
    z_table = data_pkg.AtomicNumberTable(atomic_numbers)
    dataset = data_pkg.HDF5Dataset(dataset_path, mode='r')
    loader = data_pkg.StreamingGraphDataLoader(
        datasets=[dataset],
        dataset_specs=[spec],
        z_table=z_table,
        r_max=2.5,
        n_node=None,
        n_edge=None,
        head_to_index={'Default': 0},
        num_workers=0,
    )

    def _collect_ids(process_index: int) -> set[int]:
        ids: set[int] = set()
        for batch in loader.iter_batches(
            epoch=0,
            seed=0,
            process_count=3,
            process_index=process_index,
        ):
            mask = np.asarray(jraph.get_graph_padding_mask(batch))
            energies = np.asarray(batch.globals.energy).reshape(-1)
            ids.update(int(v) for v in energies[mask])
        return ids

    try:
        ids0 = _collect_ids(0)
        ids1 = _collect_ids(1)
        ids2 = _collect_ids(2)
    finally:
        loader.close()

    assert ids0.isdisjoint(ids1)
    assert ids0.isdisjoint(ids2)
    assert ids1.isdisjoint(ids2)
    assert sorted(ids0 | ids1 | ids2) == list(range(6))


def test_train_evaluates_each_head(monkeypatch, tmp_path):
    head_A_train = tmp_path / 'head_A_train.h5'
    head_B_train = tmp_path / 'head_B_train.h5'
    head_A_valid = tmp_path / 'head_A_valid.h5'
    head_B_valid = tmp_path / 'head_B_valid.h5'
    _write_native_hdf5(head_A_train, num_structures=4, seed=31)
    _write_native_hdf5(head_B_train, num_structures=4, seed=32)
    _write_native_hdf5(head_A_valid, num_structures=2, seed=41)
    _write_native_hdf5(head_B_valid, num_structures=2, seed=42)

    head_cfgs = {
        'Default': {
            'train_path': str(head_A_train),
            'valid_path': str(head_A_valid),
        },
        'Surface': {
            'train_path': str(head_B_train),
            'valid_path': str(head_B_valid),
        },
    }
    gin.bind_parameter('mace_jax.tools.gin_datasets.datasets.head_configs', head_cfgs)
    gin.bind_parameter(
        'mace_jax.tools.gin_datasets.datasets.heads', ['Default', 'Surface']
    )
    gin.bind_parameter('mace_jax.tools.gin_model.model.heads', ['Default', 'Surface'])

    eval_calls: list[tuple[str, tuple[str, ...]]] = []

    def _fake_evaluate(**kwargs):
        loader = kwargs['data_loader']
        name = kwargs['name']
        heads = tuple(getattr(loader, 'heads', ()) or ())
        eval_calls.append((name, heads))
        return 0.0, {}

    monkeypatch.setattr(gin_functions.tools, 'evaluate', _fake_evaluate)
    _run_gin_training(tmp_path)

    valid_calls = [heads for name, heads in eval_calls if name.startswith('valid')]
    assert set(valid_calls) == {('Default',), ('Surface',)}


def test_eval_interval_controls_frequency(monkeypatch, dataset_paths):
    gin.bind_parameter('mace_jax.tools.gin_functions.train.eval_interval', 2)
    gin.bind_parameter(
        'mace_jax.tools.gin_datasets.datasets.valid_path',
        str(dataset_paths['valid']),
    )

    def _multi_interval_train(*, params, optimizer_state, **kwargs):
        for idx in range(3):
            yield idx, params, optimizer_state, params

    monkeypatch.setattr(gin_functions.tools, 'train', _multi_interval_train)
    monkeypatch.setattr(
        gin_functions,
        'optimizer',
        lambda *args, **kwargs: (optax.identity(), 1),
    )

    eval_calls = []

    def _fake_evaluate(**kwargs):
        eval_calls.append(kwargs['name'])
        return 0.0, {}

    monkeypatch.setattr(gin_functions.tools, 'evaluate', _fake_evaluate)
    _run_gin_training(Path(dataset_paths['valid']).parent)
    assert eval_calls.count('valid') == 2


def test_streaming_loader_auto_estimates_caps():
    gin.bind_parameter('mace_jax.tools.gin_datasets.datasets.n_edge', 16)
    train_loader, _, _, _, _ = gin_datasets.datasets()
    assert train_loader._n_node >= 1
    assert train_loader._n_edge >= 1


def test_streaming_loader_preserves_order_without_shuffle(dataset_paths):
    dataset_path = Path(dataset_paths['train'])
    spec = data_pkg.StreamingDatasetSpec(path=dataset_path)
    atomic_numbers = gin_datasets._unique_atomic_numbers_from_hdf5([dataset_path])
    z_table = data_pkg.AtomicNumberTable(atomic_numbers)
    head_to_index = {'Default': 0}
    stats, _ = gin_datasets._compute_streaming_stats(
        dataset_path,
        spec=spec,
        z_table=z_table,
        r_max=2.5,
        head_to_index=head_to_index,
        sample_limit=0,
        edge_cap=16,
        node_percentile=None,
        collect_metadata=False,
        stats_workers=None,
    )
    dataset = data_pkg.HDF5Dataset(dataset_path, mode='r')
    num_graphs = len(dataset)
    try:
        loader = data_pkg.StreamingGraphDataLoader(
            datasets=[dataset],
            dataset_specs=[spec],
            z_table=z_table,
            r_max=2.5,
            n_node=stats.n_nodes,
            n_edge=stats.n_edges,
            head_to_index=head_to_index,
            num_workers=0,
            pad_graphs=stats.n_graphs,
        )
        energies = []
        for batch in loader:
            mask = np.asarray(jraph.get_graph_padding_mask(batch))
            batch_energy = np.asarray(batch.globals.energy).reshape(-1)
            energies.extend(batch_energy[mask].tolist())
    finally:
        dataset.close()

    assert energies == list(range(num_graphs))


def test_streaming_loader_iter_batches_deterministic(tmp_path):
    dataset_path = tmp_path / 'deterministic_stream.h5'
    _write_native_hdf5(dataset_path, num_structures=8, seed=0, atoms_per_structure=2)
    spec = data_pkg.StreamingDatasetSpec(path=dataset_path)
    atomic_numbers = gin_datasets._unique_atomic_numbers_from_hdf5([dataset_path])
    z_table = data_pkg.AtomicNumberTable(atomic_numbers)
    dataset = data_pkg.HDF5Dataset(dataset_path, mode='r')
    loader = data_pkg.StreamingGraphDataLoader(
        datasets=[dataset],
        dataset_specs=[spec],
        z_table=z_table,
        r_max=2.5,
        n_node=None,
        n_edge=None,
        head_to_index={'Default': 0},
        num_workers=0,
    )

    def _collect_batches(epoch, process_count, process_index):
        batches = list(
            loader.iter_batches(
                epoch=epoch,
                seed=123,
                process_count=process_count,
                process_index=process_index,
            )
        )
        assert batches
        return batches

    def _ids_from_batches(batches):
        ids = set()
        for batch in batches:
            mask = np.asarray(jraph.get_graph_padding_mask(batch))
            energies = np.asarray(batch.globals.energy).reshape(-1)
            ids.update(int(v) for v in energies[mask])
        return ids

    def _sequence_signature(batches):
        signatures = []
        for batch in batches:
            mask = np.asarray(jraph.get_graph_padding_mask(batch))
            energies = np.asarray(batch.globals.energy).reshape(-1)
            signatures.append(tuple(sorted(int(v) for v in energies[mask])))
        return signatures

    try:
        rank0_batches = _collect_batches(epoch=0, process_count=2, process_index=0)
        rank1_batches = _collect_batches(epoch=0, process_count=2, process_index=1)
        ids_rank0 = _ids_from_batches(rank0_batches)
        ids_rank1 = _ids_from_batches(rank1_batches)
        assert ids_rank0.isdisjoint(ids_rank1)
        assert sorted(ids_rank0 | ids_rank1) == list(range(8))

        seq_epoch0 = _sequence_signature(
            _collect_batches(epoch=0, process_count=1, process_index=0)
        )
        seq_epoch1 = _sequence_signature(
            _collect_batches(epoch=1, process_count=1, process_index=0)
        )
        assert seq_epoch0 == seq_epoch1
    finally:
        loader.close()


def test_gin_model_torch_checkpoint(monkeypatch, tmp_path):
    import numpy as np
    import torch

    try:
        from mace.tools import scripts_utils
    except Exception:  # pragma: no cover - optional dependency path
        pytest.skip('cuequivariance_ops unavailable in this environment')

    import mace_jax.cli.mace_jax_from_torch as jax_from_torch_mod

    ckpt = tmp_path / 'dummy.pt'
    ckpt.write_bytes(b'checkpoint')

    class _DummyTorchModel:
        heads = ['Default', 'Surface']

        def eval(self):
            return self

    dummy_params = {'param': jnp.array(1.0)}

    class _DummyModule:
        def apply(
            self,
            params,
            data,
            *,
            compute_force: bool = True,
            compute_stress: bool = False,
        ):
            del params
            num_nodes = data['positions'].shape[0]
            energy = jnp.asarray([42.0], dtype=jnp.float64)
            forces = jnp.zeros((num_nodes, 3), dtype=jnp.float64)
            return {'energy': energy, 'forces': forces, 'stress': None}

    def _fake_load(path, map_location=None):
        assert Path(path) == ckpt
        assert map_location == 'cpu'
        return {'model': _DummyTorchModel()}

    def _fake_extract(model):
        assert isinstance(model, _DummyTorchModel)
        return {
            'atomic_numbers': [1],
            'num_interactions': 3,
            'heads': ['Default', 'Surface'],
        }

    def _fake_convert(model, config):
        assert config['heads'] == ['Default', 'Surface']
        return _DummyModule(), dummy_params, None

    monkeypatch.setattr(torch, 'load', _fake_load)
    monkeypatch.setattr(jax_from_torch_mod, 'convert_model', _fake_convert)
    monkeypatch.setattr(scripts_utils, 'extract_config_mace_model', _fake_extract)

    apply_fn, params, num_interactions = gin_model.model(
        r_max=2.0,
        torch_checkpoint=str(ckpt),
        torch_head='Surface',
        torch_param_dtype='float64',
    )

    assert num_interactions == 3
    assert params == dummy_params

    configuration = Configuration(
        atomic_numbers=np.array([1], dtype=int),
        positions=np.zeros((1, 3)),
        energy=np.array(0.0),
        forces=np.zeros((1, 3)),
        stress=np.zeros((3, 3)),
        cell=np.eye(3),
        pbc=(False, False, False),
    )
    graph = graph_from_configuration(configuration, cutoff=2.0)
    outputs = apply_fn(params, graph, compute_force=True, compute_stress=False)
    np.testing.assert_allclose(np.asarray(outputs['energy']), np.array([42.0]))
    np.testing.assert_allclose(
        np.asarray(outputs['forces']), np.zeros((graph.nodes.positions.shape[0], 3))
    )


@pytest.mark.parametrize(
    'loss_config',
    [
        'loss = @mace_jax.modules.loss.WeightedEnergyForcesLoss()',
        'loss = @mace_jax.modules.loss.WeightedForcesLoss()',
        'loss = @mace_jax.modules.loss.WeightedHuberEnergyForcesStressLoss()',
        'loss = @mace_jax.modules.loss.WeightedEnergyForcesL1L2Loss()',
        'loss = @mace_jax.modules.loss.WeightedEnergyForcesVirialsLoss()',
        'loss = @mace_jax.modules.loss.WeightedEnergyForcesDipoleLoss()',
        'loss = @mace_jax.modules.loss.UniversalLoss()',
        'loss = @mace_jax.modules.loss.DipoleSingleLoss()',
        'loss = @mace_jax.modules.loss.DipolePolarLoss()',
    ],
)
def test_gin_training_with_alternate_losses(tmp_path, loss_config):
    gin.parse_config('import mace_jax.modules.loss')
    gin.parse_config(loss_config)
    _run_gin_training(tmp_path)
