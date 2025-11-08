import shutil
from pathlib import Path

import gin
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from ase import io as ase_io

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
mace_jax.tools.gin_datasets.datasets.train_path = 'tests/test_data/simple.xyz'
mace_jax.tools.gin_datasets.datasets.valid_fraction = 0.0
mace_jax.tools.gin_datasets.datasets.test_path = None
mace_jax.tools.gin_datasets.datasets.n_node = 8
mace_jax.tools.gin_datasets.datasets.n_edge = 16
mace_jax.tools.gin_datasets.datasets.n_graph = 2
mace_jax.tools.gin_datasets.datasets.min_n_graph = 2
mace_jax.tools.gin_datasets.datasets.r_max = 2.5

mace_jax.tools.gin_functions.flags.seed = 0
mace_jax.tools.gin_functions.optimizer.steps_per_interval = 1
mace_jax.tools.gin_functions.optimizer.max_num_intervals = 1
mace_jax.tools.gin_functions.train.progress_bar = False
"""


@pytest.fixture(autouse=True)
def reset_gin():
    gin.clear_config()
    gin.parse_config(GIN_CONFIG)
    data_path = Path(__file__).resolve().parent / 'test_data' / 'simple.xyz'
    gin.bind_parameter(
        'mace_jax.tools.gin_datasets.datasets.train_path', str(data_path)
    )
    gin.bind_parameter('mace_jax.tools.gin_datasets.datasets.test_path', None)
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
    def _optimizer():
        return optax.identity(), 1, 1

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


def _run_gin_training(tmp_path):
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

    gradient_transform, steps_per_interval, max_num_intervals = (
        gin_functions.optimizer()
    )
    assert steps_per_interval == 1
    assert max_num_intervals == 1

    optimizer_state = gradient_transform.init(params)

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
        max_num_intervals=max_num_intervals,
        steps_per_interval=steps_per_interval,
        logger=logger,
        directory=directory,
        tag=tag,
        patience=None,
        eval_train=False,
        eval_test=False,
        log_errors='PerAtomRMSE',
    )

    assert ema_params is not None
    return ema_params


def test_multihead_training(tmp_path):
    simple_xyz = Path(__file__).resolve().parent / 'test_data' / 'simple.xyz'
    head_default = tmp_path / 'head_default.xyz'
    head_surface = tmp_path / 'head_surface.xyz'
    shutil.copy(simple_xyz, head_default)
    shutil.copy(simple_xyz, head_surface)

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


def test_graph_dataloader_split_by_heads(tmp_path):
    simple_xyz = Path(__file__).resolve().parent / 'test_data' / 'simple.xyz'
    head_cfgs = {
        'Default': {'train_path': str(simple_xyz), 'valid_fraction': 0.5},
        'Surface': {'train_path': str(simple_xyz), 'valid_fraction': 0.5},
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


def test_train_evaluates_each_head(monkeypatch, tmp_path):
    simple_xyz = Path(__file__).resolve().parent / 'test_data' / 'simple.xyz'
    head_cfgs = {
        'Default': {'train_path': str(simple_xyz), 'valid_fraction': 0.5},
        'Surface': {'train_path': str(simple_xyz), 'valid_fraction': 0.5},
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

    valid_calls = [heads for name, heads in eval_calls if name.startswith('eval_valid')]
    assert set(valid_calls) == {('Default',), ('Surface',)}


def test_head_replay_paths_and_weights(tmp_path):
    simple_xyz = Path(__file__).resolve().parent / 'test_data' / 'simple.xyz'
    replay_xyz = tmp_path / 'replay.xyz'
    shutil.copy(simple_xyz, replay_xyz)
    n_base = len(ase_io.read(simple_xyz, ':'))

    head_cfgs = {
        'Default': {
            'train_path': str(simple_xyz),
            'replay_paths': [str(replay_xyz)],
            'replay_weight': 0.25,
        }
    }
    gin.bind_parameter('mace_jax.tools.gin_datasets.datasets.head_configs', head_cfgs)
    gin.bind_parameter('mace_jax.tools.gin_datasets.datasets.heads', ['Default'])
    train_loader, _, _, _, _ = gin_datasets.datasets()

    assert len(train_loader.graphs) == 2 * n_base
    replay_weight = float(train_loader.graphs[-1].globals.weight[0])
    assert np.isclose(replay_weight, 0.25)


def test_head_pseudolabel_replay(monkeypatch, tmp_path):
    simple_xyz = Path(__file__).resolve().parent / 'test_data' / 'simple.xyz'
    unlabeled_xyz = tmp_path / 'unlabeled.xyz'
    atoms = ase_io.read(simple_xyz, ':')
    for at in atoms:
        at.info.pop('energy', None)
    ase_io.write(unlabeled_xyz, atoms)

    def fake_loader(*_, **__):
        def _predict(graph):
            n_atoms = int(graph.n_node[0])
            return {
                'energy': np.asarray([99.0]),
                'forces': np.ones((n_atoms, 3)),
                'stress': np.zeros((1, 3, 3)),
            }

        return _predict

    monkeypatch.setattr(
        gin_datasets, '_load_pseudolabel_predictor', fake_loader, raising=True
    )

    head_cfgs = {
        'Default': {
            'train_path': str(simple_xyz),
            'replay_paths': [str(unlabeled_xyz)],
            'replay_allow_unlabeled': True,
            'pseudolabel_checkpoint': 'dummy.pt',
            'pseudolabel_targets': 'replay',
        }
    }
    gin.bind_parameter('mace_jax.tools.gin_datasets.datasets.head_configs', head_cfgs)
    gin.bind_parameter('mace_jax.tools.gin_datasets.datasets.heads', ['Default'])

    train_loader, _, _, _, _ = gin_datasets.datasets()
    replay_graph = train_loader.graphs[-1]
    assert np.isclose(float(replay_graph.globals.energy[0]), 99.0)
    forces = np.asarray(replay_graph.nodes.forces)
    assert forces.shape[1] == 3 and np.allclose(forces, 1.0)


def test_gin_model_torch_checkpoint(monkeypatch, tmp_path):
    import numpy as np
    import torch
    from mace.tools import scripts_utils

    import mace_jax.cli.mace_torch2jax as torch2jax_mod

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
            assert compute_force
            assert not compute_stress
            assert 'head' in data and data['head'].shape == (1,)
            energy = jnp.asarray([42.0], dtype=jnp.float64)
            forces = jnp.zeros((data['positions'].shape[0], 3), dtype=jnp.float64)
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
    monkeypatch.setattr(torch2jax_mod, 'convert_model', _fake_convert)
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
