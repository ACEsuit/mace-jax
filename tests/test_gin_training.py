from pathlib import Path

import gin
import jax
import jax.numpy as jnp
import optax
import pytest

import mace_jax.data as data_pkg
import mace_jax.data.utils as data_utils
from mace_jax.data import neighborhood
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
            return {'energy': energy, 'forces': forces, 'stress': stress}

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
