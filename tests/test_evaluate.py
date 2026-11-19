"""Evaluation semantics and compilation reuse across training epochs."""

import pickle

import gin
import jax
import jax.numpy as jnp
import jraph
import numpy as np
import optax
import pytest

from mace_jax import tools
from mace_jax.data import GraphEdges, GraphGlobals, GraphNodes
from mace_jax.modules.loss import WeightedEnergyForcesLoss
from mace_jax.tools import gin_functions
from mace_jax.tools.train import SWAConfig, _make_eval_step


class _Loader:
    def __init__(self, energies=(2.0,)):
        graph = jraph.GraphsTuple(
            nodes=GraphNodes(
                positions=jnp.zeros((2, 3)),
                forces=jnp.zeros((2, 3)),
                species=jnp.ones((2,), dtype=jnp.int32),
            ),
            edges=GraphEdges(shifts=jnp.zeros((0, 3)), unit_shifts=jnp.zeros((0, 3))),
            globals=GraphGlobals(
                cell=jnp.eye(3)[None],
                energy=jnp.array([2.0]),
                stress=jnp.zeros((1, 3, 3)),
                weight=jnp.ones((1,)),
            ),
            senders=jnp.zeros((0,), dtype=jnp.int32),
            receivers=jnp.zeros((0,), dtype=jnp.int32),
            n_node=jnp.array([2]),
            n_edge=jnp.array([0]),
        )
        self.graphs = [
            jraph.pad_with_graphs(
                graph._replace(globals=graph.globals._replace(energy=jnp.array([e]))),
                n_node=3,
                n_edge=1,
                n_graph=2,
            )
            for e in energies
        ]

    def iter_batches(self, **kwargs):
        return iter(self.graphs)

    def approx_length(self):
        return len(self.graphs)


class _Logger:
    def __init__(self):
        self.rows = []

    def log(self, metrics):
        self.rows.append(metrics)


@pytest.fixture(autouse=True)
def _reset_gin():
    gin.clear_config()
    yield
    gin.clear_config()


def _predict(params, graph):
    return {
        'energy': jnp.full_like(graph.globals.energy, params['w']),
        'forces': jnp.zeros_like(graph.nodes.forces),
    }


@pytest.mark.parametrize('prefetch_batches', [0, 2])
def test_evaluate_default_step(prefetch_batches):
    loss, metrics = tools.evaluate(
        predictor=_predict,
        params={'w': jnp.array(0.0)},
        loss_fn=WeightedEnergyForcesLoss(energy_weight=1.0, forces_weight=0.0),
        data_loader=_Loader(),
        progress_bar=False,
        device_prefetch_batches=prefetch_batches,
    )
    assert loss == pytest.approx(1.0)
    assert metrics['rmse_e'] == pytest.approx(2.0)


@pytest.mark.parametrize('prefetch_batches', [0, 2])
def test_cached_evaluation_reduces_distinct_graphs(prefetch_batches):
    loader = _Loader(energies=(2.0, 4.0, 8.0))
    # Uneven graph counts on two devices exercise both reduction and padding.
    loader.graphs = [
        jraph.pad_with_graphs(
            jraph.batch([jraph.unpad_with_graphs(g) for g in loader.graphs]),
            n_node=7,
            n_edge=1,
            n_graph=4,
        )
    ]
    loss_fn = WeightedEnergyForcesLoss(energy_weight=1.0, forces_weight=0.0)
    traces = []

    def predictor(params, graph):
        traces.append(None)
        return _predict(params, graph)

    eval_step = _make_eval_step(predictor, loss_fn)
    for value in (0.0, 1.0):
        loss, metrics = tools.evaluate(
            predictor=predictor,
            params={'w': jnp.array(value)},
            loss_fn=loss_fn,
            data_loader=loader,
            progress_bar=False,
            device_prefetch_batches=prefetch_batches,
            eval_step=eval_step,
        )
        mse = np.mean((np.array([2.0, 4.0, 8.0]) - value) ** 2)
        assert loss == pytest.approx(mse / 4.0)
        assert metrics['rmse_e'] == pytest.approx(np.sqrt(mse))
    assert len(traces) == 1


def _run_training(tmp_path, monkeypatch, predictor, *, params=None, swa_config=None):
    loader = _Loader()
    logger = _Logger()

    def epochs(*, params, optimizer_state, **kwargs):
        # Keep optimization out of these tests while exercising real evaluation
        # on changing parameters and repeated batches of the same shape.
        for epoch in range(4):
            current = {**params, 'w': params['w'] + 0.5 * epoch}
            yield epoch, current, optimizer_state, current

    monkeypatch.setattr(gin_functions.tools, 'train', epochs)
    monkeypatch.setattr(
        gin_functions,
        'loss',
        lambda: WeightedEnergyForcesLoss(energy_weight=1.0, forces_weight=0.0),
    )
    gin_functions.train(
        predictor=predictor,
        params={'w': jnp.array(0.0)} if params is None else params,
        optimizer_state={},
        train_loader=loader,
        valid_loader=loader,
        test_loader=None,
        gradient_transform=optax.sgd(0.0),
        max_epochs=3,
        logger=logger,
        directory=str(tmp_path),
        tag='evaluation',
        eval_train=False,
        swa_config=swa_config,
    )
    return [row['loss'] for row in logger.rows if row['mode'] == 'valid']


def test_training_reuses_evaluation_step(tmp_path, monkeypatch):
    traces = []

    def predictor(params, graph):
        traces.append(None)
        return _predict(params, graph)

    losses = _run_training(tmp_path, monkeypatch, predictor)
    np.testing.assert_allclose(losses, [1.0, 0.5625, 0.25, 0.0625])
    assert len(traces) == 1


def test_training_evaluation_preserves_config(tmp_path, monkeypatch):
    def predictor(bundle, graph):
        value = bundle['params']['w'] + bundle['config']['offset']
        return _predict({'w': value}, graph)

    losses = _run_training(
        tmp_path,
        monkeypatch,
        predictor,
        params={'params': {'w': jnp.array(0.0)}, 'config': {'offset': jnp.array(2.0)}},
    )
    np.testing.assert_allclose(losses, [0.0, 0.0625, 0.25, 0.5625])


def test_training_recompiles_evaluation_only_when_loss_changes(tmp_path, monkeypatch):
    traces = []

    def predictor(params, graph):
        traces.append(None)
        return _predict(params, graph)

    losses = _run_training(
        tmp_path,
        monkeypatch,
        predictor,
        swa_config=SWAConfig(
            start_interval=1,
            stage_loss_factory=WeightedEnergyForcesLoss,
            stage_loss_kwargs={'energy_weight': 10.0, 'forces_weight': 0.0},
        ),
    )
    np.testing.assert_allclose(losses, [1.0, 0.5625, 2.5, 0.625])
    assert len(traces) == 2


@pytest.mark.parametrize('eval_train', [False, True])
@pytest.mark.parametrize('resume', [False, True])
def test_evaluation_with_real_training_and_ema(
    tmp_path, monkeypatch, eval_train, resume
):
    """Exercise the optimizer, EMA, and restored config without mocking epochs."""
    loader = _Loader(energies=(2.0,) * jax.local_device_count())
    logger = _Logger()
    optimizer = optax.sgd(0.2)
    params = {'w': jnp.array(0.0, dtype=jnp.float64)}
    config = {'offset': 0.5, 'source': 'initial'}
    start_epoch = 0
    expected_weight = 0.0
    expected_ema = 0.0
    resume_path = None
    if resume:
        start_epoch = 2
        expected_weight = 0.4
        expected_ema = 0.2
        restored_config = {'offset': 1.0, 'source': 'checkpoint'}
        resume_path = tmp_path / 'checkpoint.pkl'
        with resume_path.open('wb') as handle:
            pickle.dump(
                {
                    'epoch': start_epoch,
                    'params': {
                        'params': {'w': jnp.array(expected_weight)},
                        'config': restored_config,
                    },
                    'eval_params': {
                        'params': {'w': jnp.array(expected_ema)},
                        'config': restored_config,
                    },
                    'optimizer_state': optimizer.init(params),
                },
                handle,
            )
    expected_config = restored_config if resume else config

    def predictor(bundle, graph):
        assert bundle['config'] == expected_config
        return _predict(
            {'w': bundle['params']['w'] + bundle['config']['offset']}, graph
        )

    monkeypatch.setattr(
        gin_functions,
        'loss',
        lambda: WeightedEnergyForcesLoss(energy_weight=1.0, forces_weight=0.0),
    )
    epoch, final_bundle = gin_functions.train(
        predictor=predictor,
        params={'params': params, 'config': config},
        optimizer_state=optimizer.init(params),
        train_loader=loader,
        valid_loader=loader,
        test_loader=None,
        gradient_transform=optimizer,
        max_epochs=start_epoch + 3,
        logger=logger,
        directory=str(tmp_path),
        tag='real_training',
        eval_train=eval_train,
        ema_decay=0.5,
        resume_from=None if resume_path is None else str(resume_path),
        progress_bar=False,
    )
    target = 2.0 - expected_config['offset']
    expected_losses = [((expected_ema - target) / 2.0) ** 2]
    for step in range(1, 4):
        expected_weight -= 0.2 * (expected_weight - target) / 2.0
        decay = min(0.5, (1 + step) / (10 + step))
        expected_ema = decay * expected_ema + (1 - decay) * expected_weight
        expected_losses.append(((expected_ema - target) / 2.0) ** 2)
    losses = [row['loss'] for row in logger.rows if row['mode'] == 'valid']
    np.testing.assert_allclose(losses, expected_losses)
    assert epoch == start_epoch + 3
    assert float(final_bundle['params']['w']) == pytest.approx(expected_ema)
    assert final_bundle['config'] == expected_config
