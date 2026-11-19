"""Evaluation semantics and compilation reuse across training epochs."""

import gin
import jax.numpy as jnp
import jraph
import numpy as np
import optax
import pytest

from mace_jax import tools
from mace_jax.data import GraphEdges, GraphGlobals, GraphNodes
from mace_jax.modules.loss import WeightedEnergyForcesLoss
from mace_jax.tools import gin_functions
from mace_jax.tools.train import SWAConfig


class _Loader:
    def __init__(self):
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
        self.graphs = [jraph.pad_with_graphs(graph, n_node=3, n_edge=0, n_graph=2)]

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
