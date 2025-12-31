import jax
import jax.numpy as jnp
import jraph
import numpy as np
import optax
import pytest

from mace_jax.data import (
    AtomicNumberTable,
    Configuration,
    HDF5Dataset,
    StreamingDatasetSpec,
    StreamingGraphDataLoader,
)
from mace_jax.tools.train import train as train_loop
from tests.conftest import _write_hdf5_from_configs

jax.config.update('jax_enable_x64', True)


@pytest.fixture(autouse=True)
def _force_single_device(monkeypatch):
    """Keep unit tests deterministic on multi-device hosts."""
    monkeypatch.setattr(jax, 'local_device_count', lambda: 1)


def _make_streaming_loader(tmp_path, targets: list[float]) -> StreamingGraphDataLoader:
    configs = [
        Configuration(
            atomic_numbers=np.array([1], dtype=np.int32),
            positions=np.zeros((1, 3), dtype=np.float64),
            energy=float(target),
            forces=np.zeros((1, 3), dtype=np.float64),
            stress=np.zeros((3, 3), dtype=np.float64),
        )
        for target in targets
    ]
    dataset_path = tmp_path / 'train_controls_targets.h5'
    _write_hdf5_from_configs(dataset_path, configs)
    dataset = HDF5Dataset(dataset_path, mode='r')
    assignments = [[[idx] for idx in range(len(configs))]]
    return StreamingGraphDataLoader(
        datasets=[dataset],
        dataset_specs=[StreamingDatasetSpec(path=dataset_path)],
        z_table=AtomicNumberTable([1]),
        r_max=2.5,
        n_node=None,
        n_edge=None,
        head_to_index={'Default': 0},
        shuffle=False,
        seed=0,
        num_workers=0,
        batch_assignments=assignments,
    )


def _loss_fn(params, graph: jraph.GraphsTuple) -> jnp.ndarray:
    graph_mask = jraph.get_graph_padding_mask(graph).astype(jnp.float64)
    targets = jnp.asarray(graph.globals.energy, dtype=jnp.float64)
    prediction = params['w'][0]
    preds = jnp.full_like(targets, prediction)
    diff = (preds - targets) * graph_mask
    return 0.5 * diff * diff


def _run_training(tmp_path, targets, *, steps, learning_rate=0.3, **train_kwargs):
    params = {'w': jnp.zeros((1,), dtype=jnp.float64)}
    gradient_transform = optax.sgd(learning_rate=learning_rate)
    optimizer_state = gradient_transform.init(params)
    loader = _make_streaming_loader(tmp_path, targets)

    try:
        trainer = train_loop(
            params=params,
            total_loss_fn=_loss_fn,
            train_loader=loader,
            gradient_transform=gradient_transform,
            optimizer_state=optimizer_state,
            progress_bar=False,
            **train_kwargs,
        )

        final_eval = None
        for interval, _, _, eval_params in trainer:
            if interval == steps:
                final_eval = float(eval_params['w'][0])
                break
    finally:
        loader.close()

    assert final_eval is not None
    return final_eval


def test_gradient_clipping_limits_update_magnitude(tmp_path):
    unclipped = _run_training(tmp_path, [1.0], steps=1, max_grad_norm=None)
    clipped = _run_training(tmp_path, [1.0], steps=1, max_grad_norm=0.1)

    assert np.isclose(unclipped, 0.3, atol=1e-6)
    assert np.isclose(clipped, 0.03, atol=1e-6)
    assert clipped < unclipped


def _simulate_params(targets, learning_rate):
    value = 0.0
    history = []
    for target in targets:
        grad = value - target
        value = value - learning_rate * grad
        history.append(value)
    return history


def _expected_ema(raw_params, ema_decay):
    ema = 0.0
    for idx, value in enumerate(raw_params, start=1):
        decay = min(ema_decay, (1 + idx) / (10 + idx))
        ema = ema * decay + value * (1 - decay)
    return ema


def test_ema_eval_uses_smoothed_parameters(tmp_path):
    targets = [1.0, -0.5]
    ema_decay = 0.5
    learning_rate = 0.3
    steps = len(targets)
    expected = _expected_ema(
        _simulate_params(targets * steps, learning_rate),
        ema_decay,
    )

    eval_param = _run_training(
        tmp_path,
        targets,
        steps=steps,
        learning_rate=learning_rate,
        ema_decay=ema_decay,
    )

    assert np.isclose(eval_param, expected, atol=1e-8)


def test_schedule_free_eval_fn_overrides_eval_params(tmp_path):
    params = {'w': jnp.zeros((1,), dtype=jnp.float64)}
    gradient_transform = optax.sgd(learning_rate=0.0)
    optimizer_state = gradient_transform.init(params)
    loader = _make_streaming_loader(tmp_path, [0.0])

    eval_marker = jnp.asarray([42.0], dtype=jnp.float64)

    def _eval_fn(state, current_params):
        assert state is not None
        assert 'w' in current_params
        return {'w': eval_marker}

    try:
        trainer = train_loop(
            params=params,
            total_loss_fn=_loss_fn,
            train_loader=loader,
            gradient_transform=gradient_transform,
            optimizer_state=optimizer_state,
            progress_bar=False,
            schedule_free_eval_fn=_eval_fn,
        )

        epoch_zero = next(trainer)
        assert epoch_zero[3]['w'][0] == eval_marker[0]
        epoch_one = next(trainer)
        assert epoch_one[3]['w'][0] == eval_marker[0]
    finally:
        loader.close()
