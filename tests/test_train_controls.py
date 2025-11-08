import jax
import jax.numpy as jnp
import jraph
import numpy as np
import optax

from mace_jax.tools.train import train as train_loop


jax.config.update('jax_enable_x64', True)


def _make_graph(target: float) -> jraph.GraphsTuple:
    base = jraph.GraphsTuple(
        nodes=jnp.zeros((1, 1), dtype=jnp.float64),
        senders=jnp.array([0], dtype=jnp.int32),
        receivers=jnp.array([0], dtype=jnp.int32),
        edges=jnp.zeros((1, 1), dtype=jnp.float64),
        n_node=jnp.array([1], dtype=jnp.int32),
        n_edge=jnp.array([1], dtype=jnp.int32),
        globals={'target': jnp.array([target], dtype=jnp.float64)},
    )
    padded = jraph.pad_with_graphs(
        base,
        n_node=2,
        n_edge=2,
        n_graph=2,
    )
    padded_targets = jnp.asarray(padded.globals['target'], dtype=jnp.float64)
    padded_targets = padded_targets.at[1].set(0.0)
    return padded._replace(globals={'target': padded_targets})


class _Loader:
    def __init__(self, targets: list[float]):
        self._targets = list(targets)
        self._index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if not self._targets:
            raise StopIteration
        graph = _make_graph(self._targets[self._index])
        self._index = (self._index + 1) % len(self._targets)
        return graph


def _loss_fn(params, graph: jraph.GraphsTuple) -> jnp.ndarray:
    diff = params['w'][0] - graph.globals['target'][0]
    return jnp.array([0.5 * diff * diff])


def _run_training(targets, *, steps, learning_rate=0.3, **train_kwargs):
    params = {'w': jnp.zeros((1,), dtype=jnp.float64)}
    gradient_transform = optax.sgd(learning_rate=learning_rate)
    optimizer_state = gradient_transform.init(params)
    loader = _Loader(targets)

    trainer = train_loop(
        params=params,
        total_loss_fn=_loss_fn,
        train_loader=loader,
        gradient_transform=gradient_transform,
        optimizer_state=optimizer_state,
        steps_per_interval=1,
        progress_bar=False,
        **train_kwargs,
    )

    final_eval = None
    for interval, _, _, eval_params in trainer:
        if interval == steps:
            final_eval = float(eval_params['w'][0])
            break
    assert final_eval is not None
    return final_eval


def test_gradient_clipping_limits_update_magnitude():
    unclipped = _run_training([1.0], steps=1, max_grad_norm=None)
    clipped = _run_training([1.0], steps=1, max_grad_norm=0.1)

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


def test_ema_eval_uses_smoothed_parameters():
    targets = [1.0, -0.5]
    ema_decay = 0.5
    learning_rate = 0.3
    expected = _expected_ema(
        _simulate_params(targets, learning_rate),
        ema_decay,
    )

    eval_param = _run_training(
        targets,
        steps=len(targets),
        learning_rate=learning_rate,
        ema_decay=ema_decay,
    )

    assert np.isclose(eval_param, expected, atol=1e-8)


def test_schedule_free_eval_fn_overrides_eval_params():
    params = {'w': jnp.zeros((1,), dtype=jnp.float64)}
    gradient_transform = optax.sgd(learning_rate=0.0)
    optimizer_state = gradient_transform.init(params)
    loader = _Loader([0.0])

    eval_marker = jnp.asarray([42.0], dtype=jnp.float64)

    def _eval_fn(state, current_params):
        assert state is not None
        assert 'w' in current_params
        return {'w': eval_marker}

    trainer = train_loop(
        params=params,
        total_loss_fn=_loss_fn,
        train_loader=loader,
        gradient_transform=gradient_transform,
        optimizer_state=optimizer_state,
        steps_per_interval=1,
        progress_bar=False,
        schedule_free_eval_fn=_eval_fn,
    )

    epoch_zero = next(trainer)
    assert epoch_zero[3]['w'][0] == eval_marker[0]
    epoch_one = next(trainer)
    assert epoch_one[3]['w'][0] == eval_marker[0]
