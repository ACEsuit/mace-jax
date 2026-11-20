"""Global reductions, tail batches, and distributed iterator coordination."""

import os
import pickle
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import jraph
import numpy as np
import optax
import pytest

from mace_jax import tools
from mace_jax.modules.loss import WeightedEnergyForcesLoss
from mace_jax.tools import distributed, gin_model
from mace_jax.tools.train import _device_batches
from tests.test_evaluate import _Loader, _predict


class SquaredEnergyLoss:
    energy_weight = 1.0
    forces_weight = 0.0

    def __call__(self, graph, predictions):
        return 0.5 * (predictions['energy'] - graph.globals.energy) ** 2


@pytest.mark.parametrize('prefetch_batches', [0, 2])
@pytest.mark.parametrize('layout', ['individual', 'packed', 'stacked'])
def test_streaming_prediction_keeps_all_graphs(layout, prefetch_batches):
    loader = _Loader((2.0, 4.0, 8.0))
    loader.graphs = [
        graph._replace(
            globals=graph.globals._replace(
                graph_id=np.array([graph_id, -1]),
                # Prediction must retain real graphs even with zero loss weight.
                weight=np.zeros(2),
            )
        )
        for graph_id, graph in zip((2, 0, 1), loader.graphs)
    ]
    if layout == 'packed':
        loader.graphs = [
            jraph.pad_with_graphs(
                jraph.batch([jraph.unpad_with_graphs(g) for g in loader.graphs]),
                n_node=7,
                n_edge=1,
                n_graph=4,
            )
        ]
    elif layout == 'stacked':
        loader.graphs = [
            jax.tree_util.tree_map(lambda *xs: np.stack(xs), *loader.graphs[:2]),
            loader.graphs[2],
        ]

    def predict(params, graph):
        del params
        mask = jraph.get_graph_padding_mask(graph)
        return {
            'energy': jnp.where(mask, graph.globals.energy, 0.0),
            'forces': graph.nodes.positions,
        }

    ids, outputs = tools.predict_streaming(
        predict,
        {},
        loader,
        progress_bar=False,
        device_prefetch_batches=prefetch_batches,
    )
    assert ids == [0, 1, 2]
    np.testing.assert_array_equal(outputs['energy'], [4.0, 8.0, 2.0])
    assert [value.shape for value in outputs['forces']] == [(2, 3)] * 3


def _train_epoch(
    loader, *, collect_metrics=False, params=None, optimizer=None, ema_decay=None
):
    params = {'w': jnp.array(0.0)} if params is None else params
    optimizer = optax.sgd(0.1) if optimizer is None else optimizer
    loss_fn = SquaredEnergyLoss()
    trainer = tools.train(
        params=params,
        total_loss_fn=lambda p, g: loss_fn(g, _predict(p, g)),
        train_loader=loader,
        gradient_transform=optimizer,
        optimizer_state=optimizer.init(params),
        ema_decay=ema_decay,
        progress_bar=False,
        collect_metrics=collect_metrics,
        metrics_predictor=_predict,
        metrics_loss_fn=loss_fn,
    )
    try:
        initial = next(trainer)
        final = next(trainer)
        return initial, final
    finally:
        trainer.close()


@pytest.mark.parametrize('placement', ['host', 'default', 'committed'])
def test_replicated_parameter_shape_is_preserved(placement):
    mesh = distributed.DeviceMesh()
    shape = (len(jax.devices()), 3)
    original = {'zeros': np.zeros(shape), 'ones': np.ones(shape)}
    if placement == 'default':
        original = jax.device_put(original)
    elif placement == 'committed':
        original = jax.device_put(original, jax.local_devices()[0])
    replicated = mesh.replicate(original)
    roundtrip = jax.device_get(replicated)
    for name, value in original.items():
        assert replicated[name].shape == shape
        assert replicated[name].is_fully_replicated
        np.testing.assert_array_equal(roundtrip[name], value)


@pytest.mark.parametrize('collect_metrics', [False, True])
@pytest.mark.parametrize('missing_weights', [False, True])
def test_training_keeps_incomplete_device_group(collect_metrics, missing_weights):
    count = len(jax.local_devices())
    targets = tuple(float(i + 1) for i in range(count + 1))
    loader = _Loader(targets)
    if missing_weights:
        loader.graphs = [
            graph._replace(globals=graph.globals._replace(weight=None))
            for graph in loader.graphs
        ]
    initial, final = _train_epoch(loader, collect_metrics=collect_metrics)
    after_first = 0.1 * np.mean(targets[:count])
    expected = after_first + 0.1 * (targets[-1] - after_first)
    assert float(initial[1]['w']) == 0.0
    assert float(final[1]['w']) == pytest.approx(expected)
    if collect_metrics:
        errors = [*targets[:count], targets[-1] - after_first]
        expected_loss = np.mean(np.square(errors)) / 2.0
        assert final[4]['loss'] == pytest.approx(expected_loss)


def test_gradients_are_weighted_by_real_graph_count():
    if len(jax.local_devices()) < 2:
        pytest.skip('requires two local devices')
    graphs = _Loader((1.0, 3.0, 3.0, 3.0)).graphs
    graphs = [jraph.unpad_with_graphs(g) for g in graphs]
    loader = _Loader()
    loader.graphs = [
        jraph.pad_with_graphs(jraph.batch(part), n_node=7, n_edge=1, n_graph=4)
        for part in (graphs[:1], graphs[1:])
    ]
    _, final = _train_epoch(loader, collect_metrics=True)
    # Four graphs contribute equally, despite the 1:3 device imbalance.
    assert float(final[1]['w']) == pytest.approx(0.25)
    assert final[4]['loss'] == pytest.approx(3.5)
    assert final[4]['rmse_e'] == pytest.approx(np.sqrt(7.0))


def test_training_reduces_each_head_with_tail_padding():
    loader = _Loader((1.0, 3.0, 5.0))
    loader.heads = ('first', 'second')
    loader._head_to_index = {'first': 0, 'second': 1}
    loader.graphs = [
        graph._replace(globals=graph.globals._replace(head=jnp.array([head, 0])))
        for head, graph in zip((0, 1, 1), loader.graphs)
    ]
    _, final = _train_epoch(loader, collect_metrics=True, optimizer=optax.sgd(0.0))
    assert final[4]['loss'] == pytest.approx(35 / 6)
    head_metrics = final[4]['head_metrics']
    assert head_metrics['first']['loss'] == pytest.approx(0.5)
    assert head_metrics['first']['rmse_e'] == pytest.approx(1.0)
    assert head_metrics['second']['loss'] == pytest.approx(8.5)
    assert head_metrics['second']['rmse_e'] == pytest.approx(np.sqrt(17.0))


def test_zero_weight_batches_do_not_advance_optimizer_or_change_ema():
    loader = _Loader((2.0,))
    loader.graphs = [distributed.empty_graph_like(loader.graphs[0])]
    params = {'w': jnp.array(1.0)}
    initial, final = _train_epoch(
        loader,
        params=params,
        optimizer=optax.adamw(0.1, weight_decay=0.2),
        ema_decay=0.5,
    )
    for before, after in zip(
        jax.tree_util.tree_leaves(initial[1:]),
        jax.tree_util.tree_leaves(final[1:]),
    ):
        np.testing.assert_array_equal(before, after)


def test_zero_weight_step_does_not_change_later_ema_decay():
    count = len(jax.local_devices())
    loader = _Loader((2.0,) * count)
    _, expected = _train_epoch(loader, ema_decay=0.9)
    loader.graphs = [
        distributed.empty_graph_like(g) for g in loader.graphs
    ] + loader.graphs
    _, actual = _train_epoch(loader, ema_decay=0.9)
    for before, after in zip(
        jax.tree_util.tree_leaves(expected[1:]),
        jax.tree_util.tree_leaves(actual[1:]),
    ):
        np.testing.assert_array_equal(before, after)


def test_mace_energy_force_update_matches_serial_gradient():
    """Cover nested force differentiation and masked devices with a real MACE."""
    loader = _Loader((2.0,))
    graph = jraph.unpad_with_graphs(loader.graphs[0])
    graph = graph._replace(
        nodes=graph.nodes._replace(
            positions=jnp.array([[0.0, 0.0, 0.0], [1.2, 0.1, 0.0]]),
            species=jnp.zeros(2, dtype=jnp.int32),
        ),
        edges=graph.edges._replace(
            shifts=jnp.zeros((2, 3)), unit_shifts=jnp.zeros((2, 3))
        ),
        senders=jnp.array([0, 1], dtype=jnp.int32),
        receivers=jnp.array([1, 0], dtype=jnp.int32),
        n_edge=jnp.array([2], dtype=jnp.int32),
    )
    graph = jraph.pad_with_graphs(graph, n_node=3, n_edge=3, n_graph=2)
    loader.graphs = [graph]
    predictor, bundle, _ = gin_model.model(
        r_max=3.0,
        num_species=1,
        atomic_numbers=(1,),
        atomic_energies='zero',
        avg_num_neighbors=1.0,
        num_interactions=2,
        num_bessel=2,
        num_polynomial_cutoff=5,
        max_ell=1,
        correlation=2,
        gate=jax.nn.silu,
        hidden_irreps='2x0e+2x1o',
        MLP_irreps='2x0e',
    )
    params = bundle['params']
    loss_fn = WeightedEnergyForcesLoss(energy_weight=1.0, forces_weight=1.0)

    def objective(p, g):
        per_graph = loss_fn(g, predictor(p, g))
        return jnp.sum(jnp.where(jraph.get_graph_padding_mask(g), per_graph, 0.0))

    expected_loss, grad = jax.jit(jax.value_and_grad(objective))(params, graph)
    grad_norm = float(optax.global_norm(grad))
    assert np.isfinite(grad_norm) and grad_norm > 0
    expected_params = jax.tree_util.tree_map(lambda p, g: p - 0.001 * g, params, grad)
    optimizer = optax.sgd(0.001)
    trainer = tools.train(
        params=params,
        total_loss_fn=lambda p, g: loss_fn(g, predictor(p, g)),
        train_loader=loader,
        gradient_transform=optimizer,
        optimizer_state=optimizer.init(params),
        progress_bar=False,
        collect_metrics=True,
        metrics_predictor=predictor,
        metrics_loss_fn=loss_fn,
    )
    try:
        next(trainer)
        result = next(trainer)
    finally:
        trainer.close()
    assert result[4]['loss'] == pytest.approx(float(expected_loss), rel=1e-6)
    for expected, actual in zip(
        jax.tree_util.tree_leaves(expected_params), jax.tree_util.tree_leaves(result[1])
    ):
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize('prefetch', [0, 2])
def test_evaluation_accepts_prepared_device_batches(prefetch):
    count = len(jax.local_devices())
    loader = _Loader(tuple(float(i + 1) for i in range(count)))
    loader.graphs = [jax.tree_util.tree_map(lambda *xs: np.stack(xs), *loader.graphs)]
    loss, _ = tools.evaluate(
        _predict,
        {'w': jnp.array(0.0)},
        SquaredEnergyLoss(),
        loader,
        progress_bar=False,
        device_prefetch_batches=prefetch,
    )
    assert loss == pytest.approx(np.mean(np.arange(1, count + 1) ** 2) / 2)


def _prepared_item():
    mesh = distributed.DeviceMesh()
    graphs = _Loader((2.0,) * mesh.local_device_count).graphs
    batch = mesh.prepare_local_batch(graphs)
    count = mesh.local_device_count
    return batch, (3 * count, count, 2 * count, count, count, count)


def _mock_processes(monkeypatch, remote_active, reference):
    """Simulate the other process's control-plane messages, not model reductions."""
    monkeypatch.setattr(distributed.jax, 'process_count', lambda: 2)
    monkeypatch.setattr(distributed.jax, 'process_index', lambda: 0)
    states = iter(remote_active)
    calls = []

    def gather(local, *, tiled):
        assert threading.current_thread() is threading.main_thread()
        assert tiled is False
        calls.append(np.array(local))
        remote = np.concatenate(
            ([next(states)], distributed._batch_signature(reference))
        )
        return np.stack([local, remote])

    monkeypatch.setattr(distributed.multihost_utils, 'process_allgather', gather)
    return calls


def test_exhausted_process_keeps_participating(monkeypatch):
    item = _prepared_item()
    calls = _mock_processes(monkeypatch, [1, 1, 0], item[0])
    chunks = list(
        distributed.synchronize_batches(iter([item]), len(jax.local_devices()))
    )
    assert len(chunks) == 2
    np.testing.assert_array_equal(chunks[0][0].globals.weight, item[0].globals.weight)
    assert not np.any(chunks[1][0].globals.weight)
    assert len(calls) == 3


def test_initially_empty_process_receives_padding_template(monkeypatch):
    item = _prepared_item()
    _mock_processes(monkeypatch, [1, 1, 0], item[0])
    payload = pickle.dumps(jax.device_get(item[0]))
    messages = iter([np.asarray(len(payload)), np.frombuffer(payload, dtype=np.uint8)])

    def broadcast(_value, *, is_source):
        assert not is_source
        return next(messages)

    monkeypatch.setattr(distributed.multihost_utils, 'broadcast_one_to_all', broadcast)
    chunks = list(distributed.synchronize_batches(iter([]), len(jax.local_devices())))
    assert len(chunks) == 2
    for graph, _ in chunks:
        assert graph.nodes.positions.shape == item[0].nodes.positions.shape
        assert not np.any(graph.globals.weight)


def test_prefetch_does_not_run_distributed_collectives_in_worker(monkeypatch):
    mesh = distributed.DeviceMesh()
    loader = _Loader((2.0,) * mesh.local_device_count)
    reference = mesh.prepare_local_batch(loader.graphs)
    calls = _mock_processes(monkeypatch, [1, 0], reference)
    batches = list(_device_batches(iter(loader.graphs), mesh, 2))
    assert len(batches) == 1
    assert len(calls) == 2


def test_mismatched_process_batch_shapes_fail_before_model_step(monkeypatch):
    item = _prepared_item()
    other = item[0]._replace(senders=np.zeros((len(jax.local_devices()), 4), np.int32))
    _mock_processes(monkeypatch, [1], other)
    with pytest.raises(ValueError, match='matching fixed batch shapes'):
        list(distributed.synchronize_batches(iter([item]), len(jax.local_devices())))


@pytest.mark.parametrize('failed_process', [0, 1])
def test_loader_failures_are_reported_on_all_processes(monkeypatch, failed_process):
    item = _prepared_item()
    _mock_processes(monkeypatch, [-1 if failed_process == 1 else 1], item[0])

    def failing_loader():
        raise ValueError('broken loader')
        yield  # pragma: no cover

    source = failing_loader() if failed_process == 0 else iter([item])
    with pytest.raises(RuntimeError, match=f'processes \\[{failed_process}\\]'):
        list(distributed.synchronize_batches(source, len(jax.local_devices())))


@pytest.mark.parametrize('prefetch', [0, 2])
def test_eager_loader_failure_is_coordinated(monkeypatch, prefetch):
    mesh = distributed.DeviceMesh()
    item = _prepared_item()
    calls = _mock_processes(monkeypatch, [1], item[0])

    class BrokenLoader:
        def iter_batches(self, **kwargs):
            raise ValueError('Cannot open dataset')

    source = distributed.iter_loader_batches(BrokenLoader())
    with pytest.raises(RuntimeError, match=r'processes \[0\]'):
        list(_device_batches(source, mesh, prefetch))
    assert calls[0][0] == -1


def test_training_across_two_processes(tmp_path):
    """Exercise real collectives in fresh CPU workers, even on GPU test hosts."""
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        address = f'127.0.0.1:{sock.getsockname()[1]}'
    env = {k: v for k, v in os.environ.items() if not k.lower().endswith('_proxy')}
    env.update(
        JAX_PLATFORMS='cpu',
        JAX_PLATFORM_NAME='cpu',
        JAX_NUM_CPU_DEVICES='1',
        CUDA_VISIBLE_DEVICES='',
        JAX_CPU_COLLECTIVES_IMPLEMENTATION='gloo',
        XLA_FLAGS='--xla_force_host_platform_device_count=1',
        OMP_NUM_THREADS='1',
        OPENBLAS_NUM_THREADS='1',
        MKL_NUM_THREADS='1',
        TMPDIR=str(tmp_path),
    )
    processes = []
    logs = []
    timed_out = False
    try:
        for rank in range(2):
            log = (tmp_path / f'worker-{rank}.log').open('w')
            logs.append(log)
            processes.append(
                subprocess.Popen(
                    [
                        sys.executable,
                        '-m',
                        'tests.test_distributed_training',
                        str(rank),
                        address,
                    ],
                    cwd=Path(__file__).resolve().parents[1],
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            )
        deadline = time.monotonic() + 60
        for process in processes:
            process.wait(timeout=max(1, deadline - time.monotonic()))
    except subprocess.TimeoutExpired:
        timed_out = True
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.wait()
        for log in logs:
            log.close()
    output = '\n'.join(
        (tmp_path / f'worker-{rank}.log').read_text() for rank in range(2)
    )
    # Only skip the specific restricted-host failure; model/collective failures fail.
    if (
        all(p.returncode == 77 for p in processes)
        and 'GLOO_PERMISSION_DENIED' in output
    ):
        pytest.skip('CPU distributed backend cannot access Gloo system information')
    assert not timed_out, f'Distributed workers timed out:\n{output}'
    assert all(p.returncode == 0 for p in processes), output
    assert output.count('DISTRIBUTED_CHECKS_PASSED') == 2, output


def _distributed_worker(rank, address):
    jax.distributed.initialize(
        coordinator_address=address,
        num_processes=2,
        process_id=rank,
        initialization_timeout=20,
    )
    try:
        jax.devices()
    except RuntimeError as exc:
        if 'gloo/common/linux.cc' in str(exc) and 'Permission denied' in str(exc):
            print(f'GLOO_PERMISSION_DENIED: {exc}', flush=True)
            sys.exit(77)
        raise
    assert jax.device_count() == 2
    # Standalone evaluation must also accept parameters explicitly placed on
    # each rank's local device, without trying to device_put onto remote devices.
    committed_params = jax.device_put({'w': 0.0}, jax.local_devices()[0])
    loss, _ = tools.evaluate(
        _predict,
        committed_params,
        SquaredEnergyLoss(),
        _Loader((1.0 + 2 * rank,)),
        progress_bar=False,
    )
    np.testing.assert_allclose(loss, 2.5, rtol=1e-6)
    for initially_empty in (False, True):
        targets = (() if initially_empty else (1.0,)) if rank == 0 else (3.0, 5.0)
        loader = _Loader(targets)
        # Different seeds must not produce different initial state across ranks.
        params = {'w': jnp.array(7.0 * rank), 'unused': jnp.full((2, 3), float(rank))}
        initial, final = _train_epoch(loader, collect_metrics=True, params=params)
        assert float(initial[1]['w']) == 0.0
        np.testing.assert_array_equal(initial[1]['unused'], np.zeros((2, 3)))
        expected_weight = 0.77 if initially_empty else 0.68
        np.testing.assert_allclose(final[1]['w'], expected_weight, rtol=1e-6)
        assert final[1]['unused'].shape == (2, 3)
        expected_loss = (
            (4.5 + 4.7**2 / 2) / 2 if initially_empty else (5 + 4.8**2 / 2) / 3
        )
        np.testing.assert_allclose(final[4]['loss'], expected_loss, rtol=1e-6)
        loss, metrics = tools.evaluate(
            _predict,
            initial[1],
            SquaredEnergyLoss(),
            loader,
            progress_bar=False,
            device_prefetch_batches=2,
        )
        mse = 17.0 if initially_empty else 35.0 / 3
        np.testing.assert_allclose(loss, mse / 2, rtol=1e-6)
        np.testing.assert_allclose(metrics['rmse_e'], np.sqrt(mse), rtol=1e-6)
    loss, metrics = tools.evaluate(
        _predict,
        initial[1],
        SquaredEnergyLoss(),
        _Loader(()),
        progress_bar=False,
    )
    assert loss == 0.0 and metrics == {}
    print('DISTRIBUTED_CHECKS_PASSED', flush=True)
    jax.distributed.shutdown()


if __name__ == '__main__':
    _distributed_worker(int(sys.argv[1]), sys.argv[2])
