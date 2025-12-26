import dataclasses
import itertools
import logging
import math
import threading
import time
from collections.abc import Callable
from functools import partial
from queue import Queue
from typing import Any

import gin
import jax
import jax.numpy as jnp
import jraph
import numpy as np
import optax
import tqdm

from mace_jax import data


@gin.register
@dataclasses.dataclass
class SWAConfig:
    """Configuration for simple Stochastic Weight Averaging."""

    start_interval: int = 0
    update_interval: int = 1
    min_snapshots_for_eval: int = 1
    max_snapshots: int | None = None
    prefer_swa_params: bool = True
    stage_loss_factory: Callable | None = None
    stage_loss_kwargs: dict[str, float] | None = None

    def __post_init__(self) -> None:
        if self.start_interval < 0:
            raise ValueError('SWA start_interval must be non-negative')
        if self.update_interval <= 0:
            raise ValueError('SWA update_interval must be positive')
        if self.min_snapshots_for_eval <= 0:
            raise ValueError('SWA min_snapshots_for_eval must be positive')

    def should_collect(self, interval: int, num_snapshots: int) -> bool:
        if interval < self.start_interval:
            return False
        if self.max_snapshots is not None and num_snapshots >= self.max_snapshots:
            return False
        offset = interval - self.start_interval
        return offset % self.update_interval == 0

    def should_use(self, num_snapshots: int) -> bool:
        if not self.prefer_swa_params:
            return False
        return num_snapshots >= self.min_snapshots_for_eval


def _sanitize_grads(grads):
    """Replace NaNs/inf/float0 leaves to keep the optimizer stable."""

    def _sanitize(leaf):
        arr = jnp.asarray(leaf)
        if 'float0' in str(arr.dtype):
            arr = jnp.zeros_like(arr, dtype=jnp.float32)
        return jnp.nan_to_num(arr)

    return jax.tree_util.tree_map(_sanitize, grads)


def _prefetch_iterator(iterator, capacity: int):
    if capacity is None or capacity <= 0:
        return iterator
    queue: Queue = Queue(maxsize=capacity)
    sentinel = object()
    total_hint = getattr(iterator, 'total_batches_hint', 0)

    def _producer():
        try:
            for item in iterator:
                queue.put(item)
        finally:
            queue.put(sentinel)

    threading.Thread(target=_producer, daemon=True).start()

    class _Prefetched:
        def __init__(self):
            self._done = False
            self.total_batches_hint = total_hint

        def __iter__(self):
            return self

        def __next__(self):
            if self._done:
                raise StopIteration
            item = queue.get()
            if item is sentinel:
                self._done = True
                raise StopIteration
            return item

    return _Prefetched()


@dataclasses.dataclass
class _MetricAccumulator:
    count: float = 0.0
    sum_abs_delta: float = 0.0
    sum_abs_target: float = 0.0
    sum_sq_delta: float = 0.0
    sum_sq_target: float = 0.0

    def update(self, batch: dict | None) -> None:
        if not batch:
            return
        self.count += float(batch['count'])
        self.sum_abs_delta += float(batch['sum_abs_delta'])
        self.sum_abs_target += float(batch['sum_abs_target'])
        self.sum_sq_delta += float(batch['sum_sq_delta'])
        self.sum_sq_target += float(batch['sum_sq_target'])

    def finalize(self) -> dict[str, float] | None:
        if self.count <= 0:
            return None
        mae = self.sum_abs_delta / self.count
        target_abs_mean = self.sum_abs_target / self.count
        rmse = math.sqrt(self.sum_sq_delta / self.count)
        target_rms = math.sqrt(self.sum_sq_target / self.count)
        return {
            'mae': mae,
            'rel_mae': mae / (target_abs_mean + 1e-30),
            'rmse': rmse,
            'rel_rmse': rmse / (target_rms + 1e-30),
        }


def _metric_stats(delta, target, mask):
    broadcast_mask = jnp.broadcast_to(mask, delta.shape)
    mask_float = broadcast_mask.astype(delta.dtype)
    count = jnp.sum(mask_float)
    abs_delta = jnp.abs(delta)
    abs_target = jnp.abs(target)
    sum_abs_delta = jnp.sum(abs_delta * mask_float)
    sum_abs_target = jnp.sum(abs_target * mask_float)
    sum_sq_delta = jnp.sum(jnp.square(delta) * mask_float)
    sum_sq_target = jnp.sum(jnp.square(target) * mask_float)
    return {
        'count': count,
        'sum_abs_delta': sum_abs_delta,
        'sum_abs_target': sum_abs_target,
        'sum_sq_delta': sum_sq_delta,
        'sum_sq_target': sum_sq_target,
    }


def _compute_eval_batch_metrics(
    loss_fn, graph, pred_graph, pred_outputs, mask_override=None
):
    graph_mask = jraph.get_graph_padding_mask(graph).astype(jnp.float32)
    if graph_mask.shape == graph.n_node.shape:
        non_empty = jnp.asarray(graph.n_node > 0, dtype=jnp.float32)
        graph_mask = graph_mask * non_empty
    weights = getattr(graph.globals, 'weight', None)
    if weights is not None:
        weights_arr = jnp.asarray(weights)
        graph_mask = graph_mask * (weights_arr > 0).astype(jnp.float32)
    if mask_override is not None:
        override = jnp.asarray(mask_override, dtype=jnp.float32)
        graph_mask = graph_mask * override
    node_mask = jraph.get_node_padding_mask(graph).astype(jnp.float32)
    total_nodes = node_mask.shape[0]
    nodes_per_graph_mask = jnp.repeat(
        graph_mask,
        graph.n_node,
        axis=0,
        total_repeat_length=total_nodes,
    )
    node_mask = node_mask * nodes_per_graph_mask
    atom_counts = jnp.maximum(jnp.asarray(graph.n_node), 1.0)

    per_graph_loss = loss_fn(graph, pred_outputs)
    loss = jnp.sum(per_graph_loss * graph_mask)

    result = {
        'loss': loss,
        'graph_count': jnp.sum(graph_mask),
        'energy': None,
        'energy_per_atom': None,
        'forces': None,
        'stress': None,
        'virials': None,
        'virials_per_atom': None,
        'dipole': None,
        'dipole_per_atom': None,
        'polar': None,
        'polar_per_atom': None,
    }

    ref_energy = getattr(graph.globals, 'energy', None)
    pred_energy = getattr(pred_graph.globals, 'energy', None)
    if ref_energy is not None and pred_energy is not None:
        ref_energy = jnp.asarray(ref_energy)
        pred_energy = jnp.asarray(pred_energy)
        delta_energy = ref_energy - pred_energy
        energy_stats = _metric_stats(delta_energy, ref_energy, graph_mask)
        atom_counts_safe = atom_counts.astype(delta_energy.dtype)
        per_atom_delta = delta_energy / atom_counts_safe
        per_atom_ref = ref_energy / atom_counts_safe
        energy_pa_stats = _metric_stats(per_atom_delta, per_atom_ref, graph_mask)
        result['energy'] = energy_stats
        result['energy_per_atom'] = energy_pa_stats

    ref_forces = getattr(graph.nodes, 'forces', None)
    pred_forces = getattr(pred_graph.nodes, 'forces', None)
    if ref_forces is not None and pred_forces is not None:
        ref_forces = jnp.asarray(ref_forces)
        pred_forces = jnp.asarray(pred_forces)
        delta_forces = ref_forces - pred_forces
        force_stats = _metric_stats(delta_forces, ref_forces, node_mask[:, None])
        result['forces'] = force_stats

    ref_stress = getattr(graph.globals, 'stress', None)
    pred_stress = getattr(pred_graph.globals, 'stress', None)
    if ref_stress is not None and pred_stress is not None:
        ref_stress = jnp.asarray(ref_stress)
        pred_stress = jnp.asarray(pred_stress)
        delta_stress = ref_stress - pred_stress
        stress_stats = _metric_stats(
            delta_stress, ref_stress, graph_mask[:, None, None]
        )
        result['stress'] = stress_stats

    ref_virials = getattr(graph.globals, 'virials', None)
    pred_virials = getattr(pred_graph.globals, 'virials', None)
    if ref_virials is not None and pred_virials is not None:
        ref_virials = jnp.asarray(ref_virials)
        pred_virials = jnp.asarray(pred_virials)
        delta_virials = ref_virials - pred_virials
        virial_stats = _metric_stats(
            delta_virials, ref_virials, graph_mask[:, None, None]
        )
        atoms_matrix = atom_counts[:, None, None].astype(delta_virials.dtype)
        delta_virials_per_atom = delta_virials / atoms_matrix
        ref_virials_per_atom = ref_virials / atoms_matrix
        virial_pa_stats = _metric_stats(
            delta_virials_per_atom, ref_virials_per_atom, graph_mask[:, None, None]
        )
        result['virials'] = virial_stats
        result['virials_per_atom'] = virial_pa_stats

    ref_dipole = getattr(graph.globals, 'dipole', None)
    pred_dipole = getattr(pred_graph.globals, 'dipole', None)
    if ref_dipole is not None and pred_dipole is not None:
        ref_dipole = jnp.asarray(ref_dipole)
        pred_dipole = jnp.asarray(pred_dipole)
        delta_dipole = ref_dipole - pred_dipole
        dipole_stats = _metric_stats(delta_dipole, ref_dipole, graph_mask[:, None])
        atoms_vector = atom_counts[:, None].astype(delta_dipole.dtype)
        delta_dipole_per_atom = delta_dipole / atoms_vector
        ref_dipole_per_atom = ref_dipole / atoms_vector
        dipole_pa_stats = _metric_stats(
            delta_dipole_per_atom, ref_dipole_per_atom, graph_mask[:, None]
        )
        result['dipole'] = dipole_stats
        result['dipole_per_atom'] = dipole_pa_stats

    ref_polar = getattr(graph.globals, 'polarizability', None)
    pred_polar = getattr(pred_graph.globals, 'polarizability', None)
    if ref_polar is not None and pred_polar is not None:
        ref_polar = jnp.asarray(ref_polar)
        pred_polar = jnp.asarray(pred_polar)
        delta_polar = ref_polar - pred_polar
        polar_stats = _metric_stats(delta_polar, ref_polar, graph_mask[:, None, None])
        atoms_matrix = atom_counts[:, None, None].astype(delta_polar.dtype)
        delta_polar_per_atom = delta_polar / atoms_matrix
        ref_polar_per_atom = ref_polar / atoms_matrix
        polar_pa_stats = _metric_stats(
            delta_polar_per_atom, ref_polar_per_atom, graph_mask[:, None, None]
        )
        result['polar'] = polar_stats
        result['polar_per_atom'] = polar_pa_stats

    return result


def _make_eval_batch_metrics(loss_fn):
    def _batch(graph, pred_graph, pred_outputs, graph_mask):
        return _compute_eval_batch_metrics(
            loss_fn, graph, pred_graph, pred_outputs, graph_mask
        )

    return jax.jit(_batch)


def train(
    params: Any,
    total_loss_fn: Callable,  # loss_fn(params, graph) -> [num_graphs]
    train_loader: data.GraphDataLoader,  # device parallel done on the (optional) extra dimension
    gradient_transform: Any,
    optimizer_state: dict[str, Any],
    ema_decay: float | None = None,
    progress_bar: bool = True,
    swa_config: SWAConfig | None = None,
    max_grad_norm: float | None = None,
    schedule_free_eval_fn: Callable | None = None,
    *,
    start_interval: int = 0,
    data_seed: int | None = None,
):
    """
    for interval, params, optimizer_state, ema_params in train(...):
        # do something
    """
    num_updates = 0
    ema_params = params
    eval_params = params
    swa_state = None
    if swa_config is not None:
        swa_state = {'params': None, 'count': 0}

    logging.info('Started training')

    local_device_count = max(1, jax.local_device_count())

    def _prepare_graph_for_devices(graph: jraph.GraphsTuple):
        """Ensure ``graph`` has a leading axis matching local device count."""

        if local_device_count == 1:
            if graph.n_node.ndim == 1:
                return jax.tree_util.tree_map(lambda x: x[None, ...], graph)
            return graph

        if graph.n_node.ndim == 1:
            return data.prepare_sharded_batch(graph, local_device_count)

        if graph.n_node.shape[0] != local_device_count:
            raise ValueError(
                'Expected microbatches with leading axis equal to the number of local '
                f'devices ({local_device_count}), got axis size {graph.n_node.shape[0]}.'
            )
        return graph

    @partial(jax.pmap, in_axes=(None, 0), out_axes=None, axis_name='devices')
    # @partial(jax.vmap, in_axes=(None, 0), out_axes=0)
    def grad_fn(params, graph: jraph.GraphsTuple):
        # graph is assumed to be padded by jraph.pad_with_graphs
        mask = jraph.get_graph_padding_mask(graph)  # [n_graphs,]
        if mask.ndim == graph.n_node.ndim:
            non_empty = jnp.asarray(graph.n_node > 0, dtype=mask.dtype)
            mask = mask * non_empty
        weights = getattr(graph.globals, 'weight', None)
        if weights is not None:
            weight_mask = jnp.asarray(weights > 0, dtype=mask.dtype)
            if weight_mask.shape == mask.shape:
                mask = mask * weight_mask

        def _loss_fn(trainable):
            return jnp.sum(jnp.where(mask, total_loss_fn(trainable, graph), 0.0))

        loss, grad = jax.value_and_grad(_loss_fn)(params)
        loss = jax.lax.psum(loss, 'devices')
        grad = jax.tree_util.tree_map(lambda g: jax.lax.psum(g, 'devices'), grad)
        num_graphs = jax.lax.psum(jnp.sum(mask), 'devices')
        return num_graphs, loss, grad

    # jit-of-pmap is not recommended but so far it seems faster
    @jax.jit
    def update_fn(
        params, optimizer_state, ema_params, num_updates: int, graph: jraph.GraphsTuple
    ) -> tuple[float, Any, Any]:
        n, loss, grad = grad_fn(params, graph)
        loss = loss / n
        grad = jax.tree_util.tree_map(lambda x: x / n, grad)
        grad = _sanitize_grads(grad)

        if max_grad_norm is not None:
            grad_norm = jnp.sqrt(
                sum(jnp.sum(jnp.square(x)) for x in jax.tree_util.tree_leaves(grad))
            )
            clip_coef = jnp.minimum(1.0, max_grad_norm / (grad_norm + 1e-12))
            grad = jax.tree_util.tree_map(lambda g: g * clip_coef, grad)

        updates, optimizer_state = gradient_transform.update(
            grad, optimizer_state, params
        )
        params = optax.apply_updates(params, updates)
        if ema_decay is not None:
            decay = jnp.minimum(ema_decay, (1 + num_updates) / (10 + num_updates))
            ema_params = jax.tree_util.tree_map(
                lambda x, y: x * decay + y * (1 - decay), ema_params, params
            )
        else:
            ema_params = params
        return loss, params, optimizer_state, ema_params

    last_cache_size = update_fn._cache_size()
    logged_compile_count = 0

    def _swa_average(current_avg, new_tree, new_count: int):
        if current_avg is None:
            return new_tree
        alpha = 1.0 / float(new_count)
        return jax.tree_util.tree_map(
            lambda avg, new: avg + (new - avg) * alpha,
            current_avg,
            new_tree,
        )

    process_index = getattr(jax, 'process_index', lambda: 0)()
    is_primary = process_index == 0
    process_count = getattr(jax, 'process_count', lambda: 1)()
    supports_iter_batches = hasattr(train_loader, 'iter_batches')
    legacy_steps_cache: int | None = None

    def _epoch_batches(epoch: int):
        iterator = train_loader.iter_batches(
            epoch=epoch,
            seed=data_seed,
            process_count=process_count,
            process_index=process_index,
        )
        prefetch_cap = getattr(train_loader, '_prefetch_batches', None)
        iterator = _prefetch_iterator(iterator, int(prefetch_cap or 0))
        total_batches = getattr(iterator, 'total_batches_hint', 0)
        if not total_batches:
            approx_length = getattr(train_loader, 'approx_length', None)
            if callable(approx_length):
                try:
                    total_batches = int(approx_length())
                except Exception:  # pragma: no cover - defensive fallback
                    total_batches = 0
        if not total_batches:
            raise ValueError(
                'Training loader did not provide a total batch count; ensure it '
                'implements iter_batches() with a total_batches_hint or approx_length().'
            )
        return iterator, total_batches

    def _legacy_interval_loader():
        while True:
            yield from train_loader

    def _legacy_effective_steps():
        nonlocal legacy_steps_cache
        if legacy_steps_cache is not None:
            return legacy_steps_cache
        approx_length = getattr(train_loader, 'approx_length', None)
        if callable(approx_length):
            value = int(approx_length())
            if value > 0:
                legacy_steps_cache = value
                return legacy_steps_cache
        length = None
        try:
            length = int(len(train_loader))  # type: ignore[arg-type]
        except Exception:  # pragma: no cover - length optional
            length = None
        if length is None or length <= 0:
            raise ValueError(
                'Unable to determine number of batches for the training loader; provide '
                'a loader with iter_batches() support or implement approx_length().'
            )
        legacy_steps_cache = length
        return legacy_steps_cache

    initial_eval = eval_params
    if schedule_free_eval_fn is not None:
        initial_eval = schedule_free_eval_fn(optimizer_state, params)
    yield start_interval, params, optimizer_state, initial_eval

    for epoch in itertools.count(start_interval):
        if supports_iter_batches:
            epoch_batches_iter, available_steps = _epoch_batches(epoch)
            effective_steps = available_steps
            if effective_steps <= 0:
                raise ValueError(
                    'iter_batches() produced no data; reduce process_count or provide more data.'
                )
            p_bar = tqdm.tqdm(
                total=effective_steps,
                desc=f'Epoch {epoch + 1}',
                disable=not (progress_bar and is_primary),
            )
            batches_in_epoch = 0
            for graph in epoch_batches_iter:
                graph = _prepare_graph_for_devices(graph)
                num_updates += 1
                start_time = time.time()
                loss, params, optimizer_state, ema_params = update_fn(
                    params, optimizer_state, ema_params, num_updates, graph
                )
                loss = float(loss)
                if is_primary:
                    p_bar.set_postfix({'loss': f'{loss:7.3f}'})
                p_bar.update(1)

                if last_cache_size != update_fn._cache_size():
                    last_cache_size = update_fn._cache_size()

                    logging.info(
                        f'Compilation time: {time.time() - start_time:.3f}s, cache size: {last_cache_size}'
                    )
                batches_in_epoch += 1
            p_bar.close()
            if batches_in_epoch <= 0:
                raise ValueError(
                    'iter_batches() produced no data; reduce process_count or provide more data.'
                )

        else:
            legacy_iter = _legacy_interval_loader()
            setter = getattr(train_loader, 'set_epoch', None)
            if callable(setter):
                setter(epoch)

            def _next_batch():
                return next(legacy_iter)

            effective_steps = _legacy_effective_steps()

            p_bar = tqdm.tqdm(
                range(effective_steps),
                desc=f'Epoch {epoch + 1}',
                disable=not (progress_bar and is_primary),
            )
            for _ in p_bar:
                graph = _next_batch()
                graph = _prepare_graph_for_devices(graph)
                num_updates += 1
                start_time = time.time()
                loss, params, optimizer_state, ema_params = update_fn(
                    params, optimizer_state, ema_params, num_updates, graph
                )
                loss = float(loss)
                if is_primary:
                    p_bar.set_postfix({'loss': f'{loss:7.3f}'})

                if last_cache_size != update_fn._cache_size():
                    last_cache_size = update_fn._cache_size()

                    logging.info(
                        f'Compilation time: {time.time() - start_time:.3f}s, cache size: {last_cache_size}'
                    )

        eval_params = ema_params
        if swa_state is not None and swa_config is not None:
            if swa_config.should_collect(epoch, swa_state['count']):
                new_count = swa_state['count'] + 1
                swa_state['params'] = _swa_average(
                    swa_state['params'], params, new_count
                )
                swa_state['count'] = new_count
                logging.info(
                    'Updated SWA parameters at epoch %s (snapshots=%s).',
                    epoch,
                    swa_state['count'],
                )
            if swa_state['params'] is not None and swa_config.should_use(
                swa_state['count']
            ):
                eval_params = swa_state['params']
        if schedule_free_eval_fn is not None:
            eval_params = schedule_free_eval_fn(optimizer_state, params)

        yield epoch + 1, params, optimizer_state, eval_params


def evaluate(
    predictor: Callable,
    params: Any,
    loss_fn: Any,
    data_loader: data.GraphDataLoader,
    name: str = 'Evaluation',
    progress_bar: bool = True,
) -> tuple[float, dict[str, Any]]:
    r"""Evaluate the predictor on the given data loader.

    Args:
        predictor: function of signature `predictor(params, graph) -> {energy: [num_graphs], forces: [num_nodes, 3], stress: [num_graphs, 3, 3]}`
        params: parameters of the predictor
        loss_fn: function of signature `loss_fn(graph, output) -> loss` where `output` is the output of `predictor`
        data_loader: data loader
    """
    total_loss = 0.0
    num_graphs = 0.0

    metric_accumulators = {
        'energy': _MetricAccumulator(),
        'energy_per_atom': _MetricAccumulator(),
        'forces': _MetricAccumulator(),
        'stress': _MetricAccumulator(),
        'virials': _MetricAccumulator(),
        'virials_per_atom': _MetricAccumulator(),
        'dipole': _MetricAccumulator(),
        'dipole_per_atom': _MetricAccumulator(),
        'polar': _MetricAccumulator(),
        'polar_per_atom': _MetricAccumulator(),
    }
    batch_metrics_fn = _make_eval_batch_metrics(loss_fn)

    cache_support = hasattr(predictor, '_cache_size')
    if cache_support:
        last_cache_size = predictor._cache_size()
    else:
        last_cache_size = None

    start_time = time.time()

    process_index = getattr(jax, 'process_index', lambda: 0)()
    process_count = getattr(jax, 'process_count', lambda: 1)()
    supports_iter_batches = hasattr(data_loader, 'iter_batches')

    if supports_iter_batches:
        iterator = data_loader.iter_batches(
            epoch=0,
            seed=None,
            process_count=process_count,
            process_index=process_index,
        )
        total_hint = getattr(iterator, 'total_batches_hint', 0)
    else:
        iterator = iter(data_loader)
        total_hint = 0

    eval_iterator = _prefetch_iterator(
        iterator, int(getattr(data_loader, '_prefetch_batches', 0) or 0)
    )

    p_bar = tqdm.tqdm(
        eval_iterator,
        desc=name,
        total=total_hint or data_loader.approx_length(),
        disable=not progress_bar,
    )

    for ref_graph in p_bar:
        output = predictor(params, ref_graph)
        nodes = ref_graph.nodes
        if output.get('forces') is not None:
            nodes = nodes._replace(forces=output.get('forces'))
        globals_updates = {}
        for key in ('energy', 'stress', 'virials', 'dipole', 'polarizability'):
            if key in output and output[key] is not None:
                globals_updates[key] = output[key]
        globals_attr = (
            ref_graph.globals._replace(**globals_updates)
            if globals_updates
            else ref_graph.globals
        )
        pred_graph = ref_graph._replace(nodes=nodes, globals=globals_attr)

        if last_cache_size is not None and last_cache_size != predictor._cache_size():
            last_cache_size = predictor._cache_size()

            logging.info('Compiled function `predictor`.')
            logging.info(f'-> cache size: {last_cache_size}')

        graph_mask = np.asarray(jraph.get_graph_padding_mask(ref_graph))
        if graph_mask.shape == ref_graph.n_node.shape:
            non_empty = np.asarray(ref_graph.n_node > 0)
            graph_mask = graph_mask & non_empty
        weights = getattr(ref_graph.globals, 'weight', None)
        if weights is not None:
            graph_mask = graph_mask & (np.asarray(weights) > 0)
        if not graph_mask.any():
            continue
        device_graph_mask = jnp.asarray(graph_mask, dtype=jnp.float32)

        pred_outputs = {
            'energy': pred_graph.globals.energy,
            'forces': pred_graph.nodes.forces,
            'stress': pred_graph.globals.stress,
        }
        virials_value = getattr(pred_graph.globals, 'virials', None)
        if virials_value is not None:
            pred_outputs['virials'] = virials_value
        dipole_value = getattr(pred_graph.globals, 'dipole', None)
        if dipole_value is not None:
            pred_outputs['dipole'] = dipole_value
        polar_value = getattr(pred_graph.globals, 'polarizability', None)
        if polar_value is not None:
            pred_outputs['polarizability'] = polar_value

        batch_metrics = batch_metrics_fn(
            ref_graph, pred_graph, pred_outputs, device_graph_mask
        )
        batch_metrics = jax.device_get(batch_metrics)
        total_loss += float(batch_metrics['loss'])
        num_graphs += float(batch_metrics['graph_count'])
        metric_accumulators['energy'].update(batch_metrics.get('energy'))
        metric_accumulators['energy_per_atom'].update(
            batch_metrics.get('energy_per_atom')
        )
        metric_accumulators['forces'].update(batch_metrics.get('forces'))
        metric_accumulators['stress'].update(batch_metrics.get('stress'))
        metric_accumulators['virials'].update(batch_metrics.get('virials'))
        metric_accumulators['virials_per_atom'].update(
            batch_metrics.get('virials_per_atom')
        )
        metric_accumulators['dipole'].update(batch_metrics.get('dipole'))
        metric_accumulators['dipole_per_atom'].update(
            batch_metrics.get('dipole_per_atom')
        )
        metric_accumulators['polar'].update(batch_metrics.get('polar'))
        metric_accumulators['polar_per_atom'].update(
            batch_metrics.get('polar_per_atom')
        )
        p_bar.set_postfix({'n': int(num_graphs)})

    if num_graphs <= 0:
        logging.warning(f'No graphs in data_loader ! Returning 0.0 for {name}')
        return 0.0, {}

    avg_loss = total_loss / num_graphs

    aux = {
        'loss': avg_loss,
        'time': time.time() - start_time,
        'mae_e': None,
        'rel_mae_e': None,
        'mae_e_per_atom': None,
        'rel_mae_e_per_atom': None,
        'rmse_e': None,
        'rel_rmse_e': None,
        'rmse_e_per_atom': None,
        'rel_rmse_e_per_atom': None,
        'mae_f': None,
        'rel_mae_f': None,
        'rmse_f': None,
        'rel_rmse_f': None,
        'mae_s': None,
        'rel_mae_s': None,
        'rmse_s': None,
        'rel_rmse_s': None,
        'mae_stress': None,
        'rel_mae_stress': None,
        'rmse_stress': None,
        'rel_rmse_stress': None,
        'mae_virials': None,
        'rmse_virials': None,
        'rmse_virials_per_atom': None,
        'mae_mu': None,
        'mae_mu_per_atom': None,
        'rel_mae_mu': None,
        'rmse_mu': None,
        'rmse_mu_per_atom': None,
        'rel_rmse_mu': None,
        'mae_polarizability': None,
        'mae_polarizability_per_atom': None,
        'rmse_polarizability': None,
        'rmse_polarizability_per_atom': None,
    }

    energy_stats = metric_accumulators['energy'].finalize()
    if energy_stats:
        aux['mae_e'] = energy_stats['mae']
        aux['rel_mae_e'] = energy_stats['rel_mae']
        aux['rmse_e'] = energy_stats['rmse']
        aux['rel_rmse_e'] = energy_stats['rel_rmse']
    energy_pa_stats = metric_accumulators['energy_per_atom'].finalize()
    if energy_pa_stats:
        aux['mae_e_per_atom'] = energy_pa_stats['mae']
        aux['rel_mae_e_per_atom'] = energy_pa_stats['rel_mae']
        aux['rmse_e_per_atom'] = energy_pa_stats['rmse']
        aux['rel_rmse_e_per_atom'] = energy_pa_stats['rel_rmse']

    force_stats = metric_accumulators['forces'].finalize()
    if force_stats:
        aux['mae_f'] = force_stats['mae']
        aux['rel_mae_f'] = force_stats['rel_mae']
        aux['rmse_f'] = force_stats['rmse']
        aux['rel_rmse_f'] = force_stats['rel_rmse']

    stress_stats = metric_accumulators['stress'].finalize()
    if stress_stats:
        aux['mae_s'] = stress_stats['mae']
        aux['rel_mae_s'] = stress_stats['rel_mae']
        aux['rmse_s'] = stress_stats['rmse']
        aux['rel_rmse_s'] = stress_stats['rel_rmse']
        aux['mae_stress'] = aux['mae_s']
        aux['rel_mae_stress'] = aux['rel_mae_s']
        aux['rmse_stress'] = aux['rmse_s']
        aux['rel_rmse_stress'] = aux['rel_rmse_s']

    virial_stats = metric_accumulators['virials'].finalize()
    if virial_stats:
        aux['mae_virials'] = virial_stats['mae']
        aux['rmse_virials'] = virial_stats['rmse']
    virial_pa_stats = metric_accumulators['virials_per_atom'].finalize()
    if virial_pa_stats:
        aux['rmse_virials_per_atom'] = virial_pa_stats['rmse']

    dipole_stats = metric_accumulators['dipole'].finalize()
    if dipole_stats:
        aux['mae_mu'] = dipole_stats['mae']
        aux['rel_mae_mu'] = dipole_stats['rel_mae']
        aux['rmse_mu'] = dipole_stats['rmse']
        aux['rel_rmse_mu'] = dipole_stats['rel_rmse']
    dipole_pa_stats = metric_accumulators['dipole_per_atom'].finalize()
    if dipole_pa_stats:
        aux['mae_mu_per_atom'] = dipole_pa_stats['mae']
        aux['rmse_mu_per_atom'] = dipole_pa_stats['rmse']

    polar_stats = metric_accumulators['polar'].finalize()
    if polar_stats:
        aux['mae_polarizability'] = polar_stats['mae']
        aux['rmse_polarizability'] = polar_stats['rmse']
    polar_pa_stats = metric_accumulators['polar_per_atom'].finalize()
    if polar_pa_stats:
        aux['mae_polarizability_per_atom'] = polar_pa_stats['mae']
        aux['rmse_polarizability_per_atom'] = polar_pa_stats['rmse']

    return avg_loss, aux
