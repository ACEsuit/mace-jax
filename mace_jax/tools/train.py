"""Low-level training and evaluation loops for JAX MACE.

The routines in this file assume that batches are padded to fixed shapes
(`n_node`, `n_edge`, `n_graph`). Fixed shapes are essential for JAX/XLA because
the model is compiled per input shape; a changing batch shape would trigger
recompilation and stall training. The streaming loader plus packing logic
ensures those fixed shapes, while these loops focus on efficient updates and
metric aggregation.
"""

import dataclasses
import itertools
import logging
import math
import threading
import time
from collections.abc import Callable, Sequence
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
    """Configuration for simple Stochastic Weight Averaging (SWA).

    Attributes:
        start_interval: Epoch/interval index to begin collecting SWA snapshots.
        update_interval: Frequency (in intervals) for collecting SWA snapshots.
        min_snapshots_for_eval: Minimum snapshots before using SWA parameters.
        max_snapshots: Optional cap on the number of stored snapshots.
        prefer_swa_params: Whether to use SWA params for evaluation once available.
        stage_loss_factory: Optional loss factory to swap in at SWA stage start.
        stage_loss_kwargs: Optional kwargs for the stage-two loss.
    """

    start_interval: int = 0
    update_interval: int = 1
    min_snapshots_for_eval: int = 1
    max_snapshots: int | None = None
    prefer_swa_params: bool = True
    stage_loss_factory: Callable | None = None
    stage_loss_kwargs: dict[str, float] | None = None

    def __post_init__(self) -> None:
        """Validate SWA configuration values."""
        if self.start_interval < 0:
            raise ValueError('SWA start_interval must be non-negative')
        if self.update_interval <= 0:
            raise ValueError('SWA update_interval must be positive')
        if self.min_snapshots_for_eval <= 0:
            raise ValueError('SWA min_snapshots_for_eval must be positive')

    def should_collect(self, interval: int, num_snapshots: int) -> bool:
        """Return True if a snapshot should be collected at this interval."""
        if interval < self.start_interval:
            return False
        if self.max_snapshots is not None and num_snapshots >= self.max_snapshots:
            return False
        offset = interval - self.start_interval
        return offset % self.update_interval == 0

    def should_use(self, num_snapshots: int) -> bool:
        """Return True if SWA parameters should be used for evaluation."""
        if not self.prefer_swa_params:
            return False
        return num_snapshots >= self.min_snapshots_for_eval


def _sanitize_grads(grads):
    """Replace NaNs/inf/float0 leaves to keep the optimizer stable.

    Args:
        grads: Gradient pytree produced by JAX/Optax.

    Returns:
        Sanitized gradient pytree with finite floating-point values.
    """

    def _sanitize(leaf):
        """Convert unsupported dtypes and replace NaNs/infs."""
        arr = jnp.asarray(leaf)
        if 'float0' in str(arr.dtype):
            arr = jnp.zeros_like(arr, dtype=jnp.float32)
        return jnp.nan_to_num(arr)

    return jax.tree_util.tree_map(_sanitize, grads)


def _prefetch_to_device(iterator, capacity: int, device_put_fn: Callable[[Any], Any]):
    """Prefetch items onto device(s) using a bounded queue.

    Args:
        iterator: Source iterator yielding host batches.
        capacity: Max items to buffer; <=0 disables device prefetching.
        device_put_fn: Callable that moves one item to device(s).

    Returns:
        Iterator yielding device-resident items.
    """
    if capacity is None or capacity <= 0:
        return iterator
    queue: Queue = Queue(maxsize=capacity)
    sentinel = object()
    total_hint = getattr(iterator, 'total_batches_hint', 0)

    def _producer():
        """Fill the queue with device-resident items."""
        try:
            for item in iterator:
                queue.put(device_put_fn(item))
        finally:
            queue.put(sentinel)

    threading.Thread(target=_producer, daemon=True).start()

    class _Prefetched:
        """Iterator wrapper that yields device-prefetched items."""

        def __init__(self):
            """Initialize the prefetched iterator wrapper."""
            self._done = False
            self.total_batches_hint = total_hint

        def __iter__(self):
            """Return self as an iterator."""
            return self

        def __next__(self):
            """Return the next device-prefetched item or stop iteration."""
            if self._done:
                raise StopIteration
            item = queue.get()
            if item is sentinel:
                self._done = True
                raise StopIteration
            return item

    return _Prefetched()


def _group_batches(iterator, group_size: int, total_hint: int = 0):
    """Group consecutive batches into fixed-size chunks.

    Args:
        iterator: Iterator yielding individual batches.
        group_size: Number of batches per grouped chunk.
        total_hint: Optional total batch hint to propagate.

    Returns:
        Iterator yielding lists of length `group_size`.
    """
    if group_size <= 1:
        return iterator

    class _Grouped:
        """Iterator wrapper that yields fixed-size groups of batches."""

        def __init__(self):
            """Initialize the grouped iterator wrapper."""
            self._iter = iter(iterator)
            self.total_batches_hint = total_hint

        def __iter__(self):
            """Return self as an iterator."""
            return self

        def __next__(self):
            """Return the next grouped chunk or stop iteration."""
            chunk = []
            for _ in range(group_size):
                try:
                    chunk.append(next(self._iter))
                except StopIteration:
                    raise StopIteration
            return chunk

    return _Grouped()


@dataclasses.dataclass
class _MetricAccumulator:
    """Accumulate metric sums across batches for final reduction."""

    count: float = 0.0
    sum_abs_delta: float = 0.0
    sum_abs_target: float = 0.0
    sum_sq_delta: float = 0.0
    sum_sq_target: float = 0.0

    def update(self, batch: dict | None) -> None:
        """Update accumulator with a batch-level metric dict."""
        if not batch:
            return
        self.count += float(batch['count'])
        self.sum_abs_delta += float(batch['sum_abs_delta'])
        self.sum_abs_target += float(batch['sum_abs_target'])
        self.sum_sq_delta += float(batch['sum_sq_delta'])
        self.sum_sq_target += float(batch['sum_sq_target'])

    def finalize(self) -> dict[str, float] | None:
        """Finalize accumulated metrics into MAE/RMSE and relative variants."""
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
    """Compute masked error statistics for a single tensor.

    Args:
        delta: Difference between reference and prediction.
        target: Reference tensor (used for relative error stats).
        mask: Broadcastable mask indicating valid entries.

    Returns:
        Dict of aggregated counts and sums used to derive MAE/RMSE.
    """
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


def _graph_mask(graph: jraph.GraphsTuple) -> jnp.ndarray:
    """Build a per-graph mask that excludes padded or zero-weight graphs."""
    mask = jraph.get_graph_padding_mask(graph).astype(jnp.float32)
    if mask.shape == graph.n_node.shape:
        non_empty = jnp.asarray(graph.n_node > 0, dtype=jnp.float32)
        mask = mask * non_empty
    weights = getattr(graph.globals, 'weight', None)
    if weights is not None:
        weights_arr = jnp.asarray(weights)
        if weights_arr.shape == mask.shape:
            mask = mask * (weights_arr > 0).astype(jnp.float32)
    return mask


def _compute_eval_batch_metrics(
    loss_fn, graph, pred_graph, pred_outputs, mask_override=None
):
    """Compute loss and per-target metric stats for a single evaluation batch.

    Args:
        loss_fn: Loss function returning per-graph losses.
        graph: Reference graph with target values.
        pred_graph: Graph with predicted values attached.
        pred_outputs: Raw model outputs (energy/forces/stress, etc.).
        mask_override: Optional mask to further filter graphs.

    Returns:
        Dict containing loss and metric statistic aggregates for the batch.
    """
    graph_mask = _graph_mask(graph)
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
    """Create a jitted per-batch metric computation function."""

    def _batch(graph, pred_graph, pred_outputs, graph_mask):
        """Compute metric stats for a single batch."""
        return _compute_eval_batch_metrics(
            loss_fn, graph, pred_graph, pred_outputs, graph_mask
        )

    return jax.jit(_batch)


def _apply_metric_stats(
    aux: dict[str, Any],
    stats: dict[str, float] | None,
    *,
    mae_key: str | None = None,
    rel_mae_key: str | None = None,
    rmse_key: str | None = None,
    rel_rmse_key: str | None = None,
) -> None:
    """Populate an aux dict with derived metrics from a stats dict."""
    if not stats:
        return
    if mae_key:
        aux[mae_key] = stats['mae']
    if rel_mae_key:
        aux[rel_mae_key] = stats['rel_mae']
    if rmse_key:
        aux[rmse_key] = stats['rmse']
    if rel_rmse_key:
        aux[rel_rmse_key] = stats['rel_rmse']


def _padding_amounts(graph) -> tuple[int, int, int, int, int, int]:
    """Compute padding statistics for a (possibly grouped) batch.

    Args:
        graph: GraphsTuple or sequence of GraphsTuple items.

    Returns:
        Tuple of (node_cap, edge_cap, graph_cap, pad_nodes, pad_edges, pad_graphs).
    """
    if isinstance(graph, Sequence) and not isinstance(graph, jraph.GraphsTuple):
        totals = [0, 0, 0, 0, 0, 0]
        for item in graph:
            values = _padding_amounts(item)
            totals = [left + right for left, right in zip(totals, values)]
        return tuple(totals)
    node_ref = getattr(graph.nodes, 'positions', None)
    if node_ref is None:
        node_ref = graph.nodes[0]
    node_cap = int(np.asarray(node_ref).shape[0])
    edge_cap = int(np.asarray(graph.senders).shape[0])
    graph_cap = int(np.asarray(graph.n_node).shape[0])
    pad_nodes = int(np.asarray(jraph.get_number_of_padding_with_graphs_nodes(graph)))
    pad_edges = int(np.asarray(jraph.get_number_of_padding_with_graphs_edges(graph)))
    pad_graphs = int(np.asarray(jraph.get_number_of_padding_with_graphs_graphs(graph)))
    return node_cap, edge_cap, graph_cap, pad_nodes, pad_edges, pad_graphs


def train(
    params: Any,
    total_loss_fn: Callable,  # loss_fn(params, graph) -> [num_graphs]
    train_loader: data.StreamingGraphDataLoader,
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
    lr_scale_by_graphs: bool = True,
    device_prefetch_batches: int | None = None,
):
    """Yield training state for each epoch while updating parameters.

    This is the low-level training iterator used by `gin_functions.train`. It
    performs gradient updates, optional EMA/SWA tracking, padding diagnostics,
    and yields updated parameters and optimizer state after each epoch. Fixed
    batch shapes are assumed so that the compiled model can be reused across
    epochs without recompilation.

    Args:
        params: Trainable parameters (and optional config state).
        total_loss_fn: Callable returning per-graph loss values.
        train_loader: Streaming data loader yielding graph batches.
        gradient_transform: Optax gradient transformation.
        optimizer_state: Initial optimizer state.
        ema_decay: Optional EMA decay for evaluation parameters.
        progress_bar: Whether to show a progress bar during training.
        swa_config: Optional SWA configuration.
        max_grad_norm: Optional gradient clipping threshold.
        schedule_free_eval_fn: Optional schedule-free eval parameter extractor.
        start_interval: Starting epoch/interval index.
        data_seed: Optional data shuffling seed for the loader.
        lr_scale_by_graphs: Whether to scale LR per batch size.
        device_prefetch_batches: Optional number of batches to prefetch to devices.
            Defaults to max(loader prefetch, 2) when None.

    Yields:
        Tuple of (epoch, trainable_params, optimizer_state, eval_params).
    """
    num_updates = 0
    ema_params = params
    eval_params = params
    swa_state = None
    if swa_config is not None:
        swa_state = {'params': None, 'count': 0}

    logging.info('Started training')

    local_devices = jax.local_devices()
    local_device_count = max(1, len(local_devices))

    def _prepare_graph_for_devices(graph: jraph.GraphsTuple):
        """Ensure `graph` has a leading axis matching local device count.

        Args:
            graph: Batch graph or list of per-device microbatches.

        Returns:
            GraphsTuple with leading device axis suitable for pmap.
        """

        if local_device_count == 1:
            if graph.n_node.ndim == 1:
                return jax.tree_util.tree_map(lambda x: x[None, ...], graph)
            return graph

        if isinstance(graph, Sequence) and not isinstance(graph, jraph.GraphsTuple):
            return data.prepare_sharded_batch(graph, local_device_count)

        if graph.n_node.ndim == 1:
            return data.prepare_sharded_batch(graph, local_device_count)

        if graph.n_node.shape[0] != local_device_count:
            raise ValueError(
                'Expected microbatches with leading axis equal to the number of local '
                f'devices ({local_device_count}), got axis size {graph.n_node.shape[0]}.'
            )
        return graph

    @partial(
        jax.pmap,
        in_axes=(None, 0),
        out_axes=None,
        axis_name='devices',
        devices=local_devices,
    )
    def grad_fn(params, graph: jraph.GraphsTuple):
        """Compute per-device loss and gradients for a sharded batch."""
        # graph is assumed to be padded by jraph.pad_with_graphs
        mask = _graph_mask(graph)

        def _loss_fn(trainable):
            """Compute masked total loss for gradient evaluation."""
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
    ) -> tuple[float, Any, Any, Any, jnp.ndarray]:
        """Apply one optimizer step and update EMA parameters."""
        n, loss, grad = grad_fn(params, graph)
        loss = loss / n
        grad = jax.tree_util.tree_map(lambda x: x / n, grad)
        if lr_scale_by_graphs:
            grad = jax.tree_util.tree_map(lambda g: g * n, grad)
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
        return loss, params, optimizer_state, ema_params, n

    process_index = getattr(jax, 'process_index', lambda: 0)()
    is_primary = process_index == 0
    process_count = getattr(jax, 'process_count', lambda: 1)()

    def _resolve_lr(step: int) -> float | None:
        """Return the current learning rate for a given step if available."""
        schedule = getattr(gradient_transform, 'lr_schedule', None)
        if schedule is None:
            schedule = getattr(gradient_transform, 'schedule', None)
        if schedule is None:
            schedule = getattr(gradient_transform, 'learning_rate', None)
        if schedule is None:
            return None
        try:
            value = schedule(step) if callable(schedule) else schedule
        except Exception:  # pragma: no cover - best effort for custom schedules
            return None
        return float(np.asarray(value))

    def _swa_average(current_avg, new_tree, new_count: int):
        """Update an SWA running average tree with a new snapshot."""
        if current_avg is None:
            return new_tree
        alpha = 1.0 / float(new_count)
        return jax.tree_util.tree_map(
            lambda avg, new: avg + (new - avg) * alpha,
            current_avg,
            new_tree,
        )

    def _epoch_batches(epoch: int):
        """Build an epoch batch iterator and total batch count."""
        iterator = train_loader.iter_batches(
            epoch=epoch,
            seed=data_seed,
            process_count=process_count,
            process_index=process_index,
        )
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
        if local_device_count > 1:
            total_batches = total_batches // local_device_count
            iterator = _group_batches(iterator, local_device_count, total_batches)
        return iterator, total_batches

    initial_eval = eval_params
    if schedule_free_eval_fn is not None:
        initial_eval = schedule_free_eval_fn(optimizer_state, params)
    yield start_interval, params, optimizer_state, initial_eval

    for epoch in itertools.count(start_interval):
        epoch_batches_iter, effective_steps = _epoch_batches(epoch)
        if effective_steps <= 0:
            raise ValueError(
                'iter_batches() produced no data; reduce process_count or provide more data.'
            )
        if is_primary:
            lr_value = _resolve_lr(num_updates)
            if lr_value is not None:
                logging.info('Epoch %s learning rate %.6e', epoch + 1, lr_value)
        host_prefetch_cap = int(getattr(train_loader, '_prefetch_batches', 0) or 0)
        device_prefetch_cap = device_prefetch_batches
        if device_prefetch_cap is None:
            device_prefetch_cap = max(host_prefetch_cap, 2)
        device_prefetch_cap = int(device_prefetch_cap or 0)
        device_prefetch_active = device_prefetch_cap > 0
        if device_prefetch_active:
            source_iter = epoch_batches_iter

            def _device_put(graph):
                """Prepare/pad and move graphs to device while preserving stats."""
                padding = _padding_amounts(graph)
                graph = _prepare_graph_for_devices(graph)
                if local_device_count <= 1:
                    graph = jax.device_put(graph)
                return graph, padding

            epoch_batches_iter = _prefetch_to_device(
                source_iter, device_prefetch_cap, _device_put
            )
        p_bar = tqdm.tqdm(
            total=effective_steps,
            desc=f'Epoch {epoch + 1}',
            disable=not (progress_bar and is_primary),
        )
        batches_in_epoch = 0
        padding_stats = {
            'pad_nodes': 0,
            'node_cap': 0,
            'pad_edges': 0,
            'edge_cap': 0,
            'pad_graphs': 0,
            'graph_cap': 0,
        }
        for item in epoch_batches_iter:
            if device_prefetch_active:
                graph, padding = item
                node_cap, edge_cap, graph_cap, pad_nodes, pad_edges, pad_graphs = (
                    padding
                )
            else:
                graph = item
                node_cap, edge_cap, graph_cap, pad_nodes, pad_edges, pad_graphs = (
                    _padding_amounts(graph)
                )
            padding_stats['node_cap'] += node_cap
            padding_stats['edge_cap'] += edge_cap
            padding_stats['graph_cap'] += graph_cap
            padding_stats['pad_nodes'] += pad_nodes
            padding_stats['pad_edges'] += pad_edges
            padding_stats['pad_graphs'] += pad_graphs
            if not device_prefetch_active:
                graph = _prepare_graph_for_devices(graph)
            num_updates += 1
            num_updates_value = jnp.asarray(num_updates, dtype=jnp.int32)
            start_time = time.time()
            loss, params, optimizer_state, ema_params, n_graphs = update_fn(
                params, optimizer_state, ema_params, num_updates_value, graph
            )
            loss = float(loss)
            n_graphs_value = float(n_graphs)
            if is_primary and n_graphs_value == 0.0:
                logging.warning(
                    'Batch %s contains no valid graphs; gradients are zero.',
                    num_updates,
                )
            if is_primary:
                p_bar.set_postfix({'loss': f'{loss:7.3f}'})
            p_bar.update(1)
            batches_in_epoch += 1
        if p_bar.total is not None and p_bar.n != p_bar.total:
            p_bar.total = p_bar.n
            p_bar.refresh()
        p_bar.close()
        if is_primary and padding_stats['node_cap'] > 0:
            if getattr(train_loader, 'streaming', False):
                logging.info(
                    'Streaming padding: batches=%s nodes avg_pad=%.1f (%.1f%%) '
                    'edges avg_pad=%.1f (%.1f%%) graphs avg_pad=%.2f (%.1f%%)',
                    batches_in_epoch,
                    padding_stats['pad_nodes'] / batches_in_epoch,
                    100.0 * padding_stats['pad_nodes'] / padding_stats['node_cap'],
                    padding_stats['pad_edges'] / batches_in_epoch,
                    100.0 * padding_stats['pad_edges'] / padding_stats['edge_cap'],
                    padding_stats['pad_graphs'] / batches_in_epoch,
                    100.0 * padding_stats['pad_graphs'] / padding_stats['graph_cap'],
                )
            else:
                logging.debug(
                    'Epoch %s padding: nodes=%.6f%% edges=%.6f%% graphs=%.2f%%',
                    epoch + 1,
                    100.0 * padding_stats['pad_nodes'] / padding_stats['node_cap'],
                    100.0 * padding_stats['pad_edges'] / padding_stats['edge_cap'],
                    100.0 * padding_stats['pad_graphs'] / padding_stats['graph_cap'],
                )
        if batches_in_epoch <= 0:
            raise ValueError(
                'iter_batches() produced no data; reduce process_count or provide more data.'
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
    data_loader: data.StreamingGraphDataLoader,
    name: str = 'Evaluation',
    progress_bar: bool = True,
    device_prefetch_batches: int | None = None,
) -> tuple[float, dict[str, Any]]:
    r"""Evaluate the predictor on the given data loader.

    Args:
        predictor: Callable `predictor(params, graph)` returning energy/forces/stress.
        params: Parameters used by the predictor.
        loss_fn: Loss callable `loss_fn(graph, output) -> per-graph loss`.
        data_loader: Streaming loader yielding evaluation batches.
        name: Name used for progress bar and logging.
        progress_bar: Whether to display a progress bar.
        device_prefetch_batches: Optional number of batches to prefetch to devices.
            Defaults to max(loader prefetch, 2) when None.

    Returns:
        Tuple of (average_loss, metrics_dict) aggregated over the dataset.

    Evaluation uses the same fixed-shape batches as training so the compiled
    model can be reused without shape-driven recompilation.
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

    padding_stats = {
        'pad_nodes': 0,
        'node_cap': 0,
        'pad_edges': 0,
        'edge_cap': 0,
        'pad_graphs': 0,
        'graph_cap': 0,
    }

    local_devices = jax.local_devices()
    local_device_count = max(1, len(local_devices))
    process_index = getattr(jax, 'process_index', lambda: 0)()
    process_count = getattr(jax, 'process_count', lambda: 1)()
    is_primary = process_index == 0
    iterator = data_loader.iter_batches(
        epoch=0,
        seed=None,
        process_count=process_count,
        process_index=process_index,
    )
    total_hint = getattr(iterator, 'total_batches_hint', 0)
    if not total_hint:
        approx_length = getattr(data_loader, 'approx_length', None)
        if callable(approx_length):
            try:
                total_hint = int(approx_length())
            except Exception:  # pragma: no cover - defensive fallback
                total_hint = 0
    if not total_hint:
        raise ValueError(
            'Evaluation loader did not provide a total batch count; ensure it '
            'implements iter_batches() with a total_batches_hint or approx_length().'
        )

    def _prepare_graph_for_devices(graph: jraph.GraphsTuple):
        """Prepare a graph batch for multi-device evaluation."""
        if local_device_count == 1:
            if graph.n_node.ndim == 1:
                return jax.tree_util.tree_map(lambda x: x[None, ...], graph)
            return graph
        if isinstance(graph, Sequence) and not isinstance(graph, jraph.GraphsTuple):
            return data.prepare_sharded_batch(graph, local_device_count)
        if graph.n_node.ndim == 1:
            return data.prepare_sharded_batch(graph, local_device_count)
        if graph.n_node.shape[0] != local_device_count:
            raise ValueError(
                'Expected microbatches with leading axis equal to the number of local '
                f'devices ({local_device_count}), got axis size {graph.n_node.shape[0]}.'
            )
        return graph

    def _psum_metrics(tree):
        """Sum metric trees across devices for pmap evaluation."""
        return jax.tree_util.tree_map(
            lambda x: jax.lax.psum(x, 'devices') if x is not None else None,
            tree,
            is_leaf=lambda x: x is None,
        )

    @partial(
        jax.pmap,
        in_axes=(None, 0),
        out_axes=0,
        axis_name='devices',
        devices=local_devices,
    )
    def _eval_step(params_, graph):
        """Run the predictor and compute per-batch metric stats on devices."""
        output = predictor(params_, graph)
        nodes = graph.nodes
        if output.get('forces') is not None:
            nodes = nodes._replace(forces=output.get('forces'))
        globals_updates = {}
        for key in ('energy', 'stress', 'virials', 'dipole', 'polarizability'):
            if key in output and output[key] is not None:
                globals_updates[key] = output[key]
        globals_attr = (
            graph.globals._replace(**globals_updates)
            if globals_updates
            else graph.globals
        )
        pred_graph = graph._replace(nodes=nodes, globals=globals_attr)

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

        batch_metrics = batch_metrics_fn(graph, pred_graph, pred_outputs, None)
        return _psum_metrics(batch_metrics)

    def _process_batch(
        ref_graph: jraph.GraphsTuple,
        *,
        p_bar=None,
        prepared: bool = False,
        padding: tuple[int, int, int, int, int, int] | None = None,
    ) -> None:
        """Update loss/metric accumulators from a single batch."""
        nonlocal total_loss, num_graphs
        if padding is None:
            padding = _padding_amounts(ref_graph)
        node_cap, edge_cap, graph_cap, pad_nodes, pad_edges, pad_graphs = padding
        padding_stats['node_cap'] += node_cap
        padding_stats['edge_cap'] += edge_cap
        padding_stats['graph_cap'] += graph_cap
        padding_stats['pad_nodes'] += pad_nodes
        padding_stats['pad_edges'] += pad_edges
        padding_stats['pad_graphs'] += pad_graphs

        device_graph = ref_graph if prepared else _prepare_graph_for_devices(ref_graph)
        batch_metrics = _eval_step(params, device_graph)
        batch_metrics = data.unreplicate_from_local_devices(batch_metrics)

        def _squeeze(value):
            if value is None:
                return None
            arr = np.asarray(value)
            if arr.ndim == 0:
                return value
            if arr.shape[0] == 1:
                return arr[0]
            return value

        batch_metrics = jax.tree_util.tree_map(
            _squeeze, batch_metrics, is_leaf=lambda x: x is None
        )
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
        if p_bar is not None:
            p_bar.set_postfix({'n': int(num_graphs)})

    start_time = time.time()

    eval_iterator = iterator
    host_prefetch_cap = int(getattr(data_loader, '_prefetch_batches', 0) or 0)
    device_prefetch_cap = device_prefetch_batches
    if device_prefetch_cap is None:
        device_prefetch_cap = max(host_prefetch_cap, 2)
    device_prefetch_cap = int(device_prefetch_cap or 0)
    device_prefetch_active = device_prefetch_cap > 0
    if device_prefetch_active:
        source_iter = eval_iterator

        def _device_put(graph):
            """Prepare/pad and move graphs to device while preserving stats."""
            padding = _padding_amounts(graph)
            graph = _prepare_graph_for_devices(graph)
            if local_device_count <= 1:
                graph = jax.device_put(graph)
            return graph, padding

        eval_iterator = _prefetch_to_device(
            source_iter, device_prefetch_cap, _device_put
        )

    p_bar = tqdm.tqdm(
        eval_iterator,
        desc=name,
        total=total_hint,
        disable=not (progress_bar and is_primary),
    )

    batches_seen = 0
    for item in p_bar:
        if device_prefetch_active:
            ref_graph, padding = item
            _process_batch(ref_graph, p_bar=p_bar, prepared=True, padding=padding)
        else:
            _process_batch(item, p_bar=p_bar)
        batches_seen += 1
    if p_bar.total is not None and p_bar.n != p_bar.total:
        p_bar.total = p_bar.n
        p_bar.refresh()
    p_bar.close()

    if num_graphs <= 0:
        logging.warning(f'No graphs in data_loader ! Returning 0.0 for {name}')
        return 0.0, {}

    if padding_stats['node_cap'] > 0:
        if getattr(data_loader, 'streaming', False):
            logging.info(
                'Streaming padding: batches=%s nodes avg_pad=%.1f (%.1f%%) '
                'edges avg_pad=%.1f (%.1f%%) graphs avg_pad=%.2f (%.1f%%)',
                batches_seen,
                padding_stats['pad_nodes'] / batches_seen,
                100.0 * padding_stats['pad_nodes'] / padding_stats['node_cap'],
                padding_stats['pad_edges'] / batches_seen,
                100.0 * padding_stats['pad_edges'] / padding_stats['edge_cap'],
                padding_stats['pad_graphs'] / batches_seen,
                100.0 * padding_stats['pad_graphs'] / padding_stats['graph_cap'],
            )
        else:
            logging.debug(
                '%s padding: nodes=%.6f%% edges=%.6f%% graphs=%.2f%%',
                name,
                100.0 * padding_stats['pad_nodes'] / padding_stats['node_cap'],
                100.0 * padding_stats['pad_edges'] / padding_stats['edge_cap'],
                100.0 * padding_stats['pad_graphs'] / padding_stats['graph_cap'],
            )

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

    metric_map = [
        ('energy', 'mae_e', 'rel_mae_e', 'rmse_e', 'rel_rmse_e'),
        (
            'energy_per_atom',
            'mae_e_per_atom',
            'rel_mae_e_per_atom',
            'rmse_e_per_atom',
            'rel_rmse_e_per_atom',
        ),
        ('forces', 'mae_f', 'rel_mae_f', 'rmse_f', 'rel_rmse_f'),
        ('stress', 'mae_s', 'rel_mae_s', 'rmse_s', 'rel_rmse_s'),
        ('virials', 'mae_virials', None, 'rmse_virials', None),
        ('virials_per_atom', None, None, 'rmse_virials_per_atom', None),
        ('dipole', 'mae_mu', 'rel_mae_mu', 'rmse_mu', 'rel_rmse_mu'),
        ('dipole_per_atom', 'mae_mu_per_atom', None, 'rmse_mu_per_atom', None),
        (
            'polar',
            'mae_polarizability',
            None,
            'rmse_polarizability',
            None,
        ),
        (
            'polar_per_atom',
            'mae_polarizability_per_atom',
            None,
            'rmse_polarizability_per_atom',
            None,
        ),
    ]
    for key, mae_key, rel_mae_key, rmse_key, rel_rmse_key in metric_map:
        _apply_metric_stats(
            aux,
            metric_accumulators[key].finalize(),
            mae_key=mae_key,
            rel_mae_key=rel_mae_key,
            rmse_key=rmse_key,
            rel_rmse_key=rel_rmse_key,
        )
    if aux['mae_s'] is not None:
        aux['mae_stress'] = aux['mae_s']
        aux['rel_mae_stress'] = aux['rel_mae_s']
        aux['rmse_stress'] = aux['rmse_s']
        aux['rel_rmse_stress'] = aux['rel_rmse_s']

    return avg_loss, aux


_GRAPH_OUTPUT_KEYS = {
    'energy',
    'stress',
    'virials',
    'dipole',
    'polarizability',
    'polar',
}
_NODE_OUTPUT_KEYS = {
    'forces',
}
_EDGE_OUTPUT_KEYS: set[str] = set()


def _filter_by_mask(values: Sequence[Any], mask: np.ndarray) -> list[Any]:
    if len(values) != len(mask):
        raise ValueError(
            f'Mismatched graph mask length ({len(values)} values vs {len(mask)} mask).'
        )
    return [value for value, keep in zip(values, mask) if keep]


def _split_prediction_outputs(
    outputs: dict[str, Any],
    graph: jraph.GraphsTuple,
) -> tuple[list[int], dict[str, list[Any]]]:
    graph_ids = getattr(graph.globals, 'graph_id', None)
    if graph_ids is None:
        raise ValueError(
            'Streaming prediction requires graph.globals.graph_id to be set.'
        )
    graph_ids = np.asarray(graph_ids).reshape(-1)
    graph_mask = np.asarray(jraph.get_graph_padding_mask(graph), dtype=bool)
    if graph_ids.shape[0] != graph_mask.shape[0]:
        raise ValueError(
            'graph_id length does not match graph batch size '
            f'({graph_ids.shape[0]} vs {graph_mask.shape[0]}).'
        )

    n_node = np.asarray(graph.n_node, dtype=int)
    n_edge = np.asarray(graph.n_edge, dtype=int)
    total_nodes = int(n_node.sum())
    total_edges = int(n_edge.sum())
    n_graphs = int(n_node.shape[0])

    def _split_by_counts(arr: np.ndarray, counts: np.ndarray) -> list[np.ndarray]:
        if counts.size == 0:
            return []
        total = int(counts.sum())
        if total == 0:
            return [arr[:0]] * len(counts)
        splits = np.cumsum(counts)[:-1]
        return np.split(arr[:total], splits, axis=0)

    per_graph_outputs: dict[str, list[Any]] = {}
    for key, value in outputs.items():
        if value is None:
            continue
        arr = np.asarray(value)
        if arr.ndim == 0:
            continue
        if key in _NODE_OUTPUT_KEYS:
            chunks = _split_by_counts(arr, n_node)
            per_graph_outputs[key] = _filter_by_mask(chunks, graph_mask)
            continue
        if key in _EDGE_OUTPUT_KEYS:
            chunks = _split_by_counts(arr, n_edge)
            per_graph_outputs[key] = _filter_by_mask(chunks, graph_mask)
            continue
        if key in _GRAPH_OUTPUT_KEYS:
            arr = arr[:n_graphs]
            per_graph_outputs[key] = _filter_by_mask(list(arr), graph_mask)
            continue

        if arr.shape[0] >= total_nodes and total_nodes > 0:
            chunks = _split_by_counts(arr, n_node)
            per_graph_outputs[key] = _filter_by_mask(chunks, graph_mask)
        elif arr.shape[0] >= total_edges and total_edges > 0:
            chunks = _split_by_counts(arr, n_edge)
            per_graph_outputs[key] = _filter_by_mask(chunks, graph_mask)
        elif arr.shape[0] >= n_graphs:
            arr = arr[:n_graphs]
            per_graph_outputs[key] = _filter_by_mask(list(arr), graph_mask)
        else:
            raise ValueError(
                f'Output {key} has incompatible leading dimension {arr.shape[0]} '
                f'(graphs={n_graphs}, nodes={total_nodes}, edges={total_edges}).'
            )

    graph_ids = [int(val) for val in graph_ids[graph_mask]]
    return graph_ids, per_graph_outputs


def predict_streaming(
    predictor: Callable,
    params: Any,
    data_loader: data.StreamingGraphDataLoader,
    name: str = 'Prediction',
    progress_bar: bool = True,
    device_prefetch_batches: int | None = None,
) -> tuple[list[int], dict[str, list[Any]]]:
    """Run the predictor on a streaming loader and order outputs by graph_id.

    Returns per-graph outputs sorted by graph_id for the local process. For
    multi-process runs, callers should gather results across processes.
    """
    local_devices = jax.local_devices()
    local_device_count = max(1, len(local_devices))
    process_index = getattr(jax, 'process_index', lambda: 0)()
    process_count = getattr(jax, 'process_count', lambda: 1)()
    is_primary = process_index == 0

    iterator = data_loader.iter_batches(
        epoch=0,
        seed=None,
        process_count=process_count,
        process_index=process_index,
    )
    total_hint = getattr(iterator, 'total_batches_hint', 0)
    if not total_hint:
        approx_length = getattr(data_loader, 'approx_length', None)
        if callable(approx_length):
            try:
                total_hint = int(approx_length())
            except Exception:  # pragma: no cover - defensive fallback
                total_hint = 0

    @partial(
        jax.pmap,
        in_axes=(None, 0),
        out_axes=0,
        axis_name='devices',
        devices=local_devices,
    )
    def _predict_step(params_, graph):
        return predictor(params_, graph)

    def _prepare_device_graphs(graph):
        if local_device_count <= 1:
            return graph, [graph]
        device_graphs = data.split_graphs_for_devices(graph, local_device_count)
        device_batch = data.prepare_sharded_batch(device_graphs, local_device_count)
        return device_batch, device_graphs

    host_prefetch_cap = int(getattr(data_loader, '_prefetch_batches', 0) or 0)
    device_prefetch_cap = device_prefetch_batches
    if device_prefetch_cap is None:
        device_prefetch_cap = max(host_prefetch_cap, 2)
    device_prefetch_cap = int(device_prefetch_cap or 0)
    device_prefetch_active = device_prefetch_cap > 0
    if device_prefetch_active:
        source_iter = iterator

        def _device_put(graph):
            device_batch, device_graphs = _prepare_device_graphs(graph)
            device_batch = jax.device_put(device_batch)
            return device_batch, device_graphs

        iterator = _prefetch_to_device(source_iter, device_prefetch_cap, _device_put)

    p_bar = tqdm.tqdm(
        iterator,
        desc=name,
        total=total_hint or None,
        disable=not (progress_bar and is_primary),
    )

    def _select_device_output(value, device_idx):
        if value is None:
            return None
        arr = np.asarray(value)
        if arr.ndim > 0 and arr.shape[0] == local_device_count:
            return value[device_idx]
        return value

    graph_ids: list[int] = []
    outputs: dict[str, list[Any]] = {}
    for item in p_bar:
        if device_prefetch_active:
            device_batch, device_graphs = item
        else:
            device_batch, device_graphs = _prepare_device_graphs(item)
        if local_device_count <= 1:
            raw_outputs = predictor(params, device_batch)
            raw_outputs = jax.device_get(raw_outputs)
            batch_graph_ids, batch_outputs = _split_prediction_outputs(
                raw_outputs, device_graphs[0]
            )
        else:
            raw_outputs = _predict_step(params, device_batch)
            raw_outputs = jax.device_get(raw_outputs)
            batch_graph_ids = []
            batch_outputs: dict[str, list[Any]] = {}
            for device_idx, device_graph in enumerate(device_graphs):
                device_outputs = jax.tree_util.tree_map(
                    lambda x: _select_device_output(x, device_idx),
                    raw_outputs,
                    is_leaf=lambda x: x is None,
                )
                ids, per_graph = _split_prediction_outputs(device_outputs, device_graph)
                for key in batch_outputs:
                    if key not in per_graph:
                        batch_outputs[key].extend([None] * len(ids))
                for key, values in per_graph.items():
                    if key not in batch_outputs:
                        batch_outputs[key] = [None] * len(batch_graph_ids)
                    batch_outputs[key].extend(values)
                batch_graph_ids.extend(ids)

        for key in outputs:
            if key not in batch_outputs:
                outputs[key].extend([None] * len(batch_graph_ids))
        for key, values in batch_outputs.items():
            if key not in outputs:
                outputs[key] = [None] * len(graph_ids)
            outputs[key].extend(values)
        graph_ids.extend(batch_graph_ids)

    if p_bar.total is not None and p_bar.n != p_bar.total:
        p_bar.total = p_bar.n
        p_bar.refresh()
    p_bar.close()

    if not graph_ids:
        return [], {}

    order = np.argsort(np.asarray(graph_ids))
    ordered_graph_ids = [graph_ids[idx] for idx in order]
    ordered_outputs = {
        key: [values[idx] for idx in order] for key, values in outputs.items()
    }
    return ordered_graph_ids, ordered_outputs
