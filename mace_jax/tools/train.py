import dataclasses
import itertools
import logging
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

from mace_jax import data, tools


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


def train(
    params: Any,
    total_loss_fn: Callable,  # loss_fn(params, graph) -> [num_graphs]
    train_loader: data.GraphDataLoader,  # device parallel done on the (optional) extra dimension
    gradient_transform: Any,
    optimizer_state: dict[str, Any],
    steps_per_interval: int | None,
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
            if steps_per_interval is None or steps_per_interval <= 0:
                raise ValueError(
                    'Provide steps_per_interval when total_batches_hint is unavailable.'
                )
            return iterator, steps_per_interval
        return iterator, total_batches

    def _legacy_interval_loader():
        while True:
            yield from train_loader

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
            if steps_per_interval not in (None, effective_steps):
                logging.warning(
                    'Ignoring steps_per_interval=%s in favor of total_batches=%s when using iter_batches().',
                    steps_per_interval,
                    effective_steps,
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

            effective_steps = steps_per_interval
            if effective_steps is None or effective_steps <= 0:
                raise ValueError(
                    'steps_per_interval must be specified and positive when the '
                    'training loader does not expose iter_batches().'
                )

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
    num_graphs = 0

    delta_es_list = []
    es_list = []

    delta_es_per_atom_list = []
    es_per_atom_list = []

    delta_fs_list = []
    fs_list = []

    delta_stress_list = []
    stress_list = []
    delta_virials_list = []
    delta_virials_per_atom_list = []
    virials_list = []
    delta_dipoles_list = []
    delta_dipoles_per_atom_list = []
    dipoles_list = []
    delta_polar_list = []
    delta_polar_per_atom_list = []

    if hasattr(predictor, '_cache_size'):
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
        nodes = ref_graph.nodes._replace(forces=output.get('forces'))
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

        ref_graph = jraph.unpad_with_graphs(ref_graph)
        pred_graph = jraph.unpad_with_graphs(pred_graph)
        valid_mask = np.asarray(ref_graph.n_node > 0)
        weights = getattr(ref_graph.globals, 'weight', None)
        if weights is not None:
            valid_mask = valid_mask & (np.asarray(weights) > 0)
        if not valid_mask.any():
            continue
        num_graphs += int(valid_mask.sum())
        atom_counts = jnp.maximum(ref_graph.n_node, 1.0)

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

        loss = jnp.sum(loss_fn(ref_graph, pred_outputs))
        total_loss += float(loss)
        p_bar.set_postfix({'n': num_graphs})

        if ref_graph.globals.energy is not None:
            ref_energy = jnp.asarray(ref_graph.globals.energy)
            pred_energy = jnp.asarray(pred_graph.globals.energy)
            delta_e = ref_energy - pred_energy
            delta_es_list.append(np.asarray(delta_e)[valid_mask])
            es_list.append(np.asarray(ref_energy)[valid_mask])

            per_atom_delta = delta_e / atom_counts
            delta_es_per_atom_list.append(np.asarray(per_atom_delta)[valid_mask])
            per_atom_energy = ref_energy / atom_counts
            es_per_atom_list.append(np.asarray(per_atom_energy)[valid_mask])

        if ref_graph.nodes.forces is not None:
            delta_fs_list.append(ref_graph.nodes.forces - pred_graph.nodes.forces)
            fs_list.append(ref_graph.nodes.forces)

        if ref_graph.globals.stress is not None:
            delta_stress = ref_graph.globals.stress - pred_graph.globals.stress
            delta_stress_list.append(np.asarray(delta_stress)[valid_mask])
            stress_list.append(np.asarray(ref_graph.globals.stress)[valid_mask])

        if (
            hasattr(ref_graph.globals, 'virials')
            and ref_graph.globals.virials is not None
            and pred_graph.globals.virials is not None
        ):
            delta_virials = ref_graph.globals.virials - pred_graph.globals.virials
            virials_list.append(np.asarray(ref_graph.globals.virials)[valid_mask])
            per_graph_atoms = atom_counts[:, None, None]
            delta_virials_list.append(np.asarray(delta_virials)[valid_mask])
            delta_virials_per_atom = delta_virials / per_graph_atoms
            delta_virials_per_atom_list.append(
                np.asarray(delta_virials_per_atom)[valid_mask]
            )

        if (
            hasattr(ref_graph.globals, 'dipole')
            and ref_graph.globals.dipole is not None
            and pred_graph.globals.dipole is not None
        ):
            delta_mus = ref_graph.globals.dipole - pred_graph.globals.dipole
            dipoles_list.append(np.asarray(ref_graph.globals.dipole)[valid_mask])
            per_graph_atoms = atom_counts[:, None]
            delta_dipoles_list.append(np.asarray(delta_mus)[valid_mask])
            delta_dipoles_per_atom = delta_mus / per_graph_atoms
            delta_dipoles_per_atom_list.append(
                np.asarray(delta_dipoles_per_atom)[valid_mask]
            )

        if (
            hasattr(ref_graph.globals, 'polarizability')
            and ref_graph.globals.polarizability is not None
            and pred_graph.globals.polarizability is not None
        ):
            delta_polar = (
                ref_graph.globals.polarizability - pred_graph.globals.polarizability
            )
            per_graph_atoms = atom_counts[:, None, None]
            delta_polar_list.append(np.asarray(delta_polar)[valid_mask])
            delta_polar_per_atom = delta_polar / per_graph_atoms
            delta_polar_per_atom_list.append(
                np.asarray(delta_polar_per_atom)[valid_mask]
            )

    if num_graphs == 0:
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
        'q95_e': None,
        'mae_f': None,
        'rel_mae_f': None,
        'rmse_f': None,
        'rel_rmse_f': None,
        'q95_f': None,
        'mae_s': None,
        'rel_mae_s': None,
        'rmse_s': None,
        'rel_rmse_s': None,
        'q95_s': None,
        'mae_stress': None,
        'rel_mae_stress': None,
        'rmse_stress': None,
        'rel_rmse_stress': None,
        'q95_stress': None,
        'mae_virials': None,
        'rmse_virials': None,
        'rmse_virials_per_atom': None,
        'q95_virials': None,
        'mae_mu': None,
        'mae_mu_per_atom': None,
        'rel_mae_mu': None,
        'rmse_mu': None,
        'rmse_mu_per_atom': None,
        'rel_rmse_mu': None,
        'q95_mu': None,
        'mae_polarizability': None,
        'mae_polarizability_per_atom': None,
        'rmse_polarizability': None,
        'rmse_polarizability_per_atom': None,
        'q95_polarizability': None,
    }

    if len(delta_es_list) > 0:
        delta_es = np.concatenate(delta_es_list, axis=0)
        delta_es_per_atom = np.concatenate(delta_es_per_atom_list, axis=0)
        es = np.concatenate(es_list, axis=0)
        es_per_atom = np.concatenate(es_per_atom_list, axis=0)
        aux.update(
            {
                # Mean absolute error
                'mae_e': tools.compute_mae(delta_es),
                'rel_mae_e': tools.compute_rel_mae(delta_es, es),
                'mae_e_per_atom': tools.compute_mae(delta_es_per_atom),
                'rel_mae_e_per_atom': tools.compute_rel_mae(
                    delta_es_per_atom, es_per_atom
                ),
                # Root-mean-square error
                'rmse_e': tools.compute_rmse(delta_es),
                'rel_rmse_e': tools.compute_rel_rmse(delta_es, es),
                'rmse_e_per_atom': tools.compute_rmse(delta_es_per_atom),
                'rel_rmse_e_per_atom': tools.compute_rel_rmse(
                    delta_es_per_atom, es_per_atom
                ),
                # Q_95
                'q95_e': tools.compute_q95(delta_es),
            }
        )
    if len(delta_fs_list) > 0:
        delta_fs = np.concatenate(delta_fs_list, axis=0)
        fs = np.concatenate(fs_list, axis=0)
        aux.update(
            {
                # Mean absolute error
                'mae_f': tools.compute_mae(delta_fs),
                'rel_mae_f': tools.compute_rel_mae(delta_fs, fs),
                # Root-mean-square error
                'rmse_f': tools.compute_rmse(delta_fs),
                'rel_rmse_f': tools.compute_rel_rmse(delta_fs, fs),
                # Q_95
                'q95_f': tools.compute_q95(delta_fs),
            }
        )
    if len(delta_stress_list) > 0:
        delta_stress = np.concatenate(delta_stress_list, axis=0)
        stress = np.concatenate(stress_list, axis=0)
        aux.update(
            {
                # Mean absolute error
                'mae_s': tools.compute_mae(delta_stress),
                'rel_mae_s': tools.compute_rel_mae(delta_stress, stress),
                # Root-mean-square error
                'rmse_s': tools.compute_rmse(delta_stress),
                'rel_rmse_s': tools.compute_rel_rmse(delta_stress, stress),
                # Q_95
                'q95_s': tools.compute_q95(delta_stress),
            }
        )
    aux['mae_stress'] = aux['mae_s']
    aux['rel_mae_stress'] = aux['rel_mae_s']
    aux['rmse_stress'] = aux['rmse_s']
    aux['rel_rmse_stress'] = aux['rel_rmse_s']
    aux['q95_stress'] = aux['q95_s']

    if len(delta_virials_list) > 0:
        delta_virials = np.concatenate(delta_virials_list, axis=0)
        delta_virials_per_atom = np.concatenate(delta_virials_per_atom_list, axis=0)
        aux.update(
            {
                'mae_virials': tools.compute_mae(delta_virials),
                'rmse_virials': tools.compute_rmse(delta_virials),
                'rmse_virials_per_atom': tools.compute_rmse(delta_virials_per_atom),
                'q95_virials': tools.compute_q95(delta_virials),
            }
        )

    if len(delta_dipoles_list) > 0:
        delta_mus = np.concatenate(delta_dipoles_list, axis=0)
        delta_mus_per_atom = np.concatenate(delta_dipoles_per_atom_list, axis=0)
        mus = np.concatenate(dipoles_list, axis=0)
        aux.update(
            {
                'mae_mu': tools.compute_mae(delta_mus),
                'mae_mu_per_atom': tools.compute_mae(delta_mus_per_atom),
                'rel_mae_mu': tools.compute_rel_mae(delta_mus, mus),
                'rmse_mu': tools.compute_rmse(delta_mus),
                'rmse_mu_per_atom': tools.compute_rmse(delta_mus_per_atom),
                'rel_rmse_mu': tools.compute_rel_rmse(delta_mus, mus),
                'q95_mu': tools.compute_q95(delta_mus),
            }
        )

    if len(delta_polar_list) > 0:
        delta_polar = np.concatenate(delta_polar_list, axis=0)
        delta_polar_per_atom = np.concatenate(delta_polar_per_atom_list, axis=0)
        aux.update(
            {
                'mae_polarizability': tools.compute_mae(delta_polar),
                'mae_polarizability_per_atom': tools.compute_mae(delta_polar_per_atom),
                'rmse_polarizability': tools.compute_rmse(delta_polar),
                'rmse_polarizability_per_atom': tools.compute_rmse(
                    delta_polar_per_atom
                ),
                'q95_polarizability': tools.compute_q95(delta_polar),
            }
        )

    return avg_loss, aux
