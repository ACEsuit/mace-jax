import itertools
import logging
import time
from functools import partial
from typing import Any, Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import jraph
import numpy as np
import optax
import tqdm

from mace_jax import data, tools


def train(
    params: Any,
    total_loss_fn: Callable,  # loss_fn(params, graph) -> [num_graphs]
    train_loader: data.GraphDataLoader,  # device parallel done on the (optional) extra dimension
    gradient_transform: Any,
    optimizer_state: Dict[str, Any],
    steps_per_interval: int,
    ema_decay: Optional[float] = None,
    progress_bar: bool = True,
):
    """
    for interval, params, optimizer_state, ema_params in train(...):
        # do something
    """
    num_updates = 0
    ema_params = params

    logging.info('Started training')

    @partial(jax.pmap, in_axes=(None, 0), out_axes=0)
    # @partial(jax.vmap, in_axes=(None, 0), out_axes=0)
    def grad_fn(params, graph: jraph.GraphsTuple):
        # graph is assumed to be padded by jraph.pad_with_graphs
        mask = jraph.get_graph_padding_mask(graph)  # [n_graphs,]
        loss, grad = jax.value_and_grad(
            lambda params: jnp.sum(jnp.where(mask, total_loss_fn(params, graph), 0.0))
        )(params)
        return jnp.sum(mask), loss, grad

    # jit-of-pmap is not recommended but so far it seems faster
    @jax.jit
    def update_fn(
        params, optimizer_state, ema_params, num_updates: int, graph: jraph.GraphsTuple
    ) -> Tuple[float, Any, Any]:
        if graph.n_node.ndim == 1:
            graph = jax.tree_map(lambda x: x[None, ...], graph)

        n, loss, grad = grad_fn(params, graph)
        loss = jnp.sum(loss) / jnp.sum(n)
        grad = jax.tree_map(lambda x: jnp.sum(x, axis=0) / jnp.sum(n), grad)

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

    def interval_loader():
        i = 0
        while True:
            for graph in train_loader:
                yield graph
                i += 1
                if i >= steps_per_interval:
                    return

    for interval in itertools.count():
        yield interval, params, optimizer_state, ema_params

        # Train one interval
        p_bar = tqdm.tqdm(
            interval_loader(),
            desc=f'Train interval {interval}',
            total=steps_per_interval,
            disable=not progress_bar,
        )
        for graph in p_bar:
            num_updates += 1
            start_time = time.time()
            loss, params, optimizer_state, ema_params = update_fn(
                params, optimizer_state, ema_params, num_updates, graph
            )
            loss = float(loss)
            p_bar.set_postfix({'loss': f'{loss:7.3f}'})

            if last_cache_size != update_fn._cache_size():
                last_cache_size = update_fn._cache_size()

                logging.info('Compiled function `update_fn` for args:')
                logging.info(f'- n_node={graph.n_node} total={graph.n_node.sum()}')
                logging.info(f'- n_edge={graph.n_edge} total={graph.n_edge.sum()}')
                logging.info(f'Outout: loss= {loss:.3f}')
                logging.info(
                    f'Compilation time: {time.time() - start_time:.3f}s, cache size: {last_cache_size}'
                )


def evaluate(
    predictor: Callable,
    params: Any,
    loss_fn: Any,
    data_loader: data.GraphDataLoader,
    name: str = 'Evaluation',
    progress_bar: bool = True,
) -> Tuple[float, Dict[str, Any]]:
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

    if hasattr(predictor, '_cache_size'):
        last_cache_size = predictor._cache_size()
    else:
        last_cache_size = None

    start_time = time.time()

    p_bar = tqdm.tqdm(
        data_loader,
        desc=name,
        total=data_loader.approx_length(),
        disable=not progress_bar,
    )

    for ref_graph in p_bar:
        output = predictor(params, ref_graph)
        pred_graph = ref_graph._replace(
            nodes=ref_graph.nodes._replace(forces=output['forces']),
            globals=ref_graph.globals._replace(
                energy=output['energy'], stress=output['stress']
            ),
        )

        if last_cache_size is not None and last_cache_size != predictor._cache_size():
            last_cache_size = predictor._cache_size()

            logging.info('Compiled function `predictor` for args:')
            logging.info(f'- n_node={ref_graph.n_node} total={ref_graph.n_node.sum()}')
            logging.info(f'- n_edge={ref_graph.n_edge} total={ref_graph.n_edge.sum()}')
            logging.info(f'cache size: {last_cache_size}')

        ref_graph = jraph.unpad_with_graphs(ref_graph)
        pred_graph = jraph.unpad_with_graphs(pred_graph)

        loss = jnp.sum(
            loss_fn(
                ref_graph,
                dict(
                    energy=pred_graph.globals.energy,
                    forces=pred_graph.nodes.forces,
                    stress=pred_graph.globals.stress,
                ),
            )
        )
        total_loss += float(loss)
        num_graphs += len(ref_graph.n_edge)
        p_bar.set_postfix({'n': num_graphs})

        if ref_graph.globals.energy is not None:
            delta_es_list.append(ref_graph.globals.energy - pred_graph.globals.energy)
            es_list.append(ref_graph.globals.energy)

            delta_es_per_atom_list.append(
                (ref_graph.globals.energy - pred_graph.globals.energy)
                / ref_graph.n_node
            )
            es_per_atom_list.append(ref_graph.globals.energy / ref_graph.n_node)

        if ref_graph.nodes.forces is not None:
            delta_fs_list.append(ref_graph.nodes.forces - pred_graph.nodes.forces)
            fs_list.append(ref_graph.nodes.forces)

        if ref_graph.globals.stress is not None:
            delta_stress_list.append(
                ref_graph.globals.stress - pred_graph.globals.stress
            )
            stress_list.append(ref_graph.globals.stress)

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

    return avg_loss, aux
