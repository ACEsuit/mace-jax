import logging
import time
from typing import Any, Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import jraph
import numpy as np
import optax
import tqdm
from torch.utils.data import DataLoader

from .utils import (
    MetricsLogger,
    compute_mae,
    compute_q95,
    compute_rel_mae,
    compute_rel_rmse,
    compute_rmse,
)


def train(
    model: Callable,
    params: Dict[str, Any],
    loss_fn: Any,
    train_loader: DataLoader,
    gradient_transform: Any,
    optimizer_state: Dict[str, Any],
    start_epoch: int,
    max_num_epochs: int,
    logger: MetricsLogger,
    ema_decay: Optional[float] = None,
):
    num_updates = 0
    ema_params = params

    logging.info("Started training")

    @jax.jit
    def update_fn(
        params, optimizer_state, ema_params, num_updates: int, graph: jraph.GraphsTuple
    ) -> Tuple[float, Any, Any]:
        # graph is assumed to be padded by jraph.pad_with_graphs
        mask = jraph.get_graph_padding_mask(graph)  # [n_graphs,]
        loss, grad = jax.value_and_grad(
            lambda params: jnp.mean(loss_fn(graph, **model(params, graph)) * mask)
        )(params)
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

    for epoch in range(start_epoch, max_num_epochs):
        yield epoch, params, optimizer_state, ema_params

        # Train one epoch
        p_bar = tqdm.tqdm(train_loader, desc="Epoch {}".format(epoch))
        for graph in p_bar:
            num_updates += 1
            start_time = time.time()
            loss, params, optimizer_state, ema_params = update_fn(
                params, optimizer_state, ema_params, num_updates, graph
            )
            loss = float(loss)
            p_bar.set_postfix({"loss": loss})
            opt_metrics = {
                "loss": loss,
                "time": time.time() - start_time,
            }

            opt_metrics["mode"] = "opt"
            opt_metrics["epoch"] = epoch
            opt_metrics["num_updates"] = num_updates
            opt_metrics["epoch_"] = start_epoch + num_updates / len(train_loader)
            logger.log(opt_metrics)

            if last_cache_size != update_fn._cache_size():
                last_cache_size = update_fn._cache_size()

                logging.info("Compiled function `update_fn` for args:")
                logging.info(f"- n_node={graph.n_node} total={graph.n_node.sum()}")
                logging.info(f"- n_edge={graph.n_edge} total={graph.n_edge.sum()}")
                logging.info(f"Outout: loss= {loss:.3f}")
                logging.info(
                    f"Compilation time: {opt_metrics['time']:.3f}s, cache size: {last_cache_size}"
                )


def evaluate(
    model: Callable,
    params: Any,
    loss_fn: Any,
    data_loader: DataLoader,
) -> Tuple[float, Dict[str, Any]]:
    total_loss = 0.0
    num_graphs = 0
    delta_es_list = []
    delta_es_per_atom_list = []
    delta_fs_list = []
    fs_list = []

    start_time = time.time()
    p_bar = tqdm.tqdm(data_loader, desc="Evaluating")
    for ref_graph in p_bar:
        output = model(params, ref_graph)
        pred_graph = ref_graph._replace(
            nodes=ref_graph.nodes._replace(forces=output["forces"]),
            globals=ref_graph.globals._replace(energy=output["energy"]),
        )

        ref_graph = jraph.unpad_with_graphs(ref_graph)
        pred_graph = jraph.unpad_with_graphs(pred_graph)

        loss = jnp.sum(
            loss_fn(
                graph=ref_graph,
                energy=pred_graph.globals.energy,
                forces=pred_graph.nodes.forces,
            )
        )
        total_loss += float(loss)
        num_graphs += len(ref_graph.n_edge)
        p_bar.set_postfix({"n": num_graphs})

        delta_es_list.append(ref_graph.globals.energy - pred_graph.globals.energy)
        delta_es_per_atom_list.append(
            (ref_graph.globals.energy - pred_graph.globals.energy) / ref_graph.n_node
        )
        delta_fs_list.append(ref_graph.nodes.forces - pred_graph.nodes.forces)
        fs_list.append(ref_graph.nodes.forces)

    avg_loss = total_loss / num_graphs

    delta_es = np.concatenate(delta_es_list, axis=0)
    delta_es_per_atom = np.concatenate(delta_es_per_atom_list, axis=0)
    delta_fs = np.concatenate(delta_fs_list, axis=0)
    fs = np.concatenate(fs_list, axis=0)

    aux = {
        "loss": avg_loss,
        # Mean absolute error
        "mae_e": compute_mae(delta_es),
        "mae_e_per_atom": compute_mae(delta_es_per_atom),
        "mae_f": compute_mae(delta_fs),
        "rel_mae_f": compute_rel_mae(delta_fs, fs),
        # Root-mean-square error
        "rmse_e": compute_rmse(delta_es),
        "rmse_e_per_atom": compute_rmse(delta_es_per_atom),
        "rmse_f": compute_rmse(delta_fs),
        "rel_rmse_f": compute_rel_rmse(delta_fs, fs),
        # Q_95
        "q95_e": compute_q95(delta_es),
        "q95_f": compute_q95(delta_fs),
        # Time
        "time": time.time() - start_time,
    }

    return avg_loss, aux
