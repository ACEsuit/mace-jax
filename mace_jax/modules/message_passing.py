from typing import Callable

import e3nn_jax as e3nn
import jax.numpy as jnp


def message_passing_convolution(
    node_feats: e3nn.IrrepsArray,  # [n_nodes, irreps]
    edge_attrs: e3nn.IrrepsArray,  # [n_edges, irreps]
    edge_feats: e3nn.IrrepsArray,  # [n_edges, irreps]
    senders: jnp.ndarray,  # [n_edges, ]
    receivers: jnp.ndarray,  # [n_edges, ]
    avg_num_neighbors: float,
    target_irreps: e3nn.Irreps,
    activation: Callable,
) -> e3nn.IrrepsArray:
    messages = e3nn.Linear(target_irreps)(
        e3nn.MultiLayerPerceptron(3 * [64], activation)(edge_feats),  # [n_edges, 64]
        e3nn.tensor_product(node_feats[senders], edge_attrs),  # [n_edges, irreps]
    )  # [n_edges, irreps]

    zeros = e3nn.IrrepsArray.zeros(messages.irreps, (node_feats.shape[0],))
    node_feats = zeros.at[receivers].add(messages) / jnp.sqrt(
        avg_num_neighbors
    )  # [n_nodes, irreps]

    return node_feats
