from typing import Callable

import e3nn_jax as e3nn
import jax.numpy as jnp
import haiku as hk


class MessagePassingConvolution(hk.Module):
    def __init__(
        self,
        avg_num_neighbors: float,
        target_irreps: e3nn.Irreps,
        activation: Callable,
    ):
        super().__init__()
        self.avg_num_neighbors = avg_num_neighbors
        self.target_irreps = e3nn.Irreps(target_irreps)
        self.activation = activation

    def __call__(
        self,
        node_feats: e3nn.IrrepsArray,  # [n_nodes, feature, irreps]
        edge_attrs: e3nn.IrrepsArray,  # [n_edges, irreps]
        edge_feats: e3nn.IrrepsArray,  # [n_edges, irreps]
        senders: jnp.ndarray,  # [n_edges, ]
        receivers: jnp.ndarray,  # [n_edges, ]
    ) -> e3nn.IrrepsArray:
        assert node_feats.ndim == 3
        assert edge_attrs.ndim == 2
        assert edge_feats.ndim == 2

        messages = e3nn.Linear(self.target_irreps, weights_per_channel=True)(
            e3nn.MultiLayerPerceptron(
                3 * [64], self.activation, output_activation=True
            )(
                edge_feats
            ),  # [n_edges, 64]
            e3nn.tensor_product(
                node_feats[senders], edge_attrs[:, None, :]
            ),  # [n_edges, feature, irreps]
        )  # [n_edges, feature, irreps]

        zeros = e3nn.IrrepsArray.zeros(
            messages.irreps, node_feats.shape[:1] + messages.shape[1:2], messages.dtype
        )
        node_feats = zeros.at[receivers].add(messages) / jnp.sqrt(
            self.avg_num_neighbors
        )  # [n_nodes, feature, irreps]

        return node_feats
