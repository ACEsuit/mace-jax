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

        messages = e3nn.tensor_product(
            node_feats[senders], edge_attrs[:, None, :]
        ).filter(
            self.target_irreps
        )  # [n_edges, feature, irreps]

        mix = e3nn.MultiLayerPerceptron(
            3 * [64] + [messages.shape[1] * messages.irreps.num_irreps],
            self.activation,
            output_activation=False,
        )(
            edge_feats
        )  # [n_edges, feature * num_irreps]

        mix = mix.mul_to_axis(messages.shape[1])  # [n_edges, feature, irreps]

        messages = e3nn.elementwise_tensor_product(
            messages, mix
        )  # [n_edges, feature, irreps]

        zeros = e3nn.IrrepsArray.zeros(
            messages.irreps, node_feats.shape[:1] + messages.shape[1:2], messages.dtype
        )
        node_feats = zeros.at[receivers].add(messages)  # [n_nodes, feature, irreps]

        return node_feats / jnp.sqrt(self.avg_num_neighbors)
