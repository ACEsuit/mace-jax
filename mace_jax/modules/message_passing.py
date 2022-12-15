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
        node_feats: e3nn.IrrepsArray,  # [n_nodes, irreps]
        edge_attrs: e3nn.IrrepsArray,  # [n_edges, irreps]
        senders: jnp.ndarray,  # [n_edges, ]
        receivers: jnp.ndarray,  # [n_edges, ]
    ) -> e3nn.IrrepsArray:
        assert node_feats.ndim == 2
        assert edge_attrs.ndim == 2

        messages = node_feats[senders]

        messages = e3nn.concatenate(
            [
                messages.filter(self.target_irreps),
                e3nn.tensor_product(
                    messages,
                    edge_attrs.filter(drop="0e"),
                    filter_ir_out=self.target_irreps,
                ),
            ]
        ).regroup()  # [n_edges, irreps]

        # one = e3nn.IrrepsArray.ones("0e", edge_attrs.shape[:-1])
        # messages = e3nn.tensor_product(
        #     messages, e3nn.concatenate([one, edge_attrs.filter(drop="0e")])
        # ).filter(self.target_irreps)

        mix = e3nn.haiku.MultiLayerPerceptron(
            3 * [64] + [messages.irreps.num_irreps],
            self.activation,
            output_activation=False,
        )(
            edge_attrs.filter(keep="0e")
        )  # [n_edges, num_irreps]

        messages = messages * mix  # [n_edges, irreps]

        zeros = e3nn.IrrepsArray.zeros(
            messages.irreps, node_feats.shape[:1], messages.dtype
        )
        node_feats = zeros.at[receivers].add(messages)  # [n_nodes, irreps]

        return node_feats / jnp.sqrt(self.avg_num_neighbors)
