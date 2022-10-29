import e3nn_jax as e3nn
import jax
import jax.numpy as jnp


def message_passing_convolution(
    node_feats: e3nn.IrrepsArray,  # [n_nodes, irreps]
    edge_attrs: e3nn.IrrepsArray,  # [n_edges, irreps]
    edge_feats: e3nn.IrrepsArray,  # [n_edges, irreps]
    senders: jnp.ndarray,  # [n_edges, ]
    receivers: jnp.ndarray,  # [n_edges, ]
    avg_num_neighbors: float,
    target_irreps: e3nn.Irreps,
) -> e3nn.IrrepsArray:
    messages = (
        e3nn.tensor_product(node_feats[senders], edge_attrs).remove_nones().simplify()
    )  # [n_edges, irreps]
    linear = e3nn.FunctionalLinear(messages.irreps, target_irreps)

    # Learnable Radial
    assert edge_feats.irreps.is_scalar()
    w = e3nn.MultiLayerPerceptron(
        3 * [64] + [linear.num_weights],  # TODO (mario): make this configurable?
        jax.nn.silu,
    )(
        edge_feats.array
    )  # [n_edges, linear.num_weights]
    assert w.shape == (edge_feats.shape[0], linear.num_weights)
    w = jax.vmap(linear.split_weights)(w)  # List of [n_edges, *path_shape]

    messages = jax.vmap(linear)(w, messages).simplify()  # [n_edges, irreps]

    # Scatter sum
    message = e3nn.IrrepsArray.zeros(messages.irreps, (node_feats.shape[0],))
    node_feats = message.at[receivers].add(messages) / jnp.sqrt(
        avg_num_neighbors
    )  # [n_nodes, irreps]

    return node_feats
