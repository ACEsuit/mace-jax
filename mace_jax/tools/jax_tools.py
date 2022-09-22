from collections import namedtuple
import numpy as np
import jax.numpy as jnp
import jraph


def _nearest_bigger_power_of_two(x: int) -> int:
    """Computes the nearest power of two greater than x for padding."""
    y = 2
    while y < x:
        y *= 2
    return y


def pad_graph_to_nearest_power_of_two(
    graphs_tuple: jraph.GraphsTuple,
) -> jraph.GraphsTuple:
    """Pads a batched `GraphsTuple` to the nearest power of two.

    For example, if a `GraphsTuple` has 7 nodes, 5 edges and 3 graphs, this method
    would pad the `GraphsTuple` nodes and edges:
        7batch_sizedes --> 8 nodes (2^3)
        5 edges --> 8 edges (2^3)

    And since padding is accomplished using `jraph.pad_with_graphs`, an extra
    graph and node is added:
        8 nodes --> 9 nodes
        3 graphs --> 4 graphs

    Args:
        graphs_tuple: a batched `GraphsTuple` (can be batch size 1).

    Returns:
        A graphs_tuple batched to the nearest power of two.
    """
    # Add 1 since we need at least one padding node for pad_with_graphs.
    pad_nodes_to = _nearest_bigger_power_of_two(jnp.sum(graphs_tuple.n_node)) + 1
    pad_edges_to = _nearest_bigger_power_of_two(jnp.sum(graphs_tuple.n_edge))
    # Add 1 since we need at least one padding graph for pad_with_graphs.
    # We do not pad to nearest power of two because the batch size is fixed.
    pad_graphs_to = graphs_tuple.n_node.shape[0] + 1
    return jraph.pad_with_graphs(
        graphs_tuple, pad_nodes_to, pad_edges_to, pad_graphs_to
    )


Node = namedtuple("Node", ["positions", "attrs", "forces"])
Edge = namedtuple("Edge", ["shifts"])
Global = namedtuple("Global", ["energy", "weight", "ptr"])


def get_batched_padded_graph_tuples(batch):
    graphs = jraph.GraphsTuple(
        nodes=Node(
            positions=np.array(batch.position),
            attrs=np.array(batch.attrs),
            forces=np.array(batch.forces),
        ),
        edges=Edge(shifts=np.array(batch.shifts)),
        n_node=np.array(batch.position.shape[0]),
        n_edge=np.array(batch.edge_index.shape[1]),
        senders=np.array(batch.edge_index[0]),
        receivers=np.array(batch.edge_index[1]),
        globals=Global(
            energy=np.array(batch.energy),
            weight=np.array(batch.weight),
            ptr=np.array(batch.ptr),
        ),
    )

    labels = np.array(batch.y)
    graphs = pad_graph_to_nearest_power_of_two(graphs)  # padd the whole batch once
    return graphs, labels
