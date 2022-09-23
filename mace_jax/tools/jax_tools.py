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


### From Flax
class _EmptyNode:
    pass


empty_node = _EmptyNode()


def flatten_dict(xs, keep_empty_nodes=False, is_leaf=None, sep=None):
    """Flatten a nested dictionary.

  The nested keys are flattened to a tuple.
  See `unflatten_dict` on how to restore the
  nested dictionary structure.

  Example::

    xs = {'foo': 1, 'bar': {'a': 2, 'b': {}}}
    flat_xs = flatten_dict(xs)
    print(flat_xs)
    # {
    #   ('foo',): 1,
    #   ('bar', 'a'): 2,
    # }

  Note that empty dictionaries are ignored and
  will not be restored by `unflatten_dict`.

  Args:
    xs: a nested dictionary
    keep_empty_nodes: replaces empty dictionaries
      with `traverse_util.empty_node`. This must
      be set to `True` for `unflatten_dict` to
      correctly restore empty dictionaries.
    is_leaf: an optional function that takes the
      next nested dictionary and nested keys and
      returns True if the nested dictionary is a
      leaf (i.e., should not be flattened further).
    sep: if specified, then the keys of the returned
      dictionary will be `sep`-joined strings (if
      `None`, then keys will be tuples).
  Returns:
    The flattened dictionary.
  """
    assert isinstance(xs, dict), "expected (frozen)dict"

    def _key(path):
        if sep is None:
            return path
        return sep.join(path)

    def _flatten(xs, prefix):
        if not isinstance(xs, dict) or (is_leaf and is_leaf(prefix, xs)):
            return {_key(prefix): xs}
        result = {}
        is_empty = True
        for key, value in xs.items():
            is_empty = False
            path = prefix + (key,)
            result.update(_flatten(value, path))
        if keep_empty_nodes and is_empty:
            if prefix == ():  # when the whole input is empty
                return {}
            return {_key(prefix): empty_node}
        return result

    return _flatten(xs, ())


def unflatten_dict(xs, sep=None):
    """Unflatten a dictionary.

  See `flatten_dict`

  Example::

    flat_xs = {
      ('foo',): 1,
      ('bar', 'a'): 2,
    }
    xs = unflatten_dict(flat_xs)
    print(xs)
    # {
    #   'foo': 1
    #   'bar': {'a': 2}
    # }

  Args:
    xs: a flattened dictionary
    sep: separator (same as used with `flatten_dict()`).
  Returns:
    The nested dictionary.
  """
    assert isinstance(xs, dict), "input is not a dict"
    result = {}
    for path, value in xs.items():
        if sep is not None:
            path = path.split(sep)
        if value is empty_node:
            value = {}
        cursor = result
        for key in path[:-1]:
            if key not in cursor:
                cursor[key] = {}
            cursor = cursor[key]
        cursor[path[-1]] = value
    return result

