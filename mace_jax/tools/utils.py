import json
import logging
import os
import sys
from collections import namedtuple
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple, Union

import e3nn_jax as e3nn
import jax
import jax.numpy as jnp
import jraph
import numpy as np
import torch
from roundmantissa import ceil_mantissa

from mace_jax.tools.scatter import scatter_sum


TensorDict = Dict[str, torch.Tensor]


def to_one_hot(indices: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Generates one-hot encoding with <num_classes> classes from <indices>
    :param indices: (N x 1) tensor
    :param num_classes: number of classes
    :param device: torch device
    :return: (N x num_classes) tensor
    """
    shape = indices.shape[:-1] + (num_classes,)
    oh = torch.zeros(shape, device=indices.device).view(shape)

    # scatter_ is the in-place version of scatter
    oh.scatter_(dim=-1, index=indices, value=1)

    return oh.view(*shape)


def count_parameters(parameters) -> int:
    return sum(x.size for x in jax.tree_util.tree_leaves(parameters))


def set_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.cpu().detach().numpy()


dtype_dict = {"float32": torch.float32, "float64": torch.float64}


def set_default_dtype(dtype: str) -> None:
    torch.set_default_dtype(dtype_dict[dtype])
    jax.config.update("jax_enable_x64", dtype == "float64")


def pad_graph_to_nearest_ceil_mantissa(
    graphs_tuple: jraph.GraphsTuple,
    n_mantissa_bits: int = 2,
    n_min_nodes: int = 1,
    n_min_edges: int = 1,
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
    pad_nodes_to = ceil_mantissa(jnp.sum(graphs_tuple.n_node) + 1, n_mantissa_bits)
    pad_nodes_to = max(pad_nodes_to, n_min_nodes)
    pad_edges_to = ceil_mantissa(jnp.sum(graphs_tuple.n_edge), n_mantissa_bits)
    pad_edges_to = max(pad_edges_to, n_min_edges)
    # Add 1 since we need at least one padding graph for pad_with_graphs.
    # pad_graphs_to = ceil_mantissa(graphs_tuple.n_node.shape[0] + 1, n_mantissa_bits)
    pad_graphs_to = graphs_tuple.n_node.shape[0] + 1
    return jraph.pad_with_graphs(
        graphs_tuple, pad_nodes_to, pad_edges_to, pad_graphs_to
    )


Node = namedtuple("Node", ["positions", "species", "forces"])
Edge = namedtuple("Edge", ["shifts"])
Global = namedtuple("Global", ["energy", "weight", "cell"])


def get_jraph_graph_from_pyg(batch):
    return jraph.GraphsTuple(
        nodes=Node(
            positions=batch.positions.numpy(),
            species=batch.node_species.numpy(),
            forces=batch.forces.numpy(),
        ),
        edges=Edge(shifts=batch.shifts.numpy()),
        globals=Global(
            energy=batch.energy.numpy(),
            weight=batch.weight.numpy(),
            cell=batch.cell.numpy() if batch.cell is not None else None,
        ),
        n_node=(batch.ptr[1:] - batch.ptr[:-1]).numpy(),
        n_edge=batch.n_edge.numpy(),
        senders=batch.edge_index[0].numpy(),
        receivers=batch.edge_index[1].numpy(),
    )


def get_batched_padded_graph_tuples(
    batch, n_mantissa_bits: int = 2, n_min_nodes: int = 1, n_min_edges: int = 1
):
    graphs = get_jraph_graph_from_pyg(batch)
    graphs = pad_graph_to_nearest_ceil_mantissa(
        graphs, n_mantissa_bits, n_min_nodes, n_min_edges
    )  # padd the whole batch once
    return graphs


# From Flax
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


def safe_norm(x: jnp.ndarray, axis: int = None, keepdims=False) -> jnp.ndarray:
    """nan-safe norm."""
    x2 = jnp.sum(x**2, axis=axis, keepdims=keepdims)
    return jnp.where(x2 == 0, 1, x2) ** 0.5


def compute_mean_std_atomic_inter_energy(
    data_loader: torch.utils.data.DataLoader,
    atomic_energies: np.ndarray,
) -> Tuple[float, float]:
    atomic_energies = torch.from_numpy(atomic_energies)

    atom_energy_list = []

    for batch in data_loader:
        node_e0 = atomic_energies[batch.node_species]
        graph_e0s = scatter_sum(
            src=node_e0, index=batch.batch, dim=-1, dim_size=batch.num_graphs
        )
        graph_sizes = batch.ptr[1:] - batch.ptr[:-1]
        atom_energy_list.append(
            (batch.energy - graph_e0s) / graph_sizes
        )  # {[n_graphs], }

    atom_energies = torch.cat(atom_energy_list, dim=0)  # [total_n_graphs]

    mean = to_numpy(torch.mean(atom_energies)).item()
    std = to_numpy(torch.std(atom_energies)).item()

    return mean, std


def compute_mean_rms_energy_forces(
    data_loader: torch.utils.data.DataLoader,
    atomic_energies: np.ndarray,
) -> Tuple[float, float]:
    mean, _ = compute_mean_std_atomic_inter_energy(data_loader, atomic_energies)

    atomic_energies = torch.from_numpy(atomic_energies)

    forces = torch.cat(
        [batch.forces for batch in data_loader], dim=0
    )  # [total_n_graphs * n_atoms, 3]

    rms = torch.sqrt(torch.mean(torch.square(forces))).item()

    return mean, rms


def compute_avg_num_neighbors(data_loader: torch.utils.data.DataLoader) -> float:
    num_neighbors = []

    for graph in data_loader:
        graph = jraph.unpad_with_graphs(graph)
        _, counts = np.unique(graph.receivers, return_counts=True)
        num_neighbors.append(counts)

    return np.mean(np.concatenate(num_neighbors)).item()


def compute_avg_min_neighbor_distance(
    data_loader: torch.utils.data.DataLoader,
) -> float:
    min_neighbor_distances = []

    for graph in data_loader:
        graph = jraph.unpad_with_graphs(graph)
        vectors = get_edge_relative_vectors(
            graph.nodes.positions,
            graph.senders,
            graph.receivers,
            graph.edges.shifts,
            graph.globals.cell,
            graph.n_edge,
        )
        length = np.linalg.norm(vectors, axis=-1)
        min_neighbor_distances.append(length.min())

    return np.mean(min_neighbor_distances).item()


def sum_nodes_of_the_same_graph(
    graph: jraph.GraphsTuple, node_quantities: jnp.ndarray
) -> jnp.ndarray:
    """Sum node quantities and return a graph quantity."""
    return e3nn.scatter_sum(node_quantities, nel=graph.n_node)  # [ n_graphs,]


def get_edge_vectors(
    positions: np.ndarray,  # [n_nodes, 3]
    senders: np.ndarray,  # [n_edges]
    receivers: np.ndarray,  # [n_edges]
    shifts: np.ndarray,  # [n_edges, 3]
    cell: Optional[np.ndarray],  # [n_graph, 3, 3]
    n_edge: np.ndarray,  # [n_graph]
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the positions of the sender and receiver nodes of each edge.

    This function assumes that the shift is done to the sender node.

    Args:
        positions: The positions of the nodes.
        senders: The sender nodes of each edge. ``j`` output of ``ase.neighborlist.primitive_neighbor_list``.
        receivers: The receiver nodes of each edge. ``i`` output of ``ase.neighborlist.primitive_neighbor_list``.
        shifts: The shift vectors of each edge. ``S`` output of ``ase.neighborlist.primitive_neighbor_list``.
        cell: The cell of each graph. Array of shape ``[n_graph, 3, 3]``.
        n_edge: The number of edges of each graph. Array of shape ``[n_graph]``.

    Returns:
        The positions of the sender and receiver nodes of each edge.
    """
    # From ASE docs: With the shift vector S, the distances D between atoms can be computed from
    # D = positions[j]-positions[i]+S.dot(cell)
    vectors_senders = positions[senders]  # [n_edges, 3]

    if cell is not None:
        num_edges = receivers.shape[0]
        shifts = jnp.einsum(
            "ei,eij->ej",
            shifts,  # [n_edges, 3]
            jnp.repeat(
                cell,  # [n_graph, 3, 3]
                n_edge,  # [n_graph]
                axis=0,
                total_repeat_length=num_edges,
            ),  # [n_edges, 3, 3]
        )  # [n_edges, 3]
        vectors_senders += shifts

    return vectors_senders, positions[receivers]  # [n_edges, 3]


def get_edge_relative_vectors(
    positions: np.ndarray,  # [n_nodes, 3]
    senders: np.ndarray,  # [n_edges]
    receivers: np.ndarray,  # [n_edges]
    shifts: np.ndarray,  # [n_edges, 3]
    cell: Optional[np.ndarray],  # [n_graph, 3, 3]
    n_edge: np.ndarray,  # [n_graph]
) -> np.ndarray:
    vectors_senders, vectors_receivers = get_edge_vectors(
        positions=positions,
        senders=senders,
        receivers=receivers,
        shifts=shifts,
        cell=cell,
        n_edge=n_edge,
    )
    return vectors_receivers - vectors_senders


def compute_mae(delta: np.ndarray) -> float:
    return np.mean(np.abs(delta)).item()


def compute_rel_mae(delta: np.ndarray, target_val: np.ndarray) -> float:
    target_norm = np.mean(np.abs(target_val))
    return np.mean(np.abs(delta)).item() / (target_norm + 1e-9) * 100


def compute_rmse(delta: np.ndarray) -> float:
    return np.sqrt(np.mean(np.square(delta))).item()


def compute_rel_rmse(delta: np.ndarray, target_val: np.ndarray) -> float:
    target_norm = np.sqrt(np.mean(np.square(target_val))).item()
    return np.sqrt(np.mean(np.square(delta))).item() / (target_norm + 1e-9) * 100


def compute_q95(delta: np.ndarray) -> float:
    return np.percentile(np.abs(delta), q=95)


def compute_c(delta: np.ndarray, eta: float) -> float:
    return np.mean(np.abs(delta) < eta).item()


def setup_logger(
    level: Union[int, str] = logging.INFO,
    filename: Optional[str] = None,
    directory: Optional[str] = None,
    name: Optional[str] = None,
):
    logger = logging.getLogger()
    logger.setLevel(level)

    # remove all handlers
    for handler in logger.handlers:
        logger.removeHandler(handler)

    fmt = "%(asctime)s.%(msecs)03d %(levelname)s: %(message)s"
    if name is not None:
        fmt = f"{name} {fmt}"
    formatter = logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S")

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if (directory is not None) and (filename is not None):
        os.makedirs(name=directory, exist_ok=True)
        path = os.path.join(directory, filename)
        fh = logging.FileHandler(path)
        fh.setFormatter(formatter)

        logger.addHandler(fh)


class AtomicNumberTable:
    def __init__(self, zs: Sequence[int]):
        self.zs = zs

    def __len__(self) -> int:
        return len(self.zs)

    def __str__(self):
        return f"AtomicNumberTable: {tuple(s for s in self.zs)}"

    def index_to_z(self, index: int) -> int:
        return self.zs[index]

    def z_to_index(self, atomic_number: str) -> int:
        return self.zs.index(atomic_number)


def get_atomic_number_table_from_zs(zs: Iterable[int]) -> AtomicNumberTable:
    return AtomicNumberTable(sorted(set(zs)))


def atomic_numbers_to_indices(
    atomic_numbers: np.ndarray, z_table: AtomicNumberTable
) -> np.ndarray:
    to_index_fn = np.vectorize(z_table.z_to_index)
    return to_index_fn(atomic_numbers)


class UniversalEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, torch.Tensor):
            return to_numpy(o)
        return json.JSONEncoder.default(self, o)


class MetricsLogger:
    def __init__(self, directory: str, filename: str) -> None:
        self.directory = directory
        self.filename = filename
        self.path = os.path.join(self.directory, self.filename)

    def log(self, d: Dict[str, Any]) -> None:
        logging.debug(f"Saving info: {self.path}")
        os.makedirs(name=self.directory, exist_ok=True)
        with open(self.path, mode="a", encoding="utf-8") as f:
            f.write(json.dumps(d, cls=UniversalEncoder))
            f.write("\n")
