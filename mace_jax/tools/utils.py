import json
import logging
import os
import sys
from collections.abc import Sequence
from typing import Any

import e3nn_jax as e3nn
import jax
import jax.numpy as jnp
import jraph
import numpy as np
from jax import config as jax_config


def count_parameters(parameters) -> int:
    return sum(x.size for x in jax.tree_util.tree_leaves(parameters))


def set_seeds(seed: int) -> None:
    np.random.seed(seed)


def is_primary_process() -> bool:
    try:
        return getattr(jax, 'process_index', lambda: 0)() == 0
    except Exception:
        return True


def log_info_primary(message: str, *args) -> None:
    if is_primary_process():
        logging.info(message, *args)


def pt_head_first(
    heads: Sequence[str], pt_head_name: str = 'pt_head'
) -> tuple[str, ...]:
    if not heads:
        return ()
    head_list = [str(head) for head in heads]
    if not pt_head_name:
        return tuple(head_list)
    pt_heads = [head for head in head_list if head == pt_head_name]
    if not pt_heads:
        return tuple(head_list)
    rest = [head for head in head_list if head != pt_head_name]
    return tuple(pt_heads + rest)


def set_default_dtype(dtype: str) -> None:
    jax_config.update('jax_enable_x64', dtype == 'float64')


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
    assert isinstance(xs, dict), 'expected (frozen)dict'

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
    assert isinstance(xs, dict), 'input is not a dict'
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
    return jnp.where(x2 == 0.0, 0.0, jnp.where(x2 == 0, 1.0, x2) ** 0.5)


def compute_mean_std_atomic_inter_energy(
    graphs: list[jraph.GraphsTuple],
    atomic_energies: np.ndarray,
) -> tuple[float, float]:
    energies = np.asarray(atomic_energies, dtype=np.float64)
    if energies.ndim == 1:
        energies = energies[None, :]
    num_heads = energies.shape[0]

    per_head: list[list[float]] = [[] for _ in range(num_heads)]
    for graph in graphs:
        if graph.nodes is None:
            continue
        species = np.asarray(graph.nodes.species, dtype=int)
        if species.size == 0:
            continue
        energy_value = getattr(graph.globals, 'energy', None)
        if energy_value is None:
            continue
        head = getattr(graph.globals, 'head', None)
        if head is None:
            head_index = 0
        else:
            head_values = np.asarray(head).reshape(-1)
            head_index = int(head_values[0]) if head_values.size else 0
        if head_index >= energies.shape[0]:
            raise ValueError(
                f'Head index {head_index} exceeds atomic_energies heads '
                f'({energies.shape[0]}).'
            )
        e0_sum = float(energies[head_index, species].sum())
        energy_scalar = float(np.asarray(energy_value).reshape(-1)[0])
        per_head[head_index].append((energy_scalar - e0_sum) / float(species.size))

    if not any(per_head):
        return 0.0, 1.0

    means = np.array(
        [float(np.mean(values)) if values else 0.0 for values in per_head],
        dtype=np.float64,
    )
    stds = np.array(
        [float(np.std(values)) if values else 0.0 for values in per_head],
        dtype=np.float64,
    )
    if np.any(stds == 0.0):
        logging.warning(
            'Standard deviation of the scaling is zero, changing to no scaling.'
        )
        stds = np.where(stds == 0.0, 1.0, stds)

    if num_heads == 1:
        return float(means[0]), float(stds[0])
    return means, stds


def compute_mean_rms_energy_forces(
    graphs: list[jraph.GraphsTuple],
    atomic_energies: np.ndarray,
) -> tuple[float, float]:
    energies = np.asarray(atomic_energies, dtype=np.float64)
    if energies.ndim == 1:
        energies = energies[None, :]
    num_heads = energies.shape[0]

    per_head: list[list[float]] = [[] for _ in range(num_heads)]
    force_sq_sum = np.zeros(num_heads, dtype=np.float64)
    force_count = np.zeros(num_heads, dtype=np.int64)

    for graph in graphs:
        if graph.nodes is None:
            continue
        species = np.asarray(graph.nodes.species, dtype=int)
        if species.size == 0:
            continue

        energy_value = getattr(graph.globals, 'energy', None)
        head = getattr(graph.globals, 'head', None)
        if head is None:
            head_index = 0
        else:
            head_values = np.asarray(head).reshape(-1)
            head_index = int(head_values[0]) if head_values.size else 0
        if head_index >= energies.shape[0]:
            raise ValueError(
                f'Head index {head_index} exceeds atomic_energies heads '
                f'({energies.shape[0]}).'
            )

        if energy_value is not None:
            e0_sum = float(energies[head_index, species].sum())
            energy_scalar = float(np.asarray(energy_value).reshape(-1)[0])
            per_head[head_index].append((energy_scalar - e0_sum) / float(species.size))

        forces = getattr(graph.nodes, 'forces', None)
        if forces is not None:
            forces_array = np.asarray(forces, dtype=np.float64)
            force_sq_sum[head_index] += float(np.sum(np.square(forces_array)))
            force_count[head_index] += int(forces_array.size)

    means = np.array(
        [float(np.mean(values)) if values else 0.0 for values in per_head],
        dtype=np.float64,
    )
    rms = np.ones(num_heads, dtype=np.float64)
    nonzero = force_count > 0
    rms[nonzero] = np.sqrt(force_sq_sum[nonzero] / force_count[nonzero])
    if np.any(rms == 0.0):
        logging.warning(
            'Standard deviation of the scaling is zero, changing to no scaling.'
        )
        rms = np.where(rms == 0.0, 1.0, rms)

    if num_heads == 1:
        return float(means[0]), float(rms[0])
    return means, rms


def compute_avg_num_neighbors(graphs: list[jraph.GraphsTuple]) -> float:
    num_neighbors = []

    for graph in graphs:
        _, counts = np.unique(graph.receivers, return_counts=True)
        num_neighbors.append(counts)

    return np.mean(np.concatenate(num_neighbors)).item()


def compute_avg_min_neighbor_distance(graphs: list[jraph.GraphsTuple]) -> float:
    min_neighbor_distances = []

    for graph in graphs:
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
    cell: np.ndarray | None,  # [n_graph, 3, 3]
    n_edge: np.ndarray,  # [n_graph]
) -> tuple[np.ndarray, np.ndarray]:
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
    vectors_receivers = positions[receivers]  # [n_edges, 3]

    if cell is not None:
        num_edges = receivers.shape[0]
        shifts = jnp.einsum(
            'ei,eij->ej',
            shifts,  # [n_edges, 3]
            jnp.repeat(
                cell,  # [n_graph, 3, 3]
                n_edge,  # [n_graph]
                axis=0,
                total_repeat_length=num_edges,
            ),  # [n_edges, 3, 3]
        )  # [n_edges, 3]
        vectors_senders += shifts

    return vectors_senders, vectors_receivers  # [n_edges, 3]


def get_edge_relative_vectors(
    positions: np.ndarray,  # [n_nodes, 3]
    senders: np.ndarray,  # [n_edges]
    receivers: np.ndarray,  # [n_edges]
    shifts: np.ndarray,  # [n_edges, 3]
    cell: np.ndarray | None,  # [n_graph, 3, 3]
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
    return np.mean(np.abs(delta)).item() / (target_norm + 1e-30)


def compute_rmse(delta: np.ndarray) -> float:
    return np.sqrt(np.mean(np.square(delta))).item()


def compute_rel_rmse(delta: np.ndarray, target_val: np.ndarray) -> float:
    target_norm = np.sqrt(np.mean(np.square(target_val))).item()
    return np.sqrt(np.mean(np.square(delta))).item() / (target_norm + 1e-30)


def compute_c(delta: np.ndarray, eta: float) -> float:
    return np.mean(np.abs(delta) < eta).item()


def setup_logger(
    level: int | str = logging.INFO,
    filename: str | None = None,
    directory: str | None = None,
    name: str | None = None,
    stream: bool = True,
):
    logger = logging.getLogger()
    logger.setLevel(level)

    # remove all handlers
    for handler in logger.handlers:
        logger.removeHandler(handler)

    fmt = '%(asctime)s.%(msecs)03d %(levelname)s: %(message)s'
    if name is not None:
        fmt = f'{name} {fmt}'
    formatter = logging.Formatter(fmt, datefmt='%Y-%m-%d %H:%M:%S')

    if stream:
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    if (directory is not None) and (filename is not None):
        os.makedirs(name=directory, exist_ok=True)
        path = os.path.join(directory, filename)
        fh = logging.FileHandler(path)
        fh.setFormatter(formatter)

        logger.addHandler(fh)


class UniversalEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return json.JSONEncoder.default(self, o)


class MetricsLogger:
    def __init__(self, directory: str, filename: str) -> None:
        self.directory = directory
        self.filename = filename
        self.path = os.path.join(self.directory, self.filename)

    def log(self, d: dict[str, Any]) -> None:
        logging.debug(f'Saving info: {self.path}')
        os.makedirs(name=self.directory, exist_ok=True)
        with open(self.path, mode='a', encoding='utf-8') as f:
            f.write(json.dumps(d, cls=UniversalEncoder))
            f.write('\n')
