import json
import logging
import os
import sys
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple, Union

import jax.numpy as jnp
import numpy as np
import torch

from .torch_tools import to_numpy


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


def get_tag(name: str, seed: int) -> str:
    return f"{name}_run-{seed}"


def setup_logger(
    level: Union[int, str] = logging.INFO,
    tag: Optional[str] = None,
    directory: Optional[str] = None,
):
    logger = logging.getLogger()
    logger.setLevel(level)

    # remove all handlers
    for handler in logger.handlers:
        logger.removeHandler(handler)

    formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03d %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if (directory is not None) and (tag is not None):
        os.makedirs(name=directory, exist_ok=True)
        path = os.path.join(directory, tag + ".log")
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


def get_optimizer(
    name: str,
    amsgrad: bool,
    learning_rate: float,
    weight_decay: float,
    parameters: Iterable[torch.Tensor],
) -> torch.optim.Optimizer:
    if name == "adam":
        return torch.optim.Adam(
            parameters, lr=learning_rate, amsgrad=amsgrad, weight_decay=weight_decay
        )

    if name == "adamw":
        return torch.optim.AdamW(
            parameters, lr=learning_rate, amsgrad=amsgrad, weight_decay=weight_decay
        )

    raise RuntimeError(f"Unknown optimizer '{name}'")


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
    def __init__(self, directory: str, tag: str) -> None:
        self.directory = directory
        self.filename = tag + ".txt"
        self.path = os.path.join(self.directory, self.filename)

    def log(self, d: Dict[str, Any]) -> None:
        logging.debug(f"Saving info: {self.path}")
        os.makedirs(name=self.directory, exist_ok=True)
        with open(self.path, mode="a", encoding="utf-8") as f:
            f.write(json.dumps(d, cls=UniversalEncoder))
            f.write("\n")
