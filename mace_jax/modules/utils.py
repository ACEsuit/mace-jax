from typing import Callable, Tuple, Optional
from jax_md.partition import neighbor_list
from jax_md import partition
import e3nn_jax as e3nn
import jax.numpy as jnp
import jraph
import numpy as np
import torch
from functools import partial
from jax import vmap

from mace_jax.tools import to_numpy
from mace_jax.tools.scatter import scatter_sum

from .blocks import AtomicEnergiesBlock


def safe_norm(x: jnp.ndarray, axis: int = None, keepdims=False) -> jnp.ndarray:
    """nan-safe norm."""
    x2 = jnp.sum(x**2, axis=axis, keepdims=keepdims)
    return jnp.where(x2 == 0, 1, x2) ** 0.5


def get_edge_vectors_and_lengths(
    positions: np.ndarray,  # [n_nodes, 3]
    neighbour_list: neighbor_list,
    disp_fn: Callable
    # receivers: np.ndarray,  # [n_edges]
    # senders: np.ndarray,  # [n_edges]
    # shifts: np.ndarray,  # [n_edges, 3]
    # cell: Optional[np.ndarray],  # [n_graph, 3, 3]
    # n_edge: np.ndarray,  # [n_graph]
) -> Tuple[np.ndarray, np.ndarray]:
    # vectors = positions[receivers] - positions[senders]  # [n_edges, 3]

    # if cell is not None:
    #     # From the docs: With the shift vector S, the distances D between atoms can be computed from
    #     # D = positions[j]-positions[i]+S.dot(cell)
    #     num_edges = receivers.shape[0]
    #     shifts = jnp.einsum(
    #         "ei,eij->ej",
    #         shifts,  # [n_edges, 3]
    #         jnp.repeat(
    #             cell,  # [n_graph, 3, 3]
    #             n_edge,  # [n_graph]
    #             axis=0,
    #             total_repeat_length=num_edges,
    #         ),  # [n_edges, 3, 3]
    #     )  # [n_edges, 3]
    #     vectors += shifts
    mask = partition.neighbor_list_mask(neighbour_list)

    d = vmap(partial(disp_fn))
    vectors = d(
        positions[neighbour_list.idx[0, :]], positions[neighbour_list.idx[1, :]]
    )

    mask = partition.neighbor_list_mask(neighbour_list)
    vectors = jnp.where(mask[:, None], vectors, 0)

    lengths = safe_norm(vectors, axis=-1, keepdims=True)  # [n_edges, 1]
    return vectors, lengths


def compute_mean_std_atomic_inter_energy(
    data_loader,
    atomic_energies: np.ndarray,
) -> Tuple[float, float]:
    atomic_energies_fn = AtomicEnergiesBlock(atomic_energies=atomic_energies)

    avg_atom_inter_es_list = []

    for batch in data_loader:
        node_e0 = atomic_energies_fn(batch.node_attrs)
        graph_e0s = scatter_sum(
            src=node_e0, index=batch.batch, dim=-1, dim_size=batch.num_graphs
        )
        graph_sizes = batch.ptr[1:] - batch.ptr[:-1]
        avg_atom_inter_es_list.append(
            (batch.energy - graph_e0s) / graph_sizes
        )  # {[n_graphs], }

    avg_atom_inter_es = torch.cat(avg_atom_inter_es_list)  # [total_n_graphs]
    mean = to_numpy(torch.mean(avg_atom_inter_es)).item()
    std = to_numpy(torch.std(avg_atom_inter_es)).item()

    return mean, std


def compute_mean_rms_energy_forces(
    data_loader: torch.utils.data.DataLoader,
    atomic_energies: np.ndarray,
) -> Tuple[float, float]:
    atomic_energies_fn = AtomicEnergiesBlock(atomic_energies=atomic_energies)

    atom_energy_list = []
    forces_list = []

    for batch in data_loader:
        node_e0 = atomic_energies_fn(batch.node_attrs)
        graph_e0s = scatter_sum(
            src=node_e0, index=batch.batch, dim=-1, dim_size=batch.num_graphs
        )
        graph_sizes = batch.ptr[1:] - batch.ptr[:-1]
        atom_energy_list.append(
            (batch.energy - graph_e0s) / graph_sizes
        )  # {[n_graphs], }
        forces_list.append(batch.forces)  # {[n_graphs*n_atoms,3], }

    atom_energies = torch.cat(atom_energy_list, dim=0)  # [total_n_graphs]
    forces = torch.cat(forces_list, dim=0)  # {[total_n_graphs*n_atoms,3], }

    mean = to_numpy(torch.mean(atom_energies)).item()
    rms = to_numpy(torch.sqrt(torch.mean(torch.square(forces)))).item()

    return mean, rms


def compute_avg_num_neighbors(data_loader: torch.utils.data.DataLoader) -> float:
    num_neighbors = []

    for batch in data_loader:
        _, receivers = batch.edge_index
        _, counts = torch.unique(receivers, return_counts=True)
        num_neighbors.append(counts)

    avg_num_neighbors = torch.mean(
        torch.cat(num_neighbors, dim=0).type(torch.get_default_dtype())
    )
    return to_numpy(avg_num_neighbors).item()


def sum_nodes_of_the_same_graph(
    graph: jraph.GraphsTuple, node_quantities: jnp.ndarray
) -> jnp.ndarray:
    """Sum node quantities and return a graph quantity."""
    num_graphs = graph.n_node.shape[0]
    num_nodes = graph.nodes.positions.shape[0]
    graph_index = jnp.repeat(
        jnp.arange(num_graphs), graph.n_node, total_repeat_length=num_nodes
    )  # [n_nodes,]
    graph_quantities = e3nn.index_add(
        indices=graph_index, input=node_quantities, out_dim=num_graphs
    )  # [ n_graphs,]
    return graph_quantities
