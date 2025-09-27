from typing import Any, NamedTuple, Optional

import jax.numpy as jnp


def get_symmetric_displacement(
    positions: jnp.ndarray,
    unit_shifts: jnp.ndarray,
    cell: Optional[jnp.ndarray],
    edge_index: jnp.ndarray,
    num_graphs: int,
    batch: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    if cell is None:
        # shape: (num_graphs * 3, 3)
        cell = jnp.zeros((num_graphs * 3, 3), dtype=positions.dtype)

    sender = edge_index[0]

    displacement = jnp.zeros((num_graphs, 3, 3), dtype=positions.dtype)

    # symmetric version: (num_graphs, 3, 3)
    symmetric_displacement = 0.5 * (displacement + jnp.swapaxes(displacement, -1, -2))

    # apply symmetric displacement to positions
    # "be,bec->bc": positions (N,3), sym_disp[batch] (N,3,3) -> (N,3)
    positions = positions + jnp.einsum(
        'be,bec->bc', positions, symmetric_displacement[batch]
    )

    # reshape cell to (-1,3,3) like in torch.view
    cell = cell.reshape(-1, 3, 3)

    # update cell
    cell = cell + jnp.matmul(cell, symmetric_displacement)

    # shifts: "be,bec->bc"
    shifts = jnp.einsum(
        'be,bec->bc',
        unit_shifts,
        cell[batch[sender]],
    )

    return positions, shifts, displacement


def get_edge_vectors_and_lengths(
    positions: jnp.ndarray,  # [n_nodes, 3]
    edge_index: jnp.ndarray,  # [2, n_edges]
    shifts: jnp.ndarray,  # [n_edges, 3]
    normalize: bool = False,
    eps: float = 1e-9,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    sender = edge_index[0]  # [n_edges]
    receiver = edge_index[1]  # [n_edges]

    # edge vectors
    vectors = positions[receiver] - positions[sender] + shifts  # [n_edges, 3]

    # lengths (keep last dim)
    lengths = jnp.linalg.norm(vectors, axis=-1, keepdims=True)  # [n_edges, 1]

    if normalize:
        vectors = vectors / (lengths + eps)

    return vectors, lengths


class InteractionKwargs(NamedTuple):
    lammps_class: Optional[Any]
    lammps_natoms: tuple[int, int]


class GraphContext(NamedTuple):
    num_graphs: int
    num_atoms_arange: jnp.ndarray
    displacement: Optional[jnp.ndarray]
    positions: jnp.ndarray
    vectors: jnp.ndarray
    lengths: jnp.ndarray
    cell: jnp.ndarray
    node_heads: jnp.ndarray
    interaction_kwargs: InteractionKwargs


def prepare_graph(
    data: dict[str, jnp.ndarray],
    compute_virials: bool = False,
    compute_stress: bool = False,
    compute_displacement: bool = False,
) -> GraphContext:
    if 'head' in data:
        node_heads = data['head'][data['batch']]
    else:
        node_heads = jnp.zeros_like(data['batch'])

    positions = data['positions']  # no requires_grad
    cell = data['cell']
    num_atoms_arange = jnp.arange(positions.shape[0])
    num_graphs = int(data['ptr'].shape[0] - 1)
    displacement = jnp.zeros((num_graphs, 3, 3), dtype=positions.dtype)

    if compute_virials or compute_stress or compute_displacement:
        positions, shifts, displacement = get_symmetric_displacement(
            positions=positions,
            unit_shifts=data['unit_shifts'],
            cell=cell,
            edge_index=data['edge_index'],
            num_graphs=num_graphs,
            batch=data['batch'],
        )
        data = dict(data)
        data['positions'] = positions
        data['shifts'] = shifts

    vectors, lengths = get_edge_vectors_and_lengths(
        positions=data['positions'],
        edge_index=data['edge_index'],
        shifts=data['shifts'],
    )
    ikw = InteractionKwargs(None, (0, 0))

    return GraphContext(
        num_graphs=num_graphs,
        num_atoms_arange=num_atoms_arange,
        displacement=displacement,
        positions=positions,
        vectors=vectors,
        lengths=lengths,
        cell=cell,
        node_heads=node_heads,
        interaction_kwargs=ikw,
    )
