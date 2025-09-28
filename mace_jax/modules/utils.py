from typing import Any, Callable, NamedTuple, Optional

import jax
import jax.numpy as jnp


def compute_forces(
    energy_fn: Callable[[jnp.ndarray], jnp.ndarray],
    positions: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute total energy and forces as the negative gradient of energy w.r.t. positions.

    Args:
        energy_fn: function taking positions -> energy (scalar or [n_graphs])
        positions: [n_nodes, 3] array of positions

    Returns:
        energy: scalar total energy (sum over graphs if energy_fn returns [n_graphs])
        forces: [n_nodes, 3] array of forces
    """
    # Compute energy and gradient in one pass
    grad_pos = jax.grad(lambda pos: jnp.sum(energy_fn(pos)))(positions)

    # Forces = -∂E/∂R
    forces = -grad_pos
    return forces


def get_outputs(
    energy_fn,
    positions: jnp.ndarray,
    compute_force: bool = True,
) -> tuple[
    Optional[jnp.ndarray],  # forces
    Optional[jnp.ndarray],  # virials
    Optional[jnp.ndarray],  # stress
    Optional[jnp.ndarray],  # hessian
    Optional[jnp.ndarray],  # edge_forces
]:
    total_energy = energy_fn(positions)
    forces = None

    if compute_force:
        forces = compute_forces(
            energy_fn=energy_fn,
            positions=positions,
        )

    return total_energy, forces


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
