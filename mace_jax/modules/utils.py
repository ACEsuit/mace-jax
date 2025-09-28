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
        forces: [n_nodes, 3] array of forces
    """
    # Compute energy and gradient in one pass
    grad_pos = jax.grad(lambda pos: jnp.sum(energy_fn(pos)))(positions)

    # Forces = -∂E/∂R
    forces = -grad_pos
    return forces


def compute_forces_and_stress(
    energy_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    positions: jnp.ndarray,  # [n_nodes, 3]
    cell: jnp.ndarray,  # [n_graphs, 3, 3]
    num_graphs: int,
    batch: jnp.ndarray,  # [n_nodes] with graph indices
) -> tuple[jnp.ndarray, Optional[jnp.ndarray], Optional[jnp.ndarray], jnp.ndarray]:
    """
    JAX equivalent of the PyTorch compute_forces_virials.
    Computes forces, virials, stress, and returns the per-graph energies.

    Args:
        energy_fn: function returning per-graph energies [n_graphs]
        positions: [n_nodes, 3]
        cell: [n_graphs, 3, 3]
        num_graphs: int
        batch: [n_nodes] with graph indices
    """

    # Define a scalar-valued loss = sum over graphs to mimic grad_outputs=ones_like(...)
    def energy_displacement_fn(positions, displacement):
        # Make the displacement symmetric:
        symmetric_displacement = 0.5 * (
            displacement + jnp.swapaxes(displacement, -1, -2)
        )  # shape: [n_graphs, 3, 3]

        # Apply symmetric deformation to positions:
        deformed_positions = positions + jnp.einsum(
            'be,bec->bc', positions, symmetric_displacement[batch]
        )

        return jnp.sum(energy_fn(deformed_positions))

    # Create a zero displacement tensor of shape [num_graphs, 3, 3]
    displacement = jnp.zeros((num_graphs, 3, 3), dtype=positions.dtype)

    # Compute total energy (scalar) and gradients wrt both inputs
    grad_pos, grad_disp = jax.grad(energy_displacement_fn, argnums=(0, 1))(
        positions, displacement
    )

    # Forces and virials follow the physics convention
    forces = -grad_pos
    virials = -grad_disp

    # Stress computation
    volume = jnp.abs(jnp.linalg.det(cell.reshape(-1, 3, 3))).reshape(-1, 1, 1)
    stress = virials / volume
    stress = jnp.where(jnp.abs(stress) < 1e10, stress, jnp.zeros_like(stress))

    return forces, stress


def get_outputs(
    energy_fn,
    data: dict[str, jnp.ndarray],
    compute_force: bool = True,
    compute_stress: bool = False,
) -> tuple[
    Optional[jnp.ndarray],  # energy
    Optional[jnp.ndarray],  # forces
    Optional[jnp.ndarray],  # stress
]:
    positions = data['positions']
    cell = data['cell']
    num_graphs = int(jnp.size(data['ptr']) - 1)
    batch = data['batch']

    total_energy = energy_fn(positions)
    forces = None
    stress = None

    if compute_force and not compute_stress:
        forces = compute_forces(
            energy_fn=energy_fn,
            positions=positions,
        )
    elif compute_stress:
        forces, stress = compute_forces_and_stress(
            energy_fn=energy_fn,
            positions=positions,
            cell=cell,
            num_graphs=num_graphs,
            batch=batch,
        )

    return total_energy, forces, stress


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


def add_output_interface(cls=None):
    """
    Decorator that injects a __call__ method into a class that already defines
    __call_energy__(self, data, training=False) -> dict with at least 'energy'
    and possibly other intermediate values.
    """

    def wrap(cls):
        def __call__(
            self,
            data: dict[str, jnp.ndarray],
            compute_force: bool = True,
            compute_stress: bool = False,
        ) -> dict[str, Optional[jnp.ndarray]]:
            def energy_fn(positions):
                # Replace the positions in `data` with `pos` before recomputing
                new_data = dict(data)
                new_data['positions'] = positions

                return self._energy_fn(
                    new_data,
                )

            total_energy, forces, stress = get_outputs(
                energy_fn=energy_fn,
                data=data,
                compute_force=compute_force,
                compute_stress=compute_stress,
            )

            return {
                'energy': total_energy,
                'forces': forces,
                'stress': stress,
            }

        # Move __call__ to _energy_fn
        setattr(cls, '_energy_fn', getattr(cls, '__call__'))
        # Attach the new __call__
        setattr(cls, '__call__', __call__)
        return cls

    # Allow decorator to be used with or without parentheses
    return wrap if cls is None else wrap(cls)
