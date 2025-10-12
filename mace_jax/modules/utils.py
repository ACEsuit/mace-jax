from typing import Any, Callable, NamedTuple, Optional

import jax
import jax.numpy as jnp

from mace_jax.tools.dtype import default_dtype


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
    unit_shifts: jnp.ndarray,  # [n_edges, 3]
    edge_index: jnp.ndarray,  # [2, n_edges]
    batch: jnp.ndarray,  # [n_nodes] with graph indices
    num_graphs: int,
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
        cell_reshaped = cell.reshape(-1, 3, 3)  # [n_graphs, 3, 3]
        cell_reshaped = cell_reshaped + jnp.matmul(
            cell_reshaped, symmetric_displacement
        )  # [n_graphs, 3, 3]
        shifts = jnp.einsum(
            'be,bec->bc',
            unit_shifts,  # [n_edges, 3]
            cell_reshaped[batch[edge_index[0]]],  # [n_edges, 3, 3]
        )  # → [n_edges, 3]

        return jnp.sum(energy_fn(deformed_positions, shifts=shifts))

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
    stress = -virials / volume
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
    unit_shifts = data['unit_shifts']
    edge_index = data['edge_index']
    batch = data['batch']
    num_graphs = int(jnp.size(data['ptr']) - 1)

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
            unit_shifts=unit_shifts,
            edge_index=edge_index,
            batch=batch,
            num_graphs=num_graphs,
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
    is_lammps: bool
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
    *,
    compute_virials: bool = False,
    compute_stress: bool = False,
    compute_displacement: bool = False,
    lammps_mliap: bool = False,
) -> GraphContext:
    batch = jnp.asarray(data['batch'], dtype=jnp.int32)

    if 'head' in data:
        heads = jnp.asarray(data['head'], dtype=jnp.int32)
        node_heads = heads[batch]
    else:
        node_heads = jnp.zeros_like(batch)

    if lammps_mliap:
        node_attrs = jnp.asarray(data['node_attrs'])
        n_real = int(node_attrs.shape[0])
        n_ghosts = 0
        vectors = jnp.asarray(data['vectors'], dtype=default_dtype())
        lengths = jnp.linalg.norm(vectors, axis=-1, keepdims=True)
        num_graphs = 2  # match torch behaviour: real and ghost graph

        positions = jnp.zeros((n_real, 3), dtype=vectors.dtype)
        displacement = None
        cell = jnp.zeros((num_graphs, 3, 3), dtype=vectors.dtype)
        num_atoms_arange = jnp.arange(n_real, dtype=jnp.int32)
        node_heads = node_heads[:n_real]
        ikw = InteractionKwargs(None, (n_real, n_ghosts))

        return GraphContext(
            is_lammps=True,
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

    positions = jnp.asarray(data['positions'], dtype=default_dtype())
    cell = jnp.asarray(data['cell'], dtype=positions.dtype)
    shifts = jnp.asarray(data['shifts'], dtype=positions.dtype)
    edge_index = jnp.asarray(data['edge_index'], dtype=jnp.int32)

    num_atoms_arange = jnp.arange(positions.shape[0], dtype=jnp.int32)
    num_graphs = int(data['ptr'].shape[0] - 1)
    displacement = jnp.zeros((num_graphs, 3, 3), dtype=positions.dtype)

    vectors, lengths = get_edge_vectors_and_lengths(
        positions=positions,
        edge_index=edge_index,
        shifts=shifts,
    )
    ikw = InteractionKwargs(None, (0, 0))

    return GraphContext(
        is_lammps=False,
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
            **model_kwargs,
        ) -> dict[str, Optional[jnp.ndarray]]:
            raw_out = self._energy_fn(
                data,
                **model_kwargs,
            )

            energy_arr = raw_out['energy'] if isinstance(raw_out, dict) else raw_out

            def energy_fn(positions, shifts=None):
                # Replace the positions in `data` with `pos` before recomputing
                new_data = dict(data)
                new_data['positions'] = positions

                if shifts is not None:
                    new_data['shifts'] = shifts

                out = self._energy_fn(
                    new_data,
                    **model_kwargs,
                )
                return out['energy'] if isinstance(out, dict) else out

            total_energy, forces, stress = get_outputs(
                energy_fn=energy_fn,
                data=data,
                compute_force=compute_force,
                compute_stress=compute_stress,
            )

            result = (
                dict(raw_out) if isinstance(raw_out, dict) else {'energy': energy_arr}
            )
            result.update(
                {
                    'energy': total_energy,
                    'forces': forces,
                    'stress': stress,
                }
            )
            return result

        # Move __call__ to _energy_fn
        setattr(cls, '_energy_fn', getattr(cls, '__call__'))
        # Attach the new __call__
        setattr(cls, '__call__', __call__)
        return cls

    # Allow decorator to be used with or without parentheses
    return wrap if cls is None else wrap(cls)
