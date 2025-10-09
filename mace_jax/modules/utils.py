from typing import Any, Callable, NamedTuple, Optional

import jax
import jax.numpy as jnp


def compute_forces(
    energy_fn: Callable[[jnp.ndarray], jnp.ndarray],
    positions: jnp.ndarray,
) -> jnp.ndarray:
    """Return forces as ``-∂E/∂R`` for the provided configuration.

    We reduce the per-graph energies down to a scalar so that ``jax.grad`` mirrors the
    scalar-valued backwards pass used by Torch, ensuring the gradient is accumulated
    across graphs with identical weights.
    """

    grad_pos = jax.grad(lambda pos: jnp.sum(energy_fn(pos)))(positions)
    return -grad_pos


def compute_forces_and_stress(
    energy_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    positions: jnp.ndarray,  # [n_nodes, 3]
    cell: jnp.ndarray,  # [n_graphs, 3, 3]
    unit_shifts: jnp.ndarray,  # [n_edges, 3]
    edge_index: jnp.ndarray,  # [2, n_edges]
    batch: jnp.ndarray,  # [n_nodes] with graph indices
    num_graphs: int,
) -> tuple[jnp.ndarray, Optional[jnp.ndarray], Optional[jnp.ndarray], jnp.ndarray]:
    """Replicate Torch's symmetric deformation trick to obtain stress.

    Torch back-propagates a symmetric strain tensor against the energy in order to
    compute virials.  We mimic that procedure in JAX to keep sign conventions and
    scaling identical, ensuring round-trips through ``import_from_torch`` continue to
    match reference outputs.
    """

    cell = cell.reshape(-1, 3, 3)

    def deformation_energy(pos, displacement):
        symmetric = 0.5 * (displacement + jnp.swapaxes(displacement, -1, -2))
        deformed_pos = pos + jnp.einsum('be,bec->bc', pos, symmetric[batch])
        deformed_cell = cell + jnp.matmul(cell, symmetric)
        shifts = jnp.einsum(
            'be,bec->bc', unit_shifts, deformed_cell[batch[edge_index[0]]]
        )
        return energy_fn(deformed_pos, shifts=shifts)

    displacement0 = jnp.zeros((num_graphs, 3, 3), dtype=positions.dtype)

    def scalar_energy(pos, displacement):
        return jnp.sum(deformation_energy(pos, displacement))

    grad_pos, grad_disp = jax.grad(scalar_energy, argnums=(0, 1))(
        positions, displacement0
    )

    forces = -grad_pos
    virials = -grad_disp

    volume = jnp.abs(jnp.linalg.det(cell)).reshape(-1, 1, 1)
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
    """Invoke the appropriate autograd routine to obtain forces and stress.

    The signature mirrors the Torch helper so higher-level modules can request the
    same combination of outputs regardless of backend.
    """
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
        # Stress computation also returns forces so we can share the deformation pass.
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
            def energy_fn(positions, shifts=None):
                # Replace the positions in `data` with `pos` before recomputing
                new_data = dict(data)
                new_data['positions'] = positions

                if shifts is not None:
                    new_data['shifts'] = shifts

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
