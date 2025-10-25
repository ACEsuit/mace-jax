from collections.abc import Callable
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax import lax
from jax.errors import TracerBoolConversionError

from mace_jax.tools.dtype import default_dtype


class Outputs(NamedTuple):
    """Container describing energy, forces, stress, and their availability masks."""

    energy: jnp.ndarray
    forces: jnp.ndarray | None
    stress: jnp.ndarray | None
    force_mask: bool | jnp.ndarray
    stress_mask: bool | jnp.ndarray


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
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    JAX equivalent of the PyTorch compute_forces_virials.
    Computes forces and stress tensors using a symmetric displacement scheme.

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

        per_graph_energy = energy_fn(deformed_positions, shifts=shifts)
        return jnp.sum(per_graph_energy), per_graph_energy

    # Create a zero displacement tensor of shape [num_graphs, 3, 3]
    displacement = jnp.zeros((num_graphs, 3, 3), dtype=positions.dtype)

    # Compute total energy (scalar) and gradients wrt both inputs
    (_, _), (grad_pos, grad_disp) = jax.value_and_grad(
        energy_displacement_fn,
        argnums=(0, 1),
        has_aux=True,
    )(positions, displacement)

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
) -> Outputs:
    """
    Evaluate per-graph energies together with optional forces and stress tensors.

    The function works with both plain Python booleans and traced JAX booleans.
    When `compute_force` / `compute_stress` are Python `bool`s we only execute
    the strictly necessary derivative computation and return `None` for any
    disabled quantity. When the flags are traced (for example, inside a JIT
    compiled control flow) the same logic is reproduced with `lax.cond`; in that
    case the function always returns arrays but accompanies them with boolean
    masks so callers can tell which entries are meaningful.

    Args:
        energy_fn: Callable that accepts positions (and optional shifts) and
            returns per-graph energies with shape `[n_graphs]`.
        data: Graph batch containing at least `positions`, `cell`, `unit_shifts`,
            `edge_index`, `batch`, and `ptr`.
        compute_force: Whether to return forces. May be a Python bool or a JAX
            boolean scalar.
        compute_stress: Whether to return stresses. As above, can be Python or
            traced.

    Returns:
        An `Outputs` tuple where:
            * `energy` is a `[n_graphs]` array of per-graph energies.
            * `forces` is `[n_nodes, 3]` or `None` depending on the flags.
            * `stress` is `[n_graphs, 3, 3]` or `None` depending on the flags.
            * `force_mask` / `stress_mask` indicate availability; when the input
              flags were traced these masks are JAX booleans aligned with the
              returned arrays.
    """
    positions = data['positions']
    cell = data['cell']
    unit_shifts = data['unit_shifts']
    edge_index = data['edge_index']
    batch = data['batch']
    num_graphs = int(jnp.size(data['ptr']) - 1)

    displacement = jnp.zeros((num_graphs, 3, 3), dtype=positions.dtype)

    # Helper closures
    def energy_displacement_fn(pos, disp):
        """Return total energy and per-graph energies for a symmetric displacement."""
        symmetric_displacement = 0.5 * (disp + jnp.swapaxes(disp, -1, -2))
        deformed_positions = pos + jnp.einsum(
            'be,bec->bc', pos, symmetric_displacement[batch]
        )
        cell_reshaped = cell.reshape(-1, 3, 3)
        cell_reshaped = cell_reshaped + jnp.matmul(
            cell_reshaped, symmetric_displacement
        )
        shifts = jnp.einsum(
            'be,bec->bc',
            unit_shifts,
            cell_reshaped[batch[edge_index[0]]],
        )
        per_graph_energy = energy_fn(deformed_positions, shifts=shifts)
        return jnp.sum(per_graph_energy), per_graph_energy

    def energy_sum_fn(pos):
        """Return (total energy, per-graph energy) for the un-deformed geometry."""
        per_graph_energy = energy_fn(pos)
        return jnp.sum(per_graph_energy), per_graph_energy

    def _ensure_energy(value: jnp.ndarray) -> jnp.ndarray:
        return jnp.atleast_1d(jnp.asarray(value))

    def _compute_stress_raw() -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Gradient-based computation of (energy, forces, stress) for stress path."""
        (_, per_graph_energy), (grad_pos, grad_disp) = jax.value_and_grad(
            energy_displacement_fn,
            argnums=(0, 1),
            has_aux=True,
        )(positions, displacement)

        forces = -grad_pos
        virials = -grad_disp
        volume = jnp.abs(jnp.linalg.det(cell.reshape(-1, 3, 3))).reshape(-1, 1, 1)
        stress = -virials / volume
        stress = jnp.where(
            jnp.abs(stress) < 1e10,
            stress,
            jnp.zeros_like(stress),
        )
        total_energy = _ensure_energy(per_graph_energy)
        return total_energy, forces, stress

    def _compute_force_raw() -> tuple[jnp.ndarray, jnp.ndarray]:
        """Gradient-based computation of (energy, forces) for the force-only path."""
        (_, per_graph_energy), grad_pos = jax.value_and_grad(
            energy_sum_fn,
            has_aux=True,
        )(positions)
        forces = -grad_pos
        total_energy = _ensure_energy(per_graph_energy)
        return total_energy, forces

    def _compute_energy_raw() -> jnp.ndarray:
        """Energy-only evaluation used when no derivatives are requested."""
        return _ensure_energy(energy_fn(positions))

    def _coerce_flag(flag):
        """Return `(flag_value, is_traced)` for Python bools and traced JAX values."""
        try:
            return bool(flag), False
        except TracerBoolConversionError:
            return flag, True

    force_flag, force_traced = _coerce_flag(compute_force)
    stress_flag, stress_traced = _coerce_flag(compute_stress)

    if not (force_traced or stress_traced):
        if stress_flag:
            total_energy, forces, stress = _compute_stress_raw()
            return Outputs(total_energy, forces, stress, True, True)
        if force_flag:
            total_energy, forces = _compute_force_raw()
            return Outputs(total_energy, forces, None, True, False)
        total_energy = _compute_energy_raw()
        return Outputs(total_energy, None, None, False, False)

    compute_force_mask = jnp.asarray(compute_force, dtype=bool)
    compute_stress_mask = jnp.asarray(compute_stress, dtype=bool)

    zero_forces = jnp.zeros_like(positions)
    zero_stress = jnp.zeros((num_graphs, 3, 3), dtype=positions.dtype)
    true_scalar = jnp.asarray(True, dtype=bool)
    false_scalar = jnp.asarray(False, dtype=bool)

    def _stress_branch(_):
        total_energy, forces, stress = _compute_stress_raw()
        return Outputs(total_energy, forces, stress, true_scalar, true_scalar)

    def _force_branch(_):
        total_energy, forces = _compute_force_raw()
        return Outputs(total_energy, forces, zero_stress, true_scalar, false_scalar)

    def _energy_only_branch(_):
        total_energy = _compute_energy_raw()
        return Outputs(
            total_energy, zero_forces, zero_stress, false_scalar, false_scalar
        )

    # lazily route traced flags through the right derivative path so only the
    # requested branch is evaluated when inside a JIT-compiled context
    return lax.cond(
        compute_stress_mask,
        _stress_branch,
        lambda _: lax.cond(
            compute_force_mask,
            _force_branch,
            _energy_only_branch,
            operand=None,
        ),
        operand=None,
    )


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
    lammps_class: Any | None
    lammps_natoms: tuple[int, int]


class GraphContext(NamedTuple):
    is_lammps: bool
    num_graphs: int
    num_atoms_arange: jnp.ndarray
    displacement: jnp.ndarray | None
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
    lammps_class: Any | None = None,
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
        natoms = data.get('natoms', (n_real, 0))
        if isinstance(natoms, tuple):
            n_ghosts = int(natoms[1]) if len(natoms) > 1 else 0
        else:
            n_ghosts = int(natoms) if natoms else 0
        vectors = jnp.asarray(data['vectors'], dtype=default_dtype())
        lengths = jnp.linalg.norm(vectors, axis=-1, keepdims=True)
        num_graphs = 2  # match torch behaviour: real and ghost graph

        positions = jnp.zeros((n_real, 3), dtype=vectors.dtype)
        displacement = None
        cell = jnp.zeros((num_graphs, 3, 3), dtype=vectors.dtype)
        num_atoms_arange = jnp.arange(n_real, dtype=jnp.int32)
        node_heads = node_heads[:n_real]
        lammps_cls = (
            lammps_class if lammps_class is not None else data.get('lammps_class')
        )
        ikw = InteractionKwargs(lammps_cls, (n_real, n_ghosts))

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
        ) -> dict[str, jnp.ndarray | None]:
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

            total_energy, forces, stress, has_force, has_stress = get_outputs(
                energy_fn=energy_fn,
                data=data,
                compute_force=compute_force,
                compute_stress=compute_stress,
            )

            result = (
                dict(raw_out) if isinstance(raw_out, dict) else {'energy': energy_arr}
            )
            try:
                force_flag = bool(has_force)
                stress_flag = bool(has_stress)
            except TracerBoolConversionError:
                result.update({
                    'energy': total_energy,
                    'forces': forces,
                    'stress': stress,
                    'forces_mask': has_force,
                    'stress_mask': has_stress,
                })
            else:
                result.update({
                    'energy': total_energy,
                    'forces': forces if force_flag else None,
                    'stress': stress if stress_flag else None,
                })
            return result

        # Move __call__ to _energy_fn
        setattr(cls, '_energy_fn', getattr(cls, '__call__'))
        # Attach the new __call__
        setattr(cls, '__call__', __call__)
        return cls

    # Allow decorator to be used with or without parentheses
    return wrap if cls is None else wrap(cls)
