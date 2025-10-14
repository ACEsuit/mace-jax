import logging
import os
import time
from contextlib import contextmanager
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from ase.data import chemical_symbols

try:
    from lammps.mliap.mliap_unified_abc import MLIAPUnified
except ImportError:  # pragma: no cover

    class MLIAPUnified:  # type: ignore
        def __init__(self):
            pass


class MACELammpsConfig:
    """Configuration settings for MACE-LAMMPS integration."""

    def __init__(self) -> None:
        self.debug_time = self._get_env_bool('MACE_TIME', False)
        self.debug_profile = self._get_env_bool('MACE_PROFILE', False)
        self.profile_start_step = int(os.environ.get('MACE_PROFILE_START', '5'))
        self.profile_end_step = int(os.environ.get('MACE_PROFILE_END', '10'))
        self.allow_cpu = self._get_env_bool('MACE_ALLOW_CPU', False)
        self.force_cpu = self._get_env_bool('MACE_FORCE_CPU', False)

    @staticmethod
    def _get_env_bool(var_name: str, default: bool) -> bool:
        return os.environ.get(var_name, str(default)).lower() in {
            'true',
            '1',
            't',
            'yes',
        }


@contextmanager
def timer(name: str, enabled: bool = False):
    """Context manager for timing code blocks."""
    if not enabled:
        yield
        return

    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        logging.info('Timer - %s: %.3f ms', name, elapsed * 1000.0)


class MACEEdgeForcesWrapper:
    """Wrapper that adds per-pair force computation to a MACE model."""

    def __init__(self, model: Any, params: Any, **kwargs: Any) -> None:
        self.model = model
        self.params = params

        self.atomic_numbers = np.asarray(model.atomic_numbers, dtype=np.int64)
        self.r_max = float(model.r_max)
        self.num_interactions = int(model.num_interactions)

        heads = getattr(model, '_heads', None)
        if not heads:
            heads = getattr(model, 'heads', None)
        if not heads:
            heads = ('Default',)
        self.heads = tuple(heads)
        head_name = kwargs.get('head', self.heads[-1])
        self.head_idx = self.heads.index(head_name)

        def _apply_fn(p, d, lammps_class, lammps_natoms):
            data = d if lammps_natoms is None else dict(d, natoms=lammps_natoms)
            return self.model.apply(
                p,
                data,
                compute_force=False,
                compute_stress=False,
                lammps_mliap=True,
                lammps_class=lammps_class,
            )

        self._apply = jax.jit(_apply_fn, static_argnums=(2, 3))

        def energy_sum(p, d, lammps_class, lammps_natoms, vectors):
            batch = dict(d)
            if lammps_natoms is not None:
                batch['natoms'] = lammps_natoms
            batch['vectors'] = vectors
            out = self.model.apply(
                p,
                batch,
                compute_force=False,
                compute_stress=False,
                lammps_mliap=True,
                lammps_class=lammps_class,
            )
            energy = out['energy']
            return jnp.sum(energy)

        self._grad_edge_forces = jax.jit(
            jax.grad(energy_sum, argnums=4),
            static_argnums=(2, 3),
        )

    def __call__(
        self, data: dict[str, Any]
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        batch = dict(data)
        lammps_class = batch.pop('lammps_class', None)
        lammps_natoms = batch.pop('natoms', None)
        batch['head'] = jnp.asarray([self.head_idx], dtype=jnp.int32)

        outputs = self._apply(
            self.params,
            batch,
            lammps_class,
            lammps_natoms,
        )
        energy = outputs['energy']
        node_energy = outputs['node_energy']

        vectors = batch['vectors']
        grad_vectors = self._grad_edge_forces(
            self.params,
            batch,
            lammps_class,
            lammps_natoms,
            vectors,
        )
        pair_forces = -grad_vectors

        return energy, node_energy, pair_forces


class LAMMPS_MLIAP_MACE(MLIAPUnified):
    """MACE integration for LAMMPS using the MLIAP interface (JAX version)."""

    def __init__(self, model: Any, params: Any, **kwargs: Any) -> None:
        super().__init__()
        self.config = MACELammpsConfig()
        self.model = MACEEdgeForcesWrapper(model, params, **kwargs)

        self.element_types = [
            chemical_symbols[int(z)] for z in self.model.atomic_numbers
        ]
        self.num_species = len(self.element_types)
        self.rcutfac = 0.5 * float(self.model.r_max)
        self.ndescriptors = 1
        self.nparams = 1
        self.dtype = jnp.asarray(self.model.r_max).dtype

        self.device = 'cpu'
        self.initialized = False
        self.step = 0

    def _initialize_device(self, data: Any) -> None:
        using_kokkos = 'kokkos' in data.__class__.__module__.lower()

        if using_kokkos and not self.config.force_cpu:
            # Kokkos tensors implement the array interface, detect device type if possible
            elems = np.asarray(data.elems)
            if elems.dtype != np.int64 and not self.config.allow_cpu:
                raise ValueError(
                    'GPU requested but data appears to be on CPU. '
                    'Set MACE_ALLOW_CPU=true to force CPU computation.'
                )
        self.initialized = True
        logging.info('MACE model initialized on device: %s', self.device)

    def compute_forces(self, data: Any) -> None:
        natoms = int(data.nlocal)
        ntotal = int(data.ntotal)
        nghosts = ntotal - natoms
        npairs = int(data.npairs)
        species = jnp.asarray(np.asarray(data.elems), dtype=jnp.int64)

        if not self.initialized:
            self._initialize_device(data)

        self.step += 1
        self._manage_profiling()

        if natoms == 0 or npairs <= 1:
            return

        with timer('total_step', enabled=self.config.debug_time):
            with timer('prepare_batch', enabled=self.config.debug_time):
                batch = self._prepare_batch(data, natoms, nghosts, species)

            with timer('model_forward', enabled=self.config.debug_time):
                energy, atom_energies, pair_forces = self.model(batch)

            with timer('update_lammps', enabled=self.config.debug_time):
                self._update_lammps_data(
                    data, atom_energies, pair_forces, natoms, energy
                )

    def _prepare_batch(
        self,
        data: Any,
        natoms: int,
        nghosts: int,
        species: jnp.ndarray,
    ) -> dict[str, Any]:
        vectors = jnp.asarray(np.asarray(data.rij), dtype=self.dtype)
        pair_j = jnp.asarray(np.asarray(data.pair_j), dtype=jnp.int32)
        pair_i = jnp.asarray(np.asarray(data.pair_i), dtype=jnp.int32)

        batch = jnp.zeros(natoms, dtype=jnp.int32)
        ptr = jnp.asarray([0, natoms, natoms + nghosts], dtype=jnp.int32)

        raw_positions = getattr(data, 'positions', None)
        if raw_positions is None:
            raw_positions = getattr(data, 'x', None)
        if raw_positions is not None:
            positions_arr = np.asarray(raw_positions, dtype=float)
            if positions_arr.ndim != 2 or positions_arr.shape[1] != 3:
                raise ValueError('LAMMPS data.positions must have shape (N, 3)')
            if positions_arr.shape[0] < natoms:
                raise ValueError(
                    f'LAMMPS positions only provided for {positions_arr.shape[0]} atoms, '
                    f'but nlocal={natoms}'
                )
            positions = jnp.asarray(positions_arr[:natoms], dtype=self.dtype)
        else:
            positions = jnp.zeros((natoms, 3), dtype=self.dtype)

        raw_cell = getattr(data, 'cell', None)
        if raw_cell is not None:
            cell_arr = np.asarray(raw_cell, dtype=float)
            if cell_arr.ndim == 2 and cell_arr.shape == (3, 3):
                cell = jnp.asarray(np.stack((cell_arr, cell_arr)), dtype=self.dtype)
            elif cell_arr.ndim == 3 and cell_arr.shape[-2:] == (3, 3):
                if cell_arr.shape[0] == 1:
                    cell = jnp.asarray(np.repeat(cell_arr, 2, axis=0), dtype=self.dtype)
                elif cell_arr.shape[0] == 2:
                    cell = jnp.asarray(cell_arr, dtype=self.dtype)
                else:
                    cell = jnp.asarray(cell_arr[:2], dtype=self.dtype)
            else:
                raise ValueError('LAMMPS cell must be of shape (3,3) or (2,3,3)')
        else:
            cell = jnp.zeros((2, 3, 3), dtype=self.dtype)

        raw_unit_shifts = getattr(data, 'unit_shifts', None)
        if raw_unit_shifts is not None:
            unit_shifts_arr = np.asarray(raw_unit_shifts, dtype=float)
            if unit_shifts_arr.shape != vectors.shape:
                raise ValueError(
                    'LAMMPS unit_shifts must match the shape of rij vectors'
                )
            unit_shifts = jnp.asarray(unit_shifts_arr, dtype=self.dtype)
        else:
            unit_shifts = jnp.zeros_like(vectors)

        raw_shifts = getattr(data, 'shifts', None)
        if raw_shifts is not None:
            shifts_arr = np.asarray(raw_shifts, dtype=float)
            if shifts_arr.shape != vectors.shape:
                raise ValueError('LAMMPS shifts must match the shape of rij vectors')
            shifts = jnp.asarray(shifts_arr, dtype=self.dtype)
        else:
            if raw_unit_shifts is not None and raw_cell is not None:
                shifts = unit_shifts @ cell[0]
            else:
                shifts = jnp.zeros_like(vectors)

        node_attrs = jax.nn.one_hot(
            species, num_classes=self.num_species, dtype=self.dtype
        )

        return {
            'vectors': vectors,
            'node_attrs': node_attrs,
            'edge_index': jnp.stack((pair_j, pair_i), axis=0),
            'batch': batch,
            'natoms': (natoms, nghosts),
            'ptr': ptr,
            'positions': positions,
            'unit_shifts': unit_shifts,
            'shifts': shifts,
            'cell': cell,
            'lammps_class': data,
        }

    def _update_lammps_data(
        self,
        data: Any,
        atom_energies: jnp.ndarray,
        pair_forces: jnp.ndarray,
        natoms: int,
        total_energy: jnp.ndarray,
    ) -> None:
        eatoms = np.asarray(data.eatoms)
        per_atom = np.array(atom_energies[:natoms])
        np.copyto(eatoms[:natoms], per_atom)
        data.energy = float(per_atom.sum())
        data.update_pair_forces_gpu(np.array(pair_forces, dtype=np.float64))

    def _manage_profiling(self) -> None:
        if not self.config.debug_profile:
            return

        if self.step == self.config.profile_start_step:
            logging.info('Starting profiler at step %s', self.step)
        if self.step == self.config.profile_end_step:
            logging.info('Stopping profiler at step %s', self.step)
            logging.info('Profiling complete. Exiting.')
            raise SystemExit()

    def compute_descriptors(
        self, data: Any
    ) -> None:  # pragma: no cover - interface stub
        del data

    def compute_gradients(self, data: Any) -> None:  # pragma: no cover - interface stub
        del data


def create_lammps_mliap_calculator(
    model: Any,
    params: Any,
    **kwargs: Any,
) -> LAMMPS_MLIAP_MACE:
    """
    Convenience factory mirroring the torch create_lammps_model helper.

    Args:
        model: Instantiated Flax module (e.g. `MACE` or `ScaleShiftMACE`).
        params: PyTree of trained parameters for `model`.
        **kwargs: Optional keyword arguments such as `head`.

    Returns:
        Initialised `LAMMPS_MLIAP_MACE` wrapper ready for use inside LAMMPS.
    """

    return LAMMPS_MLIAP_MACE(model, params, **kwargs)
