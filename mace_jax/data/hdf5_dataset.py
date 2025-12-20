"""Minimal reader for native MACE-style HDF5 datasets."""

from __future__ import annotations

from pathlib import Path
import re

import h5py
import numpy as np
from ase import Atoms


class _CachedCalc:
    """Tiny replacement for ASE's SinglePointCalculator."""

    def __init__(self, energy, forces, stress):
        self._energy = float(energy)
        self._forces = np.asarray(forces)
        self._stress = np.asarray(stress)

    def get_potential_energy(self, apply_constraint: bool = False):
        del apply_constraint
        return self._energy

    def get_forces(self, apply_constraint: bool = False):
        del apply_constraint
        return self._forces

    def get_stress(self, apply_constraint: bool = False):
        del apply_constraint
        return self._stress


def _decode_field(value):
    """Best-effort conversion of HDF5 scalars/arrays into native Python types."""
    if isinstance(value, np.ndarray) and value.shape == ():
        value = value[()]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (bytes, np.bytes_)):
        value = value.decode('utf-8')
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == 'None':
            return None
        return stripped
    return value


def _voigt_to_matrix(stress):
    if stress is None:
        return None
    arr = np.asarray(stress)
    if arr.shape == (3, 3):
        return arr
    if arr.shape == (6,):
        xx, yy, zz, yz, xz, xy = arr
        return np.asarray(
            [
                [xx, xy, xz],
                [xy, yy, yz],
                [xz, yz, zz],
            ]
        )
    return arr


def _sorted_numeric(names, prefix):
    def _key(name):
        match = re.search(r'(\d+)$', name)
        return int(match.group(1)) if match else name

    return sorted((name for name in names if name.startswith(prefix)), key=_key)


class HDF5Dataset:
    """Random-access view over a native MACE streaming HDF5 store."""

    def __init__(self, filename: Path | str, mode: str = 'r'):
        self._filename = Path(filename)
        self._handle = h5py.File(self._filename, mode)
        self._index = self._build_index()

    def _build_index(self) -> list[tuple[str, str]]:
        index: list[tuple[str, str]] = []
        batch_names = _sorted_numeric(self._handle.keys(), 'config_batch_')
        for batch in batch_names:
            group = self._handle[batch]
            for config in _sorted_numeric(group.keys(), 'config_'):
                index.append((batch, config))
        if not index:
            raise ValueError(
                f"HDF5 file '{self._filename}' does not contain any MACE batches."
            )
        return index

    @property
    def filename(self) -> Path:
        return self._filename

    def __len__(self) -> int:
        return len(self._index)

    def _read_entry(self, index: int) -> Atoms:
        batch_name, config_name = self._index[index]
        group = self._handle[batch_name][config_name]
        atomic_numbers = np.asarray(group['atomic_numbers']).astype(np.int32, copy=False)
        positions = np.asarray(group['positions'])
        cell = _decode_field(group['cell'][()])
        if cell is None:
            cell = np.zeros((3, 3), dtype=float)
        pbc_data = _decode_field(group['pbc'][()])
        if pbc_data is None:
            pbc = (False, False, False)
        else:
            pbc_array = np.asarray(pbc_data).astype(bool)
            pbc = tuple(bool(x) for x in np.reshape(pbc_array, (-1,))[:3])

        atoms = Atoms(
            numbers=atomic_numbers,
            positions=positions,
            cell=np.asarray(cell),
            pbc=pbc,
        )
        info = atoms.info

        config_type = group.get('config_type')
        if config_type is not None:
            decoded = _decode_field(config_type[()])
            if decoded is not None:
                info['config_type'] = str(decoded)

        weight = group.get('weight')
        if weight is not None:
            decoded = _decode_field(weight[()])
            if decoded is not None:
                info['config_weight'] = float(decoded)

        properties = group.get('properties')
        if properties is not None:
            for key in properties.keys():
                value = _decode_field(properties[key][()])
                if value is None:
                    continue
                if key == 'forces':
                    atoms.arrays['forces'] = np.asarray(value)
                else:
                    info[key] = np.asarray(value)

        prop_weights = group.get('property_weights')
        if prop_weights is not None:
            for key in prop_weights.keys():
                value = _decode_field(prop_weights[key][()])
                if value is None:
                    continue
                info[f'{key}_weight'] = float(value)

        head = group.attrs.get('head') or group.get('head')
        if head is not None:
            decoded = _decode_field(head[()] if hasattr(head, '__getitem__') else head)
            if decoded is not None:
                info['head'] = str(decoded)

        energy_value = info.get('energy', 0.0)
        forces_value = atoms.arrays.get('forces')
        if forces_value is None:
            forces_value = np.zeros_like(atoms.positions)
        stress_value = info.get('stress')
        if stress_value is None:
            stress_value = np.zeros((3, 3))
        atoms.calc = _CachedCalc(
            float(np.asarray(energy_value).reshape(())),
            np.asarray(forces_value),
            np.asarray(stress_value),
        )

        return atoms

    def __getitem__(self, index: int) -> Atoms:
        return self._read_entry(index)

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        del exc_type, exc, tb
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


__all__ = ['HDF5Dataset']
