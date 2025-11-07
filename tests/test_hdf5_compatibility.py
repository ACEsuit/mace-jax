"""Ensure mace-jax loads HDF5 datasets produced by the Torch MACE tooling."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import h5py

from mace.data.utils import Configuration as TorchConfiguration
from mace.data.utils import save_configurations_as_HDF5
from mace.tools.utils import AtomicNumberTable

from mace_jax import data as jax_data


def _make_dummy_configuration() -> TorchConfiguration:
    atomic_numbers = np.array([1, 8], dtype=int)
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.9, 0.0, 0.0],
        ],
        dtype=float,
    )
    properties = {
        'energy': np.array([[-10.0]]),
        'forces': np.vstack((positions * 0.1, positions * -0.1)),
    }
    property_weights = {'energy': 1.0, 'forces': 1.0}
    cell = np.eye(3) * 5.0
    pbc = (False, False, False)
    return TorchConfiguration(
        atomic_numbers=atomic_numbers,
        positions=positions,
        properties=properties,
        property_weights=property_weights,
        cell=cell,
        pbc=pbc,
        weight=1.0,
        config_type='Default',
    )


def test_loads_torch_hdf5_dataset(tmp_path: Path):
    config = _make_dummy_configuration()
    hdf5_path = tmp_path / 'torch_dataset.h5'
    with h5py.File(hdf5_path, 'w') as handle:
        save_configurations_as_HDF5([config], None, handle)

    atomic_energies, configs = jax_data.load_from_hdf5(hdf5_path)

    assert atomic_energies == {}
    assert len(configs) == 1
    loaded = configs[0]
    np.testing.assert_array_equal(loaded.atomic_numbers, config.atomic_numbers)
    np.testing.assert_allclose(loaded.positions, config.positions)
    assert loaded.energy == pytest.approx(config.properties['energy'].reshape(-1)[0])
    np.testing.assert_allclose(loaded.forces, config.properties['forces'])
    np.testing.assert_allclose(loaded.cell, config.cell)
    assert loaded.pbc == config.pbc
