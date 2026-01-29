from pathlib import Path

import ase.io
import h5py
import numpy as np
import pytest
import torch
from jax import config as jax_config
from contextlib import contextmanager

from mace_jax.data.utils import config_from_atoms

# Register safe globals for torch before imports in test files
torch.serialization.add_safe_globals([slice])

# Set default dtype for JAX and Torch
jax_config.update('jax_enable_x64', True)
torch.set_default_dtype(torch.float64)


@contextmanager
def preserve_jax_x64():
    """Temporarily preserve the global jax_enable_x64 setting."""
    prev = jax_config.jax_enable_x64
    try:
        yield
    finally:
        jax_config.update('jax_enable_x64', prev)


@pytest.fixture(autouse=True)
def _preserve_global_precisions_per_test():
    """Ensure tests cannot leak global precision settings across the suite."""
    prev_jax_x64 = jax_config.jax_enable_x64
    prev_torch_dtype = torch.get_default_dtype()
    try:
        yield
    finally:
        jax_config.update('jax_enable_x64', prev_jax_x64)
        torch.set_default_dtype(prev_torch_dtype)


@pytest.fixture(scope='session')
def simple_xyz_configs():
    """Load the tiny XYZ sample as Configuration objects for reuse in tests."""
    data_path = Path(__file__).resolve().parent / 'test_data' / 'simple.xyz'
    atoms_list = ase.io.read(data_path, index=':')
    return [config_from_atoms(atoms) for atoms in atoms_list]


@pytest.fixture(scope='session')
def simple_xyz_features(simple_xyz_configs):
    """Return scalar+vector features (Z, positions) from the first config."""
    conf = simple_xyz_configs[0]
    atomic_numbers = conf.atomic_numbers.astype(np.float32).reshape(-1, 1)
    positions = conf.positions.astype(np.float32)
    return np.concatenate([atomic_numbers, positions], axis=1)


def _write_hdf5_from_configs(path: Path, configs) -> None:
    with h5py.File(path, 'w') as handle:
        batch = handle.create_group('config_batch_0')
        for idx, config in enumerate(configs):
            subgroup = batch.create_group(f'config_{idx}')
            numbers = np.asarray(config.atomic_numbers, dtype=np.int32)
            positions = np.asarray(config.positions, dtype=np.float64)
            cell = np.asarray(config.cell if config.cell is not None else np.eye(3))
            pbc = np.asarray(
                config.pbc if config.pbc is not None else (False, False, False),
                dtype=np.bool_,
            )
            subgroup.create_dataset('atomic_numbers', data=numbers)
            subgroup.create_dataset('positions', data=positions)
            subgroup.create_dataset('cell', data=cell)
            subgroup.create_dataset('pbc', data=pbc)
            subgroup.create_dataset('weight', data=np.array(config.weight))
            subgroup.create_dataset(
                'config_type',
                data=np.array(
                    config.config_type or 'Default',
                    dtype='S',
                ),
            )
            subgroup.create_dataset(
                'head', data=np.array(config.head or 'Default', dtype='S')
            )
            properties = subgroup.create_group('properties')
            energy = 0.0 if config.energy is None else float(config.energy)
            forces = (
                np.zeros((positions.shape[0], 3))
                if config.forces is None
                else np.asarray(config.forces)
            )
            stress = (
                np.zeros((3, 3)) if config.stress is None else np.asarray(config.stress)
            )
            properties.create_dataset('energy', data=np.array(energy, dtype=np.float64))
            properties.create_dataset('forces', data=forces)
            properties.create_dataset('stress', data=stress)
            prop_weights = subgroup.create_group('property_weights')
            for key in ('energy', 'forces', 'stress'):
                prop_weights.create_dataset(key, data=np.array(1.0, dtype=np.float64))


@pytest.fixture(scope='session')
def simple_hdf5_path(tmp_path_factory, simple_xyz_configs):
    """Convert the sample XYZ configs into an HDF5 dataset for streaming tests."""
    data_dir = tmp_path_factory.mktemp('hdf5_data')
    path = data_dir / 'simple.h5'
    _write_hdf5_from_configs(path, simple_xyz_configs)
    return path
