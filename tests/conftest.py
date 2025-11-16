from pathlib import Path

import ase.io
import numpy as np
import pytest
import torch
from jax import config as jax_config

from mace_jax.data.utils import config_from_atoms

# Register safe globals for torch before imports in test files
torch.serialization.add_safe_globals([slice])

# Set default dtype for JAX and Torch
jax_config.update('jax_enable_x64', True)
torch.set_default_dtype(torch.float64)


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
