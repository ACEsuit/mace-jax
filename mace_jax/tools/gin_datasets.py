import logging
from typing import Dict, Tuple

import gin
import numpy as np
from tqdm import tqdm

from mace_jax import data


@gin.configurable
def datasets(
    *,
    r_max: float,
    train_path: str,
    config_type_weights: Dict = None,
    num_train: int = None,
    valid_path: str = None,
    valid_fraction: float = None,
    test_path: str = None,
    seed: int = 1234,
    energy_key: str = "energy",
    forces_key: str = "forces",
    n_node: int = 1,
    n_edge: int = 1,
    n_graph: int = 1,
    n_mantissa_bits: int = 1,
    prefactor_stress: float = 1.0,
    remap_stress: np.ndarray = None,
) -> Tuple[data.GraphDataLoader, data.GraphDataLoader, data.GraphDataLoader, Dict[int, float], float]:
    """Load training and test dataset from xyz file"""

    atomic_energies_dict, all_train_configs = data.load_from_xyz(
        file_path=train_path,
        config_type_weights=config_type_weights,
        energy_key=energy_key,
        forces_key=forces_key,
        extract_atomic_energies=True,
        num_configs=num_train,
        prefactor_stress=prefactor_stress,
        remap_stress=remap_stress,
    )
    logging.info(
        f"Loaded {len(all_train_configs)} training configurations from '{train_path}'"
    )

    if valid_path is not None:
        _, valid_configs = data.load_from_xyz(
            file_path=valid_path,
            config_type_weights=config_type_weights,
            energy_key=energy_key,
            forces_key=forces_key,
            extract_atomic_energies=False,
            prefactor_stress=prefactor_stress,
            remap_stress=remap_stress,
        )
        logging.info(
            f"Loaded {len(valid_configs)} validation configurations from '{valid_path}'"
        )
        train_configs = all_train_configs
    elif valid_fraction is not None:
        logging.info(
            "Using random %s%% of training set for validation", 100 * valid_fraction
        )
        train_configs, valid_configs = data.random_train_valid_split(
            all_train_configs, valid_fraction, seed
        )
    else:
        logging.info("No validation set")
        train_configs = all_train_configs
        valid_configs = []
    del all_train_configs

    if test_path is not None:
        _, test_configs = data.load_from_xyz(
            file_path=test_path,
            config_type_weights=config_type_weights,
            energy_key=energy_key,
            forces_key=forces_key,
            extract_atomic_energies=False,
            prefactor_stress=prefactor_stress,
            remap_stress=remap_stress,
        )
        logging.info(
            f"Loaded {len(test_configs)} test configurations from '{test_path}'"
        )
    else:
        test_configs = []

    logging.info(
        f"Total number of configurations: "
        f"train={len(train_configs)}, "
        f"valid={len(valid_configs)}, "
        f"test={len(test_configs)}"
    )

    train_loader = data.GraphDataLoader(
        graphs=[
            data.graph_from_configuration(c, cutoff=r_max) for c in tqdm(train_configs)
        ],
        n_node=n_node,
        n_edge=n_edge,
        n_graph=n_graph,
        n_mantissa_bits=n_mantissa_bits,
        shuffle=True,
    )
    valid_loader = data.GraphDataLoader(
        graphs=[
            data.graph_from_configuration(c, cutoff=r_max) for c in tqdm(valid_configs)
        ],
        n_node=n_node,
        n_edge=n_edge,
        n_graph=n_graph,
        n_mantissa_bits=n_mantissa_bits,
        shuffle=False,
    )
    test_loader = data.GraphDataLoader(
        graphs=[
            data.graph_from_configuration(c, cutoff=r_max) for c in tqdm(test_configs)
        ],
        n_node=n_node,
        n_edge=n_edge,
        n_graph=n_graph,
        n_mantissa_bits=n_mantissa_bits,
        shuffle=False,
    )
    return train_loader, valid_loader, test_loader, atomic_energies_dict, r_max
