# This file loads an xyz dataset and prepares HDF5 files for streaming training.

from __future__ import annotations

import argparse
import ast
import json
import logging
import multiprocessing as mp
import random
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np

from mace_jax import data, tools
from mace_jax.data.utils import save_configurations_as_HDF5


def _parse_config_type_weights(value: str) -> dict:
    try:
        config_type_weights = ast.literal_eval(value)
        assert isinstance(config_type_weights, dict)
    except Exception as exc:  # pylint: disable=broad-except
        logging.warning(
            'Config type weights not specified correctly (%s), using Default',
            exc,
        )
        config_type_weights = {'Default': 1.0}
    return config_type_weights


def _normalize_paths(value) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith('[') or stripped.startswith('('):
            parsed = ast.literal_eval(stripped)
            if isinstance(parsed, (list, tuple)):
                return [str(item) for item in parsed]
        return [value]
    return [str(value)]


def _parse_atomic_numbers(value: str | None) -> list[int] | None:
    if value is None:
        return None
    parsed = ast.literal_eval(value)
    if not isinstance(parsed, (list, tuple)):
        raise ValueError('atomic_numbers must evaluate to a list of integers.')
    return [int(z) for z in parsed]


def _parse_e0s(value: str) -> dict[int, float] | str:
    lowered = value.strip().lower()
    if lowered == 'average':
        return 'average'
    if value.endswith('.json') and Path(value).exists():
        with open(value, 'r', encoding='utf-8') as handle:
            data_dict = json.load(handle)
        if not isinstance(data_dict, dict):
            raise ValueError('E0s JSON must be a dict of atomic energies.')
        return {int(k): float(v) for k, v in data_dict.items()}
    literal = ast.literal_eval(value)
    if not isinstance(literal, dict):
        raise ValueError('E0s literal must be a dict of atomic energies.')
    return {int(k): float(v) for k, v in literal.items()}


def _compute_average_e0s(
    configs: data.Configurations, z_table: data.AtomicNumberTable
) -> dict[int, float]:
    len_train = len(configs)
    len_zs = len(z_table)
    A = np.zeros((len_train, len_zs))
    B = np.zeros(len_train)
    for i, config in enumerate(configs):
        B[i] = float(np.asarray(config.energy).reshape(()))
        for j, z in enumerate(z_table.zs):
            A[i, j] = np.count_nonzero(config.atomic_numbers == z)
    try:
        e0s = np.linalg.lstsq(A, B, rcond=None)[0]
        return {z: float(e0s[i]) for i, z in enumerate(z_table.zs)}
    except np.linalg.LinAlgError:
        logging.warning(
            'Failed to compute E0s using least squares regression, using zeros.'
        )
        return {z: 0.0 for z in z_table.zs}


def _split_indices(count: int, num_splits: int) -> list[list[int]]:
    if num_splits <= 1:
        return [list(range(count))]
    indices = np.array_split(np.arange(count), num_splits)
    return [list(chunk) for chunk in indices]


def _write_hdf5_split(
    configs: data.Configurations, path: Path, drop_last: bool
) -> None:
    with h5py.File(path, 'w') as handle:
        handle.attrs['drop_last'] = drop_last
        save_configurations_as_HDF5(configs, 0, handle)


def _write_hdf5_splits(
    configs: data.Configurations,
    output_dir: Path,
    prefix: str,
    num_process: int,
) -> None:
    if not configs:
        logging.warning('No configurations provided for %s.', output_dir.name)
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    num_process = max(1, min(num_process, len(configs)))
    drop_last = len(configs) % 2 == 1
    indices = _split_indices(len(configs), num_process)
    processes: list[mp.Process] = []
    for idx, chunk in enumerate(indices):
        if not chunk:
            continue
        subset = [configs[i] for i in chunk]
        path = output_dir / f'{prefix}_{idx}.h5'
        proc = mp.Process(target=_write_hdf5_split, args=(subset, path, drop_last))
        proc.start()
        processes.append(proc)
    for proc in processes:
        proc.join()


def _random_train_valid_split(
    configs: data.Configurations,
    valid_fraction: float,
    seed: int,
    work_dir: Path,
) -> tuple[data.Configurations, data.Configurations]:
    if not 0.0 < valid_fraction < 1.0:
        raise ValueError('valid_fraction must be in (0, 1).')
    size = len(configs)
    if size < 2:
        raise ValueError('Need at least 2 configurations for a validation split.')
    train_size = min(size - int(valid_fraction * size), size - 1)
    indices = list(range(size))
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    valid_indices = indices[train_size:]
    if len(valid_indices) >= 10:
        work_dir.mkdir(parents=True, exist_ok=True)
        path = work_dir / f'valid_indices_{seed}.txt'
        with open(path, 'w', encoding='utf-8') as handle:
            for idx in valid_indices:
                handle.write(f'{idx}\n')
        logging.info(
            'Using random %.0f%% of training set for validation with indices saved in: %s',
            100 * valid_fraction,
            path,
        )
    return (
        [configs[i] for i in indices[:train_size]],
        [configs[i] for i in valid_indices],
    )


def _compute_statistics(
    configs: data.Configurations,
    z_table: data.AtomicNumberTable,
    r_max: float,
    atomic_energies: np.ndarray,
) -> tuple[float, float, float]:
    neighbor_sum = 0.0
    neighbor_count = 0
    energy_per_atom = []
    forces = []
    for config in configs:
        if config.positions is None or len(config.positions) == 0:
            continue
        edge_index, _, _, _ = data.get_neighborhood(
            positions=np.asarray(config.positions),
            cutoff=r_max,
            pbc=config.pbc,
            cell=None if config.cell is None else np.array(config.cell, copy=True),
        )
        receivers = edge_index[1]
        if receivers.size:
            _, counts = np.unique(receivers, return_counts=True)
            neighbor_sum += counts.sum()
            neighbor_count += counts.size
        if config.energy is not None:
            indices = data.atomic_numbers_to_indices(
                config.atomic_numbers, z_table
            ).astype(np.int64)
            e0_sum = float(atomic_energies[indices].sum())
            energy_value = float(np.asarray(config.energy).reshape(()))
            energy_per_atom.append(
                (energy_value - e0_sum) / float(len(config.atomic_numbers))
            )
        if config.forces is not None:
            forces.append(np.asarray(config.forces))
    avg_num_neighbors = (
        neighbor_sum / neighbor_count if neighbor_count > 0 else 0.0
    )
    mean = float(np.mean(energy_per_atom)) if energy_per_atom else 0.0
    if forces:
        forces_array = np.concatenate(forces, axis=0)
        rms = float(np.sqrt(np.mean(np.square(forces_array))))
    else:
        rms = 0.0
    return avg_num_neighbors, mean, rms


def _collect_configs(
    paths: Iterable[str],
    *,
    config_type_weights: dict,
    energy_key: str,
    forces_key: str,
    stress_key: str,
    virials_key: str,
    dipole_key: str,
    polarizability_key: str,
    extract_atomic_energies: bool,
) -> tuple[dict[int, float], data.Configurations]:
    atomic_energies_values: dict[int, list[float]] = {}
    configs: data.Configurations = []
    for path in paths:
        ae_dict, new_configs = data.load_from_xyz(
            path,
            config_type_weights=config_type_weights,
            energy_key=energy_key,
            forces_key=forces_key,
            stress_key=stress_key,
            virials_key=virials_key,
            dipole_key=dipole_key,
            polarizability_key=polarizability_key,
            extract_atomic_energies=extract_atomic_energies,
        )
        configs.extend(new_configs)
        for z, value in ae_dict.items():
            atomic_energies_values.setdefault(int(z), []).append(float(value))
    atomic_energies_dict = {
        z: float(sum(values) / len(values))
        for z, values in atomic_energies_values.items()
    }
    return atomic_energies_dict, configs


def _resolve_atomic_energies(
    *,
    atomic_energies_dict: dict[int, float],
    e0s_value: str | None,
    train_configs: data.Configurations,
    z_table: data.AtomicNumberTable,
) -> dict[int, float]:
    if atomic_energies_dict:
        return atomic_energies_dict
    if e0s_value is None:
        raise RuntimeError(
            'E0s not found in training file and not specified on the command line.'
        )
    parsed = _parse_e0s(e0s_value)
    if parsed == 'average':
        return _compute_average_e0s(train_configs, z_table)
    return parsed


def main() -> None:
    args = tools.build_preprocess_arg_parser().parse_args()
    run(args)


def run(args: argparse.Namespace) -> None:
    tools.set_seeds(args.seed)
    random.seed(args.seed)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler()],
    )

    config_type_weights = _parse_config_type_weights(args.config_type_weights)

    train_paths = _normalize_paths(args.train_file)
    valid_paths = _normalize_paths(args.valid_file)
    test_paths = _normalize_paths(args.test_file)
    if not train_paths:
        raise ValueError('No training files were provided.')

    atomic_energies_dict, train_configs = _collect_configs(
        train_paths,
        config_type_weights=config_type_weights,
        energy_key=args.energy_key,
        forces_key=args.forces_key,
        stress_key=args.stress_key,
        virials_key=args.virials_key,
        dipole_key=args.dipole_key,
        polarizability_key=args.polarizability_key,
        extract_atomic_energies=True,
    )

    valid_configs: data.Configurations = []
    if valid_paths:
        _, valid_configs = _collect_configs(
            valid_paths,
            config_type_weights=config_type_weights,
            energy_key=args.energy_key,
            forces_key=args.forces_key,
            stress_key=args.stress_key,
            virials_key=args.virials_key,
            dipole_key=args.dipole_key,
            polarizability_key=args.polarizability_key,
            extract_atomic_energies=False,
        )
    else:
        train_configs, valid_configs = _random_train_valid_split(
            train_configs,
            args.valid_fraction,
            args.seed,
            Path(args.work_dir),
        )

    test_configs_by_type: list[tuple[str | None, data.Configurations]] = []
    if test_paths:
        _, test_configs = _collect_configs(
            test_paths,
            config_type_weights=config_type_weights,
            energy_key=args.energy_key,
            forces_key=args.forces_key,
            stress_key=args.stress_key,
            virials_key=args.virials_key,
            dipole_key=args.dipole_key,
            polarizability_key=args.polarizability_key,
            extract_atomic_energies=False,
        )
        test_configs_by_type = data.test_config_types(test_configs)

    if args.shuffle:
        random.shuffle(train_configs)
        random.shuffle(valid_configs)

    if args.atomic_numbers is None:
        z_table = data.get_atomic_number_table_from_zs(
            z
            for configs in (train_configs, valid_configs)
            for config in configs
            for z in config.atomic_numbers
        )
    else:
        logging.info('Using atomic numbers from command line argument')
        z_table = data.AtomicNumberTable(_parse_atomic_numbers(args.atomic_numbers))

    prefix = Path(args.h5_prefix) if args.h5_prefix else Path('.')
    _write_hdf5_splits(
        train_configs, prefix / 'train', 'train', args.num_process
    )
    _write_hdf5_splits(valid_configs, prefix / 'val', 'val', args.num_process)
    if test_configs_by_type:
        for name, subset in test_configs_by_type:
            label = name if name is not None else 'test'
            _write_hdf5_splits(
                subset, prefix / 'test', label, args.num_process
            )

    if args.compute_statistics:
        atomic_energies_dict = _resolve_atomic_energies(
            atomic_energies_dict=atomic_energies_dict,
            e0s_value=args.E0s,
            train_configs=train_configs,
            z_table=z_table,
        )
        removed_atomic_energies = {}
        for z in list(atomic_energies_dict):
            if z not in z_table.zs:
                removed_atomic_energies[z] = atomic_energies_dict.pop(z)
        if removed_atomic_energies:
            logging.warning(
                'Atomic energies for elements not present in the atomic number table have been removed.'
            )
            logging.warning(
                'Removed atomic energies (eV): %s', removed_atomic_energies
            )
            logging.warning(
                'To include these elements, specify all atomic numbers explicitly using --atomic_numbers.'
            )
        missing = [z for z in z_table.zs if z not in atomic_energies_dict]
        if missing:
            raise ValueError(
                f'atomic_energies missing entries for atomic numbers: {missing}'
            )

        atomic_energies = np.array(
            [atomic_energies_dict[z] for z in z_table.zs], dtype=np.float64
        )
        avg_num_neighbors, mean, std = _compute_statistics(
            train_configs, z_table, args.r_max, atomic_energies
        )
        logging.info('Average number of neighbors: %s', avg_num_neighbors)
        logging.info('Mean: %s', mean)
        logging.info('Standard deviation: %s', std)

        statistics = {
            'atomic_energies': str(atomic_energies_dict),
            'avg_num_neighbors': avg_num_neighbors,
            'mean': mean,
            'std': std,
            'atomic_numbers': str([int(z) for z in z_table.zs]),
            'r_max': args.r_max,
        }
        stats_path = prefix / 'statistics.json'
        with open(stats_path, 'w', encoding='utf-8') as handle:
            json.dump(statistics, handle)


if __name__ == '__main__':
    main()
