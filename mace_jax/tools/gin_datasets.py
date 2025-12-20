import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import gin
import numpy as np
from tqdm import tqdm

from mace_jax import data
from mace_jax.data.streaming_loader import StreamingDatasetSpec


def _ensure_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    return [str(value)]


def _load_configs_from_path(
    path: str,
    *,
    config_type_weights: dict,
    energy_key: str,
    forces_key: str,
    prefactor_stress: float,
    remap_stress: np.ndarray | None,
    head_name: str,
    num_configs: int | None,
    extract_atomic_energies: bool,
    allow_unlabeled: bool,
):
    loader_kwargs = dict(
        config_type_weights=config_type_weights,
        energy_key=energy_key,
        forces_key=forces_key,
        prefactor_stress=prefactor_stress,
        remap_stress=remap_stress,
        head_name=head_name,
        num_configs=num_configs,
    )
    lower_path = path.lower()
    if lower_path.endswith(('.h5', '.hdf5')):
        return _load_native_hdf5_configs(
            path,
            config_type_weights=config_type_weights,
            energy_key=energy_key,
            forces_key=forces_key,
            prefactor_stress=prefactor_stress,
            remap_stress=remap_stress,
            head_name=head_name,
            no_data_ok=allow_unlabeled,
            num_configs=num_configs,
        )
    return data.load_from_xyz(
        file_or_path=path,
        extract_atomic_energies=extract_atomic_energies,
        no_data_ok=allow_unlabeled,
        **loader_kwargs,
    )


def _load_native_hdf5_configs(
    path: str,
    *,
    config_type_weights: dict,
    energy_key: str,
    forces_key: str,
    prefactor_stress: float,
    remap_stress: np.ndarray | None,
    head_name: str,
    no_data_ok: bool,
    num_configs: int | None = None,
):
    configs: list[data.Configuration] = []
    dataset = data.HDF5Dataset(path)
    try:
        for idx in range(len(dataset)):
            atoms = dataset[idx]
            cfg = data.config_from_atoms(
                atoms,
                energy_key=energy_key,
                forces_key=forces_key,
                stress_key='stress',
                config_type_weights=config_type_weights,
                prefactor_stress=prefactor_stress,
                remap_stress=remap_stress,
                head_name=head_name,
            )
            if (
                not no_data_ok
                and cfg.energy is None
                and cfg.forces is None
                and cfg.stress is None
            ):
                continue
            configs.append(cfg)
            if num_configs is not None and len(configs) >= num_configs:
                break
    finally:
        dataset.close()
    return {}, configs


@dataclass
class _StreamingStats:
    sample_graphs: list
    ata: np.ndarray
    atb: np.ndarray
    neighbor_sum: float
    neighbor_count: int
    min_distance_sum: float
    min_distance_count: int
    total_graphs: int


def _expand_hdf5_inputs(paths: Sequence[str]) -> list[Path]:
    expanded: list[Path] = []
    for raw in paths:
        candidate = Path(raw)
        if candidate.is_dir():
            files = sorted(
                p for p in candidate.iterdir() if p.suffix.lower() in ('.h5', '.hdf5')
            )
            if not files:
                raise ValueError(
                    f"No HDF5 files found in directory '{candidate.resolve()}'."
                )
            expanded.extend(files)
            continue
        if candidate.suffix.lower() not in ('.h5', '.hdf5'):
            raise ValueError(
                f"Path '{candidate.resolve()}' is not an HDF5 file or directory."
            )
        expanded.append(candidate)
    if not expanded:
        raise ValueError('No HDF5 files were provided for streaming.')
    return expanded


def _ensure_streaming_head_options(head_name: str, head_cfg: dict) -> None:
    unsupported = [
        'train_num',
        'valid_fraction',
        'valid_num',
        'replay_paths',
        'pseudolabel_checkpoint',
        'pseudolabel_targets',
        'pseudolabel_head',
        'pseudolabel_param_dtype',
    ]
    for key in unsupported:
        if head_cfg.get(key) not in (None, [], {}):
            raise ValueError(
                f"Head '{head_name}' option '{key}' is not supported with streaming training."
            )


def _collect_streaming_specs(
    *,
    head_names: Sequence[str],
    head_configs: dict[str, dict] | None,
    train_path: str | None,
    config_type_weights: dict | None,
    energy_key: str,
    forces_key: str,
    prefactor_stress: float,
    remap_stress: np.ndarray | None,
) -> list[StreamingDatasetSpec]:
    specs: list[StreamingDatasetSpec] = []
    if head_configs:
        for head_name in head_names:
            head_cfg = head_configs.get(head_name, {})
            _ensure_streaming_head_options(head_name, head_cfg)
            head_paths = _ensure_list(head_cfg.get('train_path', train_path))
            if not head_paths:
                raise ValueError(
                    f"Head '{head_name}' does not define a train_path and no global train_path was provided."
                )
            head_ct_weights = head_cfg.get('config_type_weights', config_type_weights)
            head_energy_key = head_cfg.get('energy_key', energy_key)
            head_forces_key = head_cfg.get('forces_key', forces_key)
            head_prefactor_stress = head_cfg.get('prefactor_stress', prefactor_stress)
            head_remap_stress = head_cfg.get('remap_stress', remap_stress)
            head_weight = float(head_cfg.get('weight', 1.0))
            for expanded in _expand_hdf5_inputs(head_paths):
                specs.append(
                    StreamingDatasetSpec(
                        path=expanded,
                        head_name=head_name,
                        config_type_weights=head_ct_weights,
                        energy_key=head_energy_key,
                        forces_key=head_forces_key,
                        prefactor_stress=head_prefactor_stress,
                        remap_stress=head_remap_stress,
                        weight=head_weight,
                    )
                )
    else:
        base_paths = _ensure_list(train_path)
        if not base_paths:
            raise ValueError('train_path must be provided for streaming training.')
        for expanded in _expand_hdf5_inputs(base_paths):
            specs.append(
                StreamingDatasetSpec(
                    path=expanded,
                    head_name=head_names[0],
                    config_type_weights=config_type_weights,
                    energy_key=energy_key,
                    forces_key=forces_key,
                    prefactor_stress=prefactor_stress,
                    remap_stress=remap_stress,
                )
            )
    return specs


def _unique_atomic_numbers_from_hdf5(paths: Sequence[Path]) -> list[int]:
    numbers: set[int] = set()
    for path in paths:
        with data.HDF5Dataset(path, mode='r') as dataset:
            for idx in range(len(dataset)):
                atoms = dataset[idx]
                numbers.update(int(z) for z in atoms.get_atomic_numbers())
    if not numbers:
        raise ValueError('No atomic species found in the provided HDF5 datasets.')
    return sorted(numbers)


def _compute_streaming_stats(
    dataset_path: Path,
    *,
    spec: StreamingDatasetSpec,
    z_table: data.AtomicNumberTable,
    r_max: float,
    head_to_index: dict[str, int],
    sample_limit: int,
) -> _StreamingStats:
    dataset = data.HDF5Dataset(dataset_path)
    sample_graphs: list = []
    num_species = len(z_table)
    ata = np.zeros((num_species, num_species), dtype=np.float64)
    atb = np.zeros((num_species,), dtype=np.float64)
    neighbor_sum = 0.0
    neighbor_count = 0
    min_distance_sum = 0.0
    min_distance_count = 0
    total_graphs = 0
    try:
        for idx in range(len(dataset)):
            atoms = dataset[idx]
            conf = data.config_from_atoms(
                atoms,
                energy_key=spec.energy_key,
                forces_key=spec.forces_key,
                stress_key=spec.stress_key,
                config_type_weights=spec.config_type_weights,
                prefactor_stress=spec.prefactor_stress,
                remap_stress=spec.remap_stress,
                head_name=spec.head_name,
            )
            if spec.weight != 1.0:
                conf.weight *= float(spec.weight)
            graph = data.graph_from_configuration(
                conf,
                cutoff=r_max,
                z_table=z_table,
                head_to_index=head_to_index,
            )
            species = np.asarray(graph.nodes.species)
            counts = np.bincount(species, minlength=num_species).astype(np.float64)
            ata += np.outer(counts, counts)
            energy = getattr(graph.globals, 'energy', None)
            if energy is not None:
                energy_value = float(np.asarray(energy).reshape(-1)[0])
                atb += counts * energy_value
            receivers = np.asarray(graph.receivers)
            if receivers.size:
                _, per_counts = np.unique(receivers, return_counts=True)
                neighbor_sum += per_counts.sum()
                neighbor_count += per_counts.size
            senders = np.asarray(graph.senders)
            if senders.size and receivers.size:
                positions = np.asarray(graph.nodes.positions)
                shifts = np.asarray(graph.edges.shifts)
                recv_pos = positions[receivers]
                vectors = positions[senders] + shifts - recv_pos
                lengths = np.linalg.norm(vectors, axis=-1)
                if lengths.size:
                    min_distance_sum += float(lengths.min())
                    min_distance_count += 1
            if len(sample_graphs) < sample_limit:
                sample_graphs.append(graph)
            total_graphs += 1
    finally:
        dataset.close()

    if total_graphs == 0:
        raise ValueError(f"No graphs found in '{dataset_path}'.")

    return _StreamingStats(
        sample_graphs=sample_graphs,
        ata=ata,
        atb=atb,
        neighbor_sum=neighbor_sum,
        neighbor_count=neighbor_count,
        min_distance_sum=min_distance_sum,
        min_distance_count=min_distance_count,
        total_graphs=total_graphs,
    )


def _build_streaming_train_loader(
    *,
    dataset_specs: Sequence[StreamingDatasetSpec],
    r_max: float,
    n_node: int,
    n_edge: int,
    n_graph: int,
    seed: int,
    head_to_index: dict[str, int],
    stream_prefetch: int | None,
    stream_workers: int,
) -> tuple[data.StreamingGraphDataLoader, dict[int, float], float]:
    if not dataset_specs:
        raise ValueError('No streaming datasets were provided.')
    atomic_numbers = _unique_atomic_numbers_from_hdf5(
        [spec.path for spec in dataset_specs]
    )
    z_table = data.AtomicNumberTable(atomic_numbers)
    num_species = len(z_table)

    ata = np.zeros((num_species, num_species), dtype=np.float64)
    atb = np.zeros((num_species,), dtype=np.float64)
    neighbor_sum = 0.0
    neighbor_count = 0
    min_distance_sum = 0.0
    min_distance_count = 0
    total_graphs = 0
    sample_graphs: list = []
    sample_cap = 32

    for spec in dataset_specs:
        remaining = max(sample_cap - len(sample_graphs), 0)
        stats = _compute_streaming_stats(
            spec.path,
            spec=spec,
            z_table=z_table,
            r_max=r_max,
            head_to_index=head_to_index,
            sample_limit=remaining if remaining else 0,
        )
        if remaining:
            sample_graphs.extend(stats.sample_graphs[:remaining])
        ata += stats.ata
        atb += stats.atb
        neighbor_sum += stats.neighbor_sum
        neighbor_count += stats.neighbor_count
        min_distance_sum += stats.min_distance_sum
        min_distance_count += stats.min_distance_count
        total_graphs += stats.total_graphs

    if not sample_graphs:
        raise ValueError('Unable to build initialization graphs from training data.')

    avg_neighbors = neighbor_sum / neighbor_count if neighbor_count else 0.0
    avg_min_distance = (
        min_distance_sum / min_distance_count if min_distance_count else 0.0
    )
    regularizer = 1e-8 * np.eye(num_species)
    atomic_energies = np.linalg.solve(ata + regularizer, atb)
    atomic_energies_dict = {
        z_table.index_to_z(i): float(atomic_energies[i]) for i in range(num_species)
    }

    datasets = [data.HDF5Dataset(spec.path, mode='r') for spec in dataset_specs]
    loader = data.StreamingGraphDataLoader(
        datasets=datasets,
        dataset_specs=dataset_specs,
        z_table=z_table,
        r_max=r_max,
        n_node=n_node,
        n_edge=n_edge,
        head_to_index=head_to_index,
        shuffle=True,
        seed=seed,
        niggli_reduce=False,
        max_batches=None,
        prefetch_batches=None if stream_prefetch is None else int(stream_prefetch),
        num_workers=int(stream_workers or 0),
        graph_multiple=n_graph,
    )
    loader.graphs = sample_graphs
    loader.atomic_numbers = tuple(atomic_numbers)
    loader.avg_num_neighbors = avg_neighbors
    loader.avg_r_min = avg_min_distance
    loader.atomic_energies = atomic_energies
    loader.total_graphs = total_graphs
    loader.streaming = True
    return loader, atomic_energies_dict, r_max


def _collect_eval_configs(
    *,
    head_names: Sequence[str],
    head_configs: dict[str, dict] | None,
    valid_path: str | None,
    test_path: str | None,
    test_num: int | None,
    config_type_weights: dict | None,
    energy_key: str,
    forces_key: str,
    prefactor_stress: float,
    remap_stress: np.ndarray | None,
) -> tuple[list[data.Configuration], list[data.Configuration]]:
    valid_configs: list[data.Configuration] = []
    test_configs: list[data.Configuration] = []

    if head_configs:
        for head_name in head_names:
            head_cfg = head_configs.get(head_name, {})
            head_ct_weights = head_cfg.get('config_type_weights', config_type_weights)
            head_energy_key = head_cfg.get('energy_key', energy_key)
            head_forces_key = head_cfg.get('forces_key', forces_key)
            head_prefactor_stress = head_cfg.get('prefactor_stress', prefactor_stress)
            head_remap_stress = head_cfg.get('remap_stress', remap_stress)

            head_valid_path = head_cfg.get('valid_path', valid_path)
            if head_valid_path is not None:
                _, head_valid = _load_configs_from_path(
                    head_valid_path,
                    config_type_weights=head_ct_weights,
                    energy_key=head_energy_key,
                    forces_key=head_forces_key,
                    prefactor_stress=head_prefactor_stress,
                    remap_stress=head_remap_stress,
                    head_name=head_name,
                    num_configs=None,
                    extract_atomic_energies=False,
                    allow_unlabeled=False,
                )
                logging.info(
                    "Loaded %s validation configurations for head '%s' from '%s'",
                    len(head_valid),
                    head_name,
                    head_valid_path,
                )
                valid_configs.extend(head_valid)

            head_test_path = head_cfg.get('test_path', test_path)
            head_test_num = head_cfg.get('test_num', test_num)
            if head_test_path is not None:
                _, head_test = _load_configs_from_path(
                    head_test_path,
                    config_type_weights=head_ct_weights,
                    energy_key=head_energy_key,
                    forces_key=head_forces_key,
                    prefactor_stress=head_prefactor_stress,
                    remap_stress=head_remap_stress,
                    head_name=head_name,
                    num_configs=head_test_num,
                    extract_atomic_energies=False,
                    allow_unlabeled=False,
                )
                logging.info(
                    "Loaded %s test configurations for head '%s' from '%s'",
                    len(head_test),
                    head_name,
                    head_test_path,
                )
                test_configs.extend(head_test)
    else:
        if valid_path is not None:
            _, valid_configs = _load_configs_from_path(
                valid_path,
                config_type_weights=config_type_weights,
                energy_key=energy_key,
                forces_key=forces_key,
                prefactor_stress=prefactor_stress,
                remap_stress=remap_stress,
                head_name=head_names[0],
                num_configs=None,
                extract_atomic_energies=False,
                allow_unlabeled=False,
            )
            logging.info(
                "Loaded %s validation configurations from '%s'",
                len(valid_configs),
                valid_path,
            )
        if test_path is not None:
            _, test_configs = _load_configs_from_path(
                test_path,
                config_type_weights=config_type_weights,
                energy_key=energy_key,
                forces_key=forces_key,
                prefactor_stress=prefactor_stress,
                remap_stress=remap_stress,
                head_name=head_names[0],
                num_configs=test_num,
                extract_atomic_energies=False,
                allow_unlabeled=False,
            )
            logging.info(
                "Loaded %s test configurations from '%s'",
                len(test_configs),
                test_path,
            )

    return valid_configs, test_configs


def _graph_loader_from_configs(
    configs: list[data.Configuration],
    *,
    r_max: float,
    head_to_index: dict[str, int],
    n_node: int,
    n_edge: int,
    n_graph: int,
    min_n_node: int,
    min_n_edge: int,
    min_n_graph: int,
    n_mantissa_bits: int,
    shuffle: bool,
    head_names: Sequence[str],
):
    if not configs:
        graphs: list = []
    else:
        graphs = [
            data.graph_from_configuration(
                cfg, cutoff=r_max, head_to_index=head_to_index
            )
            for cfg in tqdm(configs)
        ]
    return data.GraphDataLoader(
        graphs=graphs,
        n_node=n_node,
        n_edge=n_edge,
        n_graph=n_graph,
        min_n_node=min_n_node,
        min_n_edge=min_n_edge,
        min_n_graph=min_n_graph,
        n_mantissa_bits=n_mantissa_bits,
        shuffle=shuffle,
        heads=head_names,
    )


@gin.configurable
def datasets(
    *,
    r_max: float,
    train_path: str,
    config_type_weights: dict = None,
    train_num: int = None,
    valid_path: str = None,
    valid_fraction: float = None,
    valid_num: int = None,
    test_path: str = None,
    test_num: int = None,
    seed: int = 1234,
    energy_key: str = 'energy',
    forces_key: str = 'forces',
    n_node: int = 1,
    n_edge: int = 1,
    n_graph: int = 1,
    min_n_node: int = 1,
    min_n_edge: int = 1,
    min_n_graph: int = 1,
    n_mantissa_bits: int = 1,
    prefactor_stress: float = 1.0,
    remap_stress: np.ndarray = None,
    heads: Sequence[str] = ('Default',),
    head_configs: dict[str, dict] | None = None,
    stream_train_prefetch: int | None = None,
    stream_train_workers: int = 0,
) -> tuple[
    data.StreamingGraphDataLoader,
    data.GraphDataLoader,
    data.GraphDataLoader,
    dict[int, float],
    float,
]:
    """Load datasets, streaming the training split directly from HDF5."""

    head_names = tuple(heads) if heads else ('Default',)
    head_to_index = {name: idx for idx, name in enumerate(head_names)}

    if train_num is not None:
        raise ValueError('train_num is not supported with streaming datasets.')
    if valid_fraction is not None or valid_num is not None:
        raise ValueError(
            'valid_fraction and valid_num are not supported; provide explicit validation files.'
        )

    dataset_specs = _collect_streaming_specs(
        head_names=head_names,
        head_configs=head_configs,
        train_path=train_path,
        config_type_weights=config_type_weights,
        energy_key=energy_key,
        forces_key=forces_key,
        prefactor_stress=prefactor_stress,
        remap_stress=remap_stress,
    )
    valid_configs, test_configs = _collect_eval_configs(
        head_names=head_names,
        head_configs=head_configs,
        valid_path=valid_path,
        test_path=test_path,
        test_num=test_num,
        config_type_weights=config_type_weights,
        energy_key=energy_key,
        forces_key=forces_key,
        prefactor_stress=prefactor_stress,
        remap_stress=remap_stress,
    )

    train_loader, atomic_energies_dict, r_max = _build_streaming_train_loader(
        dataset_specs=dataset_specs,
        r_max=r_max,
        n_node=n_node,
        n_edge=n_edge,
        n_graph=n_graph,
        seed=seed,
        head_to_index=head_to_index,
        stream_prefetch=stream_train_prefetch,
        stream_workers=stream_train_workers,
    )
    effective_n_node = n_node
    effective_n_edge = n_edge
    if effective_n_node is None:
        effective_n_node = getattr(train_loader, '_n_node', None)
    if effective_n_edge is None:
        effective_n_edge = getattr(train_loader, '_n_edge', None)
    if effective_n_node is None or effective_n_edge is None:
        raise ValueError('Unable to determine batch size limits for evaluation loaders.')

    valid_loader = _graph_loader_from_configs(
        valid_configs,
        r_max=r_max,
        head_to_index=head_to_index,
        n_node=effective_n_node,
        n_edge=effective_n_edge,
        n_graph=n_graph,
        min_n_node=min_n_node,
        min_n_edge=min_n_edge,
        min_n_graph=min_n_graph,
        n_mantissa_bits=n_mantissa_bits,
        shuffle=False,
        head_names=head_names,
    )
    test_loader = _graph_loader_from_configs(
        test_configs,
        r_max=r_max,
        head_to_index=head_to_index,
        n_node=effective_n_node,
        n_edge=effective_n_edge,
        n_graph=n_graph,
        min_n_node=min_n_node,
        min_n_edge=min_n_edge,
        min_n_graph=min_n_graph,
        n_mantissa_bits=n_mantissa_bits,
        shuffle=False,
        head_names=head_names,
    )

    logging.info(
        'Streaming training graphs: %s from %s files',
        getattr(train_loader, 'total_graphs', 'unknown'),
        len(dataset_specs),
    )
    logging.info(
        'Validation graphs: %s | Test graphs: %s',
        len(valid_loader.graphs),
        len(test_loader.graphs),
    )
    return train_loader, valid_loader, test_loader, atomic_energies_dict, r_max
