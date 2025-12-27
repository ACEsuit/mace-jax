import hashlib
import json
import logging
import pickle
from collections.abc import Sequence
from dataclasses import dataclass
from glob import glob
from pathlib import Path

import gin
import numpy as np
from tqdm.auto import tqdm

from mace_jax import data
from mace_jax.data.streaming_loader import StreamingDatasetSpec

_STREAMING_STATS_CACHE_VERSION = 1


def _ensure_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    return [str(value)]


@dataclass
class _StreamingStats:
    batch_assignments: list[list[int]]
    n_nodes: int
    n_edges: int
    n_graphs: int


@dataclass
class _StreamingMetadata:
    sample_graphs: list
    ata: np.ndarray
    atb: np.ndarray
    neighbor_sum: float
    neighbor_count: int
    min_distance_sum: float
    min_distance_count: int
    total_graphs: int
    total_nodes: int
    total_edges: int


def _graph_has_positive_weight(graph) -> bool:
    weight = getattr(graph.globals, 'weight', None)
    if weight is None:
        return True
    value = np.asarray(weight).reshape(-1)[0]
    return float(value) > 0.0


def _graph_from_dataset_entry(
    dataset,
    idx: int,
    *,
    spec: StreamingDatasetSpec,
    z_table: data.AtomicNumberTable,
    r_max: float,
    head_to_index: dict[str, int],
):
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
    return data.graph_from_configuration(
        conf,
        cutoff=r_max,
        z_table=z_table,
        head_to_index=head_to_index,
    )


def _gather_sample_graphs(
    dataset_path: Path,
    *,
    spec: StreamingDatasetSpec,
    z_table: data.AtomicNumberTable,
    r_max: float,
    head_to_index: dict[str, int],
    sample_limit: int,
) -> list:
    if sample_limit <= 0:
        return []
    graphs: list = []
    dataset = data.HDF5Dataset(dataset_path)
    try:
        for idx in range(len(dataset)):
            graph = _graph_from_dataset_entry(
                dataset,
                idx,
                spec=spec,
                z_table=z_table,
                r_max=r_max,
                head_to_index=head_to_index,
            )
            if not _graph_has_positive_weight(graph):
                continue
            graphs.append(graph)
            if len(graphs) >= sample_limit:
                break
    finally:
        dataset.close()
    return graphs


def _extend_sample_graphs(
    sample_graphs: list,
    *,
    sample_limit: int,
    metadata: _StreamingMetadata | None,
    dataset_path: Path,
    spec: StreamingDatasetSpec,
    z_table: data.AtomicNumberTable,
    r_max: float,
    head_to_index: dict[str, int],
) -> None:
    remaining = max(sample_limit - len(sample_graphs), 0)
    if not remaining:
        return
    if metadata and metadata.sample_graphs:
        sample_graphs.extend(metadata.sample_graphs[:remaining])
        remaining = max(sample_limit - len(sample_graphs), 0)
    if not remaining:
        return
    extra_graphs = _gather_sample_graphs(
        dataset_path,
        spec=spec,
        z_table=z_table,
        r_max=r_max,
        head_to_index=head_to_index,
        sample_limit=remaining,
    )
    if extra_graphs:
        sample_graphs.extend(extra_graphs[:remaining])


def _stats_cache_path(dataset_path: Path) -> Path:
    suffix = dataset_path.suffix + '.streamstats.pkl'
    return dataset_path.with_suffix(suffix)


def _dataset_signature(dataset_path: Path) -> dict[str, float]:
    stat = dataset_path.stat()
    return {'size': stat.st_size, 'mtime': stat.st_mtime}


def _normalized_config_type_weights(weights: dict | None) -> dict | None:
    if not weights:
        return None
    return {str(key): float(value) for key, value in sorted(weights.items())}


def _spec_fingerprint(
    spec: StreamingDatasetSpec,
    *,
    r_max: float,
    atomic_numbers: Sequence[int],
    head_to_index: dict[str, int] | None = None,
    edge_cap: int | None = None,
) -> str:
    head_index = 0
    if head_to_index is not None and spec.head_name in head_to_index:
        head_index = int(head_to_index[spec.head_name])
    payload = {
        'head_name': spec.head_name,
        'head_index': head_index,
        'energy_key': spec.energy_key,
        'forces_key': spec.forces_key,
        'stress_key': spec.stress_key,
        'prefactor_stress': float(spec.prefactor_stress),
        'weight': float(spec.weight),
        'config_type_weights': _normalized_config_type_weights(
            spec.config_type_weights
        ),
        'remap_stress': spec.remap_stress.tolist()
        if spec.remap_stress is not None
        else None,
        'path': str(Path(spec.path)),
        'r_max': float(r_max),
        'atomic_numbers': [int(z) for z in atomic_numbers],
        'edge_cap': int(edge_cap) if edge_cap is not None else None,
    }
    encoded = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(encoded.encode('utf-8')).hexdigest()


def _stats_to_payload(stats: _StreamingStats) -> dict:
    return {
        'batch_assignments': stats.batch_assignments,
        'n_nodes': stats.n_nodes,
        'n_edges': stats.n_edges,
        'n_graphs': stats.n_graphs,
    }


def _stats_from_payload(payload: dict) -> _StreamingStats:
    return _StreamingStats(
        batch_assignments=payload['batch_assignments'],
        n_nodes=payload['n_nodes'],
        n_edges=payload['n_edges'],
        n_graphs=payload['n_graphs'],
    )


def _load_cached_streaming_stats(
    dataset_path: Path, fingerprint: str
) -> _StreamingStats | None:
    cache_path = _stats_cache_path(dataset_path)
    if not cache_path.exists():
        return None
    try:
        with cache_path.open('rb') as fh:
            payload = pickle.load(fh)
    except Exception as exc:  # pragma: no cover - cache corruption is unexpected
        logging.warning('Failed to load streaming stats cache %s: %s', cache_path, exc)
        return None
    if payload.get('version') != _STREAMING_STATS_CACHE_VERSION:
        return None
    dataset_sig = payload.get('dataset_signature')
    if dataset_sig != _dataset_signature(dataset_path):
        return None
    if payload.get('spec_fingerprint') != fingerprint:
        return None
    stats_payload = payload.get('stats')
    if not stats_payload:
        return None
    logging.info('Loaded cached streaming stats from %s', cache_path)
    return _stats_from_payload(stats_payload)


def _store_cached_streaming_stats(
    dataset_path: Path,
    fingerprint: str,
    stats: _StreamingStats,
) -> None:
    cache_path = _stats_cache_path(dataset_path)
    payload = {
        'version': _STREAMING_STATS_CACHE_VERSION,
        'dataset_signature': _dataset_signature(dataset_path),
        'spec_fingerprint': fingerprint,
        'stats': _stats_to_payload(stats),
    }
    try:
        with cache_path.open('wb') as fh:
            pickle.dump(payload, fh)
    except OSError as exc:  # pragma: no cover - filesystem failure
        logging.warning('Failed to write streaming stats cache %s: %s', cache_path, exc)
        return
    logging.debug('Cached streaming stats to %s', cache_path)


def _expand_hdf5_inputs(paths: Sequence[str]) -> list[Path]:
    expanded: list[Path] = []
    for raw in paths:
        if any(ch in raw for ch in '*?[]'):
            matches = [Path(p) for p in sorted(glob(raw))]
            if not matches:
                raise ValueError(f"No HDF5 files matched pattern '{raw}'.")
            candidates = matches
        else:
            candidates = [Path(raw)]
        for candidate in candidates:
            if candidate.is_dir():
                files = sorted(
                    p
                    for p in candidate.iterdir()
                    if p.suffix.lower() in ('.h5', '.hdf5')
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


def _append_streaming_specs(
    collection: list[StreamingDatasetSpec],
    *,
    head_name: str,
    paths: Sequence[str],
    config_type_weights: dict | None,
    energy_key: str,
    forces_key: str,
    prefactor_stress: float,
    remap_stress: np.ndarray | None,
    weight: float = 1.0,
    log_label: str | None = None,
) -> None:
    for expanded in _expand_hdf5_inputs(paths):
        collection.append(
            StreamingDatasetSpec(
                path=expanded,
                head_name=head_name,
                config_type_weights=config_type_weights,
                energy_key=energy_key,
                forces_key=forces_key,
                prefactor_stress=prefactor_stress,
                remap_stress=remap_stress,
                weight=float(weight),
            )
        )
        if log_label:
            logging.info(
                "Prepared streaming %s dataset for head '%s' from '%s'",
                log_label,
                head_name,
                expanded,
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
            _append_streaming_specs(
                specs,
                head_name=head_name,
                paths=head_paths,
                config_type_weights=head_cfg.get(
                    'config_type_weights', config_type_weights
                ),
                energy_key=head_cfg.get('energy_key', energy_key),
                forces_key=head_cfg.get('forces_key', forces_key),
                prefactor_stress=head_cfg.get('prefactor_stress', prefactor_stress),
                remap_stress=head_cfg.get('remap_stress', remap_stress),
                weight=head_cfg.get('weight', 1.0),
            )
    else:
        base_paths = _ensure_list(train_path)
        if not base_paths:
            raise ValueError('train_path must be provided for streaming training.')
        _append_streaming_specs(
            specs,
            head_name=head_names[0],
            paths=base_paths,
            config_type_weights=config_type_weights,
            energy_key=energy_key,
            forces_key=forces_key,
            prefactor_stress=prefactor_stress,
            remap_stress=remap_stress,
        )
    return specs


def _unique_atomic_numbers_from_hdf5(paths: Sequence[Path]) -> list[int]:
    numbers: set[int] = set()
    for path in paths:
        with data.HDF5Dataset(path, mode='r') as dataset:
            dataset_len = len(dataset)
            iterator = tqdm(
                range(dataset_len),
                desc=f'Extracting atomic species ({path.name})',
                disable=dataset_len < 1024,
                leave=False,
            )
            for idx in iterator:
                atoms = dataset[idx]
                numbers.update(int(z) for z in atoms.get_atomic_numbers())
    if not numbers:
        raise ValueError('No atomic species found in the provided HDF5 datasets.')
    return sorted(numbers)


def _pack_streaming_batches(
    graph_sizes: list[tuple[int, int, int]],
    edge_cap: int,
    node_cap: int | None = None,
) -> list[list[int]]:
    if not graph_sizes:
        return []
    order = sorted(graph_sizes, key=lambda item: item[2], reverse=True)
    batches: list[dict[str, object]] = []
    for graph_idx, nodes, edges in order:
        placed = False
        for batch in batches:
            edge_sum = batch['edge_sum']
            node_sum = batch['node_sum']
            if edge_sum + edges <= edge_cap and (
                node_cap is None or node_sum + nodes <= node_cap
            ):
                batch['edge_sum'] = edge_sum + edges
                batch['node_sum'] = node_sum + nodes
                batch['graphs'].append(graph_idx)
                placed = True
                break
        if not placed:
            batches.append(
                {
                    'edge_sum': edges,
                    'node_sum': nodes,
                    'graphs': [graph_idx],
                }
            )
    return [batch['graphs'] for batch in batches]


def _compute_streaming_stats(
    dataset_path: Path,
    *,
    spec: StreamingDatasetSpec,
    z_table: data.AtomicNumberTable,
    r_max: float,
    head_to_index: dict[str, int],
    sample_limit: int,
    edge_cap: int,
    collect_metadata: bool,
) -> tuple[_StreamingStats, _StreamingMetadata | None]:
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
    total_nodes = 0
    total_edges = 0
    max_graph_edges = 0
    max_graph_nodes = 0
    graph_sizes: list[tuple[int, int, int]] = []
    dataset_len = len(dataset)
    progress = None
    try:
        iterator = range(dataset_len)
        if dataset_len >= 1024:
            progress = tqdm(
                iterator,
                desc=f'Computing streaming stats ({dataset_path.name})',
                leave=False,
            )
            iterator = progress
        for idx in iterator:
            graph = _graph_from_dataset_entry(
                dataset,
                idx,
                spec=spec,
                z_table=z_table,
                r_max=r_max,
                head_to_index=head_to_index,
            )
            if not _graph_has_positive_weight(graph):
                continue
            g_nodes = int(graph.n_node.sum())
            g_edges = int(graph.n_edge.sum())
            if collect_metadata:
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
            total_nodes += g_nodes
            total_edges += g_edges
            max_graph_edges = max(max_graph_edges, g_edges)
            max_graph_nodes = max(max_graph_nodes, g_nodes)
            graph_sizes.append((idx, g_nodes, g_edges))
    finally:
        if progress is not None:
            progress.close()
        dataset.close()

    if total_graphs == 0:
        raise ValueError(f"No graphs found in '{dataset_path}'.")

    if max_graph_edges > edge_cap:
        logging.warning(
            'Requested max edges per batch (%s) is below the largest graph (%s) '
            'in %s. Raising the limit to fit.',
            edge_cap,
            max_graph_edges,
            dataset_path.name,
        )
        edge_cap = max_graph_edges

    batch_assignments = _pack_streaming_batches(graph_sizes, edge_cap)
    nodes_by_index = {idx: nodes for idx, nodes, _ in graph_sizes}
    node_sums = [sum(nodes_by_index[i] for i in batch) for batch in batch_assignments]
    if node_sums:
        node_target = int(np.ceil(np.percentile(node_sums, 80)))
    else:
        node_target = max_graph_nodes
    if node_target < max_graph_nodes:
        logging.warning(
            'Typical node target (%s) is below the largest graph (%s) in %s. '
            'Raising to fit.',
            node_target,
            max_graph_nodes,
            dataset_path.name,
        )
        node_target = max_graph_nodes
    if node_sums:
        max_nodes_per_batch = max(node_sums)
        if node_target < max_nodes_per_batch:
            batch_assignments = _pack_streaming_batches(
                graph_sizes, edge_cap, node_target
            )

    max_nodes_per_batch = 0
    max_graphs_per_batch = 0
    for batch in batch_assignments:
        nodes_sum = sum(nodes_by_index[i] for i in batch)
        max_nodes_per_batch = max(max_nodes_per_batch, nodes_sum)
        graphs_count = len(batch)
        max_graphs_per_batch = max(max_graphs_per_batch, graphs_count)
    n_nodes = max(max_nodes_per_batch + 1, 2)
    n_graphs = max(max_graphs_per_batch + 1, 2)
    stats = _StreamingStats(
        batch_assignments=batch_assignments,
        n_nodes=n_nodes,
        n_edges=edge_cap,
        n_graphs=n_graphs,
    )
    metadata = None
    if collect_metadata:
        metadata = _StreamingMetadata(
            sample_graphs=sample_graphs,
            ata=ata,
            atb=atb,
            neighbor_sum=neighbor_sum,
            neighbor_count=neighbor_count,
            min_distance_sum=min_distance_sum,
            min_distance_count=min_distance_count,
            total_graphs=total_graphs,
            total_nodes=total_nodes,
            total_edges=total_edges,
        )
    return stats, metadata


def _load_or_compute_streaming_stats(
    spec: StreamingDatasetSpec,
    *,
    r_max: float,
    z_table: data.AtomicNumberTable,
    head_to_index: dict[str, int],
    atomic_numbers: Sequence[int],
    sample_limit: int,
    edge_cap: int,
    collect_metadata: bool,
) -> tuple[_StreamingStats, _StreamingMetadata | None]:
    dataset_path = Path(spec.path)
    fingerprint = _spec_fingerprint(
        spec,
        r_max=r_max,
        atomic_numbers=atomic_numbers,
        head_to_index=head_to_index,
        edge_cap=edge_cap,
    )
    cached = _load_cached_streaming_stats(dataset_path, fingerprint)
    if cached is None:
        stats, metadata = _compute_streaming_stats(
            dataset_path,
            spec=spec,
            z_table=z_table,
            r_max=r_max,
            head_to_index=head_to_index,
            sample_limit=sample_limit,
            edge_cap=edge_cap,
            collect_metadata=collect_metadata,
        )
        _store_cached_streaming_stats(dataset_path, fingerprint, stats)
        return stats, metadata
    if not collect_metadata:
        return cached, None
    _, metadata = _compute_streaming_stats(
        dataset_path,
        spec=spec,
        z_table=z_table,
        r_max=r_max,
        head_to_index=head_to_index,
        sample_limit=sample_limit,
        edge_cap=edge_cap,
        collect_metadata=True,
    )
    return cached, metadata


def _build_streaming_train_loader(
    *,
    dataset_specs: Sequence[StreamingDatasetSpec],
    r_max: float,
    n_node: int | None,
    n_edge: int,
    n_graph: int,
    seed: int,
    head_to_index: dict[str, int],
    stream_prefetch: int | None,
    stream_workers: int,
    atomic_numbers_override: Sequence[int] | None = None,
    atomic_energies_override: dict[int, float] | str | None = None,
    statistics_metadata_path: str | None = None,
) -> tuple[data.StreamingGraphDataLoader, dict[int, float], float]:
    if not dataset_specs:
        raise ValueError('No streaming datasets were provided.')
    if atomic_numbers_override:
        atomic_numbers = [int(z) for z in atomic_numbers_override]
    else:
        atomic_numbers = _unique_atomic_numbers_from_hdf5(
            [spec.path for spec in dataset_specs]
        )
    if atomic_numbers_override and statistics_metadata_path:
        logging.info(
            'Using atomic numbers from statistics file %s', statistics_metadata_path
        )
    z_table = data.AtomicNumberTable(atomic_numbers)
    num_species = len(z_table)

    ata = np.zeros((num_species, num_species), dtype=np.float64)
    atb = np.zeros((num_species,), dtype=np.float64)
    neighbor_sum = 0.0
    neighbor_count = 0
    min_distance_sum = 0.0
    min_distance_count = 0
    sample_graphs: list = []
    sample_cap = 32
    batch_assignments: list[list[list[int]]] = []
    n_nodes = 0
    n_edges = 0
    n_graphs = 0

    collect_metadata = not isinstance(atomic_energies_override, dict)

    for spec in dataset_specs:
        stats, metadata = _load_or_compute_streaming_stats(
            spec,
            r_max=r_max,
            z_table=z_table,
            head_to_index=head_to_index,
            atomic_numbers=atomic_numbers,
            sample_limit=sample_cap,
            edge_cap=n_edge,
            collect_metadata=collect_metadata,
        )
        batch_assignments.append(stats.batch_assignments)
        n_nodes = max(n_nodes, int(stats.n_nodes))
        n_edges = max(n_edges, int(stats.n_edges))
        n_graphs = max(n_graphs, int(stats.n_graphs))
        _extend_sample_graphs(
            sample_graphs,
            sample_limit=sample_cap,
            metadata=metadata,
            dataset_path=Path(spec.path),
            spec=spec,
            z_table=z_table,
            r_max=r_max,
            head_to_index=head_to_index,
        )
        if metadata:
            ata += metadata.ata
            atb += metadata.atb
            neighbor_sum += metadata.neighbor_sum
            neighbor_count += metadata.neighbor_count
            min_distance_sum += metadata.min_distance_sum
            min_distance_count += metadata.min_distance_count

    if not sample_graphs:
        raise ValueError('Unable to build initialization graphs from training data.')

    avg_neighbors = neighbor_sum / neighbor_count if neighbor_count else 0.0
    avg_min_distance = (
        min_distance_sum / min_distance_count if min_distance_count else 0.0
    )
    atomic_energies_dict: dict[int, float] = {}
    atomic_energies = np.zeros((num_species,), dtype=np.float64)
    if collect_metadata and np.any(atb):
        regularizer = 1e-8 * np.eye(num_species)
        atomic_energies = np.linalg.solve(ata + regularizer, atb)
        atomic_energies_dict = {
            z_table.index_to_z(i): float(atomic_energies[i]) for i in range(num_species)
        }
    if isinstance(atomic_energies_override, str):
        if atomic_energies_override.lower() == 'average':
            atomic_energies_override = None
        else:
            raise ValueError(
                f'Unsupported atomic_energies_override string: {atomic_energies_override}'
            )
    if isinstance(atomic_energies_override, dict):
        override_dict = {int(k): float(v) for k, v in atomic_energies_override.items()}
        missing = [z for z in atomic_numbers if z not in override_dict]
        if missing:
            raise ValueError(
                f'atomic_energies_override missing entries for atomic numbers: {missing}'
            )
        atomic_energies_dict = {z: override_dict[z] for z in atomic_numbers}
        atomic_energies = np.array(
            [atomic_energies_dict[z_table.index_to_z(i)] for i in range(num_species)],
            dtype=np.float64,
        )
        if statistics_metadata_path:
            logging.info(
                'Using atomic energies from statistics file %s',
                statistics_metadata_path,
            )
    elif not atomic_energies_dict:
        atomic_energies_dict = data.compute_average_E0s(sample_graphs, z_table)
        atomic_energies = np.array(
            [
                atomic_energies_dict.get(z_table.index_to_z(i), 0.0)
                for i in range(num_species)
            ],
            dtype=np.float64,
        )

    if n_edges <= 0:
        n_edges = int(n_edge)
    if n_nodes <= 0:
        n_nodes = 2
    if n_graphs <= 0:
        n_graphs = 2

    datasets = [data.HDF5Dataset(spec.path, mode='r') for spec in dataset_specs]
    loader = data.StreamingGraphDataLoader(
        datasets=datasets,
        dataset_specs=dataset_specs,
        z_table=z_table,
        r_max=r_max,
        n_node=n_nodes,
        n_edge=n_edges,
        head_to_index=head_to_index,
        shuffle=True,
        seed=seed,
        niggli_reduce=False,
        max_batches=None,
        prefetch_batches=None if stream_prefetch is None else int(stream_prefetch),
        num_workers=int(stream_workers or 0),
        batch_assignments=batch_assignments,
        pad_graphs=n_graphs,
    )
    loader.graphs = sample_graphs
    loader.atomic_numbers = tuple(atomic_numbers)
    loader.avg_num_neighbors = avg_neighbors
    loader.avg_r_min = avg_min_distance
    loader.atomic_energies = atomic_energies
    loader.total_graphs = sum(
        len(batch) for batches in batch_assignments for batch in batches
    )
    loader.total_nodes = None
    loader.total_edges = None
    loader.streaming = True
    loader.z_table = z_table
    loader._fixed_pad_nodes = int(n_nodes)
    loader._fixed_pad_edges = int(n_edges)
    loader._fixed_pad_graphs = int(n_graphs)
    return loader, atomic_energies_dict, r_max


def _collect_eval_streaming_specs(
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
) -> tuple[list[StreamingDatasetSpec], list[StreamingDatasetSpec]]:
    if test_num not in (None, 0):
        raise ValueError(
            'test_num is not supported with streaming evaluation datasets.'
        )

    valid_specs: list[StreamingDatasetSpec] = []
    test_specs: list[StreamingDatasetSpec] = []

    if head_configs:
        for head_name in head_names:
            head_cfg = head_configs.get(head_name, {})
            head_ct_weights = head_cfg.get('config_type_weights', config_type_weights)
            head_energy_key = head_cfg.get('energy_key', energy_key)
            head_forces_key = head_cfg.get('forces_key', forces_key)
            head_prefactor_stress = head_cfg.get('prefactor_stress', prefactor_stress)
            head_remap_stress = head_cfg.get('remap_stress', remap_stress)

            head_valid_path = head_cfg.get('valid_path', valid_path)
            if head_valid_path:
                _append_streaming_specs(
                    valid_specs,
                    head_name=head_name,
                    paths=_ensure_list(head_valid_path),
                    config_type_weights=head_ct_weights,
                    energy_key=head_energy_key,
                    forces_key=head_forces_key,
                    prefactor_stress=head_prefactor_stress,
                    remap_stress=head_remap_stress,
                    log_label='validation',
                )

            head_test_path = head_cfg.get('test_path', test_path)
            head_test_num = head_cfg.get('test_num', test_num)
            if head_test_num not in (None, 0):
                raise ValueError(
                    f"Head '{head_name}' specifies test_num which is not supported with streaming evaluation datasets."
                )
            if head_test_path:
                _append_streaming_specs(
                    test_specs,
                    head_name=head_name,
                    paths=_ensure_list(head_test_path),
                    config_type_weights=head_ct_weights,
                    energy_key=head_energy_key,
                    forces_key=head_forces_key,
                    prefactor_stress=head_prefactor_stress,
                    remap_stress=head_remap_stress,
                    log_label='test',
                )
    else:
        if valid_path:
            _append_streaming_specs(
                valid_specs,
                head_name=head_names[0],
                paths=_ensure_list(valid_path),
                config_type_weights=config_type_weights,
                energy_key=energy_key,
                forces_key=forces_key,
                prefactor_stress=prefactor_stress,
                remap_stress=remap_stress,
                log_label='validation',
            )
        if test_path:
            _append_streaming_specs(
                test_specs,
                head_name=head_names[0],
                paths=_ensure_list(test_path),
                config_type_weights=config_type_weights,
                energy_key=energy_key,
                forces_key=forces_key,
                prefactor_stress=prefactor_stress,
                remap_stress=remap_stress,
                log_label='test',
            )

    return valid_specs, test_specs


def _build_eval_streaming_loader(
    dataset_specs: Sequence[StreamingDatasetSpec],
    *,
    r_max: float,
    n_node: int,
    n_edge: int,
    n_graph: int,
    head_to_index: dict[str, int],
    base_z_table: data.AtomicNumberTable | None,
    num_workers: int = 0,
    seed: int = 0,
    stream_prefetch: int | None = None,
    template_loader: data.StreamingGraphDataLoader | None = None,
) -> data.StreamingGraphDataLoader | None:
    if not dataset_specs:
        return None
    atomic_numbers_override: Sequence[int] | None = None
    atomic_energies_override: dict[int, float] | str | None = None
    if template_loader is not None:
        atomic_numbers_override = getattr(template_loader, 'atomic_numbers', None)
        atomic_energies_override = getattr(template_loader, 'atomic_energies', None)
    if atomic_numbers_override is None and base_z_table is not None:
        atomic_numbers_override = [int(z) for z in base_z_table.zs]

    loader, _, _ = _build_streaming_train_loader(
        dataset_specs=dataset_specs,
        r_max=r_max,
        n_node=n_node,
        n_edge=n_edge,
        n_graph=n_graph,
        seed=int(seed),
        head_to_index=head_to_index,
        stream_prefetch=stream_prefetch,
        stream_workers=num_workers,
        atomic_numbers_override=atomic_numbers_override,
        atomic_energies_override=atomic_energies_override,
    )
    return loader


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
    n_node: int | None = None,
    n_edge: int = 1,
    n_graph: int = 1,
    prefactor_stress: float = 1.0,
    remap_stress: np.ndarray = None,
    heads: Sequence[str] = ('Default',),
    head_configs: dict[str, dict] | None = None,
    stream_train_prefetch: int | None = None,
    stream_train_workers: int = 0,
    atomic_numbers: Sequence[int] | None = None,
    atomic_energies_override: dict[int, float] | str | None = None,
    statistics_metadata_path: str | None = None,
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
    if n_node is not None:
        raise ValueError(
            'n_node is not supported with streaming datasets; configure n_edge only.'
        )
    if n_edge is None:
        raise ValueError('n_edge must be provided for streaming datasets.')

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
    valid_specs, test_specs = _collect_eval_streaming_specs(
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
        atomic_numbers_override=atomic_numbers,
        atomic_energies_override=atomic_energies_override,
        statistics_metadata_path=statistics_metadata_path,
    )
    effective_n_node = getattr(train_loader, '_fixed_pad_nodes', None)
    if effective_n_node is None:
        effective_n_node = getattr(train_loader, '_n_node', None)
    effective_n_edge = getattr(train_loader, '_fixed_pad_edges', None)
    if effective_n_edge is None:
        effective_n_edge = getattr(train_loader, '_n_edge', None)
    if effective_n_node is None or effective_n_edge is None:
        raise ValueError(
            'Unable to determine batch size limits for evaluation loaders.'
        )

    base_z_table = getattr(train_loader, 'z_table', None)
    valid_loader = _build_eval_streaming_loader(
        valid_specs,
        r_max=r_max,
        n_node=effective_n_node,
        n_edge=effective_n_edge,
        n_graph=n_graph,
        head_to_index=head_to_index,
        base_z_table=base_z_table,
        num_workers=stream_train_workers,
        seed=seed,
        stream_prefetch=stream_train_prefetch,
        template_loader=train_loader,
    )
    test_loader = _build_eval_streaming_loader(
        test_specs,
        r_max=r_max,
        n_node=effective_n_node,
        n_edge=effective_n_edge,
        n_graph=n_graph,
        head_to_index=head_to_index,
        base_z_table=base_z_table,
        num_workers=stream_train_workers,
        seed=seed,
        stream_prefetch=stream_train_prefetch,
        template_loader=train_loader,
    )
    logging.info(
        'Streaming training graphs: %s from %s files',
        getattr(train_loader, 'total_graphs', 'unknown'),
        len(dataset_specs),
    )
    valid_count = getattr(valid_loader, 'total_graphs', 0) if valid_loader else 0
    test_count = getattr(test_loader, 'total_graphs', 0) if test_loader else 0
    logging.info('Validation graphs: %s | Test graphs: %s', valid_count, test_count)
    return train_loader, valid_loader, test_loader, atomic_energies_dict, r_max
