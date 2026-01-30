"""Gin dataset helpers for streaming HDF5 training/evaluation.

These helpers build streaming loaders that emit fixed-shape batches. Fixed
shapes are crucial for JAX/XLA: the model compiles per input shape, so varying
batch sizes across steps would trigger repeated recompilation. The streaming
stats pass scans datasets to derive fixed padding caps so every batch shares
the same shape, and the caps can be cached per split to avoid rescans.
"""

import logging
import multiprocessing as mp
from collections.abc import Sequence
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from queue import Empty

import gin
import jax
import numpy as np
from tqdm.auto import tqdm

from mace_jax import data
from mace_jax.data.streaming_loader import StreamingDatasetSpec
from mace_jax.data.streaming_stats_cache import (
    load_cached_streaming_stats,
    spec_fingerprint,
    stats_payload_from_parts,
    stats_payload_to_parts,
    store_cached_streaming_stats,
)
from mace_jax.tools.utils import log_info_primary, pt_head_first


def _is_primary_process() -> bool:
    try:
        return getattr(jax, 'process_index', lambda: 0)() == 0
    except Exception:
        return True


def _ensure_list(value) -> list[str]:
    """Normalize a scalar/sequence into a list of strings.

    Args:
        value: None, a string, or a sequence of values.

    Returns:
        List of stringified values. Returns an empty list when ``value`` is None.

    This helper keeps downstream dataset spec handling uniform across gin/CLI inputs.
    """
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    return [str(value)]


def _normalize_node_percentile(value: float | None) -> float | None:
    if value is None:
        return None
    percentile = float(value)
    if 0.0 <= percentile <= 1.0:
        percentile *= 100.0
    if percentile < 0.0 or percentile > 100.0:
        raise ValueError(f'node_percentile must be between 0 and 100, got {value}.')
    return percentile


@dataclass
class _StreamingStats:
    """Cached padding caps for a single HDF5 dataset."""

    n_nodes: int
    n_edges: int
    n_graphs: int
    n_batches: int | None = None


@dataclass
class _StreamingMetadata:
    """Metadata gathered while scanning a dataset for streaming stats.

    Used to build initialization graphs, estimate E0s/neighbor statistics, and
    compute aggregate counts when constructing streaming loaders.
    """

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
    """Check whether a graph contributes to loss/metrics.

    Args:
        graph: GraphsTuple with optional ``globals.weight``.

    Returns:
        True if the graph has no weight field or a positive scalar weight.
    """
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
    """Load an HDF5 entry, apply dataset spec, and convert it into a graph.

    Args:
        dataset: Open HDF5 dataset reader.
        idx: Index within the dataset.
        spec: Streaming dataset spec (keys, weights, head name).
        z_table: Atomic number table for species indexing.
        r_max: Cutoff used to build neighborhood edges.
        head_to_index: Mapping from head names to integer indices.

    Returns:
        A ``jraph.GraphsTuple`` representing the configuration at ``idx``.

    This is the core adapter between on-disk HDF5 configs and in-memory graphs.
    """
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


def _streaming_stats_worker_main(
    worker_id: int,
    dataset_path: Path,
    spec: StreamingDatasetSpec,
    z_table: data.AtomicNumberTable,
    r_max: float,
    head_to_index: dict[str, int],
    start_idx: int,
    end_idx: int,
    collect_metadata: bool,
    result_queue,
    progress_queue,
    progress_every: int | None,
):
    """Worker process entry point to scan graphs for streaming stats."""
    dataset = None
    try:
        dataset = data.HDF5Dataset(dataset_path)
        num_species = len(z_table)
        graph_sizes: list[tuple[int, int]] = []
        max_graph_edges = 0
        total_graphs = 0
        total_nodes = 0
        total_edges = 0
        ata = np.zeros((num_species, num_species), dtype=np.float64)
        atb = np.zeros((num_species,), dtype=np.float64)
        neighbor_sum = 0.0
        neighbor_count = 0
        min_distance_sum = 0.0
        min_distance_count = 0
        progress_count = 0
        for idx in range(start_idx, end_idx):
            graph = _graph_from_dataset_entry(
                dataset,
                idx,
                spec=spec,
                z_table=z_table,
                r_max=r_max,
                head_to_index=head_to_index,
            )
            progress_count += 1
            if (
                progress_queue is not None
                and progress_every
                and progress_count >= progress_every
            ):
                progress_queue.put(progress_count)
                progress_count = 0
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
            total_graphs += 1
            total_nodes += g_nodes
            total_edges += g_edges
            max_graph_edges = max(max_graph_edges, g_edges)
            graph_sizes.append((g_nodes, g_edges))
        if progress_queue is not None and progress_count:
            progress_queue.put(progress_count)
        result_queue.put(
            (
                worker_id,
                None,
                {
                    'start_idx': start_idx,
                    'graph_sizes': graph_sizes,
                    'max_graph_edges': max_graph_edges,
                    'total_graphs': total_graphs,
                    'total_nodes': total_nodes,
                    'total_edges': total_edges,
                    'ata': ata,
                    'atb': atb,
                    'neighbor_sum': neighbor_sum,
                    'neighbor_count': neighbor_count,
                    'min_distance_sum': min_distance_sum,
                    'min_distance_count': min_distance_count,
                },
            )
        )
    except Exception as exc:  # pragma: no cover - worker crashes are unexpected
        result_queue.put((worker_id, repr(exc), None))
    finally:
        if dataset is not None:
            dataset.close()


def _gather_sample_graphs(
    dataset_path: Path,
    *,
    spec: StreamingDatasetSpec,
    z_table: data.AtomicNumberTable,
    r_max: float,
    head_to_index: dict[str, int],
    sample_limit: int,
) -> list:
    """Collect a small set of graphs from a dataset for initialization.

    Args:
        dataset_path: Path to an HDF5 dataset.
        spec: Dataset spec describing keys and weights.
        z_table: Atomic number table for indexing species.
        r_max: Cutoff used to build edges.
        head_to_index: Mapping from head names to indices.
        sample_limit: Max number of graphs to return.

    Returns:
        A list of graphs with positive weights (may be empty).

    Used to seed model initialization and fallback E0 estimates without scanning
    the full dataset.
    """
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
    """Fill the sample graph buffer up to ``sample_limit``.

    Args:
        sample_graphs: Existing list to extend in-place.
        sample_limit: Target number of samples to keep.
        metadata: Cached metadata with precomputed samples (optional).
        dataset_path: Dataset to pull extra samples from.
        spec: Dataset spec for graph construction.
        z_table: Atomic number table.
        r_max: Cutoff used to build edges.
        head_to_index: Mapping from head names to indices.

    Returns:
        None. ``sample_graphs`` is mutated in-place.
    """
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


def _expand_hdf5_inputs(paths: Sequence[str]) -> list[Path]:
    """Expand HDF5 inputs from files, directories, or glob patterns.

    Args:
        paths: Paths or glob patterns provided by the user.

    Returns:
        List of HDF5 file paths.

    Raises:
        ValueError if no files match or if inputs are invalid.
    """
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
    """Validate that unsupported head options are not used for streaming datasets.

    Args:
        head_name: Name of the model head.
        head_cfg: Head-specific configuration dict.

    Raises:
        ValueError when a head option is incompatible with streaming training.
    """
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
    """Append streaming dataset specs for a set of paths.

    Args:
        collection: List to append to.
        head_name: Head name associated with the dataset.
        paths: HDF5 inputs (files/dirs/globs) to expand.
        config_type_weights: Optional config-type weights.
        energy_key: Energy property key.
        forces_key: Forces property key.
        prefactor_stress: Scaling factor for stress.
        remap_stress: Optional remap indices for stress tensors.
        weight: Global dataset weight.
        log_label: Optional label for info logging.

    Returns:
        None. ``collection`` is mutated in-place.
    """
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
            log_info_primary(
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
    """Build streaming dataset specs for training inputs.

    Args:
        head_names: Ordered list of head names.
        head_configs: Optional per-head configuration overrides.
        train_path: Default train path when no head overrides are given.
        config_type_weights: Optional config-type weights to apply.
        energy_key: Energy property key.
        forces_key: Forces property key.
        prefactor_stress: Scaling factor for stress.
        remap_stress: Optional remap indices for stress tensors.

    Returns:
        List of ``StreamingDatasetSpec`` entries used by the streaming loader.

    This keeps multi-head training aligned with per-head dataset definitions.
    """
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
    """Scan HDF5 datasets to collect all atomic numbers present.

    Args:
        paths: HDF5 files to scan.

    Returns:
        Sorted list of unique atomic numbers.

    Used when no explicit atomic number list is supplied.
    """
    numbers: set[int] = set()
    for path in paths:
        with data.HDF5Dataset(path, mode='r') as dataset:
            dataset_len = len(dataset)
            iterator = tqdm(
                range(dataset_len),
                desc=f'Extracting atomic species ({path.name})',
                disable=dataset_len < 1024 or not _is_primary_process(),
                leave=False,
            )
            for idx in iterator:
                atoms = dataset[idx]
                numbers.update(int(z) for z in atoms.get_atomic_numbers())
    if not numbers:
        raise ValueError('No atomic species found in the provided HDF5 datasets.')
    return sorted(numbers)


def _estimate_padding_caps(
    graph_sizes: list[tuple[int, int]],
    edge_cap: int,
    node_percentile: float | None = None,
) -> tuple[int, int]:
    """Estimate padding caps by sequential packing under an edge budget.

    Args:
        graph_sizes: Tuples of (num_nodes, num_edges).
        edge_cap: Maximum total edges per batch.

    Returns:
        (n_nodes, n_graphs) padding caps.

    This keeps batch shapes stable without precomputing batches.
    """
    if not graph_sizes:
        return 2, 2

    nodes_per_batch: list[int] = []
    graphs_per_batch: list[int] = []
    max_single_nodes = max(nodes for nodes, _ in graph_sizes)
    nodes_sum = 0
    edges_sum = 0
    graph_count = 0
    for nodes, edges in graph_sizes:
        if graph_count and edges_sum + edges > edge_cap:
            nodes_per_batch.append(nodes_sum)
            graphs_per_batch.append(graph_count)
            nodes_sum = 0
            edges_sum = 0
            graph_count = 0
        nodes_sum += nodes
        edges_sum += edges
        graph_count += 1
    if graph_count:
        nodes_per_batch.append(nodes_sum)
        graphs_per_batch.append(graph_count)

    max_nodes_per_batch = max(nodes_per_batch)
    max_graphs_per_batch = max(graphs_per_batch)
    target_nodes = max_nodes_per_batch
    if node_percentile is not None:
        percentile = _normalize_node_percentile(node_percentile)
        if percentile is not None:
            target_nodes = float(np.percentile(nodes_per_batch, percentile))
    target_nodes = max(target_nodes, max_single_nodes)
    n_nodes = max(int(np.ceil(target_nodes)) + 1, 2)
    n_graphs = max(max_graphs_per_batch + 1, 2)
    return n_nodes, n_graphs


def _estimate_batch_count(
    graph_sizes: list[tuple[int, int]],
    *,
    n_nodes: int,
    n_edges: int,
    n_graphs: int,
) -> int:
    """Estimate batches by simulating sequential packing with fixed caps."""
    if not graph_sizes:
        return 0
    max_graphs = max(int(n_graphs) - 1, 1)
    batch_count = 0
    nodes_sum = 0
    edges_sum = 0
    graph_count = 0
    for nodes, edges in graph_sizes:
        if graph_count and (
            nodes_sum + nodes >= n_nodes
            or edges_sum + edges > n_edges
            or graph_count >= max_graphs
        ):
            batch_count += 1
            nodes_sum = 0
            edges_sum = 0
            graph_count = 0
        nodes_sum += nodes
        edges_sum += edges
        graph_count += 1
        if graph_count >= max_graphs:
            batch_count += 1
            nodes_sum = 0
            edges_sum = 0
            graph_count = 0
    if graph_count:
        batch_count += 1
    return max(batch_count, 1)


def _compute_streaming_stats(
    dataset_path: Path,
    *,
    spec: StreamingDatasetSpec,
    z_table: data.AtomicNumberTable,
    r_max: float,
    head_to_index: dict[str, int],
    sample_limit: int,
    edge_cap: int,
    node_percentile: float | None,
    collect_metadata: bool,
    stats_workers: int | None,
) -> tuple[_StreamingStats, _StreamingMetadata | None]:
    """Scan a dataset to compute padding caps and optional metadata.

    Args:
        dataset_path: Path to the HDF5 dataset.
        spec: Dataset spec defining keys/weights/heads.
        z_table: Atomic number table for species mapping.
        r_max: Cutoff used to build edges for stats.
        head_to_index: Mapping from head names to indices.
        sample_limit: Number of sample graphs to keep for initialization.
        edge_cap: Edge budget for batch packing.
        node_percentile: Optional percentile for node padding cap.
        collect_metadata: Whether to compute E0/neighbor statistics metadata.

    Returns:
        A tuple of (stats, metadata). Metadata is ``None`` when not requested.

    This is the expensive pass that populates the streaming stats cache. It
    scans the dataset, computes per-graph sizes, raises caps when a graph would
    exceed the configured edge limit, and derives padding caps from greedy
    packing. These fixed shapes are required to compile the model once in JAX
    and reuse the compiled graph across epochs.
    """
    sample_graphs: list = []
    if collect_metadata:
        sample_graphs = _gather_sample_graphs(
            dataset_path,
            spec=spec,
            z_table=z_table,
            r_max=r_max,
            head_to_index=head_to_index,
            sample_limit=sample_limit,
        )
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
    graph_sizes: list[tuple[int, int]] = []
    dataset = data.HDF5Dataset(dataset_path)
    dataset_len = len(dataset)
    dataset.close()
    stats_workers = int(stats_workers or 0)
    if stats_workers > 1 and dataset_len > 0:
        stats_workers = min(stats_workers, dataset_len)
    if stats_workers <= 1 or dataset_len <= 0:
        dataset = data.HDF5Dataset(dataset_path)
        progress = None
        try:
            iterator = range(dataset_len)
            if dataset_len >= 1024:
                progress = tqdm(
                    iterator,
                    desc=f'Computing streaming stats ({dataset_path.name})',
                    disable=not _is_primary_process(),
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
                    counts = np.bincount(species, minlength=num_species).astype(
                        np.float64
                    )
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
                total_graphs += 1
                total_nodes += g_nodes
                total_edges += g_edges
                max_graph_edges = max(max_graph_edges, g_edges)
                graph_sizes.append((g_nodes, g_edges))
        finally:
            if progress is not None:
                progress.close()
            dataset.close()
    else:
        ranges: list[tuple[int, int]] = []
        base = dataset_len // stats_workers
        remainder = dataset_len % stats_workers
        start = 0
        for worker_idx in range(stats_workers):
            extra = 1 if worker_idx < remainder else 0
            end = start + base + extra
            if end > start:
                ranges.append((start, end))
            start = end
        ctx = mp.get_context('spawn')
        result_queue = ctx.Queue()
        progress_queue = None
        progress = None
        progress_every = None
        if dataset_len >= 1024 and _is_primary_process():
            progress = tqdm(
                total=dataset_len,
                desc=f'Computing streaming stats ({dataset_path.name})',
                leave=False,
            )
            progress_queue = ctx.Queue()
            progress_every = max(1024, dataset_len // max(stats_workers * 20, 1))
        processes: list[mp.Process] = []
        for worker_id, (start_idx, end_idx) in enumerate(ranges):
            proc = ctx.Process(
                target=_streaming_stats_worker_main,
                args=(
                    worker_id,
                    dataset_path,
                    spec,
                    z_table,
                    r_max,
                    head_to_index,
                    start_idx,
                    end_idx,
                    collect_metadata,
                    result_queue,
                    progress_queue,
                    progress_every,
                ),
            )
            proc.daemon = True
            proc.start()
            processes.append(proc)
        finished = 0
        ordered_sizes: list[tuple[int, list[tuple[int, int]]]] = []
        try:
            while finished < len(processes):
                if progress_queue is not None:
                    try:
                        while True:
                            progress.update(progress_queue.get_nowait())
                    except Empty:
                        pass
                if progress is None:
                    worker_id, error, payload = result_queue.get()
                else:
                    try:
                        worker_id, error, payload = result_queue.get(timeout=0.5)
                    except Empty:
                        continue
                finished += 1
                if error is not None:
                    raise RuntimeError(
                        f'Streaming stats worker {worker_id} failed: {error}'
                    )
                ordered_sizes.append(
                    (int(payload['start_idx']), payload['graph_sizes'])
                )
                max_graph_edges = max(max_graph_edges, int(payload['max_graph_edges']))
                total_graphs += int(payload['total_graphs'])
                total_nodes += int(payload['total_nodes'])
                total_edges += int(payload['total_edges'])
                if collect_metadata:
                    ata += payload['ata']
                    atb += payload['atb']
                    neighbor_sum += float(payload['neighbor_sum'])
                    neighbor_count += int(payload['neighbor_count'])
                    min_distance_sum += float(payload['min_distance_sum'])
                    min_distance_count += int(payload['min_distance_count'])
        finally:
            if progress is not None:
                if progress.n < dataset_len:
                    progress.update(dataset_len - progress.n)
                progress.close()
            for proc in processes:
                proc.join()
        for _, sizes in sorted(ordered_sizes, key=lambda item: item[0]):
            graph_sizes.extend(sizes)

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

    n_nodes, n_graphs = _estimate_padding_caps(
        graph_sizes, edge_cap, node_percentile=node_percentile
    )
    n_batches = _estimate_batch_count(
        graph_sizes,
        n_nodes=n_nodes,
        n_edges=edge_cap,
        n_graphs=n_graphs,
    )
    stats = _StreamingStats(
        n_nodes=n_nodes,
        n_edges=edge_cap,
        n_graphs=n_graphs,
        n_batches=n_batches,
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
    node_percentile: float | None,
    collect_metadata: bool,
    stats_workers: int | None,
) -> tuple[_StreamingStats, _StreamingMetadata | None]:
    """Fetch streaming stats from cache or compute them if missing.

    Args:
        spec: Dataset spec to fingerprint and compute stats for.
        r_max: Cutoff used to build edges.
        z_table: Atomic number table.
        head_to_index: Mapping from head names to indices.
        atomic_numbers: Atomic numbers used for fingerprinting.
        sample_limit: Number of sample graphs to collect.
        edge_cap: Edge budget for batch packing.
        node_percentile: Optional percentile for node padding cap.
        collect_metadata: Whether to compute E0/neighbor metadata.

    Returns:
        Cached or freshly computed (stats, metadata).

    The cache stores padding caps so we avoid an expensive full scan on every
    run. This keeps startup time low while still ensuring stable batch shapes
    for JAX compilation.
    """
    dataset_path = Path(spec.path)
    fingerprint = spec_fingerprint(
        spec,
        r_max=r_max,
        atomic_numbers=atomic_numbers,
        head_to_index=head_to_index,
        edge_cap=edge_cap,
        node_percentile=node_percentile,
    )
    cached_payload = load_cached_streaming_stats(dataset_path, fingerprint)
    if cached_payload is None:
        stats, metadata = _compute_streaming_stats(
            dataset_path,
            spec=spec,
            z_table=z_table,
            r_max=r_max,
            head_to_index=head_to_index,
            sample_limit=sample_limit,
            edge_cap=edge_cap,
            node_percentile=node_percentile,
            collect_metadata=collect_metadata,
            stats_workers=stats_workers,
        )
        store_cached_streaming_stats(
            dataset_path,
            fingerprint,
            stats_payload_from_parts(
                stats.n_nodes,
                stats.n_edges,
                stats.n_graphs,
                stats.n_batches,
            ),
        )
        return stats, metadata
    if not collect_metadata:
        n_nodes, n_edges, n_graphs, n_batches = stats_payload_to_parts(cached_payload)
        return (
            _StreamingStats(
                n_nodes=n_nodes,
                n_edges=n_edges,
                n_graphs=n_graphs,
                n_batches=n_batches,
            ),
            None,
        )
    _, metadata = _compute_streaming_stats(
        dataset_path,
        spec=spec,
        z_table=z_table,
        r_max=r_max,
        head_to_index=head_to_index,
        sample_limit=sample_limit,
        edge_cap=edge_cap,
        node_percentile=node_percentile,
        collect_metadata=True,
        stats_workers=stats_workers,
    )
    n_nodes, n_edges, n_graphs, n_batches = stats_payload_to_parts(cached_payload)
    return (
        _StreamingStats(
            n_nodes=n_nodes,
            n_edges=n_edges,
            n_graphs=n_graphs,
            n_batches=n_batches,
        ),
        metadata,
    )


def _atomic_energies_to_array(
    atomic_energies_dict: dict[int, float],
    z_table: data.AtomicNumberTable,
) -> np.ndarray:
    """Convert an atomic-energies dict into a dense array aligned to ``z_table``.

    Args:
        atomic_energies_dict: Mapping from atomic number to energy.
        z_table: Atomic number table used for ordering.

    Returns:
        ``np.ndarray`` of shape ``(num_species,)``.
    """
    return np.array(
        [
            atomic_energies_dict.get(z_table.index_to_z(i), 0.0)
            for i in range(len(z_table))
        ],
        dtype=np.float64,
    )


def _resolve_atomic_energies(
    *,
    atomic_numbers: Sequence[int],
    z_table: data.AtomicNumberTable,
    ata: np.ndarray,
    atb: np.ndarray,
    collect_metadata: bool,
    sample_graphs: list,
    atomic_energies_override: dict[int, float] | str | None,
    statistics_metadata_path: str | None,
) -> tuple[dict[int, float], np.ndarray]:
    """Resolve atomic energies from overrides, metadata, or fallback estimation.

    Args:
        atomic_numbers: Atomic numbers present in the dataset.
        z_table: Atomic number table for ordering.
        ata: Normal-equation matrix accumulated from metadata.
        atb: Normal-equation RHS accumulated from metadata.
        collect_metadata: Whether metadata was collected for solving E0s.
        sample_graphs: Sample graphs used for fallback average estimation.
        atomic_energies_override: Optional dict or string override.
        statistics_metadata_path: Optional stats path for logging context.

    Returns:
        (atomic_energies_dict, atomic_energies_array).
    """
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
        if statistics_metadata_path:
            log_info_primary(
                'Using atomic energies from statistics file %s',
                statistics_metadata_path,
            )
        return override_dict, _atomic_energies_to_array(override_dict, z_table)

    if collect_metadata and np.any(atb):
        regularizer = 1e-8 * np.eye(len(z_table))
        atomic_energies = np.linalg.solve(ata + regularizer, atb)
        atomic_energies_dict = {
            z_table.index_to_z(i): float(atomic_energies[i])
            for i in range(len(z_table))
        }
        return atomic_energies_dict, atomic_energies.astype(np.float64)

    atomic_energies_dict = data.compute_average_E0s(sample_graphs, z_table)
    return atomic_energies_dict, _atomic_energies_to_array(
        atomic_energies_dict, z_table
    )


def _build_streaming_train_loader(
    *,
    dataset_specs: Sequence[StreamingDatasetSpec],
    r_max: float,
    n_edge: int,
    seed: int,
    head_to_index: dict[str, int],
    stream_prefetch: int | None,
    num_workers: int,
    stream_stats_workers: int | None = None,
    max_batches: int | None = None,
    shuffle: bool = True,
    node_percentile: float | None = None,
    atomic_numbers_override: Sequence[int] | None = None,
    atomic_energies_override: dict[int, float] | str | None = None,
    statistics_metadata_path: str | None = None,
) -> tuple[data.StreamingGraphDataLoader, dict[int, float], float]:
    """Construct the streaming training loader and associated statistics.

    Args:
        dataset_specs: Streaming dataset specs for training.
        r_max: Cutoff used for edge construction.
        n_edge: Edge cap used for batch packing.
        seed: Random seed for downstream training utilities.
        head_to_index: Mapping from head names to indices.
        stream_prefetch: Prefetch batch count.
        num_workers: Worker count for streaming loader.
        stream_stats_workers: Optional worker count for streaming stats scans.
        max_batches: Optional cap on the number of batches per epoch.
        shuffle: Whether to shuffle training samples each epoch.
        node_percentile: Percentile for node padding cap (min is max graph size).
        atomic_numbers_override: Explicit atomic numbers (optional).
        atomic_energies_override: Atomic energies override (optional).
        statistics_metadata_path: Path to statistics metadata (for logging).

    Returns:
        (train_loader, atomic_energies_dict, r_max).

    This function computes/loads streaming stats and attaches metadata to the
    loader. The resulting loader streams graphs in file order and packs batches
    on the fly within fixed padding caps so training batches have consistent
    shapes, allowing the model to compile once and run efficiently.
    """
    if not dataset_specs:
        raise ValueError('No streaming datasets were provided.')
    if atomic_numbers_override:
        atomic_numbers = [int(z) for z in atomic_numbers_override]
    else:
        atomic_numbers = _unique_atomic_numbers_from_hdf5(
            [spec.path for spec in dataset_specs]
        )
    if atomic_numbers_override and statistics_metadata_path:
        log_info_primary(
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
    n_nodes = 0
    n_edges = 0
    n_graphs = 0
    total_graphs = 0
    total_nodes = 0
    total_edges = 0
    total_batches = 0
    per_dataset_batches: list[int | None] = []

    collect_metadata = not isinstance(atomic_energies_override, dict)

    node_percentile = _normalize_node_percentile(node_percentile)
    if stream_stats_workers is None:
        stream_stats_workers = num_workers
    if max_batches is not None:
        max_batches = int(max_batches)
        if max_batches <= 0:
            max_batches = None

    for spec in dataset_specs:
        stats, metadata = _load_or_compute_streaming_stats(
            spec,
            r_max=r_max,
            z_table=z_table,
            head_to_index=head_to_index,
            atomic_numbers=atomic_numbers,
            sample_limit=sample_cap,
            edge_cap=n_edge,
            node_percentile=node_percentile,
            collect_metadata=collect_metadata,
            stats_workers=stream_stats_workers,
        )
        n_nodes = max(n_nodes, int(stats.n_nodes))
        n_edges = max(n_edges, int(stats.n_edges))
        n_graphs = max(n_graphs, int(stats.n_graphs))
        if stats.n_batches:
            total_batches += int(stats.n_batches)
        per_dataset_batches.append(stats.n_batches)
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
            total_graphs += metadata.total_graphs
            total_nodes += metadata.total_nodes
            total_edges += metadata.total_edges
        else:
            with data.HDF5Dataset(spec.path, mode='r') as dataset:
                total_graphs += len(dataset)

    if not sample_graphs:
        raise ValueError('Unable to build initialization graphs from training data.')

    avg_neighbors = neighbor_sum / neighbor_count if neighbor_count else 0.0
    avg_min_distance = (
        min_distance_sum / min_distance_count if min_distance_count else 0.0
    )
    atomic_energies_dict, atomic_energies = _resolve_atomic_energies(
        atomic_numbers=atomic_numbers,
        z_table=z_table,
        ata=ata,
        atb=atb,
        collect_metadata=collect_metadata,
        sample_graphs=sample_graphs,
        atomic_energies_override=atomic_energies_override,
        statistics_metadata_path=statistics_metadata_path,
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
        niggli_reduce=False,
        max_batches=max_batches,
        prefetch_batches=None if stream_prefetch is None else int(stream_prefetch),
        num_workers=int(num_workers or 0),
        pad_graphs=n_graphs,
        shuffle=bool(shuffle),
    )
    loader.graphs = sample_graphs
    loader.atomic_numbers = tuple(atomic_numbers)
    loader.avg_num_neighbors = avg_neighbors
    loader.avg_r_min = avg_min_distance
    loader.atomic_energies = atomic_energies
    loader.total_graphs = total_graphs or None
    loader.total_nodes = total_nodes or None
    loader.total_edges = total_edges or None
    loader.estimated_batches = total_batches or None
    loader._dataset_estimated_batches = per_dataset_batches
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
    config_type_weights: dict | None,
    energy_key: str,
    forces_key: str,
    prefactor_stress: float,
    remap_stress: np.ndarray | None,
) -> tuple[list[StreamingDatasetSpec], list[StreamingDatasetSpec]]:
    """Build streaming dataset specs for validation and test splits.

    Args:
        head_names: Ordered list of head names.
        head_configs: Optional per-head configuration overrides.
        valid_path: Default validation path.
        test_path: Default test path.
        config_type_weights: Optional config-type weights.
        energy_key: Energy property key.
        forces_key: Forces property key.
        prefactor_stress: Scaling factor for stress.
        remap_stress: Optional stress remap indices.

    Returns:
        (valid_specs, test_specs) lists for streaming loaders.
    """
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
            head_test_num = head_cfg.get('test_num')
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
    n_edge: int,
    head_to_index: dict[str, int],
    base_z_table: data.AtomicNumberTable | None,
    num_workers: int = 0,
    seed: int = 0,
    stream_prefetch: int | None = None,
    stream_stats_workers: int | None = None,
    max_batches: int | None = None,
    shuffle: bool = False,
    node_percentile: float | None = None,
    template_loader: data.StreamingGraphDataLoader | None = None,
) -> data.StreamingGraphDataLoader | None:
    """Construct a streaming loader for evaluation splits.

    Args:
        dataset_specs: Specs for the evaluation datasets.
        r_max: Cutoff used for edge construction.
        n_edge: Edge cap for padding.
        head_to_index: Mapping from head names to indices.
        base_z_table: Base atomic-number table from training (optional).
        num_workers: Worker count.
        seed: Random seed for downstream training utilities.
        stream_prefetch: Prefetch batch count.
        stream_stats_workers: Optional worker count for streaming stats scans.
        shuffle: Whether to shuffle evaluation samples (default False).
        node_percentile: Percentile for node padding cap (min is max graph size).
        template_loader: Training loader used to inherit atomic numbers/energies.

    Returns:
        A ``StreamingGraphDataLoader`` or ``None`` if no specs are provided.

    Evaluation loaders reuse the same fixed padding caps as training to avoid
    shape-driven recompilation during validation/test passes.
    """
    if not dataset_specs:
        return None
    atomic_numbers_override: Sequence[int] | None = None
    atomic_energies_override: dict[int, float] | str | None = None
    if template_loader is not None:
        atomic_numbers_override = getattr(template_loader, 'atomic_numbers', None)
        atomic_energies_override = getattr(template_loader, 'atomic_energies', None)
    if atomic_numbers_override is None and base_z_table is not None:
        atomic_numbers_override = [int(z) for z in base_z_table.zs]
    if (
        atomic_numbers_override is not None
        and isinstance(atomic_energies_override, np.ndarray)
        and atomic_energies_override.size
    ):
        atomic_energies_override = {
            int(z): float(atomic_energies_override[i])
            for i, z in enumerate(atomic_numbers_override)
        }

    loader, _, _ = _build_streaming_train_loader(
        dataset_specs=dataset_specs,
        r_max=r_max,
        n_edge=n_edge,
        seed=int(seed),
        head_to_index=head_to_index,
        stream_prefetch=stream_prefetch,
        num_workers=num_workers,
        stream_stats_workers=stream_stats_workers,
        max_batches=max_batches,
        shuffle=shuffle,
        node_percentile=node_percentile,
        atomic_numbers_override=atomic_numbers_override,
        atomic_energies_override=atomic_energies_override,
    )
    return loader


@gin.configurable
def datasets(
    *,
    r_max: float | None = None,
    train_path: str,
    config_type_weights: dict = None,
    valid_path: str = None,
    test_path: str = None,
    seed: int = 1234,
    energy_key: str = 'energy',
    forces_key: str = 'forces',
    n_edge: int = 1,
    prefactor_stress: float = 1.0,
    remap_stress: np.ndarray = None,
    heads: Sequence[str] = ('Default',),
    head_configs: dict[str, dict] | None = None,
    stream_train_prefetch: int | None = None,
    stream_train_stats_workers: int | None = None,
    stream_train_max_batches: int | None = None,
    stream_train_shuffle: bool = True,
    stream_train_node_percentile: float = 90.0,
    num_workers: int = 0,
    atomic_numbers: Sequence[int] | None = None,
    atomic_energies_override: dict[int, float] | str | None = None,
    statistics_metadata_path: str | None = None,
) -> tuple[
    data.StreamingGraphDataLoader,
    data.StreamingGraphDataLoader | None,
    data.StreamingGraphDataLoader | None,
    dict[int, float],
    float,
]:
    """Load datasets for gin, streaming the training split directly from HDF5.

    Args:
        r_max: Cutoff used for neighbor construction. When None, uses the
            model cutoff (mace_jax.tools.gin_model.model.r_max) if available.
        train_path: Training HDF5 path(s).
        config_type_weights: Optional per-config-type weights.
        valid_path: Validation HDF5 path(s).
        test_path: Test HDF5 path(s).
        seed: Random seed for downstream training utilities.
        energy_key: Energy property key.
        forces_key: Forces property key.
        n_edge: Edge cap for batch packing.
        prefactor_stress: Scaling factor for stress.
        remap_stress: Optional stress remap indices.
        heads: Head names for multihead datasets.
        head_configs: Optional per-head configuration overrides.
        stream_train_prefetch: Prefetch batch count.
        stream_train_stats_workers: Optional worker count for streaming stats scans.
        stream_train_max_batches: Optional cap on streaming batches per epoch.
        stream_train_shuffle: Whether to shuffle training samples each epoch.
        stream_train_node_percentile: Percentile for node padding cap (min is max graph size).
        num_workers: Worker count (per process/device).
        atomic_numbers: Optional explicit atomic numbers list.
        atomic_energies_override: Optional atomic energies override.
        statistics_metadata_path: Optional stats path for logging context.

    Returns:
        (train_loader, valid_loader, test_loader, atomic_energies_dict, r_max).

    This is the primary dataset entry point referenced from gin configs. It
    builds fixed-shape streaming loaders for train/valid/test so JAX compilation
    happens once and remains stable across epochs and evaluation runs.
    """

    if r_max is None:
        try:
            r_max = float(gin.query_parameter('mace_jax.tools.gin_model.model.r_max'))
        except Exception as exc:
            raise ValueError(
                'datasets.r_max is required unless '
                'mace_jax.tools.gin_model.model.r_max is configured.'
            ) from exc

    head_names = tuple(heads) if heads else ('Default',)
    head_names = pt_head_first(head_names)
    head_to_index = {name: idx for idx, name in enumerate(head_names)}
    local_device_count = getattr(jax, 'local_device_count', lambda: 1)()
    effective_workers = int(num_workers or 0)
    if local_device_count > 1 and effective_workers > 0:
        effective_workers *= int(local_device_count)
    stats_workers = stream_train_stats_workers
    if stats_workers is None:
        stats_workers = effective_workers

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
        config_type_weights=config_type_weights,
        energy_key=energy_key,
        forces_key=forces_key,
        prefactor_stress=prefactor_stress,
        remap_stress=remap_stress,
    )

    train_loader, atomic_energies_dict, r_max = _build_streaming_train_loader(
        dataset_specs=dataset_specs,
        r_max=r_max,
        n_edge=n_edge,
        seed=seed,
        head_to_index=head_to_index,
        stream_prefetch=stream_train_prefetch,
        num_workers=effective_workers,
        stream_stats_workers=stats_workers,
        max_batches=stream_train_max_batches,
        shuffle=stream_train_shuffle,
        node_percentile=stream_train_node_percentile,
        atomic_numbers_override=atomic_numbers,
        atomic_energies_override=atomic_energies_override,
        statistics_metadata_path=statistics_metadata_path,
    )
    effective_n_edge = getattr(train_loader, '_fixed_pad_edges', None)
    if effective_n_edge is None:
        effective_n_edge = getattr(train_loader, '_n_edge', None)
    if effective_n_edge is None:
        raise ValueError(
            'Unable to determine batch size limits for evaluation loaders.'
        )

    base_z_table = getattr(train_loader, 'z_table', None)
    valid_loader = _build_eval_streaming_loader(
        valid_specs,
        r_max=r_max,
        n_edge=effective_n_edge,
        head_to_index=head_to_index,
        base_z_table=base_z_table,
        seed=seed,
        stream_prefetch=stream_train_prefetch,
        num_workers=effective_workers,
        stream_stats_workers=stats_workers,
        max_batches=stream_train_max_batches,
        shuffle=False,
        node_percentile=stream_train_node_percentile,
        template_loader=train_loader,
    )
    test_loader = _build_eval_streaming_loader(
        test_specs,
        r_max=r_max,
        n_edge=effective_n_edge,
        head_to_index=head_to_index,
        base_z_table=base_z_table,
        seed=seed,
        stream_prefetch=stream_train_prefetch,
        num_workers=effective_workers,
        stream_stats_workers=stats_workers,
        max_batches=stream_train_max_batches,
        shuffle=False,
        node_percentile=stream_train_node_percentile,
        template_loader=train_loader,
    )
    log_info_primary(
        'Streaming training graphs: %s from %s files',
        getattr(train_loader, 'total_graphs', 'unknown'),
        len(dataset_specs),
    )
    valid_count = getattr(valid_loader, 'total_graphs', 0) if valid_loader else 0
    test_count = getattr(test_loader, 'total_graphs', 0) if test_loader else 0
    log_info_primary('Validation graphs: %s | Test graphs: %s', valid_count, test_count)
    return train_loader, valid_loader, test_loader, atomic_energies_dict, r_max
