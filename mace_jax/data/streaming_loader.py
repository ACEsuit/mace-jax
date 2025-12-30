"""Streaming HDF5 data loader for native MACE datasets.

This loader is designed for JAX/XLA training where the model is compiled per
input shape. We therefore prefer fixed-size (padded) batches so that the model
compiles once and is reused across epochs, avoiding recompilation stalls and
shape-driven performance regressions. The streaming pipeline scans datasets,
packs graphs into batches that respect `n_node`/`n_edge` caps, and pads each
batch to consistent sizes to keep shapes stable.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import threading
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, replace
from glob import glob
from pathlib import Path
from queue import Empty

import jraph
import numpy as np

from .hdf5_dataset import HDF5Dataset
from .streaming_stats_cache import (
    load_cached_streaming_stats,
    spec_fingerprint,
    stats_payload_from_parts,
    stats_payload_to_parts,
    store_cached_streaming_stats,
)
from .utils import (
    AtomicNumberTable,
    GraphDataLoader,
    config_from_atoms,
    graph_from_configuration,
)

_RESULT_DATA = 'data'
_RESULT_DONE = 'done'
_RESULT_ERROR = 'error'
_RESULT_SKIP = 'skip'
_TASK_BATCH = 'batch'


class BatchIteratorWrapper:
    """Wrap an iterator and expose a total_batches_hint attribute."""

    def __init__(self, iterator, total_batches_hint: int):
        """Initialize the wrapper with a source iterator and batch hint."""
        self._iterator = iterator
        self.total_batches_hint = int(total_batches_hint or 0)

    def __iter__(self):
        """Return self as an iterator."""
        return self

    def __next__(self):
        """Yield the next batch from the wrapped iterator."""
        return next(self._iterator)


@dataclass(frozen=True)
class StreamingDatasetSpec:
    """Configuration describing one HDF5 dataset stream."""

    path: Path
    head_name: str = 'Default'
    config_type_weights: dict[str, float] | None = None
    energy_key: str = 'energy'
    forces_key: str = 'forces'
    stress_key: str = 'stress'
    prefactor_stress: float = 1.0
    remap_stress: np.ndarray | None = None
    weight: float = 1.0


@dataclass(frozen=True)
class _StreamingStats:
    """Cached batch assignment summary for a single HDF5 dataset."""

    batch_assignments: list[list[int]]
    n_nodes: int
    n_edges: int
    n_graphs: int


def _niggli_reduce_inplace(atoms):
    """Apply Niggli reduction to ASE atoms if periodic and available.

    Args:
        atoms: ASE Atoms-like object.

    Returns:
        The same atoms instance (possibly modified in-place).
    """
    try:
        from ase.build.tools import niggli_reduce as _niggli_reduce  # noqa: PLC0415
    except ImportError:  # pragma: no cover - ase is an install dependency
        return atoms
    pbc = getattr(atoms, 'pbc', None)
    if pbc is None or not np.any(pbc):
        return atoms
    _niggli_reduce(atoms)
    return atoms


def _graph_worker_main(
    worker_id: int,
    dataset_paths: list[Path],
    dataset_specs: list[StreamingDatasetSpec],
    cutoff: float,
    z_table: AtomicNumberTable,
    head_to_index: dict[str, int],
    niggli_reduce: bool,
    task_queue,
    result_queue,
    stop_event,
):
    """Worker process entry point to convert dataset samples into graphs."""
    local_datasets: dict[int, HDF5Dataset] = {}
    try:
        while not stop_event.is_set():
            task = task_queue.get()
            if task is None:
                break
            filter_weight = True
            if isinstance(task, tuple) and task and task[0] == _TASK_BATCH:
                (
                    _,
                    seq_id,
                    ds_idx,
                    batch_indices,
                    filter_weight,
                    pad_graphs,
                    n_node,
                    n_edge,
                ) = task
                dataset = local_datasets.get(ds_idx)
                if dataset is None:
                    dataset = HDF5Dataset(dataset_paths[ds_idx], mode='r')
                    local_datasets[ds_idx] = dataset
                spec = dataset_specs[ds_idx]
                graphs: list[jraph.GraphsTuple] = []
                for sample_idx in batch_indices:
                    atoms = dataset[int(sample_idx)]
                    graph = _atoms_to_graph(
                        atoms=atoms,
                        spec=spec,
                        cutoff=cutoff,
                        z_table=z_table,
                        head_to_index=head_to_index,
                        niggli_reduce=niggli_reduce,
                    )
                    if filter_weight and not _has_positive_weight(graph):
                        continue
                    graphs.append(graph)
                if not graphs:
                    result_queue.put((_RESULT_SKIP, seq_id, None))
                    continue
                graph_count = len(graphs)
                local_pad_graphs = pad_graphs
                if local_pad_graphs is None:
                    local_pad_graphs = max(len(graphs) + 1, 2)
                batched = jraph.batch_np(graphs)
                result_queue.put(
                    (
                        _RESULT_DATA,
                        seq_id,
                        (
                            jraph.pad_with_graphs(
                                batched,
                                n_node=int(n_node),
                                n_edge=int(n_edge),
                                n_graph=int(local_pad_graphs),
                            ),
                            graph_count,
                        ),
                    )
                )
                continue
            if isinstance(task, tuple) and len(task) == 4:
                seq_id, ds_idx, sample_idx, filter_weight = task
            else:
                seq_id, ds_idx, sample_idx = task
            dataset = local_datasets.get(ds_idx)
            if dataset is None:
                dataset = HDF5Dataset(dataset_paths[ds_idx], mode='r')
                local_datasets[ds_idx] = dataset
            spec = dataset_specs[ds_idx]
            atoms = dataset[sample_idx]
            graph = _atoms_to_graph(
                atoms=atoms,
                spec=spec,
                cutoff=cutoff,
                z_table=z_table,
                head_to_index=head_to_index,
                niggli_reduce=niggli_reduce,
            )
            if filter_weight and not _has_positive_weight(graph):
                result_queue.put((_RESULT_SKIP, seq_id, None))
                continue
            result_queue.put((_RESULT_DATA, seq_id, graph))
    except Exception as exc:  # pragma: no cover - worker crashes are unexpected
        stop_event.set()
        result_queue.put((_RESULT_ERROR, worker_id, repr(exc)))
    finally:
        for dataset in local_datasets.values():
            dataset.close()
        result_queue.put((_RESULT_DONE, worker_id, None))


def _atoms_to_graph(
    *,
    atoms,
    spec: StreamingDatasetSpec,
    cutoff: float,
    z_table: AtomicNumberTable,
    head_to_index: dict[str, int],
    niggli_reduce: bool,
):
    """Convert an ASE atoms object into a jraph GraphsTuple.

    Args:
        atoms: ASE atoms to convert.
        spec: Dataset-specific configuration and weighting.
        cutoff: Interaction cutoff radius.
        z_table: Atomic number table for encoding species.
        head_to_index: Mapping from head names to integer indices.
        niggli_reduce: Whether to apply Niggli reduction before conversion.

    Returns:
        GraphsTuple representation of the configuration.
    """
    if niggli_reduce:
        atoms = atoms.copy()
        _niggli_reduce_inplace(atoms)
    conf = config_from_atoms(
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
    return graph_from_configuration(
        conf,
        cutoff=cutoff,
        z_table=z_table,
        head_to_index=head_to_index,
    )


def _has_positive_weight(graph: jraph.GraphsTuple) -> bool:
    """Return True if graph weight is positive or missing."""
    weight = getattr(graph.globals, 'weight', None)
    if weight is None:
        return True
    value = np.asarray(weight).reshape(-1)[0]
    return float(value) > 0.0


def _estimate_caps(
    *,
    datasets: Sequence[HDF5Dataset],
    dataset_specs: Sequence[StreamingDatasetSpec],
    z_table: AtomicNumberTable,
    r_max: float,
    head_to_index: dict[str, int],
    niggli_reduce: bool,
) -> tuple[int, int]:
    """Scan datasets to estimate max nodes/edges for padding limits."""
    max_nodes = max_edges = 1
    for dataset, spec in zip(datasets, dataset_specs):
        for idx in range(len(dataset)):
            atoms = dataset[idx]
            graph = _atoms_to_graph(
                atoms=atoms,
                spec=spec,
                cutoff=r_max,
                z_table=z_table,
                head_to_index=head_to_index,
                niggli_reduce=niggli_reduce,
            )
            if not _has_positive_weight(graph):
                continue
            nodes = int(graph.n_node.sum())
            edges = int(graph.n_edge.sum())
            max_nodes = max(max_nodes, nodes)
            max_edges = max(max_edges, edges)
    return max_nodes, max_edges


def _pack_streaming_batches(
    graph_sizes: list[tuple[int, int, int]],
    edge_cap: int,
    node_cap: int | None = None,
) -> list[list[int]]:
    """Pack graphs into batches under an edge (and optional node) budget."""
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
    z_table: AtomicNumberTable,
    r_max: float,
    head_to_index: dict[str, int],
    edge_cap: int | None,
    niggli_reduce: bool,
) -> _StreamingStats:
    """Scan a dataset to compute batch assignments and padding caps."""
    dataset = HDF5Dataset(dataset_path, mode='r')
    graph_sizes: list[tuple[int, int, int]] = []
    max_graph_edges = 0
    max_graph_nodes = 0
    try:
        for idx in range(len(dataset)):
            graph = _atoms_to_graph(
                atoms=dataset[idx],
                spec=spec,
                cutoff=r_max,
                z_table=z_table,
                head_to_index=head_to_index,
                niggli_reduce=niggli_reduce,
            )
            if not _has_positive_weight(graph):
                continue
            g_nodes = int(graph.n_node.sum())
            g_edges = int(graph.n_edge.sum())
            max_graph_edges = max(max_graph_edges, g_edges)
            max_graph_nodes = max(max_graph_nodes, g_nodes)
            graph_sizes.append((idx, g_nodes, g_edges))
    finally:
        dataset.close()

    if not graph_sizes:
        raise ValueError(f"No graphs found in '{dataset_path}'.")

    if edge_cap is None or edge_cap <= 0:
        edge_cap = max_graph_edges
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
        max_graphs_per_batch = max(max_graphs_per_batch, len(batch))
    n_nodes = max(max_nodes_per_batch + 1, 2)
    n_graphs = max(max_graphs_per_batch + 1, 2)
    return _StreamingStats(
        batch_assignments=batch_assignments,
        n_nodes=n_nodes,
        n_edges=int(edge_cap),
        n_graphs=n_graphs,
    )


def _load_or_compute_streaming_stats(
    spec: StreamingDatasetSpec,
    *,
    r_max: float,
    z_table: AtomicNumberTable,
    head_to_index: dict[str, int],
    edge_cap: int | None,
    niggli_reduce: bool,
) -> _StreamingStats:
    """Fetch streaming stats from cache or compute them if missing."""
    dataset_path = Path(spec.path)
    fingerprint = spec_fingerprint(
        spec,
        r_max=r_max,
        atomic_numbers=z_table.zs,
        head_to_index=head_to_index,
        edge_cap=edge_cap,
    )
    cached_payload = load_cached_streaming_stats(dataset_path, fingerprint)
    if cached_payload is not None:
        batch_assignments, n_nodes, n_edges, n_graphs = stats_payload_to_parts(
            cached_payload
        )
        return _StreamingStats(
            batch_assignments=batch_assignments,
            n_nodes=n_nodes,
            n_edges=n_edges,
            n_graphs=n_graphs,
        )
    stats = _compute_streaming_stats(
        dataset_path,
        spec=spec,
        z_table=z_table,
        r_max=r_max,
        head_to_index=head_to_index,
        edge_cap=edge_cap,
        niggli_reduce=niggli_reduce,
    )
    store_cached_streaming_stats(
        dataset_path,
        fingerprint,
        stats_payload_from_parts(
            stats.batch_assignments,
            stats.n_nodes,
            stats.n_edges,
            stats.n_graphs,
        ),
    )
    return stats


def _expand_hdf5_paths(paths: Sequence[Path | str]) -> list[Path]:
    """Expand glob patterns, directories, and file inputs into HDF5 paths."""
    expanded: list[Path] = []
    for raw in paths:
        raw_str = str(raw)
        if any(ch in raw_str for ch in '*?[]'):
            matches = [Path(p) for p in sorted(glob(raw_str))]
            if not matches:
                raise ValueError(f"No HDF5 files matched pattern '{raw_str}'.")
            candidates = matches
        else:
            candidates = [Path(raw_str)]
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


class StreamingGraphDataLoader:
    """Stream HDF5-backed graphs with precomputed assignments and prefetch hints.

    The key goal is stable batch shapes: JAX compiles the model per input shape,
    so changing `n_node`/`n_edge`/`n_graph` across batches can trigger repeated
    recompilations. This loader precomputes or enforces padding caps so that
    batches are uniform in shape, letting compilation happen once and keeping
    throughput stable. It also supports multiprocessing conversion and optional
    precomputed batch assignments to preserve order or enable batch-level
    shuffling without re-packing every epoch.
    """

    def __init__(
        self,
        *,
        datasets: Sequence[HDF5Dataset],
        dataset_specs: Sequence[StreamingDatasetSpec] | None = None,
        z_table: AtomicNumberTable,
        r_max: float,
        n_node: int | None,
        n_edge: int | None,
        head_to_index: dict[str, int] | None = None,
        shuffle: bool = False,
        seed: int | None = None,
        niggli_reduce: bool = False,
        max_batches: int | None = None,
        prefetch_batches: int | None = None,
        num_workers: int | None = None,
        batch_assignments: Sequence[Sequence[Sequence[int]]] | None = None,
        pad_graphs: int | None = None,
    ):
        """Initialize the streaming loader.

        Args:
            datasets: Open HDF5Dataset objects to stream from.
            dataset_specs: Per-dataset configuration overrides.
            z_table: Atomic number table to map species to indices.
            r_max: Cutoff radius used for graph construction.
            n_node: Node padding cap (None triggers a scan).
            n_edge: Edge padding cap (None triggers a scan).
            head_to_index: Optional mapping of head names to indices.
            shuffle: Whether to shuffle graphs or batches each epoch.
            seed: Seed for reproducible shuffling.
            niggli_reduce: Whether to apply Niggli reduction to periodic cells.
            max_batches: Optional cap on batches yielded per epoch.
            prefetch_batches: Prefetch depth hint for downstream training/eval loops.
            num_workers: Number of multiprocessing workers to use for conversion.
            batch_assignments: Optional precomputed batch index assignments.
            pad_graphs: Optional fixed number of graphs per padded batch.

        The loader uses precomputed batch assignments (from streaming stats)
        and pads to fixed caps so downstream JAX compilation sees stable tensor
        shapes.
        """
        if not datasets:
            raise ValueError('Expected at least one dataset.')
        self._datasets = list(datasets)
        base_paths = [Path(ds.filename) for ds in self._datasets]
        if dataset_specs is None:
            dataset_specs = [StreamingDatasetSpec(path=path) for path in base_paths]
        if len(dataset_specs) != len(self._datasets):
            raise ValueError(
                'dataset_specs must match datasets length '
                f'({len(dataset_specs)} vs {len(self._datasets)}).'
            )
        normalized_specs: list[StreamingDatasetSpec] = []
        for spec in dataset_specs:
            if isinstance(spec.path, Path):
                normalized_specs.append(spec)
            else:
                normalized_specs.append(replace(spec, path=Path(spec.path)))
        self._dataset_specs = normalized_specs
        self._dataset_paths = [Path(spec.path) for spec in self._dataset_specs]
        self._z_table = z_table
        self._head_to_index = dict(head_to_index or {'Default': 0})
        self.heads = tuple(sorted(self._head_to_index, key=self._head_to_index.get))
        self._cutoff = float(r_max)
        self._shuffle = bool(shuffle)
        self._seed = None if seed is None else int(seed)
        self._epoch = 0
        self._niggli_reduce = bool(niggli_reduce)
        self._max_batches = max_batches
        prefetched = prefetch_batches
        worker_count = int(num_workers or 0)
        if prefetched is None:
            prefetched = 2 * worker_count
        self._prefetch_batches = max(int(prefetched or 0), 0)
        self._num_workers = max(worker_count, 0)
        self._pack_info: dict | None = None
        self._history: list[tuple[int, int]] = []
        self._fixed_pad_nodes: int | None = None
        self._fixed_pad_edges: int | None = None
        self._fixed_pad_graphs: int | None = None
        self._batch_assignments: list[list[list[int]]] | None = None
        self._worker_ctx = None
        self._worker_task_queue = None
        self._worker_result_queue = None
        self._worker_stop_event = None
        self._worker_procs: list[mp.Process] | None = None
        self._worker_lock = threading.Lock()
        computed_stats: list[_StreamingStats] | None = None
        if batch_assignments is not None:
            self._batch_assignments = [
                [list(batch) for batch in batches] for batches in batch_assignments
            ]
        else:
            computed_stats = [
                _load_or_compute_streaming_stats(
                    spec,
                    r_max=self._cutoff,
                    z_table=self._z_table,
                    head_to_index=self._head_to_index,
                    edge_cap=n_edge,
                    niggli_reduce=self._niggli_reduce,
                )
                for spec in self._dataset_specs
            ]
            self._batch_assignments = [
                stats.batch_assignments for stats in computed_stats
            ]
        if pad_graphs is not None:
            self._fixed_pad_graphs = int(pad_graphs)
        # Attach optional metadata for downstream consumers.
        self.graphs = getattr(self, 'graphs', None)
        self.streaming = True
        self.total_graphs = getattr(self, 'total_graphs', None)
        self.total_nodes = getattr(self, 'total_nodes', None)
        self.total_edges = getattr(self, 'total_edges', None)

        if computed_stats is not None:
            stats_n_nodes = max(int(stats.n_nodes) for stats in computed_stats)
            stats_n_edges = max(int(stats.n_edges) for stats in computed_stats)
            stats_n_graphs = max(int(stats.n_graphs) for stats in computed_stats)
            if n_node is None:
                n_node = stats_n_nodes
            else:
                n_node = max(int(n_node), stats_n_nodes)
            if n_edge is None:
                n_edge = stats_n_edges
            else:
                n_edge = max(int(n_edge), stats_n_edges)
            if self._fixed_pad_graphs is None:
                self._fixed_pad_graphs = stats_n_graphs
            self._fixed_pad_nodes = int(n_node)
            self._fixed_pad_edges = int(n_edge)
        elif n_node is None or n_edge is None:
            est_nodes, est_edges = _estimate_caps(
                datasets=self._datasets,
                dataset_specs=self._dataset_specs,
                z_table=self._z_table,
                r_max=self._cutoff,
                head_to_index=self._head_to_index,
                niggli_reduce=self._niggli_reduce,
            )
            if n_node is None:
                n_node = est_nodes
            if n_edge is None:
                n_edge = est_edges
        if n_node is None or n_edge is None:
            raise ValueError('Failed to determine n_node and n_edge limits.')
        self._n_node = int(max(n_node, 1))
        self._n_edge = int(max(n_edge, 1))
        if self._batch_assignments is not None:
            total_batches = sum(len(batches) for batches in self._batch_assignments)
            self._pack_info = {'total_batches': total_batches}

    @property
    def epoch_history(self) -> list[tuple[int, int]]:
        """List of (graphs, batches) pairs recorded per epoch."""
        return list(self._history)

    def set_epoch(self, epoch: int) -> None:
        """Set the current epoch for deterministic shuffling."""
        self._epoch = int(epoch)

    def _effective_seed(self, override: int | None) -> int | None:
        """Compute the epoch-specific seed using optional overrides."""
        base = self._seed if override is None else override
        if base is None:
            return None
        return int(base) + int(self._epoch)

    def _convert_atoms_to_graph(self, atoms, spec: StreamingDatasetSpec):
        """Convert a raw atoms object into a graph using dataset spec settings."""
        return _atoms_to_graph(
            atoms=atoms,
            spec=spec,
            cutoff=self._cutoff,
            z_table=self._z_table,
            head_to_index=self._head_to_index,
            niggli_reduce=self._niggli_reduce,
        )

    def _iter_graphs_single_from_tasks(self, task_iter: Iterator[tuple]):
        """Yield graphs for a provided task iterator (single-process)."""
        for task in task_iter:
            if len(task) == 4:
                _, ds_idx, sample_idx, filter_weight = task
            else:
                _, ds_idx, sample_idx = task
                filter_weight = True
            atoms = self._datasets[ds_idx][sample_idx]
            spec = self._dataset_specs[ds_idx]
            graph = self._convert_atoms_to_graph(atoms, spec)
            if filter_weight and not _has_positive_weight(graph):
                continue
            yield graph

    def _iter_graphs_parallel_from_tasks(self, task_iter: Iterator[tuple]):
        """Yield graphs from tasks using multiprocessing workers."""
        return self._run_task_stream(
            task_iter,
            ordered=True,
            allow_skip=False,
            expect_all=True,
        )

    def _ensure_worker_pool(self) -> None:
        """Create worker processes and queues if needed."""
        if self._num_workers <= 1 or self._worker_procs is not None:
            return
        worker_count = max(self._num_workers, 1)
        ctx = mp.get_context('spawn')
        self._worker_ctx = ctx
        self._worker_task_queue = ctx.Queue(max(worker_count * 4, 1))
        self._worker_result_queue = ctx.Queue(max(worker_count * 4, 1))
        self._worker_stop_event = ctx.Event()
        processes: list[mp.Process] = []
        for worker_idx in range(worker_count):
            proc = ctx.Process(
                target=_graph_worker_main,
                args=(
                    worker_idx,
                    self._dataset_paths,
                    self._dataset_specs,
                    self._cutoff,
                    self._z_table,
                    self._head_to_index,
                    self._niggli_reduce,
                    self._worker_task_queue,
                    self._worker_result_queue,
                    self._worker_stop_event,
                ),
            )
            proc.daemon = True
            proc.start()
            processes.append(proc)
        self._worker_procs = processes

    def _shutdown_workers(self) -> None:
        """Terminate worker processes and release shared resources."""
        if self._worker_procs is None:
            return
        worker_count = len(self._worker_procs)
        if self._worker_stop_event is not None:
            self._worker_stop_event.set()
        if self._worker_task_queue is not None:
            for _ in range(worker_count):
                self._worker_task_queue.put(None)
        for proc in self._worker_procs:
            proc.join(timeout=1)
        self._worker_ctx = None
        self._worker_task_queue = None
        self._worker_result_queue = None
        self._worker_stop_event = None
        self._worker_procs = None

    def _run_task_stream(
        self,
        task_iter: Iterator[tuple[int, int, int]],
        *,
        ordered: bool,
        allow_skip: bool,
        expect_all: bool,
    ):
        """Stream graphs from a task iterator, optionally in parallel.

        Args:
            task_iter: Iterator yielding (seq_id, dataset_idx, sample_idx) tuples.
            ordered: Whether to preserve task ordering in output graphs.
            allow_skip: Whether workers may skip graphs (e.g., zero-weight).
            expect_all: Whether to warn if fewer graphs than tasks are produced.

        Returns:
            Iterator over `jraph.GraphsTuple` items.
        """
        if self._num_workers <= 1:
            return self._iter_graphs_single_from_tasks(task_iter)
        self._ensure_worker_pool()

        def _iter():
            """Drive producer/consumer queues and yield graphs to callers."""
            self._worker_lock.acquire()
            worker_count = len(self._worker_procs or [])
            task_queue = self._worker_task_queue
            result_queue = self._worker_result_queue
            stop_event = self._worker_stop_event
            total_tasks = {'value': 0}
            producer_done = threading.Event()
            producer_exc: dict[str, Exception | None] = {'value': None}

            def _producer():
                """Feed tasks into the worker queue."""
                try:
                    for task in task_iter:
                        if stop_event is not None and stop_event.is_set():
                            break
                        task_queue.put(task)
                        total_tasks['value'] += 1
                except Exception as exc:
                    producer_exc['value'] = exc
                    if stop_event is not None:
                        stop_event.set()
                finally:
                    producer_done.set()

            producer = threading.Thread(target=_producer, daemon=True)
            producer.start()

            finished_workers = 0
            processed = 0
            produced = 0
            next_seq = 0
            pending: dict[int, jraph.GraphsTuple] = {}
            skipped: set[int] | None = set() if allow_skip and ordered else None

            def _advance_pending():
                """Emit buffered graphs in sequence order when available."""
                nonlocal next_seq
                while True:
                    if skipped is not None and next_seq in skipped:
                        skipped.remove(next_seq)
                        next_seq += 1
                        continue
                    graph = pending.pop(next_seq, None)
                    if graph is None:
                        break
                    yield graph
                    next_seq += 1

            def _check_worker_health():
                """Detect and surface worker process failures."""
                if self._worker_procs is None:
                    return
                for idx, proc in enumerate(self._worker_procs):
                    if proc.is_alive():
                        continue
                    exit_code = proc.exitcode
                    if exit_code is None:
                        continue
                    if stop_event is not None:
                        stop_event.set()
                    self._shutdown_workers()
                    raise RuntimeError(
                        f'Graph worker {idx} exited unexpectedly with code {exit_code}.'
                    )

            try:
                while True:
                    if producer_done.is_set() and processed >= total_tasks['value']:
                        break
                    if finished_workers >= worker_count:
                        break
                    try:
                        tag, payload_a, payload_b = result_queue.get(timeout=1.0)
                    except Empty:
                        if producer_done.is_set() and processed >= total_tasks['value']:
                            break
                        if producer_exc['value'] is not None and producer_done.is_set():
                            break
                        _check_worker_health()
                        continue
                    if tag == _RESULT_DONE:
                        finished_workers += 1
                        continue
                    if tag == _RESULT_ERROR:
                        if stop_event is not None:
                            stop_event.set()
                        self._shutdown_workers()
                        raise RuntimeError(
                            f'Graph worker {payload_a} failed: {payload_b}'
                        )
                    seq_id = payload_a
                    if tag == _RESULT_SKIP:
                        if not allow_skip:
                            raise RuntimeError(
                                'Graph worker skipped a graph unexpectedly.'
                            )
                        processed += 1
                        if not ordered:
                            continue
                        if seq_id == next_seq:
                            next_seq += 1
                            for graph in _advance_pending():
                                yield graph
                        else:
                            if skipped is not None:
                                skipped.add(seq_id)
                        continue
                    graph = payload_b
                    processed += 1
                    produced += 1
                    if not ordered:
                        yield graph
                        continue
                    if seq_id == next_seq:
                        yield graph
                        next_seq += 1
                        yield from _advance_pending()
                    else:
                        pending[seq_id] = graph
                if producer_exc['value'] is not None:
                    raise RuntimeError('Task iterator failed.') from producer_exc[
                        'value'
                    ]
                if expect_all and producer_done.is_set():
                    expected = total_tasks['value']
                    if produced < expected:
                        logging.warning(
                            'Assignment stream produced fewer graphs than expected '
                            '(%s/%s).',
                            produced,
                            expected,
                        )
                if ordered and producer_done.is_set():
                    final_total = total_tasks['value']
                    while next_seq < final_total:
                        if skipped is not None and next_seq in skipped:
                            skipped.remove(next_seq)
                            next_seq += 1
                            continue
                        graph = pending.pop(next_seq, None)
                        if graph is None:
                            break
                        yield graph
                        next_seq += 1
                    if pending:
                        for seq_id in sorted(pending):
                            if seq_id >= next_seq:
                                yield pending[seq_id]
            finally:
                producer.join(timeout=1)
                if not producer_done.is_set() or processed < total_tasks['value']:
                    if stop_event is not None:
                        stop_event.set()
                    if self._worker_procs is not None:
                        self._shutdown_workers()
                self._worker_lock.release()

        return _iter()

    def _graph_iterator_from_tasks(self, task_iter: Iterator[tuple[int, int, int]]):
        """Select the appropriate task iterator backend (single vs parallel)."""
        if self._num_workers <= 1:
            return self._iter_graphs_single_from_tasks(task_iter)
        return self._iter_graphs_parallel_from_tasks(task_iter)

    def _ordered_batches(self, *, seed_override: int | None):
        """Return ordered batch plans derived from precomputed assignments."""
        if self._batch_assignments is None:
            return None
        seed = self._effective_seed(seed_override)
        batches: list[tuple[int, list[int]]] = []
        if self._shuffle:
            for ds_idx, ds_batches in enumerate(self._batch_assignments):
                for batch in ds_batches:
                    batches.append((ds_idx, list(batch)))
            if len(batches) > 1:
                rng = np.random.default_rng(seed)
                rng.shuffle(batches)
            return batches

        for ds_idx, ds_batches in enumerate(self._batch_assignments):
            if not ds_batches:
                continue
            graph_to_batch: dict[int, int] = {}
            for batch_id, batch in enumerate(ds_batches):
                for graph_idx in batch:
                    graph_to_batch[graph_idx] = batch_id
            if not graph_to_batch:
                continue
            current_batch_id = None
            current_graphs: list[int] = []
            for graph_idx in sorted(graph_to_batch):
                batch_id = graph_to_batch[graph_idx]
                if current_batch_id is None:
                    current_batch_id = batch_id
                if batch_id != current_batch_id:
                    if current_graphs:
                        batches.append((ds_idx, current_graphs))
                    current_batch_id = batch_id
                    current_graphs = []
                current_graphs.append(graph_idx)
            if current_graphs:
                batches.append((ds_idx, current_graphs))
        return batches

    def _task_iterator_from_batches(
        self, batches: Sequence[tuple[int, Sequence[int]]]
    ) -> Iterator[tuple[int, int, int]]:
        """Yield task tuples for explicit graph index batches."""
        seq = 0
        for ds_idx, graph_indices in batches:
            for graph_idx in graph_indices:
                yield seq, ds_idx, int(graph_idx), False
                seq += 1

    def _batch_task_iterator_from_batches(
        self,
        batches: Sequence[tuple[int, Sequence[int]]],
        *,
        pad_graphs: int | None,
        filter_weight: bool,
    ) -> Iterator[tuple]:
        """Yield batch tasks to offload batching/padding into workers."""
        seq = 0
        for ds_idx, graph_indices in batches:
            if not graph_indices:
                continue
            yield (
                _TASK_BATCH,
                seq,
                ds_idx,
                [int(idx) for idx in graph_indices],
                filter_weight,
                pad_graphs,
                self._n_node,
                self._n_edge,
            )
            seq += 1

    def _iter_batches_from_assignments(
        self,
        *,
        seed_override: int | None = None,
        include_counts: bool = False,
    ):
        """Iterate batches based on precomputed graph assignments."""
        batch_plan = self._ordered_batches(seed_override=seed_override)
        if not batch_plan:
            return iter(())
        pad_graphs = self._fixed_pad_graphs
        if self._num_workers > 1:
            task_iter = self._batch_task_iterator_from_batches(
                batch_plan,
                pad_graphs=pad_graphs,
                filter_weight=False,
            )
            graph_iter = self._run_task_stream(
                task_iter,
                ordered=True,
                allow_skip=False,
                expect_all=True,
            )
            if include_counts:
                return graph_iter

            def _iter():
                for item in graph_iter:
                    if (
                        isinstance(item, tuple)
                        and len(item) == 2
                        and isinstance(item[0], jraph.GraphsTuple)
                    ):
                        yield item[0]
                    else:
                        yield item

            return _iter()
        task_iter = self._task_iterator_from_batches(batch_plan)
        graph_iter = self._graph_iterator_from_tasks(task_iter)

        def _iter():
            """Yield padded batches following the assignment plan."""
            for _, graph_indices in batch_plan:
                graphs: list[jraph.GraphsTuple] = []
                for _ in range(len(graph_indices)):
                    try:
                        graphs.append(next(graph_iter))
                    except StopIteration:
                        if graphs:
                            logging.warning(
                                'Assignment stream ended early; yielding partial batch '
                                '(%s/%s graphs).',
                                len(graphs),
                                len(graph_indices),
                            )
                            local_pad_graphs = pad_graphs
                            if local_pad_graphs is None:
                                local_pad_graphs = max(len(graphs) + 1, 2)
                            batched = jraph.batch_np(graphs)
                            batch = jraph.pad_with_graphs(
                                batched,
                                n_node=self._n_node,
                                n_edge=self._n_edge,
                                n_graph=local_pad_graphs,
                            )
                            if include_counts:
                                yield batch, len(graphs)
                            else:
                                yield batch
                        return
                if not graphs:
                    continue
                local_pad_graphs = pad_graphs
                if local_pad_graphs is None:
                    local_pad_graphs = max(len(graphs) + 1, 2)
                batched = jraph.batch_np(graphs)
                batch = jraph.pad_with_graphs(
                    batched,
                    n_node=self._n_node,
                    n_edge=self._n_edge,
                    n_graph=local_pad_graphs,
                )
                if include_counts:
                    yield batch, len(graphs)
                else:
                    yield batch

        return _iter()

    def _limit_batches(self, iterable: Iterable[jraph.GraphsTuple]):
        """Yield at most `max_batches` items from an iterable."""
        if self._max_batches is None:
            yield from iterable
            return
        produced = 0
        for item in iterable:
            if produced >= self._max_batches:
                break
            yield item
            produced += 1

    def __iter__(self):
        """Iterate over batches for the current epoch."""
        if self._batch_assignments is None:
            raise RuntimeError('Streaming loader requires precomputed assignments.')
        batches = self._iter_batches_from_assignments(seed_override=None)
        limited_iter = self._limit_batches(batches)
        yield from limited_iter

    def iter_batches(
        self,
        *,
        epoch: int,
        seed: int | None,
        process_count: int,
        process_index: int,
    ) -> Iterator[jraph.GraphsTuple]:
        """Yield batches for the given epoch and process shard.

        Args:
            epoch: Epoch index to seed shuffling.
            seed: Optional override seed.
            process_count: Total number of data-parallel processes.
            process_index: Index of the current process.

        Returns:
            Iterator over padded graph batches.
        """
        self.set_epoch(epoch)
        previous_info = getattr(self, '_pack_info', None)
        if self._batch_assignments is None:
            raise RuntimeError('Streaming loader requires precomputed assignments.')
        batches_iter = self._iter_batches_from_assignments(
            seed_override=seed,
            include_counts=True,
        )
        info = {'total_batches': len(self._ordered_batches(seed_override=seed) or [])}
        limited_iter = self._limit_batches(batches_iter)
        if process_count > 1:

            def _filtered():
                """Yield only batches assigned to this process."""
                graphs_count = 0
                batches_count = 0
                try:
                    for idx, item in enumerate(limited_iter):
                        if idx % process_count != process_index:
                            continue
                        if (
                            isinstance(item, tuple)
                            and len(item) == 2
                            and isinstance(item[0], jraph.GraphsTuple)
                        ):
                            batch, graph_count = item
                            graphs_count += int(graph_count)
                        else:
                            batch = item
                            graphs_count += int(
                                np.asarray(jraph.get_graph_padding_mask(batch)).sum()
                            )
                        batches_count += 1
                        yield batch
                finally:
                    self._history.append((graphs_count, batches_count))

            iterator = _filtered()
        else:

            def _single():
                """Yield all batches and record history."""
                graphs_count = 0
                batches_count = 0
                try:
                    for item in limited_iter:
                        if (
                            isinstance(item, tuple)
                            and len(item) == 2
                            and isinstance(item[0], jraph.GraphsTuple)
                        ):
                            batch, graph_count = item
                            graphs_count += int(graph_count)
                        else:
                            batch = item
                            graphs_count += int(
                                np.asarray(jraph.get_graph_padding_mask(batch)).sum()
                            )
                        batches_count += 1
                        yield batch
                finally:
                    self._history.append((graphs_count, batches_count))

            iterator = _single()

        total_batches_hint = int(info.get('total_batches') or 0)
        if total_batches_hint <= 0 and previous_info:
            total_batches_hint = int(previous_info.get('total_batches') or 0)
        return BatchIteratorWrapper(iterator, total_batches_hint)

    def __len__(self):
        """Return the number of batches for the current packing configuration."""
        if self._pack_info and self._pack_info.get('total_batches') is not None:
            total = int(self._pack_info['total_batches'])
            if self._max_batches is not None:
                return min(total, int(self._max_batches))
            return total
        if self._batch_assignments is None:
            raise RuntimeError('Streaming loader requires precomputed assignments.')
        total = len(self._ordered_batches(seed_override=None) or [])
        if self._max_batches is not None:
            return min(total, int(self._max_batches))
        return total

    def pack_info(self) -> dict:
        """Return cached packing metadata."""
        if not self._pack_info:
            if self._batch_assignments is None:
                return {}
            return {
                'total_batches': sum(
                    len(batches) for batches in self._batch_assignments
                )
            }
        return dict(self._pack_info)

    def close(self) -> None:
        """Close datasets and terminate worker processes."""
        for dataset in self._datasets:
            dataset.close()
        self._shutdown_workers()

    def approx_length(self) -> int:
        """Estimate number of batches without forcing a prepass."""
        if self._pack_info and self._pack_info.get('total_batches') is not None:
            total = int(self._pack_info['total_batches'])
            if self._max_batches is not None:
                return max(1, min(total, int(self._max_batches)))
            return max(1, total)
        if self._batch_assignments is None:
            raise RuntimeError('Streaming loader requires precomputed assignments.')
        approx = len(self._ordered_batches(seed_override=None) or [])
        if self._max_batches is not None:
            approx = min(approx, int(self._max_batches))
        return max(1, approx)

    def subset(self, i):
        """Approximate GraphDataLoader.subset using cached initialization graphs."""
        cached = list(getattr(self, 'graphs', None) or [])
        if isinstance(i, slice):
            graphs = cached[i]
        elif isinstance(i, list):
            graphs = [cached[j] for j in i]
        elif isinstance(i, float):
            count = max(int(len(cached) * i), 0)
            graphs = cached[:count]
        elif isinstance(i, int):
            graphs = cached[: max(i, 0)]
        else:
            raise TypeError(f'Unsupported subset specifier: {i!r}')
        head_names = tuple(sorted(self._head_to_index, key=self._head_to_index.get))
        pad_graphs = max(int(self._fixed_pad_graphs or 2), 2)
        return GraphDataLoader(
            graphs=graphs,
            n_node=self._n_node,
            n_edge=self._n_edge,
            n_graph=pad_graphs,
            shuffle=False,
            heads=head_names or None,
        )

    def split_by_heads(self) -> dict[str, StreamingGraphDataLoader]:
        """Split the loader into per-head StreamingGraphDataLoader instances."""
        if len(self._head_to_index) <= 1:
            return {}
        grouped: dict[str, list[StreamingDatasetSpec]] = {}
        for spec in self._dataset_specs:
            grouped.setdefault(spec.head_name, []).append(spec)
        result: dict[str, StreamingGraphDataLoader] = {}
        assignments = self._batch_assignments
        spec_to_assignment: dict[int, list[list[int]]] = {}
        if assignments is not None:
            for idx, batches in enumerate(assignments):
                spec_to_assignment[idx] = batches
        for head_name, specs in grouped.items():
            datasets = [HDF5Dataset(spec.path, mode='r') for spec in specs]
            head_assignments = None
            if assignments is not None:
                head_assignments = [
                    spec_to_assignment[idx]
                    for idx, spec in enumerate(self._dataset_specs)
                    if spec.head_name == head_name
                ]
            loader = StreamingGraphDataLoader(
                datasets=datasets,
                dataset_specs=specs,
                z_table=self._z_table,
                r_max=self._cutoff,
                n_node=self._n_node,
                n_edge=self._n_edge,
                head_to_index={head_name: self._head_to_index[head_name]},
                shuffle=self._shuffle,
                seed=self._seed,
                niggli_reduce=self._niggli_reduce,
                max_batches=self._max_batches,
                prefetch_batches=self._prefetch_batches,
                num_workers=self._num_workers,
                batch_assignments=head_assignments,
                pad_graphs=self._fixed_pad_graphs,
            )
            cached_graphs = getattr(self, 'graphs', None)
            if cached_graphs is not None:
                expected_head = self._head_to_index[head_name]
                head_graphs: list[jraph.GraphsTuple] = []
                for graph in cached_graphs:
                    head_value = getattr(graph.globals, 'head', None)
                    if head_value is None:
                        continue
                    head_idx = int(np.asarray(head_value).reshape(-1)[0])
                    if head_idx == expected_head:
                        head_graphs.append(graph)
                loader.graphs = head_graphs
            else:
                loader.graphs = None
            loader.streaming = getattr(self, 'streaming', True)
            loader.total_graphs = getattr(self, 'total_graphs', None)
            result[head_name] = loader
        return result


def get_hdf5_dataloader(
    *,
    data_file: Path | str | Sequence[Path | str],
    atomic_numbers: AtomicNumberTable,
    r_max: float,
    shuffle: bool,
    max_nodes: int,
    max_edges: int,
    seed: int | None = None,
    niggli_reduce: bool = False,
    max_batches: int | None = None,
    prefetch_batches: int | None = None,
    num_workers: int | None = None,
    dataset_specs: Sequence[StreamingDatasetSpec] | None = None,
    head_to_index: dict[str, int] | None = None,
) -> StreamingGraphDataLoader:
    """Create a StreamingGraphDataLoader from one or more HDF5 files.

    The resulting loader yields fixed-shape padded batches suitable for JAX
    compilation reuse. This wrapper expands paths/globs, opens datasets, and
    forwards options to `StreamingGraphDataLoader`.

    Args:
        data_file: Path(s), directory, or glob patterns for HDF5 files.
        atomic_numbers: Atomic number table for species encoding.
        r_max: Cutoff radius for graph construction.
        shuffle: Whether to shuffle graphs/batches each epoch.
        max_nodes: Node padding cap.
        max_edges: Edge padding cap.
        seed: Optional shuffle seed.
        niggli_reduce: Whether to apply Niggli reduction to periodic cells.
        max_batches: Optional cap on batches per epoch.
        prefetch_batches: Prefetch depth hint for downstream training/eval loops.
        num_workers: Number of multiprocessing workers for graph conversion.
        dataset_specs: Optional per-file dataset specifications.
        head_to_index: Optional mapping of head names to indices.

    Returns:
        Configured StreamingGraphDataLoader instance.
    """
    if data_file is None:
        raise ValueError('data_file must be provided.')
    if isinstance(data_file, (list, tuple)):
        files = _expand_hdf5_paths(data_file)
    else:
        files = _expand_hdf5_paths([data_file])
    datasets = [HDF5Dataset(path, mode='r') for path in files]
    if dataset_specs is None:
        dataset_specs = [StreamingDatasetSpec(path=path) for path in files]
    return StreamingGraphDataLoader(
        datasets=datasets,
        dataset_specs=dataset_specs,
        z_table=atomic_numbers,
        r_max=r_max,
        n_node=max_nodes,
        n_edge=max_edges,
        head_to_index=head_to_index,
        shuffle=shuffle,
        seed=seed,
        niggli_reduce=niggli_reduce,
        max_batches=max_batches,
        prefetch_batches=prefetch_batches,
        num_workers=num_workers,
    )


__all__ = [
    'StreamingDatasetSpec',
    'StreamingGraphDataLoader',
    'get_hdf5_dataloader',
]
