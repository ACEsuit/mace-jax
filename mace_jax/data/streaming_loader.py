"""Streaming HDF5 data loader for native MACE datasets.

This module provides fixed-shape batch packing for JAX by reading MACE-style
HDF5 shards and padding batches to (n_node, n_edge, n_graph) caps. Graphs are
tagged with a stable global graph_id so prediction outputs can be reordered to
match the original HDF5 order. The loader supports deterministic per-epoch
shuffling, per-process round-robin sharding for distributed runs, and optional
multi-process workers that build batches directly from the HDF5 files.

Primary API:
- StreamingDatasetSpec: per-shard metadata (head name, property keys, weights).
- StreamingGraphDataLoader: produces padded jraph.GraphsTuple batches.
- get_hdf5_dataloader: convenience constructor that expands paths.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import threading
from bisect import bisect_right
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, replace
from glob import glob
from pathlib import Path
from queue import Empty, Full

import jraph
import numpy as np

from .hdf5_dataset import HDF5Dataset
from .utils import AtomicNumberTable, config_from_atoms, graph_from_configuration

_RESULT_BATCH = 'batch'
_RESULT_DONE = 'done'
_RESULT_ERROR = 'error'


class BatchIteratorWrapper:
    """Wrap an iterator and expose a total_batches_hint attribute."""

    def __init__(self, iterator, total_batches_hint: int):
        """Initialize the wrapper with a source iterator and batch hint."""
        self._iterator = iterator
        self.total_batches_hint = int(total_batches_hint or 0)
        self._lock = threading.Lock()

    def __iter__(self):
        """Return self as an iterator."""
        return self

    def __next__(self):
        """Yield the next batch from the wrapped iterator."""
        with self._lock:
            return next(self._iterator)


@dataclass(frozen=True)
class StreamingDatasetSpec:
    """Configuration describing one HDF5 dataset stream.

    Each spec maps a single HDF5 file to a model head and defines which property
    keys and weights to extract when converting atoms -> graphs.
    """

    path: Path
    head_name: str = 'Default'
    config_type_weights: dict[str, float] | None = None
    energy_key: str = 'energy'
    forces_key: str = 'forces'
    stress_key: str = 'stress'
    prefactor_stress: float = 1.0
    remap_stress: np.ndarray | None = None
    weight: float = 1.0


def _niggli_reduce_inplace(atoms):
    """Apply Niggli reduction to ASE atoms if periodic and available."""
    try:
        from ase.build.tools import niggli_reduce as _niggli_reduce  # noqa: PLC0415
    except ImportError:  # pragma: no cover - ase is an install dependency
        return atoms
    pbc = getattr(atoms, 'pbc', None)
    if pbc is None or not np.any(pbc):
        return atoms
    _niggli_reduce(atoms)
    return atoms


def _atoms_to_graph(
    *,
    atoms,
    spec: StreamingDatasetSpec,
    cutoff: float,
    z_table: AtomicNumberTable,
    head_to_index: dict[str, int],
    niggli_reduce: bool,
):
    """Convert an ASE atoms object into a jraph GraphsTuple."""
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


def _with_graph_id(graph: jraph.GraphsTuple, graph_id: int) -> jraph.GraphsTuple:
    """Attach a per-graph identifier to globals for optional reordering."""
    globals_attr = getattr(graph, 'globals', None)
    if globals_attr is None or not hasattr(globals_attr, '_replace'):
        return graph
    graph_id_arr = np.asarray([int(graph_id)], dtype=np.int64)
    return graph._replace(globals=globals_attr._replace(graph_id=graph_id_arr))


def _mark_padding_graph_ids(
    graph: jraph.GraphsTuple, graph_count: int
) -> jraph.GraphsTuple:
    """Set graph_id=-1 for padded graphs so predictions can filter them out."""
    globals_attr = getattr(graph, 'globals', None)
    if globals_attr is None:
        return graph
    graph_ids = getattr(globals_attr, 'graph_id', None)
    if graph_ids is None:
        return graph
    graph_ids = np.asarray(graph_ids)
    if graph_ids.ndim == 0:
        return graph
    if graph_ids.shape[0] <= int(graph_count):
        return graph
    graph_ids = graph_ids.copy()
    graph_ids[int(graph_count) :] = -1
    if hasattr(globals_attr, '_replace'):
        globals_attr = globals_attr._replace(graph_id=graph_ids)
    elif hasattr(globals_attr, 'items'):
        globals_attr = dict(globals_attr)
        globals_attr['graph_id'] = graph_ids
    else:
        return graph
    return graph._replace(globals=globals_attr)


def _pack_sizes_by_edge_cap(
    graph_sizes: list[tuple[int, int]],
    edge_cap: int,
) -> list[dict[str, int]]:
    """Greedily pack (nodes, edges) sizes into batches under an edge budget."""
    if not graph_sizes:
        return []

    batches: list[dict[str, int]] = []
    order = sorted(graph_sizes, key=lambda item: item[1], reverse=True)
    for nodes, edges in order:
        placed = False
        for batch in batches:
            if batch['edge_sum'] + edges <= edge_cap:
                batch['edge_sum'] += edges
                batch['node_sum'] += nodes
                batch['graph_count'] += 1
                placed = True
                break
        if not placed:
            batches.append({'edge_sum': edges, 'node_sum': nodes, 'graph_count': 1})
    return batches


def _scan_caps_for_datasets(
    *,
    dataset_paths: Sequence[Path],
    dataset_specs: Sequence[StreamingDatasetSpec],
    z_table: AtomicNumberTable,
    r_max: float,
    head_to_index: dict[str, int],
    edge_cap: int | None,
    niggli_reduce: bool,
) -> tuple[int, int, int]:
    """Scan datasets to determine padding caps when none are provided.

    This performs a full read of the input shards to estimate node/edge/graph
    caps using greedy packing under an edge budget. It is a fallback path; in
    typical training runs the caps are computed once via streaming stats and
    cached (see streaming_stats_cache).
    """
    max_nodes = 1
    max_edges = 1
    max_graphs = 1
    for spec in dataset_specs:
        dataset = HDF5Dataset(spec.path, mode='r')
        try:
            graph_sizes: list[tuple[int, int]] = []
            max_graph_edges = 0
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
                graph_sizes.append((g_nodes, g_edges))
            local_edge_cap = edge_cap
            if local_edge_cap is None or local_edge_cap <= 0:
                local_edge_cap = max_graph_edges
            if local_edge_cap <= 0:
                local_edge_cap = 1
            if max_graph_edges > local_edge_cap:
                logging.warning(
                    'Requested max edges per batch (%s) is below the largest graph (%s) '
                    'in %s. Raising the limit to fit.',
                    local_edge_cap,
                    max_graph_edges,
                    Path(spec.path).name,
                )
                local_edge_cap = max_graph_edges
            batches = _pack_sizes_by_edge_cap(graph_sizes, int(local_edge_cap))
            max_nodes_per_batch = 0
            max_graphs_per_batch = 0
            for batch in batches:
                max_nodes_per_batch = max(max_nodes_per_batch, batch['node_sum'])
                max_graphs_per_batch = max(max_graphs_per_batch, batch['graph_count'])
            max_nodes = max(max_nodes, max_nodes_per_batch + 1, 2)
            max_edges = max(max_edges, int(local_edge_cap))
            max_graphs = max(max_graphs, max_graphs_per_batch + 1, 2)
        finally:
            dataset.close()
    return max_nodes, max(max_edges, 1), max(max_graphs, 2)


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


def _graph_worker_main(
    worker_id: int,
    *,
    index_queue,
    result_queue,
    stop_event,
    dataset_specs: Sequence[StreamingDatasetSpec],
    dataset_offsets: Sequence[int],
    dataset_lengths: Sequence[int],
    z_table: AtomicNumberTable,
    r_max: float,
    head_to_index: dict[str, int],
    niggli_reduce: bool,
    n_node: int,
    n_edge: int,
    n_graph: int,
    **_,
):
    """Convert atoms to graphs, pack into batches, and return padded batches."""
    graphs: list[jraph.GraphsTuple] = []
    nodes_sum = 0
    edges_sum = 0
    graph_count = 0
    max_graphs = max(int(n_graph) - 1, 1)
    datasets = [HDF5Dataset(spec.path, mode='r') for spec in dataset_specs]
    total_graphs = int(sum(int(length) for length in dataset_lengths))

    def _result_put(item):
        result_queue.put(item)

    def _flush():
        nonlocal graphs, nodes_sum, edges_sum, graph_count
        if not graphs:
            return
        batched = jraph.batch_np(graphs)
        batch = jraph.pad_with_graphs(
            batched,
            n_node=int(n_node),
            n_edge=int(n_edge),
            n_graph=int(n_graph),
        )
        batch = _mark_padding_graph_ids(batch, graph_count)
        _result_put((_RESULT_BATCH, batch, graph_count))
        graphs = []
        nodes_sum = 0
        edges_sum = 0
        graph_count = 0

    try:
        if total_graphs <= 0:
            return
        while True:
            if stop_event is not None and stop_event.is_set():
                break
            try:
                graph_id = index_queue.get(timeout=1.0)
            except Empty:
                continue
            if graph_id is None:
                break
            graph_id = int(graph_id)
            if graph_id < 0 or graph_id >= total_graphs:
                continue
            ds_idx = bisect_right(dataset_offsets, graph_id) - 1
            ds_idx = max(ds_idx, 0)
            local_idx = graph_id - int(dataset_offsets[ds_idx])
            if local_idx < 0 or local_idx >= int(dataset_lengths[ds_idx]):
                continue
            atoms = datasets[ds_idx][int(local_idx)]
            spec = dataset_specs[int(ds_idx)]
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
            graph = _with_graph_id(graph, graph_id)
            nodes = int(graph.n_node.sum())
            edges = int(graph.n_edge.sum())
            if nodes >= int(n_node) or edges > int(n_edge):
                _result_put(
                    (
                        _RESULT_ERROR,
                        worker_id,
                        f'Graph exceeds padding limits (nodes={nodes} edges={edges}, '
                        f'caps n_node={n_node} n_edge={n_edge}).',
                    )
                )
                if stop_event is not None:
                    stop_event.set()
                break
            if graphs and (
                nodes_sum + nodes >= int(n_node)
                or edges_sum + edges > int(n_edge)
                or graph_count >= max_graphs
            ):
                _flush()
            graphs.append(graph)
            graph_count += 1
            nodes_sum += nodes
            edges_sum += edges
            if graph_count >= max_graphs:
                _flush()
    except Exception as exc:  # pragma: no cover - worker crashes are unexpected
        _result_put((_RESULT_ERROR, worker_id, repr(exc)))
        if stop_event is not None:
            stop_event.set()
    finally:
        for dataset in datasets:
            dataset.close()
        if graphs:
            _flush()
        _result_put((_RESULT_DONE, worker_id, None))


class StreamingGraphDataLoader:
    """Stream HDF5-backed graphs with fixed-size padded batches.

    The loader assigns each graph a stable global graph_id (based on file order),
    packs graphs into batches until a padding cap is reached, and then pads the
    batch to fixed shapes suitable for JAX/XLA compilation. It supports optional
    deterministic shuffling and per-process sharding (round-robin by graph_id).

    Key methods:
    - iter_batches(epoch, seed, process_count, process_index): yield batches for
      a specific epoch and process shard.
    - approx_length(): rough batch count estimate without a prepass.
    - split_by_heads(): produce per-head loaders for multi-head datasets.
    - close(): release dataset resources.
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
        niggli_reduce: bool = False,
        max_batches: int | None = None,
        prefetch_batches: int | None = None,
        num_workers: int | None = None,
        pad_graphs: int | None = None,
        shuffle: bool = False,
    ):
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
        self._niggli_reduce = bool(niggli_reduce)
        self._max_batches = max_batches
        self._shuffle = bool(shuffle)
        worker_count = int(num_workers or 0)
        worker_count = max(0, worker_count)
        self._num_workers = worker_count
        prefetched = prefetch_batches
        if prefetched is None:
            prefetched = 10 * max(worker_count, 1)
        self._prefetch_batches = max(int(prefetched or 0), 0)
        self._dataset_offsets: list[int] = []
        self._dataset_lengths: list[int] = []
        offset = 0
        for dataset in self._datasets:
            length = len(dataset)
            self._dataset_offsets.append(offset)
            self._dataset_lengths.append(length)
            offset += length

        if n_node is None or n_edge is None or pad_graphs is None:
            est_nodes, est_edges, est_graphs = _scan_caps_for_datasets(
                dataset_paths=self._dataset_paths,
                dataset_specs=self._dataset_specs,
                z_table=self._z_table,
                r_max=self._cutoff,
                head_to_index=self._head_to_index,
                edge_cap=n_edge,
                niggli_reduce=self._niggli_reduce,
            )
            if n_node is None:
                n_node = est_nodes
            if n_edge is None:
                n_edge = est_edges
            if pad_graphs is None:
                pad_graphs = est_graphs

        if n_node is None or n_edge is None or pad_graphs is None:
            raise ValueError('Failed to determine n_node, n_edge, and n_graph limits.')

        self._n_node = int(max(n_node, 1))
        self._n_edge = int(max(n_edge, 1))
        self._n_graph = int(max(pad_graphs, 2))
        self._fixed_pad_nodes = int(self._n_node)
        self._fixed_pad_edges = int(self._n_edge)
        self._fixed_pad_graphs = int(self._n_graph)
        self._last_padding_summary: dict[str, int] | None = None

        for dataset in self._datasets:
            dataset.close()
        self._datasets = []

        self.graphs = getattr(self, 'graphs', None)
        self.streaming = True
        self.total_graphs = getattr(self, 'total_graphs', None)
        self.total_nodes = getattr(self, 'total_nodes', None)
        self.total_edges = getattr(self, 'total_edges', None)

    def _graph_ids_for_epoch(
        self,
        *,
        epoch: int,
        seed: int | None,
        process_count: int,
        process_index: int,
    ) -> Sequence[int]:
        total_graphs = int(sum(self._dataset_lengths))
        if total_graphs <= 0:
            return ()
        if not self._shuffle:
            return range(process_index, total_graphs, process_count)
        seed_value = int(seed or 0) + int(epoch)
        rng = np.random.default_rng(seed_value)
        indices = np.arange(total_graphs, dtype=np.int64)
        rng.shuffle(indices)
        if process_count > 1:
            indices = indices[process_index::process_count]
        return indices

    def _iter_single_process(
        self,
        *,
        graph_ids: Sequence[int],
    ) -> Iterator[tuple[jraph.GraphsTuple, int]]:
        """Yield batches and graph counts without multiprocessing."""
        max_graphs = max(int(self._n_graph) - 1, 1)
        graphs: list[jraph.GraphsTuple] = []
        nodes_sum = 0
        edges_sum = 0
        graph_count = 0

        def _flush():
            nonlocal graphs, nodes_sum, edges_sum, graph_count
            if not graphs:
                return
            batched = jraph.batch_np(graphs)
            batch = jraph.pad_with_graphs(
                batched,
                n_node=self._n_node,
                n_edge=self._n_edge,
                n_graph=self._n_graph,
            )
            batch = _mark_padding_graph_ids(batch, graph_count)
            result = (batch, graph_count)
            graphs = []
            nodes_sum = 0
            edges_sum = 0
            graph_count = 0
            return result

        datasets = [HDF5Dataset(spec.path, mode='r') for spec in self._dataset_specs]
        try:
            for graph_id in graph_ids:
                ds_idx = bisect_right(self._dataset_offsets, int(graph_id)) - 1
                ds_idx = max(ds_idx, 0)
                local_idx = int(graph_id) - int(self._dataset_offsets[ds_idx])
                if local_idx < 0 or local_idx >= int(self._dataset_lengths[ds_idx]):
                    continue
                atoms = datasets[ds_idx][int(local_idx)]
                spec = self._dataset_specs[ds_idx]
                graph = _atoms_to_graph(
                    atoms=atoms,
                    spec=spec,
                    cutoff=self._cutoff,
                    z_table=self._z_table,
                    head_to_index=self._head_to_index,
                    niggli_reduce=self._niggli_reduce,
                )
                if not _has_positive_weight(graph):
                    continue
                graph = _with_graph_id(graph, int(graph_id))
                nodes = int(graph.n_node.sum())
                edges = int(graph.n_edge.sum())
                if nodes >= self._n_node or edges > self._n_edge:
                    raise ValueError(
                        'Graph exceeds padding limits '
                        f'(nodes={nodes} edges={edges}, '
                        f'caps n_node={self._n_node} n_edge={self._n_edge}).'
                    )
                if graphs and (
                    nodes_sum + nodes >= self._n_node
                    or edges_sum + edges > self._n_edge
                    or graph_count >= max_graphs
                ):
                    flushed = _flush()
                    if flushed is not None:
                        yield flushed
                graphs.append(graph)
                graph_count += 1
                nodes_sum += nodes
                edges_sum += edges
                if graph_count >= max_graphs:
                    flushed = _flush()
                    if flushed is not None:
                        yield flushed
        finally:
            for dataset in datasets:
                dataset.close()
        flushed = _flush()
        if flushed is not None:
            yield flushed

    def _iter_multi_process(
        self,
        *,
        graph_ids: Sequence[int],
    ) -> Iterator[tuple[jraph.GraphsTuple, int]]:
        """Yield batches from worker processes that read HDF5 directly."""
        ctx = mp.get_context('spawn')
        worker_count = max(self._num_workers, 1)
        index_queue_capacity = max(worker_count * 64, 1)
        index_queue = ctx.Queue(index_queue_capacity)
        result_queue_capacity = max(worker_count * 32, 1)
        result_queue = ctx.Queue(result_queue_capacity)
        stop_event = ctx.Event()

        worker_procs: list[mp.Process] = []
        for worker_id in range(worker_count):
            proc = ctx.Process(
                target=_graph_worker_main,
                args=(worker_id,),
                kwargs={
                    'index_queue': index_queue,
                    'result_queue': result_queue,
                    'stop_event': stop_event,
                    'dataset_specs': self._dataset_specs,
                    'dataset_offsets': self._dataset_offsets,
                    'dataset_lengths': self._dataset_lengths,
                    'z_table': self._z_table,
                    'r_max': self._cutoff,
                    'head_to_index': self._head_to_index,
                    'niggli_reduce': self._niggli_reduce,
                    'n_node': self._n_node,
                    'n_edge': self._n_edge,
                    'n_graph': self._n_graph,
                },
            )
            proc.daemon = True
            proc.start()
            worker_procs.append(proc)

        def _index_feeder():
            for graph_id in graph_ids:
                while True:
                    if stop_event.is_set():
                        return
                    try:
                        index_queue.put(int(graph_id), timeout=1.0)
                        break
                    except Full:
                        continue
            for _ in range(worker_count):
                while True:
                    if stop_event.is_set():
                        return
                    try:
                        index_queue.put(None, timeout=1.0)
                        break
                    except Full:
                        continue

        feeder = threading.Thread(target=_index_feeder, daemon=True)
        feeder.start()

        def _check_worker_health():
            for idx, proc in enumerate(worker_procs):
                if proc.is_alive():
                    continue
                exit_code = proc.exitcode
                if exit_code is None:
                    continue
                stop_event.set()
                raise RuntimeError(
                    f'Graph worker {idx} exited unexpectedly with code {exit_code}.'
                )

        finished_workers = 0
        try:
            while finished_workers < worker_count:
                try:
                    tag, payload_a, payload_b = result_queue.get(timeout=1.0)
                except Empty:
                    if stop_event.is_set():
                        break
                    _check_worker_health()
                    continue
                if tag == _RESULT_DONE:
                    finished_workers += 1
                    continue
                if tag == _RESULT_ERROR:
                    stop_event.set()
                    raise RuntimeError(f'Graph worker {payload_a} failed: {payload_b}')
                if tag == _RESULT_BATCH:
                    yield payload_a, payload_b
        finally:
            stop_event.set()
            for proc in worker_procs:
                proc.join(timeout=1)
            feeder.join(timeout=1)

    def iter_batches(
        self,
        *,
        epoch: int,
        seed: int | None,
        process_count: int,
        process_index: int,
    ) -> Iterator[jraph.GraphsTuple]:
        """Yield batches for the given epoch and process shard.

        The returned iterator yields padded jraph.GraphsTuple batches. When
        shuffle is enabled, graph_ids are shuffled deterministically with
        seed+epoch; otherwise they are in HDF5 order and sharded round-robin by
        process_index. If max_batches is set, iteration stops early.
        """
        if process_count <= 0:
            raise ValueError('process_count must be a positive integer.')
        if process_index < 0 or process_index >= process_count:
            raise ValueError(
                f'process_index {process_index} is out of range for process_count '
                f'{process_count}.'
            )
        graph_ids = self._graph_ids_for_epoch(
            epoch=epoch,
            seed=seed,
            process_count=process_count,
            process_index=process_index,
        )
        per_dataset_batches = getattr(self, '_dataset_estimated_batches', None)
        total_batches_hint = 0
        if per_dataset_batches:
            total_batches_hint = sum(
                int(estimate) for estimate in per_dataset_batches if estimate
            )
        if not total_batches_hint:
            total_graphs = int(sum(self._dataset_lengths))
            max_graphs = max(int(self._n_graph) - 1, 1)
            total_batches_hint = int(np.ceil(float(total_graphs) / float(max_graphs)))
        if process_count > 1 and total_batches_hint:
            total_batches_hint = int(
                np.ceil(float(total_batches_hint) / float(process_count))
            )
        if self._max_batches is not None:
            total_batches_hint = min(total_batches_hint, int(self._max_batches))
        total_batches_hint = max(total_batches_hint, 0)

        if self._num_workers > 0:
            source_iter = self._iter_multi_process(
                graph_ids=graph_ids,
            )
        else:
            source_iter = self._iter_single_process(
                graph_ids=graph_ids,
            )

        def _iter():
            graphs_count = 0
            nodes_count = 0
            edges_count = 0
            batches_count = 0
            produced = 0
            try:
                for batch, graph_count in source_iter:
                    if self._max_batches is not None and produced >= self._max_batches:
                        break
                    produced += 1
                    graph_total = int(graph_count)
                    graphs_count += graph_total
                    if graph_total > 0:
                        nodes_count += int(np.sum(batch.n_node[:graph_total]))
                        edges_count += int(np.sum(batch.n_edge[:graph_total]))
                    batches_count += 1
                    yield batch
            finally:
                if hasattr(source_iter, 'close'):
                    source_iter.close()
                if batches_count:
                    padded_nodes = int(self._n_node) * batches_count
                    padded_edges = int(self._n_edge) * batches_count
                    padded_graphs = int(self._n_graph) * batches_count
                    pad_nodes = max(padded_nodes - nodes_count, 0)
                    pad_edges = max(padded_edges - edges_count, 0)
                    pad_graphs = max(padded_graphs - graphs_count, 0)
                    self._last_padding_summary = {
                        'batches': batches_count,
                        'pad_nodes': pad_nodes,
                        'pad_edges': pad_edges,
                        'pad_graphs': pad_graphs,
                        'padded_nodes': padded_nodes,
                        'padded_edges': padded_edges,
                        'padded_graphs': padded_graphs,
                    }

        return BatchIteratorWrapper(_iter(), total_batches_hint)

    def __iter__(self):
        """Iterate over batches for a single-process epoch."""
        iterator = self.iter_batches(
            epoch=0,
            seed=None,
            process_count=1,
            process_index=0,
        )
        yield from iterator

    def __len__(self):
        """Return the number of batches for the current packing configuration."""
        return self.approx_length()

    def approx_length(self) -> int:
        """Estimate number of batches without forcing a prepass."""
        total_graphs = getattr(self, 'total_graphs', None)
        total_nodes = getattr(self, 'total_nodes', None)
        total_edges = getattr(self, 'total_edges', None)
        estimated_batches = getattr(self, 'estimated_batches', None)

        estimates: list[int] = []
        if estimated_batches is not None:
            estimates.append(int(estimated_batches))
        max_graphs = max(int(self._n_graph) - 1, 1)
        if total_graphs is not None:
            estimates.append(int(np.ceil(float(total_graphs) / float(max_graphs))))
        if total_nodes is not None:
            max_nodes = max(int(self._n_node) - 1, 1)
            estimates.append(int(np.ceil(float(total_nodes) / float(max_nodes))))
        if total_edges is not None:
            max_edges = max(int(self._n_edge), 1)
            estimates.append(int(np.ceil(float(total_edges) / float(max_edges))))

        if estimates:
            approx = max(estimates)
            if self._max_batches is not None:
                approx = min(approx, int(self._max_batches))
            return max(1, approx)
        return 1

    def split_by_heads(self) -> dict[str, StreamingGraphDataLoader]:
        """Split the loader into per-head StreamingGraphDataLoader instances."""
        # TODO: Review and test split_by_heads behavior thoroughly (multi-head, stats, ordering).
        if len(self._head_to_index) <= 1:
            return {}
        grouped: dict[str, list[StreamingDatasetSpec]] = {}
        for spec in self._dataset_specs:
            grouped.setdefault(spec.head_name, []).append(spec)

        result: dict[str, StreamingGraphDataLoader] = {}
        for head_name, specs in grouped.items():
            datasets = [HDF5Dataset(spec.path, mode='r') for spec in specs]
            head_to_index = {head_name: self._head_to_index[head_name]}
            loader = StreamingGraphDataLoader(
                datasets=datasets,
                dataset_specs=specs,
                z_table=self._z_table,
                r_max=self._cutoff,
                n_node=self._n_node,
                n_edge=self._n_edge,
                head_to_index=head_to_index,
                niggli_reduce=self._niggli_reduce,
                max_batches=self._max_batches,
                prefetch_batches=self._prefetch_batches,
                num_workers=self._num_workers,
                pad_graphs=self._n_graph,
                shuffle=self._shuffle,
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
            loader.total_nodes = getattr(self, 'total_nodes', None)
            loader.total_edges = getattr(self, 'total_edges', None)
            loader._fixed_pad_nodes = int(self._n_node)
            loader._fixed_pad_edges = int(self._n_edge)
            loader._fixed_pad_graphs = int(self._n_graph)
            result[head_name] = loader
        return result

    def close(self) -> None:
        """Close datasets and release cached resources."""
        for dataset in self._datasets:
            dataset.close()
        self._datasets = []


def get_hdf5_dataloader(
    *,
    data_file: Path | str | Sequence[Path | str],
    atomic_numbers: AtomicNumberTable,
    r_max: float,
    max_nodes: int | None,
    max_edges: int | None,
    max_graphs: int | None = None,
    niggli_reduce: bool = False,
    max_batches: int | None = None,
    prefetch_batches: int | None = None,
    num_workers: int | None = None,
    dataset_specs: Sequence[StreamingDatasetSpec] | None = None,
    head_to_index: dict[str, int] | None = None,
    shuffle: bool = False,
) -> StreamingGraphDataLoader:
    """Create a StreamingGraphDataLoader from one or more HDF5 files.

    Args:
        data_file: Path, directory, glob, or list of HDF5 shards.
        atomic_numbers: AtomicNumberTable for graph construction.
        r_max: Cutoff radius for neighbor construction.
        max_nodes: Fixed node padding cap (None to infer).
        max_edges: Fixed edge padding cap (None to infer).
        max_graphs: Fixed graph padding cap (None to infer).
        niggli_reduce: Apply Niggli reduction to periodic cells before graphing.
        max_batches: Optional cap on batches per epoch.
        prefetch_batches: Host prefetch depth for produced batches.
        num_workers: Worker process count for graph construction.
        dataset_specs: Optional StreamingDatasetSpec list per shard.
        head_to_index: Optional head-name to index mapping.
        shuffle: Deterministic per-epoch shuffle toggle.

    Returns:
        StreamingGraphDataLoader configured for the provided shards.
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
        niggli_reduce=niggli_reduce,
        max_batches=max_batches,
        prefetch_batches=prefetch_batches,
        num_workers=num_workers,
        pad_graphs=max_graphs,
        shuffle=shuffle,
    )


__all__ = [
    'StreamingDatasetSpec',
    'StreamingGraphDataLoader',
    'get_hdf5_dataloader',
]
