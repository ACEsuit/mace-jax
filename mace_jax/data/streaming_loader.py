"""Streaming HDF5 data loader for native MACE datasets."""

from __future__ import annotations

import math
import multiprocessing as mp
import threading
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from queue import Queue

import jraph
import numpy as np

from ._batch_iterator import BatchIteratorWrapper
from .hdf5_dataset import HDF5Dataset
from .utils import (
    AtomicNumberTable,
    GraphDataLoader,
    config_from_atoms,
    graph_from_configuration,
)

_RESULT_DATA = 'data'
_RESULT_DONE = 'done'
_RESULT_ERROR = 'error'


@dataclass(frozen=True)
class StreamingDatasetSpec:
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
    try:
        from ase.build.tools import niggli_reduce as _niggli_reduce
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
    local_datasets: dict[int, HDF5Dataset] = {}
    try:
        while not stop_event.is_set():
            task = task_queue.get()
            if task is None:
                break
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
            if not _has_positive_weight(graph):
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


class StreamingGraphDataLoader:
    """Stream HDF5-backed graphs with greedy packing and optional prefetching."""

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
        graph_multiple: int | None = None,
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
        self._cutoff = float(r_max)
        self._shuffle = bool(shuffle)
        self._seed = None if seed is None else int(seed)
        self._epoch = 0
        self._niggli_reduce = bool(niggli_reduce)
        self._max_batches = max_batches
        self._prefetch_batches = max(int(prefetch_batches or 0), 0)
        self._num_workers = max(int(num_workers or 0), 0)
        self._graph_multiple = max(int(graph_multiple or 1), 1)
        self._pack_info: dict | None = None
        self._history: list[tuple[int, int]] = []
        self._fixed_pad_nodes: int | None = None
        self._fixed_pad_edges: int | None = None
        self._fixed_pad_graphs: int | None = None
        # Attach optional metadata for downstream consumers.
        self.graphs = getattr(self, 'graphs', None)
        self.streaming = True
        self.total_graphs = getattr(self, 'total_graphs', None)
        self.total_nodes = getattr(self, 'total_nodes', None)
        self.total_edges = getattr(self, 'total_edges', None)

        if n_node is None or n_edge is None:
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

    @property
    def epoch_history(self) -> list[tuple[int, int]]:
        """List of (graphs, batches) pairs recorded per epoch."""
        return list(self._history)

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def _effective_seed(self, override: int | None) -> int | None:
        base = self._seed if override is None else override
        if base is None:
            return None
        return int(base) + int(self._epoch)

    def _task_iterator(self, seed: int | None) -> Iterator[tuple[int, int, int]]:
        rng = np.random.default_rng(seed)
        emitted = 0
        dataset_indices = list(range(len(self._datasets)))
        if self._shuffle and len(dataset_indices) > 1:
            rng.shuffle(dataset_indices)
        for ds_idx in dataset_indices:
            ds = self._datasets[ds_idx]
            indices = np.arange(len(ds))
            if self._shuffle and len(indices) > 1:
                rng.shuffle(indices)
            for idx in indices:
                yield emitted, ds_idx, int(idx)
                emitted += 1

    def _convert_atoms_to_graph(self, atoms, spec: StreamingDatasetSpec):
        return _atoms_to_graph(
            atoms=atoms,
            spec=spec,
            cutoff=self._cutoff,
            z_table=self._z_table,
            head_to_index=self._head_to_index,
            niggli_reduce=self._niggli_reduce,
        )

    def _iter_graphs_single(self, seed: int | None):
        rng = np.random.default_rng(seed)
        sources = list(zip(self._datasets, self._dataset_specs))
        if self._shuffle and len(sources) > 1:
            rng.shuffle(sources)
        for ds, spec in sources:
            indices = np.arange(len(ds))
            if self._shuffle and len(indices) > 1:
                rng.shuffle(indices)
            for idx in indices:
                atoms = ds[int(idx)]
                graph = self._convert_atoms_to_graph(atoms, spec)
                if not _has_positive_weight(graph):
                    continue
                yield graph

    def _iter_graphs_parallel(self, seed: int | None):
        worker_count = max(self._num_workers, 1)
        ctx = mp.get_context('spawn')
        task_queue = ctx.Queue(max(worker_count * 4, 1))
        result_queue = ctx.Queue(max(worker_count * 4, 1))
        stop_event = ctx.Event()

        def _producer():
            try:
                for task in self._task_iterator(seed):
                    if stop_event.is_set():
                        break
                    task_queue.put(task)
            finally:
                for _ in range(worker_count):
                    task_queue.put(None)

        producer = threading.Thread(target=_producer, daemon=True)
        producer.start()

        processes = []
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
                    task_queue,
                    result_queue,
                    stop_event,
                ),
            )
            proc.daemon = True
            proc.start()
            processes.append(proc)

        finished_workers = 0
        next_seq = 0
        pending: dict[int, jraph.GraphsTuple] = {}
        try:
            while finished_workers < worker_count:
                tag, payload_a, payload_b = result_queue.get()
                if tag == _RESULT_DONE:
                    finished_workers += 1
                    continue
                if tag == _RESULT_ERROR:
                    stop_event.set()
                    for proc in processes:
                        proc.join(timeout=1)
                    raise RuntimeError(f'Graph worker {payload_a} failed: {payload_b}')
                seq_id = payload_a
                graph = payload_b
                if seq_id == next_seq:
                    yield graph
                    next_seq += 1
                    while next_seq in pending:
                        yield pending.pop(next_seq)
                        next_seq += 1
                else:
                    pending[seq_id] = graph
        finally:
            stop_event.set()
            producer.join(timeout=1)
            for proc in processes:
                proc.join()

    def _graph_iterator(self, *, seed_override: int | None):
        seed = self._effective_seed(seed_override)
        if self._num_workers <= 1:
            return self._iter_graphs_single(seed)
        return self._iter_graphs_parallel(seed)

    def _pack(self, *, seed_override: int | None = None):
        graph_iter_fn = lambda: self._graph_iterator(seed_override=seed_override)
        reuse_info = getattr(self, '_pack_info', None)
        batches, info = pack_graphs_greedy(
            graph_iter_fn=graph_iter_fn,
            max_edges_per_batch=self._n_edge,
            max_nodes_per_batch=self._n_node,
            graph_multiple=self._graph_multiple,
            fixed_pad_nodes=getattr(self, '_fixed_pad_nodes', None),
            fixed_pad_edges=getattr(self, '_fixed_pad_edges', None),
            fixed_pad_graphs=getattr(self, '_fixed_pad_graphs', None),
            reuse_info=reuse_info,
        )
        self._fixed_pad_nodes = info.get('pad_nodes')
        self._fixed_pad_edges = info.get('pad_edges')
        self._fixed_pad_graphs = info.get('pad_graphs')
        self._pack_info = info
        return batches, info

    def __iter__(self):
        batches, _ = self._pack(seed_override=None)

        def _limited(iterable):
            if self._max_batches is None:
                yield from iterable
                return
            produced = 0
            for item in iterable:
                yield item
                produced += 1
                if produced >= self._max_batches:
                    break

        if self._prefetch_batches > 0:
            queue: Queue = Queue(maxsize=self._prefetch_batches)
            sentinel = object()

            def _producer():
                try:
                    for item in _limited(batches):
                        queue.put(item)
                finally:
                    queue.put(sentinel)

            threading.Thread(target=_producer, daemon=True).start()

            while True:
                item = queue.get()
                if item is sentinel:
                    break
                yield item
        else:
            yield from _limited(batches)

    def iter_batches(
        self,
        *,
        epoch: int,
        seed: int | None,
        process_count: int,
        process_index: int,
    ) -> Iterator[jraph.GraphsTuple]:
        self.set_epoch(epoch)
        batches_iter, _ = self._pack(seed_override=seed)

        def _limited(iterable):
            if self._max_batches is None:
                yield from iterable
                return
            produced = 0
            for item in iterable:
                if produced >= self._max_batches:
                    break
                yield item
                produced += 1

        limited_iter = _limited(batches_iter)
        if process_count > 1:

            def _filtered():
                graphs_count = 0
                batches_count = 0
                try:
                    for idx, batch in enumerate(limited_iter):
                        if idx % process_count != process_index:
                            continue
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
                graphs_count = 0
                batches_count = 0
                try:
                    for batch in limited_iter:
                        graphs_count += int(
                            np.asarray(jraph.get_graph_padding_mask(batch)).sum()
                        )
                        batches_count += 1
                        yield batch
                finally:
                    self._history.append((graphs_count, batches_count))

            iterator = _single()

        total_batches_hint = int(self._pack_info.get('total_batches') or 0)
        return BatchIteratorWrapper(iterator, total_batches_hint)

    def __len__(self):
        if self._pack_info and self._pack_info.get('total_batches') is not None:
            total = int(self._pack_info['total_batches'])
            if self._max_batches is not None:
                return min(total, int(self._max_batches))
            return total
        _, info = self._pack(seed_override=None)
        total = int(info.get('total_batches', 0))
        if self._max_batches is not None:
            return min(total, int(self._max_batches))
        return total

    def pack_info(self) -> dict:
        if not self._pack_info:
            _, info = self._pack(seed_override=None)
            return dict(info)
        return dict(self._pack_info)

    def close(self) -> None:
        for dataset in self._datasets:
            dataset.close()

    def approx_length(self) -> int:
        """Estimate number of batches without forcing a prepass."""
        if self.total_graphs is None:
            return max(1, len(self))
        nodes_cap = max(1, self._n_node)
        edges_cap = max(1, self._n_edge)
        approx = 0
        if self.total_nodes:
            approx = max(approx, math.ceil(self.total_nodes / nodes_cap))
        if self.total_edges:
            approx = max(approx, math.ceil(self.total_edges / edges_cap))
        if approx == 0:
            approx = math.ceil(self.total_graphs / max(1, self._graph_multiple))
        return max(1, approx)

    def subset(self, i):
        """Approximate GraphDataLoader.subset using cached initialization graphs."""
        cached = list(getattr(self, 'graphs', None) or [])
        if isinstance(i, slice):
            graphs = cached[i]
        elif isinstance(i, list):
            graphs = [cached[j] for j in i]
        elif isinstance(i, float):
            graphs = cached[: max(int(len(cached) * i), 0)]
        elif isinstance(i, int):
            graphs = cached[: max(i, 0)]
        else:
            raise TypeError(f'Unsupported subset specifier: {i!r}')
        head_names = tuple(sorted(self._head_to_index, key=self._head_to_index.get))
        return GraphDataLoader(
            graphs=graphs,
            n_node=self._n_node,
            n_edge=self._n_edge,
            n_graph=self._graph_multiple,
            shuffle=False,
            heads=head_names or None,
        )

    def split_by_heads(self) -> dict[str, StreamingGraphDataLoader]:
        if len(self._head_to_index) <= 1:
            return {}
        grouped: dict[str, list[StreamingDatasetSpec]] = {}
        for spec in self._dataset_specs:
            grouped.setdefault(spec.head_name, []).append(spec)
        result: dict[str, StreamingGraphDataLoader] = {}
        for head_name, specs in grouped.items():
            datasets = [HDF5Dataset(spec.path, mode='r') for spec in specs]
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
                graph_multiple=self._graph_multiple,
            )
            loader.graphs = getattr(self, 'graphs', None)
            loader.streaming = getattr(self, 'streaming', True)
            loader.total_graphs = getattr(self, 'total_graphs', None)
            result[head_name] = loader
        return result


def pack_graphs_greedy(
    *,
    graph_iter_fn: callable,
    max_edges_per_batch: int,
    max_nodes_per_batch: int | None = None,
    graph_multiple: int = 1,
    fixed_pad_nodes: int | None = None,
    fixed_pad_edges: int | None = None,
    fixed_pad_graphs: int | None = None,
    reuse_info: dict | None = None,
) -> tuple[Iterable[jraph.GraphsTuple], dict]:
    def _make_iter():
        return iter(graph_iter_fn())

    nodes_cap = (
        int(max_nodes_per_batch)
        if max_nodes_per_batch is not None
        else int(fixed_pad_nodes)
        if fixed_pad_nodes is not None
        else None
    )
    if nodes_cap is None and reuse_info and reuse_info.get('pad_nodes'):
        nodes_cap = int(reuse_info['pad_nodes'])
    if nodes_cap is None:
        raise ValueError(
            'max_nodes_per_batch must be specified when packing without a preliminary scan.'
        )
    edges_cap = int(max_edges_per_batch)
    if fixed_pad_edges is not None:
        edges_cap = max(edges_cap, int(fixed_pad_edges))
    if reuse_info and reuse_info.get('pad_edges'):
        edges_cap = max(edges_cap, int(reuse_info['pad_edges']))

    graph_multiple = max(int(graph_multiple or 1), 1)
    pad_graphs = None
    if fixed_pad_graphs is not None:
        pad_graphs = int(fixed_pad_graphs)
    elif reuse_info and reuse_info.get('pad_graphs'):
        pad_graphs = int(reuse_info['pad_graphs'])
    if pad_graphs is None:
        pad_graphs = graph_multiple + 1
    if graph_multiple > 1:
        pad_graphs = (
            (pad_graphs + graph_multiple - 1) // graph_multiple
        ) * graph_multiple
    pad_graphs = max(pad_graphs, graph_multiple + 1)
    pad_graphs += graph_multiple

    info = dict(
        dropped=0,
        total_batches=0,
        pad_graphs=pad_graphs,
        pad_nodes=nodes_cap,
        pad_edges=edges_cap,
        graphs_seen=0,
        kept_graphs=0,
        max_single_nodes=0,
        max_single_edges=0,
    )

    def _empty_graph_like(graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
        def _zero_nodes(arr):
            if arr is None:
                return None
            shape = (0,) + arr.shape[1:]
            return np.zeros(shape, dtype=arr.dtype)

        def _zero_edges(arr):
            if arr is None:
                return None
            shape = (0,) + arr.shape[1:]
            return np.zeros(shape, dtype=arr.dtype)

        def _zero_globals(value):
            if value is None:
                return None
            return np.zeros_like(value)

        nodes = graph.nodes.__class__(*(_zero_nodes(value) for value in graph.nodes))
        edges = graph.edges.__class__(*(_zero_edges(value) for value in graph.edges))
        globals_dict = graph.globals.__class__(
            *(_zero_globals(value) for value in graph.globals)
        )

        return jraph.GraphsTuple(
            nodes=nodes,
            edges=edges,
            senders=np.zeros((0,), dtype=graph.senders.dtype),
            receivers=np.zeros((0,), dtype=graph.receivers.dtype),
            globals=globals_dict,
            n_node=np.asarray([0], dtype=np.int32),
            n_edge=np.asarray([0], dtype=np.int32),
        )

    def _pad_graph_list(graphs: list[jraph.GraphsTuple]):
        if graph_multiple <= 1 or not graphs:
            return
        remainder = len(graphs) % graph_multiple
        if remainder == 0:
            return
        template = graphs[0]
        dummy = _empty_graph_like(template)
        for _ in range(graph_multiple - remainder):
            graphs.append(dummy)

    graph_cap = max(pad_graphs - graph_multiple, 1)

    def _emit_batch(batch_graphs: list[jraph.GraphsTuple]):
        nonlocal info
        if not batch_graphs:
            return None
        _pad_graph_list(batch_graphs)
        batched = jraph.batch_np(batch_graphs)
        result = jraph.pad_with_graphs(
            batched,
            n_node=nodes_cap,
            n_edge=edges_cap,
            n_graph=pad_graphs,
        )
        info['total_batches'] += 1
        return result

    def _iter():
        current: list[jraph.GraphsTuple] = []
        edge_sum = node_sum = 0
        for g in _make_iter():
            g_edges = int(g.n_edge.sum())
            g_nodes = int(g.n_node.sum())
            info['graphs_seen'] += 1
            info['max_single_nodes'] = max(info['max_single_nodes'], g_nodes)
            info['max_single_edges'] = max(info['max_single_edges'], g_edges)
            if g_edges > max_edges_per_batch or g_nodes > nodes_cap:
                info['dropped'] += 1
                continue
            info['kept_graphs'] += 1
            would_edges = edge_sum + g_edges
            would_nodes = node_sum + g_nodes
            would_graphs = len(current) + 1
            if current and (
                would_edges > max_edges_per_batch
                or would_nodes > nodes_cap
                or would_graphs > graph_cap
            ):
                emitted = _emit_batch(current)
                if emitted is not None:
                    yield emitted
                current = []
                edge_sum = node_sum = 0
            current.append(g)
            edge_sum += g_edges
            node_sum += g_nodes
        if current:
            emitted = _emit_batch(current)
            if emitted is not None:
                yield emitted

    return _iter(), info


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
    graph_multiple: int | None = None,
    dataset_specs: Sequence[StreamingDatasetSpec] | None = None,
    head_to_index: dict[str, int] | None = None,
) -> StreamingGraphDataLoader:
    if data_file is None:
        raise ValueError('data_file must be provided.')
    if isinstance(data_file, (list, tuple)):
        files = [Path(p) for p in data_file]
    else:
        files = [Path(data_file)]
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
        graph_multiple=graph_multiple,
    )


__all__ = [
    'StreamingDatasetSpec',
    'StreamingGraphDataLoader',
    'get_hdf5_dataloader',
    'pack_graphs_greedy',
]
