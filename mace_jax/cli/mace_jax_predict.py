from __future__ import annotations

import argparse
import logging
import pickle
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import jax
import jraph
import numpy as np
import tqdm

from mace_jax import data, tools
from mace_jax.data.streaming_loader import (
    BatchIteratorWrapper,
    _expand_hdf5_paths,
    _mark_padding_graph_ids,
    _pack_sizes_by_edge_cap,
    _with_graph_id,
)
from mace_jax.data.streaming_stats_cache import (
    STREAMING_STATS_CACHE_VERSION,
    dataset_signature,
    stats_cache_path,
    stats_payload_to_parts,
)
from mace_jax.tools import bundle as bundle_tools
from mace_jax.tools import gin_model

_GRAPH_OUTPUT_KEYS = {
    'energy',
    'stress',
    'virials',
    'dipole',
    'polarizability',
    'polar',
}
_NODE_OUTPUT_KEYS = {
    'forces',
}


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Run MACE-JAX predictions on XYZ or HDF5 datasets.',
    )
    parser.add_argument(
        'model',
        help=(
            'Path to a JAX model bundle (config.json + params.msgpack) '
            'or a training checkpoint (.ckpt). Pass a directory, config.json, '
            'params.msgpack, or .ckpt path.'
        ),
    )
    parser.add_argument(
        'inputs',
        nargs='+',
        help='XYZ or HDF5 inputs (HDF5 may be a file, directory, or glob).',
    )
    parser.add_argument(
        '--output',
        help=(
            'Output file or directory for predictions (.npz). '
            'Defaults to <input>.predictions.npz per input.'
        ),
    )
    parser.add_argument(
        '--dtype',
        choices=['float32', 'float64'],
        default=None,
        help='Override JAX dtype mode when loading the model bundle.',
    )
    parser.add_argument(
        '--batch-max-edges',
        type=int,
        default=None,
        help=(
            'Edge cap for streaming HDF5 predictions. If omitted, cached '
            'streaming stats are used when available.'
        ),
    )
    parser.add_argument(
        '--batch-max-nodes',
        type=int,
        default=None,
        help='Optional node cap for streaming HDF5 predictions.',
    )
    parser.add_argument(
        '--prefetch-batches',
        type=int,
        default=None,
        help='Prefetch depth for streaming predictions.',
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=0,
        help='Number of streaming worker processes to use.',
    )
    parser.add_argument(
        '--head',
        default=None,
        help='Optional head name to use for multi-head models.',
    )
    parser.add_argument(
        '--compute-forces',
        dest='compute_forces',
        action='store_true',
        help='Compute forces in addition to energies.',
    )
    parser.add_argument(
        '--no-compute-forces',
        dest='compute_forces',
        action='store_false',
        help='Disable force predictions.',
    )
    parser.set_defaults(compute_forces=True)
    parser.add_argument(
        '--compute-stress',
        dest='compute_stress',
        action='store_true',
        help='Compute stress/virials predictions.',
    )
    parser.add_argument(
        '--no-compute-stress',
        dest='compute_stress',
        action='store_false',
        help='Disable stress/virials predictions.',
    )
    parser.set_defaults(compute_stress=False)
    parser.add_argument(
        '--no-progress',
        action='store_true',
        help='Disable progress bars.',
    )
    parser.add_argument(
        '--full-outputs',
        action='store_true',
        help='Return all model outputs (slower, larger transfers).',
    )
    return parser.parse_args(argv)


def _is_hdf5_path(path: Path) -> bool:
    if path.is_dir():
        return True
    return path.suffix.lower() in ('.h5', '.hdf5')


def _is_xyz_path(path: Path) -> bool:
    return path.suffix.lower() in ('.xyz', '.extxyz')


def _resolve_head_mapping(config: dict[str, Any], head: str | None):
    heads = config.get('heads') or ['Default']
    head_names = [str(h) for h in heads]
    if head is None:
        head_name = head_names[0]
    else:
        if head not in head_names:
            raise ValueError(
                f"Requested head '{head}' not found in model heads {head_names}."
            )
        head_name = head
    head_to_index = {name: idx for idx, name in enumerate(head_names)}
    return head_name, head_to_index


def _default_output_path(input_path: Path) -> Path:
    return input_path.with_name(f'{input_path.name}.predictions.npz')


def _resolve_output_path(
    output_arg: str | None, input_path: Path, multi_input: bool
) -> Path:
    if output_arg is None:
        return _default_output_path(input_path)
    output_path = Path(output_arg).expanduser()
    if multi_input:
        if output_path.suffix:
            raise ValueError(
                'When predicting multiple inputs, --output must be a directory.'
            )
        output_path.mkdir(parents=True, exist_ok=True)
        return output_path / f'{input_path.name}.predictions.npz'
    if output_path.is_dir():
        output_path.mkdir(parents=True, exist_ok=True)
        return output_path / f'{input_path.name}.predictions.npz'
    return output_path


def _load_cached_streaming_caps(
    paths: Sequence[Path],
) -> tuple[tuple[int, int, int] | None, list[Path], list[Path]]:
    caps: list[tuple[int, int, int]] = []
    missing: list[Path] = []
    stale: list[Path] = []
    for path in paths:
        cache_path = stats_cache_path(path)
        if not cache_path.exists():
            missing.append(path)
            continue
        try:
            with cache_path.open('rb') as fh:
                payload = pickle.load(fh)
        except Exception as exc:  # pragma: no cover - cache corruption is unexpected
            logging.warning(
                'Failed to read streaming stats cache %s: %s', cache_path, exc
            )
            missing.append(path)
            continue
        if payload is None or payload.get('version') != STREAMING_STATS_CACHE_VERSION:
            missing.append(path)
            continue
        if payload.get('dataset_signature') != dataset_signature(path):
            stale.append(path)
            continue
        stats_payload = payload.get('stats')
        if not stats_payload:
            missing.append(path)
            continue
        try:
            n_nodes, n_edges, n_graphs, _ = stats_payload_to_parts(stats_payload)
        except (KeyError, TypeError, ValueError):
            missing.append(path)
            continue
        caps.append((n_nodes, n_edges, n_graphs))
    if not caps or missing or stale:
        return None, missing, stale
    max_nodes = max(item[0] for item in caps)
    max_edges = max(item[1] for item in caps)
    max_graphs = max(item[2] for item in caps)
    return (max_nodes, max_edges, max_graphs), missing, stale


def _stack_or_object(values: list[Any]) -> np.ndarray:
    if not values:
        return np.array([], dtype=np.float32)
    if any(value is None for value in values):
        return np.array(values, dtype=object)
    try:
        return np.stack(values)
    except ValueError:
        return np.array(values, dtype=object)


def _write_predictions(
    output_path: Path, graph_ids: Sequence[int], outputs: dict[str, list[Any]]
) -> None:
    payload: dict[str, Any] = {
        'graph_id': np.asarray(graph_ids, dtype=np.int64),
    }
    for key, values in outputs.items():
        payload[key] = _stack_or_object(values)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **payload)


def _build_predictor(
    module,
    *,
    model_config: dict[str, Any],
    compute_forces: bool,
    compute_stress: bool,
    full_outputs: bool,
):
    atomic_numbers = model_config.get('atomic_numbers') or []
    num_species = int(len(atomic_numbers))
    if num_species <= 0:
        raise ValueError('Model config is missing atomic_numbers for prediction.')

    keep_keys: set[str] | None = None
    if not full_outputs:
        keep_keys = {'energy', 'dipole', 'polarizability', 'polar'}
        if compute_forces:
            keep_keys.add('forces')
        if compute_stress:
            keep_keys.update({'stress', 'virials'})

    def _predict(params, graph):
        outputs = module.apply(
            params,
            gin_model._graph_to_data(graph, num_species=num_species),
            compute_force=compute_forces,
            compute_stress=compute_stress,
            compute_node_feats=full_outputs,
        )
        if full_outputs:
            return outputs
        return {key: outputs.get(key) for key in keep_keys if key in outputs}

    return _predict


def _predict_xyz(
    input_path: Path,
    *,
    predictor: Any,
    params: Any,
    model_config: dict[str, Any],
    head_name: str,
    head_to_index: dict[str, int],
    batch_max_edges: int | None,
    batch_max_nodes: int | None,
    batch_max_graphs: int | None,
    prefetch_batches: int | None,
    progress_bar: bool,
) -> tuple[list[int], dict[str, list[Any]]]:
    _, configs = data.load_from_xyz(input_path, head_name=head_name, no_data_ok=True)
    if not configs:
        raise ValueError(f'No configurations found in {input_path}.')
    z_table = data.AtomicNumberTable(model_config['atomic_numbers'])
    process_index = getattr(jax, 'process_index', lambda: 0)()
    process_count = getattr(jax, 'process_count', lambda: 1)()
    show_progress = bool(progress_bar and process_index == 0)
    graphs: list[jraph.GraphsTuple] = []
    graph_sizes: list[tuple[int, int]] = []
    max_graph_edges = 0
    config_iter = enumerate(configs)
    if show_progress:
        config_iter = tqdm.tqdm(config_iter, desc='Prepare', total=len(configs))
    for idx, config in config_iter:
        if process_count > 1 and (idx % process_count) != process_index:
            continue
        graph = data.graph_from_configuration(
            config,
            cutoff=float(model_config['r_max']),
            z_table=z_table,
            head_to_index=head_to_index,
        )
        graph = _with_graph_id(graph, idx)
        graphs.append(graph)
        nodes = int(graph.n_node.sum())
        edges = int(graph.n_edge.sum())
        graph_sizes.append((nodes, edges))
        max_graph_edges = max(max_graph_edges, edges)

    if not graphs:
        return [], {}

    edge_cap = int(batch_max_edges or max_graph_edges or 1)
    if max_graph_edges > edge_cap:
        logging.warning(
            'Requested max edges per batch (%s) is below the largest graph (%s) '
            'in %s. Raising the limit to fit.',
            edge_cap,
            max_graph_edges,
            input_path.name,
        )
        edge_cap = max_graph_edges

    batches = _pack_sizes_by_edge_cap(graph_sizes, edge_cap)
    max_nodes_per_batch = max((batch['node_sum'] for batch in batches), default=1)
    max_graphs_per_batch = max((batch['graph_count'] for batch in batches), default=1)
    inferred_nodes = max(max_nodes_per_batch + 1, 2)
    inferred_graphs = max(max_graphs_per_batch + 1, 2)

    n_node = inferred_nodes if batch_max_nodes is None else int(batch_max_nodes)
    n_graph = inferred_graphs if batch_max_graphs is None else int(batch_max_graphs)
    if n_node < inferred_nodes or n_graph < inferred_graphs:
        raise ValueError(
            'Provided batch caps are smaller than required for XYZ inputs '
            f'(n_node={n_node} required={inferred_nodes}, '
            f'n_graph={n_graph} required={inferred_graphs}).'
        )

    max_graphs = max(int(n_graph) - 1, 1)
    packed_batches: list[jraph.GraphsTuple] = []
    nodes_sum = 0
    edges_sum = 0
    graph_count = 0
    bucket: list[jraph.GraphsTuple] = []

    def _flush():
        nonlocal nodes_sum, edges_sum, graph_count, bucket
        if not bucket:
            return
        batched = jraph.batch_np(bucket)
        batch = jraph.pad_with_graphs(
            batched,
            n_node=n_node,
            n_edge=edge_cap,
            n_graph=n_graph,
        )
        batch = _mark_padding_graph_ids(batch, graph_count)
        packed_batches.append(batch)
        bucket = []
        nodes_sum = 0
        edges_sum = 0
        graph_count = 0

    for graph in graphs:
        nodes = int(graph.n_node.sum())
        edges = int(graph.n_edge.sum())
        if nodes >= n_node or edges > edge_cap:
            raise ValueError(
                'Graph exceeds padding limits '
                f'(nodes={nodes} edges={edges}, n_node={n_node} n_edge={edge_cap}).'
            )
        if bucket and (
            nodes_sum + nodes >= n_node
            or edges_sum + edges > edge_cap
            or graph_count >= max_graphs
        ):
            _flush()
        bucket.append(graph)
        graph_count += 1
        nodes_sum += nodes
        edges_sum += edges
        if graph_count >= max_graphs:
            _flush()
    _flush()

    class _GraphListLoader:
        def __init__(self, batches, prefetch):
            self._batches = list(batches)
            self._prefetch_batches = int(prefetch or 0)

        def iter_batches(self, *, epoch, seed, process_count, process_index):
            del epoch, seed, process_count, process_index

            def _iter():
                yield from self._batches

            return BatchIteratorWrapper(_iter(), len(self._batches))

        def approx_length(self):
            return len(self._batches)

    loader = _GraphListLoader(packed_batches, prefetch_batches)
    graph_ids, outputs = tools.predict_streaming(
        predictor,
        params,
        loader,
        progress_bar=progress_bar,
    )
    return graph_ids, outputs


def _predict_hdf5(
    input_path: Path,
    *,
    predictor: Any,
    params: Any,
    model_config: dict[str, Any],
    head_name: str,
    head_to_index: dict[str, int],
    batch_max_edges: int | None,
    batch_max_nodes: int | None,
    batch_max_graphs: int | None,
    prefetch_batches: int | None,
    num_workers: int,
    progress_bar: bool,
) -> tuple[list[int], dict[str, list[Any]]]:
    z_table = data.AtomicNumberTable(model_config['atomic_numbers'])
    expanded_paths = _expand_hdf5_paths([input_path])
    if batch_max_edges is None:
        caps, missing, stale = _load_cached_streaming_caps(expanded_paths)
        if caps is None:
            details: list[str] = []
            if missing:
                details.append(
                    f'missing cache for: {", ".join(path.name for path in missing)}'
                )
            if stale:
                details.append(
                    f'stale cache for: {", ".join(path.name for path in stale)}'
                )
            detail_msg = f' ({"; ".join(details)})' if details else ''
            raise ValueError(
                f'--batch-max-edges is required for streaming predictions on {input_path}'
                f' unless a valid streaming stats cache is present{detail_msg}.'
            )
        cached_nodes, cached_edges, cached_graphs = caps
        batch_max_edges = cached_edges
        if batch_max_nodes is None:
            batch_max_nodes = cached_nodes
        if batch_max_graphs is None:
            batch_max_graphs = cached_graphs
        logging.info(
            'Using cached streaming stats for %s: n_nodes=%s n_edges=%s n_graphs=%s',
            input_path,
            cached_nodes,
            cached_edges,
            cached_graphs,
        )
    dataset_specs = [
        data.StreamingDatasetSpec(path=path, head_name=head_name)
        for path in expanded_paths
    ]
    loader = data.get_hdf5_dataloader(
        data_file=expanded_paths,
        atomic_numbers=z_table,
        r_max=float(model_config['r_max']),
        max_nodes=int(batch_max_nodes) if batch_max_nodes is not None else None,
        max_edges=int(batch_max_edges),
        max_graphs=int(batch_max_graphs) if batch_max_graphs is not None else None,
        prefetch_batches=prefetch_batches,
        num_workers=num_workers,
        dataset_specs=dataset_specs,
        head_to_index=head_to_index,
    )
    try:
        graph_ids, outputs = tools.predict_streaming(
            predictor,
            params,
            loader,
            progress_bar=progress_bar,
        )
    finally:
        loader.close()
    return graph_ids, outputs


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    bundle = bundle_tools.load_model_bundle(args.model, args.dtype or '')
    head_name, head_to_index = _resolve_head_mapping(bundle.config, args.head)
    predictor = _build_predictor(
        bundle.module,
        model_config=bundle.config,
        compute_forces=args.compute_forces,
        compute_stress=args.compute_stress,
        full_outputs=bool(args.full_outputs),
    )

    inputs = [Path(path).expanduser() for path in args.inputs]
    multi_input = len(inputs) > 1
    for input_path in inputs:
        if _is_hdf5_path(input_path):
            graph_ids, outputs = _predict_hdf5(
                input_path,
                predictor=predictor,
                params=bundle.params,
                model_config=bundle.config,
                head_name=head_name,
                head_to_index=head_to_index,
                batch_max_edges=args.batch_max_edges,
                batch_max_nodes=(
                    int(args.batch_max_nodes)
                    if args.batch_max_nodes is not None
                    else None
                ),
                batch_max_graphs=None,
                prefetch_batches=(
                    int(args.prefetch_batches)
                    if args.prefetch_batches is not None
                    else None
                ),
                num_workers=int(args.num_workers or 0),
                progress_bar=not args.no_progress,
            )
        elif _is_xyz_path(input_path):
            graph_ids, outputs = _predict_xyz(
                input_path,
                predictor=predictor,
                params=bundle.params,
                model_config=bundle.config,
                head_name=head_name,
                head_to_index=head_to_index,
                batch_max_edges=args.batch_max_edges,
                batch_max_nodes=(
                    int(args.batch_max_nodes)
                    if args.batch_max_nodes is not None
                    else None
                ),
                batch_max_graphs=None,
                prefetch_batches=(
                    int(args.prefetch_batches)
                    if args.prefetch_batches is not None
                    else None
                ),
                progress_bar=not args.no_progress,
            )
        else:
            raise ValueError(
                f'Unsupported input format for {input_path}; expected .xyz or .h5/.hdf5.'
            )

        output_path = _resolve_output_path(args.output, input_path, multi_input)
        _write_predictions(output_path, graph_ids, outputs)
        logging.info('Wrote predictions for %s to %s', input_path, output_path)


if __name__ == '__main__':
    main()
