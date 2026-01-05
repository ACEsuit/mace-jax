from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Sequence

import jax
import jraph
import numpy as np

from mace_jax import data, tools
from mace_jax.tools import bundle as bundle_tools
from mace_jax.data.streaming_loader import _expand_hdf5_paths


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
            'Path to a JAX model bundle (config.json + params.msgpack). '
            'Pass a directory, config.json, or params.msgpack path.'
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
        help='Edge cap for streaming HDF5 predictions (required for HDF5 inputs).',
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


def _build_predictor(module, *, compute_forces: bool, compute_stress: bool):
    return jax.jit(
        lambda params, graph: module.apply(
            params,
            graph,
            compute_force=compute_forces,
            compute_stress=compute_stress,
        )
    )


def _append_single_prediction(
    outputs: dict[str, list[Any]],
    graph: jraph.GraphsTuple,
    raw_outputs: dict[str, Any],
) -> None:
    node_count = int(np.asarray(graph.n_node).sum())
    for key, value in raw_outputs.items():
        if value is None:
            continue
        arr = np.asarray(value)
        if arr.ndim == 0:
            val = arr.item()
        elif key in _NODE_OUTPUT_KEYS:
            val = arr[:node_count]
        elif key in _GRAPH_OUTPUT_KEYS:
            val = arr.reshape(-1)[0] if arr.shape[0] == 1 else arr
        elif arr.shape[0] == 1:
            val = arr.reshape(-1)[0]
        elif arr.shape[0] == node_count:
            val = arr[:node_count]
        else:
            val = arr
        outputs.setdefault(key, []).append(val)


def _predict_xyz(
    input_path: Path,
    *,
    predictor: Any,
    params: Any,
    model_config: dict[str, Any],
    head_name: str,
    head_to_index: dict[str, int],
) -> tuple[list[int], dict[str, list[Any]]]:
    _, configs = data.load_from_xyz(input_path, head_name=head_name, no_data_ok=True)
    if not configs:
        raise ValueError(f'No configurations found in {input_path}.')
    z_table = data.AtomicNumberTable(model_config['atomic_numbers'])
    graph_ids: list[int] = []
    outputs: dict[str, list[Any]] = {}
    for idx, config in enumerate(configs):
        graph = data.graph_from_configuration(
            config,
            cutoff=float(model_config['r_max']),
            z_table=z_table,
            head_to_index=head_to_index,
        )
        graph_ids.append(idx)
        raw_outputs = predictor(params, graph)
        raw_outputs = jax.device_get(raw_outputs)
        _append_single_prediction(outputs, graph, raw_outputs)
    return graph_ids, outputs


def _predict_hdf5(
    input_path: Path,
    *,
    predictor: Any,
    params: Any,
    model_config: dict[str, Any],
    head_name: str,
    head_to_index: dict[str, int],
    batch_max_edges: int,
    batch_max_nodes: int | None,
    prefetch_batches: int | None,
    num_workers: int,
    progress_bar: bool,
) -> tuple[list[int], dict[str, list[Any]]]:
    z_table = data.AtomicNumberTable(model_config['atomic_numbers'])
    expanded_paths = _expand_hdf5_paths([input_path])
    dataset_specs = [
        data.StreamingDatasetSpec(path=path, head_name=head_name)
        for path in expanded_paths
    ]
    loader = data.get_hdf5_dataloader(
        data_file=expanded_paths,
        atomic_numbers=z_table,
        r_max=float(model_config['r_max']),
        max_nodes=batch_max_nodes,
        max_edges=batch_max_edges,
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
        compute_forces=args.compute_forces,
        compute_stress=args.compute_stress,
    )

    inputs = [Path(path).expanduser() for path in args.inputs]
    multi_input = len(inputs) > 1
    for input_path in inputs:
        if _is_hdf5_path(input_path):
            if args.batch_max_edges is None:
                raise ValueError(
                    f'--batch-max-edges is required for streaming predictions on {input_path}.'
                )
            graph_ids, outputs = _predict_hdf5(
                input_path,
                predictor=predictor,
                params=bundle.params,
                model_config=bundle.config,
                head_name=head_name,
                head_to_index=head_to_index,
                batch_max_edges=int(args.batch_max_edges),
                batch_max_nodes=(
                    int(args.batch_max_nodes)
                    if args.batch_max_nodes is not None
                    else None
                ),
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
