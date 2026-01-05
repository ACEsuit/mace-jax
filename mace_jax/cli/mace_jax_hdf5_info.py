from __future__ import annotations

import argparse
import logging
import pickle
import re
from collections.abc import Iterable
from pathlib import Path

import h5py

from mace_jax.data.streaming_loader import _expand_hdf5_paths
from mace_jax.data.streaming_stats_cache import dataset_signature, stats_cache_path


def _sorted_numeric(names: Iterable[str], prefix: str) -> list[str]:
    def _key(name: str):
        match = re.search(r'(\d+)$', name)
        return int(match.group(1)) if match else name

    return sorted((name for name in names if name.startswith(prefix)), key=_key)


def _format_bytes(size: int) -> str:
    units = ['B', 'KiB', 'MiB', 'GiB', 'TiB']
    value = float(size)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f'{value:.1f} {unit}'
        value /= 1024.0
    return f'{value:.1f} {units[-1]}'


def _load_streaming_cache(path: Path) -> dict | None:
    cache_path = stats_cache_path(path)
    if not cache_path.exists():
        return None
    try:
        with cache_path.open('rb') as fh:
            return pickle.load(fh)
    except Exception as exc:  # pragma: no cover - corrupted cache
        logging.warning('Failed to read streaming stats cache %s: %s', cache_path, exc)
        return None


def _log_hdf5_info(path: Path) -> None:
    logging.info('HDF5 file: %s', path)
    try:
        size_bytes = path.stat().st_size
    except OSError:
        size_bytes = 0
    if size_bytes:
        logging.info('  size: %s', _format_bytes(size_bytes))

    with h5py.File(path, 'r') as handle:
        batch_names = _sorted_numeric(handle.keys(), 'config_batch_')
        if not batch_names:
            logging.info('  batches: 0 (no config_batch_* groups found)')
            return
        batch_sizes = []
        for batch in batch_names:
            group = handle[batch]
            batch_sizes.append(len(_sorted_numeric(group.keys(), 'config_')))
        total_graphs = sum(batch_sizes)
        batch_count = len(batch_names)
        logging.info('  batches: %s', batch_count)
        logging.info('  graphs: %s', total_graphs)
        logging.info(
            '  batch_size: min=%s mean=%.1f max=%s',
            min(batch_sizes),
            total_graphs / float(batch_count),
            max(batch_sizes),
        )

        drop_last = handle.attrs.get('drop_last')
        if drop_last is not None:
            logging.info('  drop_last: %s', bool(drop_last))

        first_batch = handle[batch_names[0]]
        config_names = _sorted_numeric(first_batch.keys(), 'config_')
        if config_names:
            config = first_batch[config_names[0]]
            keys = sorted(config.keys())
            logging.info('  config keys: %s', ', '.join(keys))
            if 'properties' in config:
                props = sorted(config['properties'].keys())
                logging.info('  properties: %s', ', '.join(props) or '(none)')
            if 'property_weights' in config:
                weights = sorted(config['property_weights'].keys())
                logging.info('  property_weights: %s', ', '.join(weights) or '(none)')

    cache_payload = _load_streaming_cache(path)
    if cache_payload is None:
        logging.info('  streaming_stats: none')
        return

    stats = cache_payload.get('stats') or {}
    cached_sig = cache_payload.get('dataset_signature')
    current_sig = dataset_signature(path)
    stale = cached_sig != current_sig
    logging.info(
        '  streaming_stats: n_nodes=%s n_edges=%s n_graphs=%s n_batches=%s',
        stats.get('n_nodes'),
        stats.get('n_edges'),
        stats.get('n_graphs'),
        stats.get('n_batches'),
    )
    logging.info('  streaming_stats_cache: %s', stats_cache_path(path).name)
    if stale:
        logging.info('  streaming_stats_cache: stale (dataset signature changed)')


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description='Inspect MACE-style HDF5 datasets and cached streaming stats.',
    )
    parser.add_argument(
        'paths',
        nargs='+',
        help='HDF5 files, directories, or glob patterns.',
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format='%(message)s')
    expanded: list[Path] = []
    for raw in args.paths:
        expanded.extend(_expand_hdf5_paths([raw]))
    if not expanded:
        raise ValueError('No HDF5 files found.')

    for idx, path in enumerate(expanded):
        if idx:
            logging.info('')
        _log_hdf5_info(path)


if __name__ == '__main__':
    main()
