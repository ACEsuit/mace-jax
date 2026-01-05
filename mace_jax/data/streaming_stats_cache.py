"""Shared streaming stats cache helpers.

Streaming stats (n_nodes/n_edges/n_graphs/n_batches) are computed once per
dataset/shard and reused across runs to avoid expensive rescans. The cache is
stored as a hidden sidecar file (``.<stem>.streamstats.pkl``) next to each HDF5
shard, so glob patterns like ``dir/*.h5`` do not treat the cache as data.

Cache entries are invalidated when either the dataset signature (size/mtime) or
the spec fingerprint changes (e.g. r_max, atomic_numbers, edge cap, or node
percentile). These helpers are used by gin_datasets when building streaming
train/valid/test loaders.
"""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
from collections.abc import Sequence
from pathlib import Path

STREAMING_STATS_CACHE_VERSION = 1


def stats_cache_path(dataset_path: Path) -> Path:
    """Return the cache path for streaming stats for a dataset.

    Uses a hidden sidecar file to avoid treating the cache as a shard when
    globbing directories (e.g. ``dir/*.h5``).
    """
    cache_name = f'.{dataset_path.stem}.streamstats.pkl'
    return dataset_path.with_name(cache_name)


def dataset_signature(dataset_path: Path) -> dict[str, float]:
    """Compute a lightweight signature used to invalidate cached stats.

    The signature uses file size and mtime as a fast change detector.
    """
    stat = dataset_path.stat()
    return {'size': stat.st_size, 'mtime': stat.st_mtime}


def normalized_config_type_weights(weights: dict | None) -> dict | None:
    """Normalize config type weights for hashing and cache fingerprints."""
    if not weights:
        return None
    return {str(key): float(value) for key, value in sorted(weights.items())}


def spec_fingerprint(
    spec,
    *,
    r_max: float,
    atomic_numbers: Sequence[int],
    head_to_index: dict[str, int] | None = None,
    edge_cap: int | None = None,
    node_percentile: float | None = None,
) -> str:
    """Build a stable fingerprint for dataset spec + packing parameters.

    Any change to dataset interpretation (keys, weights, r_max, head index,
    edge cap, node percentile, etc.) yields a new fingerprint and invalidates
    the cached stats.
    """
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
        'config_type_weights': normalized_config_type_weights(spec.config_type_weights),
        'remap_stress': (
            spec.remap_stress.tolist() if spec.remap_stress is not None else None
        ),
        'path': str(Path(spec.path)),
        'r_max': float(r_max),
        'atomic_numbers': [int(z) for z in atomic_numbers],
        'edge_cap': int(edge_cap) if edge_cap is not None else None,
        'node_percentile': (
            float(node_percentile) if node_percentile is not None else None
        ),
    }
    encoded = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(encoded.encode('utf-8')).hexdigest()


def stats_payload_from_parts(
    n_nodes: int,
    n_edges: int,
    n_graphs: int,
    n_batches: int | None = None,
) -> dict:
    """Serialize streaming caps to a JSON/pickle-friendly payload."""
    payload = {
        'n_nodes': int(n_nodes),
        'n_edges': int(n_edges),
        'n_graphs': int(n_graphs),
    }
    if n_batches is not None:
        payload['n_batches'] = int(n_batches)
    return payload


def stats_payload_to_parts(payload: dict) -> tuple[int, int, int, int | None]:
    """Deserialize streaming stats payload back into components."""
    n_batches = payload.get('n_batches')
    return (
        int(payload['n_nodes']),
        int(payload['n_edges']),
        int(payload['n_graphs']),
        int(n_batches) if n_batches is not None else None,
    )


def load_cached_streaming_stats(dataset_path: Path, fingerprint: str) -> dict | None:
    """Load cached streaming stats if they match the current dataset and spec.

    Returns None when the cache is missing, stale, or incompatible.
    """
    cache_path = stats_cache_path(dataset_path)
    payload = None
    if not cache_path.exists():
        return None
    try:
        with cache_path.open('rb') as fh:
            payload = pickle.load(fh)
    except Exception as exc:  # pragma: no cover - cache corruption is unexpected
        logging.warning('Failed to load streaming stats cache %s: %s', cache_path, exc)
        return None
    if payload is None:
        return None
    if payload.get('version') != STREAMING_STATS_CACHE_VERSION:
        return None
    dataset_sig = payload.get('dataset_signature')
    if dataset_sig != dataset_signature(dataset_path):
        return None
    if payload.get('spec_fingerprint') != fingerprint:
        return None
    stats_payload = payload.get('stats')
    if not stats_payload:
        return None
    logging.info('Loaded cached streaming stats from %s', cache_path)
    return stats_payload


def store_cached_streaming_stats(
    dataset_path: Path,
    fingerprint: str,
    stats_payload: dict,
) -> None:
    """Persist streaming stats to disk for reuse across runs.

    The cached payload is a pickle containing the version, dataset signature,
    spec fingerprint, and the stats payload.
    """
    cache_path = stats_cache_path(dataset_path)
    payload = {
        'version': STREAMING_STATS_CACHE_VERSION,
        'dataset_signature': dataset_signature(dataset_path),
        'spec_fingerprint': fingerprint,
        'stats': stats_payload,
    }
    try:
        with cache_path.open('wb') as fh:
            pickle.dump(payload, fh)
    except OSError as exc:  # pragma: no cover - filesystem failure
        logging.warning('Failed to write streaming stats cache %s: %s', cache_path, exc)
        return
    logging.debug('Cached streaming stats to %s', cache_path)
