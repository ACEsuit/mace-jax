#!/usr/bin/env python3
"""Run a smoke-test training job on the aspirin config/dataset."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

try:
    import torch.serialization as torch_serialization
except Exception:  # pragma: no cover - torch always available in normal installs
    torch_serialization = None
else:
    add_safe_globals = getattr(torch_serialization, 'add_safe_globals', None)
    if add_safe_globals is not None:
        # e3nn caches Wigner constants that include bare slice objects.
        add_safe_globals([slice])  # type: ignore[arg-type]

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / 'configs' / 'aspirin_small.gin'
DEFAULT_DATASET = ROOT / 'data' / '3bpa_train_300K.xyz'


def _ensure_exists(path: Path, description: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{description} '{path}' does not exist.")
    return path


def _quote_string(value: str | Path) -> str:
    return repr(str(value))


def _normalize_optional(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    if stripped.lower() == 'none':
        return None
    return stripped


def _optional_binding_path(path: Path | None) -> str:
    if path is None:
        return 'None'
    return _quote_string(path)


def _is_hdf5_path(path: Path) -> bool:
    if path.is_dir():
        return any(
            child.is_file() and child.suffix.lower() in ('.h5', '.hdf5')
            for child in path.iterdir()
        )
    return path.suffix.lower() in ('.h5', '.hdf5')


def _is_xyz_path(path: Path) -> bool:
    lowered = path.name.lower()
    return lowered.endswith('.xyz') or lowered.endswith('.xyz.gz')


def _hdf5_cache_path(source: Path, cache_dir: Path | None) -> Path:
    lowered = source.name.lower()
    if lowered.endswith('.xyz.gz'):
        stem = source.name[:-7]
    elif lowered.endswith('.xyz'):
        stem = source.name[:-4]
    else:
        stem = source.stem
    target_dir = cache_dir if cache_dir is not None else source.parent
    return target_dir / f'{stem}.h5'


def _int_env(keys: tuple[str, ...], default: int | None = None) -> int | None:
    for key in keys:
        value = os.environ.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except ValueError:
            continue
    return default


def _distributed_info() -> tuple[int, int]:
    rank = _int_env(
        ('RANK', 'LOCAL_RANK', 'SLURM_PROCID', 'OMPI_COMM_WORLD_RANK'), default=0
    )
    world_size = _int_env(
        ('WORLD_SIZE', 'SLURM_NTASKS', 'OMPI_COMM_WORLD_SIZE'), default=1
    )
    return int(rank or 0), int(world_size or 1)


def _convert_xyz_to_hdf5(source: Path, target: Path) -> None:
    import h5py

    from mace_jax.data import utils as data_utils  # noqa: PLC0415

    _, configs = data_utils.load_from_xyz(source.as_posix())
    if not configs:
        raise ValueError(f'No configurations found in {source}')
    target.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(target, 'w') as handle:
        handle.attrs['drop_last'] = False
        data_utils.save_configurations_as_HDF5(configs, 0, handle)


def _maybe_convert_dataset(path: Path, *, cache_dir: Path | None, label: str) -> Path:
    if _is_hdf5_path(path):
        return path
    if path.is_dir():
        raise ValueError(
            f"{label} dataset directory '{path}' does not contain HDF5 files."
        )
    if not _is_xyz_path(path):
        raise ValueError(f"{label} dataset '{path}' must be HDF5 (.h5/.hdf5) or XYZ.")
    target = _hdf5_cache_path(path, cache_dir)
    if target.exists():
        try:
            if target.stat().st_mtime >= path.stat().st_mtime:
                logging.info('Using cached HDF5 %s dataset: %s', label, target)
                return target
        except OSError:
            pass
    logging.info('Converting %s dataset to HDF5: %s -> %s', label, path, target)
    _convert_xyz_to_hdf5(path, target)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            'Smoke-test helper that runs configs/aspirin_small.gin on a tiny dataset.\n'
            'It mirrors the README example and is handy for CI or local validation.'
        )
    )
    parser.add_argument(
        '--config',
        default=str(DEFAULT_CONFIG),
        help='Gin configuration to run (defaults to configs/aspirin_small.gin).',
    )
    parser.add_argument(
        '--train-path',
        default=str(DEFAULT_DATASET),
        help='Training dataset to load (XYZ/HDF5).',
    )
    parser.add_argument(
        '--valid-path',
        default=None,
        help="Optional validation dataset. Defaults to 'None' (no validation set).",
    )
    parser.add_argument(
        '--test-path',
        default=None,
        help="Optional test dataset. Defaults to 'None' (no test set).",
    )
    parser.add_argument(
        '--hdf5-cache-dir',
        default=None,
        help=(
            'Optional directory to store auto-converted HDF5 datasets. '
            'Defaults to the source dataset directory.'
        ),
    )
    parser.add_argument(
        '--log-dir',
        default=None,
        help='Optional results directory override.',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='Training seed forwarded to the CLI.',
    )
    parser.add_argument(
        '--max-epochs',
        '--max-intervals',
        type=int,
        default=1,
        dest='max_epochs',
        help='Stop after this many evaluation intervals.',
    )
    parser.add_argument(
        '--batch-max-edges',
        type=str,
        default=None,
        help="Maximum number of edges per batch, or 'auto' for automatic sizing.",
    )
    parser.add_argument(
        '--device',
        choices=['auto', 'cpu', 'cuda', 'tpu'],
        default='cpu',
        help='Optional device hint forwarded to the CLI (defaults to cpu).',
    )
    parser.add_argument(
        '--launcher',
        choices=['none', 'local', 'auto'],
        default='none',
        help='Launcher mode forwarded to the CLI (defaults to none).',
    )
    parser.add_argument(
        '--distributed',
        action='store_true',
        help='Ignored compatibility flag for launcher-injected arguments.',
    )
    parser.add_argument(
        '--name',
        default=None,
        help='Ignored compatibility flag for launcher-injected arguments.',
    )
    parser.add_argument(
        '--print-config',
        action='store_true',
        help='Request the CLI to print the operative gin config.',
    )
    args, unknown = parser.parse_known_args()
    if unknown:
        raise SystemExit(f'Unexpected extra CLI arguments: {unknown}')

    rank, world_size = _distributed_info()
    if args.distributed or world_size > 1:
        if rank != 0:
            return

    if args.device in (None, 'cpu'):
        # Force JAX to prefer CPU to avoid GPU plugin init errors on systems
        # without CUDA.
        os.environ.setdefault('JAX_PLATFORMS', 'cpu')
        os.environ.setdefault('JAX_PLATFORM_NAME', 'cpu')
        os.environ.setdefault('JAX_PLUGINS', '')
        os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
        logging.getLogger('jax._src.xla_bridge').setLevel(logging.CRITICAL)

    # Delay import until after environment hints are set.
    from mace_jax.cli import mace_jax_train  # noqa: PLC0415

    config_path = _ensure_exists(Path(args.config), 'Gin config')
    cache_dir = Path(args.hdf5_cache_dir) if args.hdf5_cache_dir else None
    train_path = _ensure_exists(Path(args.train_path), 'Training dataset')
    train_path = _maybe_convert_dataset(
        train_path, cache_dir=cache_dir, label='training'
    )
    valid_path: Path | None
    valid_value = _normalize_optional(args.valid_path)
    if valid_value is None:
        valid_path = None
    else:
        valid_path = _ensure_exists(Path(valid_value), 'Validation dataset')
        valid_path = _maybe_convert_dataset(
            valid_path, cache_dir=cache_dir, label='validation'
        )
    test_path: Path | None
    test_value = _normalize_optional(args.test_path)
    if test_value is None:
        test_path = None
    else:
        test_path = _ensure_exists(Path(test_value), 'Test dataset')
        test_path = _maybe_convert_dataset(test_path, cache_dir=cache_dir, label='test')

    cli_args: list[str] = [str(config_path)]
    cli_args += ['--launcher', args.launcher]

    bindings = [
        f'mace_jax.tools.gin_datasets.datasets.train_path={_quote_string(train_path)}',
        f'mace_jax.tools.gin_datasets.datasets.valid_path={_optional_binding_path(valid_path)}',
        f'mace_jax.tools.gin_datasets.datasets.test_path={_optional_binding_path(test_path)}',
        f'mace_jax.tools.gin_functions.flags.seed={args.seed}',
        f'mace_jax.tools.gin_functions.optimizer.max_epochs={args.max_epochs}',
    ]

    def _format_batch_limit(value: str | None) -> str | None:
        if value is None:
            return None
        lowered = value.strip().lower()
        if lowered in {'auto', 'none'}:
            return 'None'
        return str(int(value))

    if args.batch_max_edges is not None:
        limit = _format_batch_limit(args.batch_max_edges)
        bindings.append(f'mace_jax.tools.gin_datasets.datasets.n_edge={limit}')

    if args.log_dir is not None:
        bindings.append(
            f'mace_jax.tools.gin_functions.logs.directory={_quote_string(Path(args.log_dir))}'
        )
    if args.device:
        bindings.append(
            f'mace_jax.tools.gin_functions.flags.device={_quote_string(args.device)}'
        )

    for binding in bindings:
        cli_args += ['--binding', binding]
    if args.print_config:
        cli_args.append('--print-config')

    mace_jax_train.main(cli_args)


if __name__ == '__main__':
    main()
