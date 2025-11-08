#!/usr/bin/env python3
"""Run a smoke-test training job on the aspirin config/dataset."""

from __future__ import annotations

import argparse
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

from mace_jax.cli import mace_jax_train

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


def _optional_path(value: str | None, *, ensure_exists: bool = False, label: str = 'dataset') -> str:
    normalized = _normalize_optional(value)
    if normalized is None:
        return 'None'
    path = Path(normalized)
    if ensure_exists:
        _ensure_exists(path, label)
    return _quote_string(path)


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
        '--steps-per-interval',
        type=int,
        default=1,
        help='Number of optimizer steps before evaluating (epoch size).',
    )
    parser.add_argument(
        '--max-intervals',
        type=int,
        default=1,
        help='Stop after this many evaluation intervals.',
    )
    parser.add_argument(
        '--device',
        choices=['auto', 'cpu', 'cuda', 'tpu'],
        default=None,
        help='Optional device hint forwarded to the CLI.',
    )
    parser.add_argument(
        '--print-config',
        action='store_true',
        help='Request the CLI to print the operative gin config.',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Only parse configs and exit without launching training.',
    )
    args, unknown = parser.parse_known_args()
    if unknown:
        raise SystemExit(f'Unexpected extra CLI arguments: {unknown}')

    config_path = _ensure_exists(Path(args.config), 'Gin config')
    train_path = _ensure_exists(Path(args.train_path), 'Training dataset')

    cli_args: list[str] = [str(config_path)]

    bindings = [
        f"mace_jax.tools.gin_datasets.datasets.train_path={_quote_string(train_path)}",
        f"mace_jax.tools.gin_datasets.datasets.valid_path={_optional_path(args.valid_path, ensure_exists=True, label='validation dataset')}",
        f"mace_jax.tools.gin_datasets.datasets.test_path={_optional_path(args.test_path, ensure_exists=True, label='test dataset')}",
        f"mace_jax.tools.gin_functions.flags.seed={args.seed}",
        f"mace_jax.tools.gin_functions.optimizer.steps_per_interval={args.steps_per_interval}",
        f"mace_jax.tools.gin_functions.optimizer.max_num_intervals={args.max_intervals}",
    ]

    if args.log_dir is not None:
        bindings.append(
            f"mace_jax.tools.gin_functions.logs.directory={_quote_string(Path(args.log_dir))}"
        )
    if args.device:
        bindings.append(
            f"mace_jax.tools.gin_functions.flags.device={_quote_string(args.device)}"
        )

    for binding in bindings:
        cli_args += ['--binding', binding]
    if args.print_config:
        cli_args.append('--print-config')
    if args.dry_run:
        cli_args.append('--dry-run')

    mace_jax_train.main(cli_args)


if __name__ == '__main__':
    main()
