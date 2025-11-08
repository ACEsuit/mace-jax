from __future__ import annotations

import argparse
import ast
import atexit
import json
import logging
import shutil
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

import gin
import jax
import optax
import torch

import mace_jax
from mace_jax import tools
from mace_jax.tools import gin_datasets, gin_functions, gin_model
from mace_jax.tools.arg_parser import build_cli_arg_parser
from mace_jax.tools.train import SWAConfig

_OPTIMIZER_ALGORITHMS = {
    'adam': optax.scale_by_adam,
    'amsgrad': tools.scale_by_amsgrad,
    'sgd': optax.identity,
}

_SCHEDULERS = {
    'constant': gin_functions.constant_schedule,
    'exponential': gin_functions.exponential_decay,
    'piecewise_constant': gin_functions.piecewise_constant_schedule,
}

_FOUNDATION_TEMP_DIRS: list[Path] = []


def _cleanup_foundation_artifacts() -> None:
    while _FOUNDATION_TEMP_DIRS:
        path = _FOUNDATION_TEMP_DIRS.pop()
        shutil.rmtree(path, ignore_errors=True)


atexit.register(_cleanup_foundation_artifacts)


def parse_args(
    argv: Sequence[str] | None = None,
) -> tuple[argparse.Namespace, list[str]]:
    parser = build_cli_arg_parser()
    args, remaining = parser.parse_known_args(argv)
    return args, remaining


def configure_gin(args: argparse.Namespace, remaining: list[str]) -> None:
    if args.configs or args.binding:
        gin.parse_config_files_and_bindings(args.configs, tuple(args.binding))
    if remaining:
        gin_functions.parse_argv([sys.argv[0], *remaining])
    if not args.configs and not args.binding and not remaining:
        raise ValueError(
            'No gin configuration supplied. '
            'Provide at least one config file or binding.'
        )


def apply_cli_overrides(args: argparse.Namespace) -> None:
    """Bind gin parameters based on CLI overrides."""
    _apply_run_options(args)
    _apply_dataset_options(args)
    _apply_training_controls(args)
    _apply_swa_options(args)
    _apply_wandb_options(args)
    _apply_optimizer_options(args)
    _prepare_foundation_checkpoint(args)
    if args.torch_checkpoint:
        gin.bind_parameter(
            'mace_jax.tools.gin_model.model.torch_checkpoint',
            str(Path(args.torch_checkpoint)),
        )
    if args.torch_head:
        gin.bind_parameter(
            'mace_jax.tools.gin_model.model.torch_head',
            args.torch_head,
        )
    if args.torch_param_dtype:
        gin.bind_parameter(
            'mace_jax.tools.gin_model.model.torch_param_dtype',
            args.torch_param_dtype,
        )
    if args.r_max is not None:
        gin.bind_parameter('mace_jax.tools.gin_datasets.datasets.r_max', args.r_max)
        gin.bind_parameter('mace_jax.tools.gin_model.model.r_max', args.r_max)

    head_configs_data = None
    if getattr(args, 'head_config', None):
        head_configs_data = _load_head_configs(Path(args.head_config))
    if getattr(args, 'heads_config', None):
        inline_configs = _parse_heads_literal(args.heads_config)
        if head_configs_data:
            raise ValueError(
                'Specify either --head-config or --heads-config, not both.'
            )
        head_configs_data = inline_configs
    if head_configs_data:
        gin.bind_parameter(
            'mace_jax.tools.gin_datasets.datasets.head_configs', head_configs_data
        )

    heads_list = None
    if head_configs_data:
        heads_list = list(head_configs_data.keys())
        if getattr(args, 'heads', None):
            explicit = list(args.heads)
            if set(explicit) != set(heads_list):
                raise ValueError(
                    '--heads must match the keys defined in --head-config/--heads-config.'
                )
            heads_list = explicit
    elif getattr(args, 'heads', None):
        heads_list = list(args.heads)

    if heads_list:
        gin.bind_parameter(
            'mace_jax.tools.gin_datasets.datasets.heads', tuple(heads_list)
        )
        gin.bind_parameter('mace_jax.tools.gin_model.model.heads', tuple(heads_list))


def _load_head_configs(path: Path) -> dict[str, dict]:
    raw_text = path.read_text(encoding='utf-8')
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore
        except ImportError as exc:  # pragma: no cover - only triggered without PyYAML
            raise ValueError(
                f'Failed to parse head config {path}. Install PyYAML to use YAML files.'
            ) from exc
        data = yaml.safe_load(raw_text)

    if not isinstance(data, dict):
        raise ValueError(f'Head configuration file {path} must define a dict.')

    normalized: dict[str, dict] = {}
    for head_name, cfg in data.items():
        if not isinstance(cfg, dict):
            raise ValueError(
                f'Head configuration for {head_name!r} must be a mapping, got {type(cfg)}.'
            )
        normalized[str(head_name)] = cfg
    return normalized


def _parse_heads_literal(literal: str) -> dict[str, dict]:
    try:
        data = ast.literal_eval(literal)
    except (ValueError, SyntaxError) as exc:
        raise ValueError(
            f'--heads expects a Python/JSON dictionary string, got {literal!r}.'
        ) from exc
    if not isinstance(data, dict):
        raise ValueError('--heads must evaluate to a dictionary mapping head names.')
    normalized = {}
    for key, value in data.items():
        if not isinstance(value, dict):
            raise ValueError(
                f'Head definition for {key!r} must be a dict, got {type(value)}.'
            )
        normalized[str(key)] = value
    return normalized


def _bind_if_not_none(param: str, value, *, transform=None) -> None:
    if value is None:
        return
    gin.bind_parameter(param, transform(value) if transform else value)


def _normalize_optional_path(
    path_str: str | None, allow_none: bool = False
) -> str | None:
    if path_str is None:
        return None
    if allow_none and path_str == 'None':
        return None
    return str(Path(path_str))


def _apply_run_options(args: argparse.Namespace) -> None:
    _bind_if_not_none('mace_jax.tools.gin_functions.logs.name', args.name)
    if args.log_dir is not None:
        _bind_if_not_none(
            'mace_jax.tools.gin_functions.logs.directory',
            str(Path(args.log_dir)),
        )
    _bind_if_not_none('mace_jax.tools.gin_functions.flags.seed', args.seed)
    _bind_if_not_none('mace_jax.tools.gin_functions.flags.dtype', args.dtype)
    if args.debug is not None:
        gin.bind_parameter('mace_jax.tools.gin_functions.flags.debug', args.debug)


def _apply_dataset_options(args: argparse.Namespace) -> None:
    if args.train_path:
        gin.bind_parameter(
            'mace_jax.tools.gin_datasets.datasets.train_path',
            _normalize_optional_path(args.train_path),
        )
    if args.valid_path is not None:
        gin.bind_parameter(
            'mace_jax.tools.gin_datasets.datasets.valid_path',
            _normalize_optional_path(args.valid_path, allow_none=True),
        )
    if args.test_path is not None:
        gin.bind_parameter(
            'mace_jax.tools.gin_datasets.datasets.test_path',
            _normalize_optional_path(args.test_path, allow_none=True),
        )
    _bind_if_not_none(
        'mace_jax.tools.gin_datasets.datasets.valid_fraction', args.valid_fraction
    )
    _bind_if_not_none('mace_jax.tools.gin_datasets.datasets.valid_num', args.valid_num)
    _bind_if_not_none('mace_jax.tools.gin_datasets.datasets.test_num', args.test_num)


def _apply_training_controls(args: argparse.Namespace) -> None:
    _bind_if_not_none(
        'mace_jax.tools.gin_functions.train.max_grad_norm', args.clip_grad
    )
    ema_decay = args.ema_decay
    if getattr(args, 'ema', False) and ema_decay is None:
        ema_decay = 0.99
    _bind_if_not_none('mace_jax.tools.gin_functions.train.ema_decay', ema_decay)


def _apply_swa_options(args: argparse.Namespace) -> None:
    requested = args.swa or any(
        value is not None
        for value in (
            args.swa_start,
            args.swa_every,
            args.swa_min_snapshots,
            args.swa_max_snapshots,
            args.swa_prefer,
        )
    )
    if not requested:
        return
    swa_config = SWAConfig(
        start_interval=args.swa_start if args.swa_start is not None else 0,
        update_interval=args.swa_every if args.swa_every is not None else 1,
        min_snapshots_for_eval=args.swa_min_snapshots
        if args.swa_min_snapshots is not None
        else 1,
        max_snapshots=args.swa_max_snapshots,
        prefer_swa_params=args.swa_prefer if args.swa_prefer is not None else True,
    )
    gin.bind_parameter('mace_jax.tools.gin_functions.train.swa_config', swa_config)


def _apply_wandb_options(args: argparse.Namespace) -> None:
    has_tags = bool(args.wandb_tags)
    wandb_requested = (
        args.wandb_enable
        or any(
            value is not None
            for value in (
                args.wandb_project,
                args.wandb_entity,
                args.wandb_group,
                args.wandb_run_name,
                args.wandb_notes,
                args.wandb_mode,
                args.wandb_dir,
            )
        )
        or has_tags
    )

    if not wandb_requested:
        if args.wandb_enable:
            gin.bind_parameter('mace_jax.tools.gin_functions.wandb_run.enabled', True)
        return

    gin.bind_parameter('mace_jax.tools.gin_functions.wandb_run.enabled', True)
    _bind_if_not_none(
        'mace_jax.tools.gin_functions.wandb_run.project', args.wandb_project
    )
    _bind_if_not_none(
        'mace_jax.tools.gin_functions.wandb_run.entity', args.wandb_entity
    )
    _bind_if_not_none('mace_jax.tools.gin_functions.wandb_run.group', args.wandb_group)
    _bind_if_not_none(
        'mace_jax.tools.gin_functions.wandb_run.name', args.wandb_run_name
    )
    _bind_if_not_none('mace_jax.tools.gin_functions.wandb_run.notes', args.wandb_notes)
    if args.wandb_tags:
        gin.bind_parameter(
            'mace_jax.tools.gin_functions.wandb_run.tags', tuple(args.wandb_tags)
        )
    _bind_if_not_none('mace_jax.tools.gin_functions.wandb_run.mode', args.wandb_mode)
    if args.wandb_dir is not None:
        gin.bind_parameter(
            'mace_jax.tools.gin_functions.wandb_run.dir', str(Path(args.wandb_dir))
        )


def _apply_optimizer_options(args: argparse.Namespace) -> None:
    if getattr(args, 'optimizer', None):
        gin.bind_parameter(
            'mace_jax.tools.gin_functions.optimizer.algorithm',
            _OPTIMIZER_ALGORITHMS[args.optimizer],
        )
    _bind_if_not_none('mace_jax.tools.gin_functions.optimizer.lr', args.lr)
    _bind_if_not_none(
        'mace_jax.tools.gin_functions.optimizer.weight_decay', args.weight_decay
    )
    if getattr(args, 'scheduler', None):
        gin.bind_parameter(
            'mace_jax.tools.gin_functions.optimizer.scheduler',
            _SCHEDULERS[args.scheduler],
        )
    if getattr(args, 'lr_scheduler_gamma', None) is not None:
        gin.bind_parameter(
            'mace_jax.tools.gin_functions.exponential_decay.decay_rate',
            args.lr_scheduler_gamma,
        )


def _load_foundation_model(name: str, *, default_dtype: str | None = None):
    from mace.calculators import foundations_models
    from mace.calculators.foundations_models import mace_mp_names

    name = name.lower()
    dtype = default_dtype or 'float32'
    mp_names = {m for m in mace_mp_names if m}
    if name in mp_names:
        calc = foundations_models.mace_mp(
            model=name,
            device='cpu',
            default_dtype=dtype,
        )
        return calc.models[0]
    if name in {'small_off', 'medium_off', 'large_off'}:
        model_type = name.split('_')[0]
        calc = foundations_models.mace_off(
            model=model_type,
            device='cpu',
            default_dtype=dtype,
        )
        return calc.models[0]
    raise ValueError(
        f"Unknown foundation_model '{name}'. Provide a valid mace-mp/mace-off name or a checkpoint path."
    )


def _prepare_foundation_checkpoint(args: argparse.Namespace) -> None:
    foundation_spec = getattr(args, 'foundation_model', None)
    if not foundation_spec:
        return
    if args.torch_checkpoint:
        raise ValueError('Specify either --torch-checkpoint or --foundation-model, not both.')
    candidate = Path(foundation_spec)
    if candidate.exists():
        args.torch_checkpoint = str(candidate)
    else:
        torch_model = _load_foundation_model(
            foundation_spec, default_dtype=args.dtype
        )
        tmp_dir = Path(tempfile.mkdtemp(prefix='mace_jax_foundation_'))
        checkpoint_path = tmp_dir / f'{foundation_spec}.pt'
        torch.save({'model': torch_model}, checkpoint_path)
        _FOUNDATION_TEMP_DIRS.append(tmp_dir)
        args.torch_checkpoint = str(checkpoint_path)
        if args.r_max is None and hasattr(torch_model, 'r_max'):
            args.r_max = float(torch_model.r_max.item())
    if args.foundation_head and not args.torch_head:
        args.torch_head = args.foundation_head


def run_training(dry_run: bool = False) -> None:
    seed = gin_functions.flags()
    directory, tag, logger = gin_functions.logs()

    Path(directory).mkdir(parents=True, exist_ok=True)
    operative_config = gin.operative_config_str()
    with open(Path(directory) / f'{tag}.gin', 'w', encoding='utf-8') as f:
        f.write(operative_config)

    logging.info('MACE-JAX version: %s', mace_jax.__version__)

    wandb_run = None
    try:
        if dry_run:
            logging.info(
                'Dry-run requested; skipping dataset construction and training.'
            )
            return

        (
            train_loader,
            valid_loader,
            test_loader,
            atomic_energies_dict,
            r_max,
        ) = gin_datasets.datasets()

        wandb_run = gin_functions.wandb(
            name=tag,
            dir=directory,
            config={
                'gin_config': operative_config,
                'mace_jax_version': mace_jax.__version__,
                'seed': seed,
            },
        )

        model_fn, params, num_message_passing = gin_model.model(
            r_max=r_max,
            atomic_energies_dict=atomic_energies_dict,
            train_graphs=train_loader.graphs,
            initialize_seed=seed,
        )
        logging.info('Number of interaction blocks: %s', num_message_passing)

        params = gin_functions.reload(params)

        predictor = jax.jit(
            lambda w, g: model_fn(w, g, compute_force=True, compute_stress=True)
        )

        if gin_functions.checks(predictor, params, train_loader):
            logging.info('Checks enabled; exiting after sanity verification.')
            return

        gradient_transform, steps_per_interval, max_num_intervals = (
            gin_functions.optimizer()
        )
        optimizer_state = gradient_transform.init(params)

        num_params = tools.count_parameters(params)
        num_opt_params = tools.count_parameters(optimizer_state)
        logging.info('Number of parameters: %s', num_params)
        logging.info('Number of optimizer parameters: %s', num_opt_params)

        if wandb_run is not None:

            def _graph_count(loader):
                if loader is None:
                    return 0
                graphs = getattr(loader, 'graphs', None)
                if graphs is None:
                    return 0
                return len(graphs)

            dataset_info = {
                'num_train_graphs': _graph_count(train_loader),
                'num_valid_graphs': _graph_count(valid_loader),
                'num_test_graphs': _graph_count(test_loader),
                'num_message_passing': num_message_passing,
                'num_parameters': num_params,
            }
            wandb_run.config.update(dataset_info, allow_val_change=True)

        gin_functions.train(
            predictor=predictor,
            params=params,
            optimizer_state=optimizer_state,
            train_loader=train_loader,
            valid_loader=valid_loader,
            test_loader=test_loader,
            gradient_transform=gradient_transform,
            max_num_intervals=max_num_intervals,
            steps_per_interval=steps_per_interval,
            logger=logger,
            directory=directory,
            tag=tag,
            wandb_run=wandb_run,
        )
    finally:
        gin_functions.finish_wandb(wandb_run)


def main(argv: Sequence[str] | None = None) -> None:
    args, remaining = parse_args(argv)
    configure_gin(args, remaining)
    apply_cli_overrides(args)

    if args.print_config:
        logging.info('Operative gin config:\n%s', gin.config_str())

    run_training(dry_run=args.dry_run)


if __name__ == '__main__':
    main()
