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
from mace_jax import modules, tools
from mace_jax.tools import gin_datasets, gin_functions, gin_model
from mace_jax.tools.arg_parser import build_cli_arg_parser
from mace_jax.tools.train import SWAConfig

_OPTIMIZER_ALGORITHMS = {
    'adam': optax.scale_by_adam,
    'adamw': optax.scale_by_adam,
    'amsgrad': tools.scale_by_amsgrad,
    'sgd': optax.identity,
    'schedulefree': optax.scale_by_adam,
}

_SCHEDULERS = {
    'constant': gin_functions.constant_schedule,
    'exponential': gin_functions.exponential_decay,
    'piecewise_constant': gin_functions.piecewise_constant_schedule,
    'plateau': gin_functions.reduce_on_plateau,
}

_LOSS_FACTORIES = {
    'weighted': modules.WeightedEnergyForcesStressLoss,
    'stress': modules.WeightedEnergyForcesStressLoss,
    'ef': modules.WeightedEnergyForcesLoss,
    'forces_only': modules.WeightedForcesLoss,
    'huber': modules.WeightedHuberEnergyForcesStressLoss,
    'virials': modules.WeightedEnergyForcesVirialsLoss,
    'dipole': modules.WeightedEnergyForcesDipoleLoss,
    'energy_forces_dipole': modules.WeightedEnergyForcesDipoleLoss,
    'l1l2': modules.WeightedEnergyForcesL1L2Loss,
}

_FOUNDATION_TEMP_DIRS: list[Path] = []


def _is_named_mp_foundation(name: str) -> bool:
    from mace.calculators.foundations_models import mace_mp_names  # noqa: PLC0415

    normalized = name.lower()
    known = {m for m in mace_mp_names if m}
    known.update({'small', 'medium', 'large', 'small_off', 'medium_off', 'large_off'})
    return normalized in known


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
    _prepare_foundation_checkpoint(args)
    _apply_dataset_options(args)
    _apply_swa_options(args)
    _apply_wandb_options(args)
    _maybe_adjust_multihead_defaults(args)
    _apply_loss_options(args)
    _apply_training_controls(args)
    _apply_optimizer_options(args)
    if args.checkpoint_dir:
        gin.bind_parameter(
            'mace_jax.tools.gin_functions.train.checkpoint_dir',
            str(Path(args.checkpoint_dir)),
        )
    if args.checkpoint_every is not None:
        gin.bind_parameter(
            'mace_jax.tools.gin_functions.train.checkpoint_every',
            args.checkpoint_every,
        )
    if args.checkpoint_keep is not None:
        gin.bind_parameter(
            'mace_jax.tools.gin_functions.train.checkpoint_keep', args.checkpoint_keep
        )
    if args.resume_from:
        gin.bind_parameter(
            'mace_jax.tools.gin_functions.train.resume_from',
            str(Path(args.resume_from)),
        )
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
    head_configs_data = _inject_pt_head_config(head_configs_data, args)
    if head_configs_data:
        gin.bind_parameter(
            'mace_jax.tools.gin_datasets.datasets.head_configs', head_configs_data
        )

    heads_list = None
    if head_configs_data:
        configs_keys = list(head_configs_data.keys())
        if getattr(args, 'heads', None):
            explicit = list(args.heads)
            for key in configs_keys:
                if key not in explicit:
                    explicit.append(key)
            heads_list = explicit
        else:
            heads_list = configs_keys
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
            import yaml  # noqa: PLC0415
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


def _inject_pt_head_config(
    head_configs: dict[str, dict] | None, args: argparse.Namespace
) -> dict[str, dict] | None:
    train_files = getattr(args, 'pt_train_file', None)
    valid_files = getattr(args, 'pt_valid_file', None)
    if not train_files and not valid_files:
        return head_configs

    def _list_paths(items):
        if not items:
            return None
        return [str(Path(p)) for p in items]

    configs = dict(head_configs) if head_configs else {}
    head_name = getattr(args, 'pt_head_name', 'pt_head') or 'pt_head'
    entry = dict(configs.get(head_name, {}))

    normalized_train = _list_paths(train_files)
    normalized_valid = _list_paths(valid_files)

    if normalized_train:
        entry['train_path'] = normalized_train
    if normalized_valid:
        entry['valid_path'] = normalized_valid

    if not entry:
        raise ValueError(
            'At least one of --pt_train_file/--pt_valid_file must be provided.'
        )

    configs[head_name] = entry
    return configs


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
    _bind_if_not_none('mace_jax.tools.gin_functions.flags.device', args.device)
    if getattr(args, 'distributed', False):
        gin.bind_parameter('mace_jax.tools.gin_functions.flags.distributed', True)
    _bind_if_not_none(
        'mace_jax.tools.gin_functions.flags.process_count', args.process_count
    )
    _bind_if_not_none(
        'mace_jax.tools.gin_functions.flags.process_index', args.process_index
    )
    _bind_if_not_none(
        'mace_jax.tools.gin_functions.flags.coordinator_address',
        args.coordinator_address,
    )
    _bind_if_not_none(
        'mace_jax.tools.gin_functions.flags.coordinator_port',
        args.coordinator_port,
    )


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
    _bind_if_not_none(
        'mace_jax.tools.gin_datasets.datasets.energy_key', args.energy_key
    )
    _bind_if_not_none(
        'mace_jax.tools.gin_datasets.datasets.forces_key', args.forces_key
    )


def _apply_training_controls(args: argparse.Namespace) -> None:
    if args.steps_per_interval is not None:
        gin.bind_parameter(
            'mace_jax.tools.gin_functions.optimizer.steps_per_interval',
            args.steps_per_interval,
        )
    if args.max_num_intervals is not None:
        gin.bind_parameter(
            'mace_jax.tools.gin_functions.optimizer.max_num_intervals',
            args.max_num_intervals,
        )
    _bind_if_not_none(
        'mace_jax.tools.gin_functions.train.max_grad_norm', args.clip_grad
    )
    ema_decay = args.ema_decay
    if getattr(args, 'ema', False) and ema_decay is None:
        ema_decay = 0.99
    _bind_if_not_none('mace_jax.tools.gin_functions.train.ema_decay', ema_decay)
    _bind_if_not_none('mace_jax.tools.gin_functions.train.patience', args.patience)
    _bind_if_not_none(
        'mace_jax.tools.gin_functions.train.eval_interval', args.eval_interval
    )
    if args.eval_train_fraction is not None:
        gin.bind_parameter(
            'mace_jax.tools.gin_functions.train.eval_train', args.eval_train_fraction
        )
    elif args.eval_train is not None:
        gin.bind_parameter(
            'mace_jax.tools.gin_functions.train.eval_train', args.eval_train
        )
    if args.eval_test is not None:
        gin.bind_parameter(
            'mace_jax.tools.gin_functions.train.eval_test', args.eval_test
        )
    if args.log_errors is not None:
        gin.bind_parameter(
            'mace_jax.tools.gin_functions.train.log_errors', args.log_errors
        )


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
    stage_loss_kwargs = _collect_swa_loss_kwargs(args)
    swa_config = SWAConfig(
        start_interval=args.swa_start if args.swa_start is not None else 0,
        update_interval=args.swa_every if args.swa_every is not None else 1,
        min_snapshots_for_eval=args.swa_min_snapshots
        if args.swa_min_snapshots is not None
        else 1,
        max_snapshots=args.swa_max_snapshots,
        prefer_swa_params=args.swa_prefer if args.swa_prefer is not None else True,
        stage_loss_factory=_resolve_swa_loss_factory(args, stage_loss_kwargs),
        stage_loss_kwargs=stage_loss_kwargs,
    )
    gin.bind_parameter('mace_jax.tools.gin_functions.train.swa_config', swa_config)
    if args.swa_lr is not None:
        gin.bind_parameter(
            'mace_jax.tools.gin_functions.optimizer.stage_two_lr', args.swa_lr
        )
        gin.bind_parameter(
            'mace_jax.tools.gin_functions.optimizer.stage_two_interval',
            args.swa_start if args.swa_start is not None else swa_config.start_interval,
        )


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
        schedule_free_selected = args.optimizer == 'schedulefree'
        gin.bind_parameter(
            'mace_jax.tools.gin_functions.optimizer.algorithm',
            _OPTIMIZER_ALGORITHMS[args.optimizer],
        )
        decoupled = True
        if args.optimizer in {'adam', 'amsgrad'}:
            decoupled = False
        gin.bind_parameter(
            'mace_jax.tools.gin_functions.optimizer.decoupled_weight_decay',
            decoupled,
        )
        if schedule_free_selected:
            gin.bind_parameter(
                'mace_jax.tools.gin_functions.optimizer.schedule_free', True
            )
            _bind_if_not_none(
                'mace_jax.tools.gin_functions.optimizer.schedule_free_b1',
                args.schedule_free_b1,
            )
            _bind_if_not_none(
                'mace_jax.tools.gin_functions.optimizer.schedule_free_weight_lr_power',
                args.schedule_free_weight_lr_power,
            )
            gin.bind_parameter('adam.b1', 0.0)
            gin.bind_parameter('amsgrad.b1', 0.0)
        elif args.beta is not None:
            gin.bind_parameter('adam.b1', args.beta)
            gin.bind_parameter('amsgrad.b1', args.beta)
    elif args.beta is not None:
        gin.bind_parameter('adam.b1', args.beta)
        gin.bind_parameter('amsgrad.b1', args.beta)
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
    if getattr(args, 'lr_factor', None) is not None:
        gin.bind_parameter(
            'mace_jax.tools.gin_functions.reduce_on_plateau.factor',
            args.lr_factor,
        )
    if getattr(args, 'scheduler_patience', None) is not None:
        gin.bind_parameter(
            'mace_jax.tools.gin_functions.reduce_on_plateau.patience',
            args.scheduler_patience,
        )


def _apply_loss_options(args: argparse.Namespace) -> None:
    loss_choice = getattr(args, 'loss', None)
    weight_params = [
        ('energy_weight', 'energy_weight'),
        ('forces_weight', 'forces_weight'),
        ('stress_weight', 'stress_weight'),
        ('virials_weight', 'virials_weight'),
        ('dipole_weight', 'dipole_weight'),
        ('polarizability_weight', 'polarizability_weight'),
        ('huber_delta', 'huber_delta'),
    ]
    configured_weights = {
        param: getattr(args, arg_name)
        for param, arg_name in weight_params
        if getattr(args, arg_name, None) is not None
    }
    if not loss_choice and not configured_weights:
        return
    factory = _LOSS_FACTORIES.get(loss_choice, modules.WeightedEnergyForcesStressLoss)
    gin.bind_parameter('loss.loss_cls', factory)
    for param_name, value in configured_weights.items():
        gin.bind_parameter(f'loss.{param_name}', value)


def _collect_swa_loss_kwargs(args: argparse.Namespace) -> dict[str, float] | None:
    fields = [
        ('energy_weight', 'swa_energy_weight'),
        ('forces_weight', 'swa_forces_weight'),
        ('stress_weight', 'swa_stress_weight'),
        ('virials_weight', 'swa_virials_weight'),
        ('dipole_weight', 'swa_dipole_weight'),
        ('polarizability_weight', 'swa_polarizability_weight'),
        ('huber_delta', 'swa_huber_delta'),
    ]
    values = {
        field: getattr(args, arg_name)
        for field, arg_name in fields
        if getattr(args, arg_name, None) is not None
    }
    return values or None


def _resolve_swa_loss_factory(
    args: argparse.Namespace, stage_loss_kwargs: dict[str, float] | None
):
    requested = getattr(args, 'swa_loss', None)
    if not requested and not stage_loss_kwargs:
        return None
    key = requested or getattr(args, 'loss', None)
    return _LOSS_FACTORIES.get(key, modules.WeightedEnergyForcesStressLoss)


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


def _maybe_adjust_multihead_defaults(args: argparse.Namespace) -> None:
    if getattr(args, '_multihead_adjusted', False):
        return
    if not getattr(args, 'multiheads_finetuning', False):
        return
    if getattr(args, 'force_mh_ft_lr', False):
        return
    if args.lr is None:
        args.lr = 1e-4
        logging.info(
            'Multihead finetuning: setting learning rate to 1e-4 (use --force_mh_ft_lr to override).'
        )
    if not getattr(args, 'ema', False):
        args.ema = True
        logging.info('Multihead finetuning: enabling EMA.')
    if args.ema_decay is None:
        args.ema_decay = 0.99999
        logging.info('Multihead finetuning: setting EMA decay to 0.99999.')
    args._multihead_adjusted = True


def _prepare_foundation_checkpoint(args: argparse.Namespace) -> None:
    foundation_spec = getattr(args, 'foundation_model', None)
    if not foundation_spec:
        return
    if args.torch_checkpoint:
        raise ValueError(
            'Specify either --torch-checkpoint or --foundation-model, not both.'
        )
    foundation_name = foundation_spec.lower()
    is_named_foundation = _is_named_mp_foundation(foundation_name)
    candidate = Path(foundation_spec)
    torch_model = None
    if candidate.exists():
        args.torch_checkpoint = str(candidate)
    else:
        torch_model = _load_foundation_model(foundation_spec, default_dtype=args.dtype)
        tmp_dir = Path(tempfile.mkdtemp(prefix='mace_jax_foundation_'))
        checkpoint_path = tmp_dir / f'{foundation_spec}.pt'
        torch.save({'model': torch_model}, checkpoint_path)
        _FOUNDATION_TEMP_DIRS.append(tmp_dir)
        args.torch_checkpoint = str(checkpoint_path)
        if args.r_max is None and hasattr(torch_model, 'r_max'):
            args.r_max = float(torch_model.r_max.item())
    if torch_model is not None:
        foundation_heads = getattr(torch_model, 'heads', None)
        if foundation_heads and not getattr(args, 'heads', None):
            args.heads = list(foundation_heads)
        if args.foundation_head is None and foundation_heads:
            args.foundation_head = foundation_heads[0]
    if (
        getattr(args, 'multiheads_finetuning', False)
        and not args.pt_train_file
        and not args.pt_valid_file
        and not is_named_foundation
    ):
        logging.warning(
            'Multihead finetuning requires pretraining data; disabling because foundation model %s '
            'is not a Materials Project checkpoint and no PT dataset was provided. '
            'Specify --pt_train_file/--pt_valid_file or choose a MP foundation.',
            foundation_spec,
        )
        args.multiheads_finetuning = False
    if args.foundation_head and not args.torch_head:
        args.torch_head = args.foundation_head
    _maybe_adjust_multihead_defaults(args)


def run_training(dry_run: bool = False) -> None:
    seed = gin_functions.flags()
    directory, tag, logger = gin_functions.logs()
    process_index = getattr(jax, 'process_index', lambda: 0)()
    is_primary = process_index == 0

    Path(directory).mkdir(parents=True, exist_ok=True)
    operative_config = gin.operative_config_str()
    if is_primary:
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

        if is_primary:
            wandb_run = gin_functions.wandb_run(
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
        if not steps_per_interval or steps_per_interval <= 0:
            auto_steps = train_loader.approx_length()
            logging.info(
                'Auto-setting steps_per_interval to %s based on training loader length.',
                auto_steps,
            )
            gin.bind_parameter(
                'mace_jax.tools.gin_functions.optimizer.steps_per_interval', auto_steps
            )
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
            data_seed=seed,
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
