from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

import gin
import jax

import mace_jax
from mace_jax import tools
from mace_jax.tools import gin_datasets, gin_functions, gin_model


def parse_args(
    argv: Sequence[str] | None = None,
) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description='Train a MACE-JAX model using gin-config files.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        'configs',
        nargs='*',
        help='gin-config files to load (evaluated in order).',
    )
    parser.add_argument(
        '-b',
        '--binding',
        action='append',
        default=[],
        metavar='KEY=VALUE',
        help='Additional gin binding applied after the config files.',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Parse the configuration and exit without launching training.',
    )
    parser.add_argument(
        '--print-config',
        action='store_true',
        help='Print the operative gin config before training starts.',
    )
    parser.add_argument(
        '--torch-checkpoint',
        help=(
            'Path to a Torch checkpoint to import via gin '
            '(binds mace_jax.tools.gin_model.model.torch_checkpoint).'
        ),
    )
    parser.add_argument(
        '--torch-head',
        help=(
            'Optional head name to select from the Torch checkpoint '
            '(binds mace_jax.tools.gin_model.model.torch_head).'
        ),
    )
    parser.add_argument(
        '--torch-param-dtype',
        choices=['float32', 'float64'],
        help=(
            'Desired dtype for imported Torch parameters '
            '(binds mace_jax.tools.gin_model.model.torch_param_dtype).'
        ),
    )
    parser.add_argument(
        '--train-path',
        help=(
            'Override gin training dataset path '
            '(binds mace_jax.tools.gin_datasets.datasets.train_path).'
        ),
    )
    parser.add_argument(
        '--valid-path',
        help=(
            'Override gin validation dataset path '
            '(binds mace_jax.tools.gin_datasets.datasets.valid_path).'
        ),
    )
    parser.add_argument(
        '--test-path',
        help=(
            'Override gin test dataset path '
            '(binds mace_jax.tools.gin_datasets.datasets.test_path).'
        ),
    )
    parser.add_argument(
        '--r-max',
        type=float,
        help=(
            'Override cutoff radius used by both dataset preparation and model '
            '(binds mace_jax.tools.gin_datasets.datasets.r_max and '
            'mace_jax.tools.gin_model.model.r_max).'
        ),
    )
    parser.add_argument(
        '--heads',
        nargs='+',
        help=(
            'Names of the heads to expose to the model/dataloader '
            '(binds mace_jax.tools.gin_datasets.datasets.heads and '
            'mace_jax.tools.gin_model.model.heads).'
        ),
    )
    parser.add_argument(
        '--head-config',
        help=(
            'Path to a JSON (or YAML if PyYAML is installed) file describing per-head '
            'dataset overrides (binds mace_jax.tools.gin_datasets.datasets.head_configs).'
        ),
    )

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
    if args.train_path:
        gin.bind_parameter(
            'mace_jax.tools.gin_datasets.datasets.train_path',
            str(Path(args.train_path)),
        )
    if args.valid_path is not None:
        gin.bind_parameter(
            'mace_jax.tools.gin_datasets.datasets.valid_path',
            None if args.valid_path == 'None' else str(Path(args.valid_path)),
        )
    if args.test_path is not None:
        gin.bind_parameter(
            'mace_jax.tools.gin_datasets.datasets.test_path',
            None if args.test_path == 'None' else str(Path(args.test_path)),
        )
    if args.r_max is not None:
        gin.bind_parameter('mace_jax.tools.gin_datasets.datasets.r_max', args.r_max)
        gin.bind_parameter('mace_jax.tools.gin_model.model.r_max', args.r_max)

    head_configs_data = None
    if getattr(args, 'head_config', None):
        head_configs_data = _load_head_configs(Path(args.head_config))
        gin.bind_parameter(
            'mace_jax.tools.gin_datasets.datasets.head_configs', head_configs_data
        )

    heads_list = None
    if getattr(args, 'heads', None):
        heads_list = list(args.heads)
    elif head_configs_data:
        heads_list = list(head_configs_data.keys())

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
            logging.info('Dry-run requested; skipping dataset construction and training.')
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
