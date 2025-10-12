from __future__ import annotations

import argparse
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


def run_training(dry_run: bool = False) -> None:
    seed = gin_functions.flags()
    directory, tag, logger = gin_functions.logs()

    Path(directory).mkdir(parents=True, exist_ok=True)
    with open(Path(directory) / f'{tag}.gin', 'w', encoding='utf-8') as f:
        f.write(gin.operative_config_str())

    logging.info('MACE-JAX version: %s', mace_jax.__version__)

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

    logging.info('Number of parameters: %s', tools.count_parameters(params))
    logging.info(
        'Number of optimizer parameters: %s',
        tools.count_parameters(optimizer_state),
    )

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
    )


def main(argv: Sequence[str] | None = None) -> None:
    args, remaining = parse_args(argv)
    configure_gin(args, remaining)

    if args.print_config:
        logging.info('Operative gin config:\n%s', gin.config_str())

    run_training(dry_run=args.dry_run)


if __name__ == '__main__':
    main()
