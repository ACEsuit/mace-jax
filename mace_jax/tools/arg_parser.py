from __future__ import annotations

import argparse


def build_cli_arg_parser() -> argparse.ArgumentParser:
    """Create the argparse parser for mace-jax training CLI."""
    parser = argparse.ArgumentParser(
        description='Train a MACE-JAX model using gin-config files.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        'configs',
        nargs='*',
        help='gin-config files to load (evaluated in order).',
    )
    parser.add_argument('--name', help='Run name used in log files.', default=None)

    parser.add_argument(
        '--log-dir',
        '--log_dir',
        help='Directory for logs/results (binds gin_functions.logs.directory).',
        default=None,
    )
    parser.add_argument(
        '--seed',
        type=int,
        help='Training seed (binds gin_functions.flags.seed).',
        default=None,
    )
    parser.add_argument(
        '--dtype',
        '--default-dtype',
        '--default_dtype',
        choices=['float32', 'float64'],
        help='Default numeric dtype (binds gin_functions.flags.dtype).',
        default=None,
    )
    parser.add_argument(
        '--debug',
        dest='debug',
        action='store_const',
        const=True,
        default=None,
        help='Enable JAX NaN/inf checks via gin_functions.flags.debug.',
    )
    parser.add_argument(
        '--no-debug',
        dest='debug',
        action='store_const',
        const=False,
        help='Disable JAX NaN/inf checks via gin_functions.flags.debug.',
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
        '--train-file',
        '--train_file',
        '--train-path',
        '--train_path',
        dest='train_path',
        help='Path to training dataset (XYZ/HDF5).',
        default=None,
    )
    parser.add_argument(
        '--valid-file',
        '--valid_file',
        '--valid-path',
        '--valid_path',
        dest='valid_path',
        help='Path to validation dataset (XYZ/HDF5) or "None".',
        default=None,
    )
    parser.add_argument(
        '--test-file',
        '--test_file',
        '--test-path',
        '--test_path',
        dest='test_path',
        help='Path to test dataset (XYZ/HDF5) or "None".',
        default=None,
    )
    parser.add_argument(
        '--valid-fraction',
        '--valid_fraction',
        type=float,
        default=None,
        help='Fraction of training configs used for validation.',
    )
    parser.add_argument(
        '--valid-num',
        '--valid_num',
        type=int,
        default=None,
        help='Number of random configs reserved for validation.',
    )
    parser.add_argument(
        '--test-num',
        '--test_num',
        type=int,
        default=None,
        help='Number of configs sampled for the test set.',
    )
    parser.add_argument(
        '--energy-key',
        '--energy_key',
        type=str,
        default=None,
        help='Key used to read energies from datasets.',
    )
    parser.add_argument(
        '--forces-key',
        '--forces_key',
        type=str,
        default=None,
        help='Key used to read forces from datasets.',
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
        '--heads-config',
        '--heads_config',
        help=(
            'Dictionary describing head-specific datasets (same syntax as the Torch CLI). '
            'Example: --heads-config "{\'Default\':{\'train_path\':\'data.xyz\'}, \'Surface\':{...}}"'
        ),
        default=None,
    )
    parser.add_argument(
        '--pt_train_file',
        action='append',
        help='Additional training file(s) for the pt_head used during multihead finetuning.',
        default=None,
    )
    parser.add_argument(
        '--pt_valid_file',
        action='append',
        help='Validation file(s) for the pt_head.',
        default=None,
    )
    parser.add_argument(
        '--pt_head_name',
        help='Name of the pretraining head when using --pt_train_file/--pt_valid_file.',
        default='pt_head',
    )
    parser.add_argument(
        '--head-config',
        help=(
            'Path to a JSON (or YAML if PyYAML is installed) file describing per-head '
            'dataset overrides (binds mace_jax.tools.gin_datasets.datasets.head_configs).'
        ),
    )
    parser.add_argument(
        '--clip-grad',
        '--clip_grad',
        type=float,
        default=None,
        help='Global gradient-norm clipping value.',
    )
    parser.add_argument(
        '--ema',
        action='store_true',
        default=False,
        help='Enable EMA with the default decay (matches Torch behavior).',
    )
    parser.add_argument(
        '--ema-decay',
        '--ema_decay',
        type=float,
        default=None,
        help='Exponential moving average decay applied to evaluation params.',
    )
    parser.add_argument(
        '--optimizer',
        help='Optimizer for parameter updates.',
        choices=['adam', 'amsgrad', 'sgd'],
        default=None,
    )
    parser.add_argument(
        '--beta',
        type=float,
        default=None,
        help='Optimizer beta parameter (mirrors Torch CLI).',
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=None,
        help='Learning rate for the optimizer.',
    )
    parser.add_argument(
        '--weight-decay',
        '--weight_decay',
        type=float,
        default=None,
        help='Weight decay applied to selected parameters.',
    )
    parser.add_argument(
        '--scheduler',
        type=str,
        default=None,
        help='Learning-rate scheduler name.',
    )
    parser.add_argument(
        '--lr-scheduler-gamma',
        '--lr_scheduler_gamma',
        type=float,
        default=None,
        help='Decay rate used by the exponential scheduler.',
    )
    parser.add_argument(
        '--loss',
        choices=[
            'weighted',
            'ef',
            'forces_only',
            'stress',
            'huber',
            'virials',
            'dipole',
            'energy_forces_dipole',
            'l1l2',
        ],
        default=None,
        help='Select loss function (mirrors Torch CLI choices).',
    )
    parser.add_argument(
        '--energy-weight',
        '--energy_weight',
        type=float,
        default=None,
        help='Weight for the energy loss term.',
    )
    parser.add_argument(
        '--forces-weight',
        '--forces_weight',
        type=float,
        default=None,
        help='Weight for the forces loss term.',
    )
    parser.add_argument(
        '--stress-weight',
        '--stress_weight',
        type=float,
        default=None,
        help='Weight for the stress loss term.',
    )
    parser.add_argument(
        '--virials-weight',
        '--virials_weight',
        type=float,
        default=None,
        help='Weight for the virials loss term.',
    )
    parser.add_argument(
        '--dipole-weight',
        '--dipole_weight',
        type=float,
        default=None,
        help='Weight for the dipole loss term.',
    )
    parser.add_argument(
        '--polarizability-weight',
        '--polarizability_weight',
        type=float,
        default=None,
        help='Weight for the polarizability loss term.',
    )
    parser.add_argument(
        '--huber-delta',
        '--huber_delta',
        type=float,
        default=None,
        help='Delta parameter for Huber losses.',
    )
    parser.add_argument(
        '--swa',
        action='store_true',
        help='Enable stochastic weight averaging with default settings.',
    )
    parser.add_argument(
        '--swa-start',
        '--start-swa',
        '--start_swa',
        type=int,
        default=None,
        help='Interval at which SWA snapshots start.',
    )
    parser.add_argument(
        '--swa-every',
        '--swa_every',
        type=int,
        default=None,
        help='Number of intervals between SWA snapshots.',
    )
    parser.add_argument(
        '--swa-min-snapshots',
        '--swa_min_snapshots',
        type=int,
        default=None,
        help='Minimum SWA snapshots required before evaluation uses the averaged params.',
    )
    parser.add_argument(
        '--swa-max-snapshots',
        '--swa_max_snapshots',
        type=int,
        default=None,
        help='Maximum number of SWA snapshots to keep.',
    )
    parser.add_argument(
        '--swa-prefer',
        '--swa_prefer',
        dest='swa_prefer',
        action='store_const',
        const=True,
        default=None,
        help='Force evaluation to prefer SWA params when available.',
    )
    parser.add_argument(
        '--swa-no-prefer',
        '--swa_no_prefer',
        dest='swa_prefer',
        action='store_const',
        const=False,
        help='Disable automatic preference for SWA params.',
    )
    parser.add_argument(
        '--multiheads_finetuning',
        action='store_true',
        help='Enable multihead finetuning defaults when using foundation models.',
    )
    parser.add_argument(
        '--force_mh_ft_lr',
        action='store_true',
        help='Prevent automatic learning-rate/EMA overrides during multihead finetuning.',
    )
    parser.add_argument(
        '--wandb',
        dest='wandb_enable',
        action='store_true',
        help='Enable Weights & Biases logging.',
    )
    parser.add_argument(
        '--wandb-project',
        '--wandb_project',
        default=None,
        help='Weights & Biases project name.',
    )
    parser.add_argument(
        '--wandb-entity',
        '--wandb_entity',
        default=None,
        help='Weights & Biases entity/workspace.',
    )
    parser.add_argument(
        '--wandb-group',
        '--wandb_group',
        default=None,
        help='Weights & Biases group name.',
    )
    parser.add_argument(
        '--wandb-run-name',
        '--wandb_run_name',
        default=None,
        help='Explicit name for the Weights & Biases run.',
    )
    parser.add_argument(
        '--wandb-notes',
        '--wandb_notes',
        default=None,
        help='Free-form notes attached to the Weights & Biases run.',
    )
    parser.add_argument(
        '--wandb-tag',
        '--wandb_tag',
        dest='wandb_tags',
        action='append',
        help='Tag to attach to the Weights & Biases run (repeatable).',
    )
    parser.add_argument(
        '--wandb-mode',
        '--wandb_mode',
        choices=['online', 'offline', 'disabled'],
        default=None,
        help='Weights & Biases mode.',
    )
    parser.add_argument(
        '--wandb-dir',
        '--wandb_dir',
        default=None,
        help='Directory used by Weights & Biases for run data.',
    )
    parser.add_argument(
        '--foundation-model',
        '--foundation_model',
        help='Name or path of a Torch foundation model checkpoint to import.',
        default=None,
    )
    parser.add_argument(
        '--foundation-head',
        '--foundation_head',
        help='Optional head to select from the foundation model.',
        default=None,
    )
    return parser
