from __future__ import annotations

import argparse
import os


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
        '--device',
        choices=['auto', 'cpu', 'cuda', 'tpu'],
        default=None,
        help='Preferred JAX platform for execution.',
    )
    parser.add_argument(
        '--enable-cueq',
        '--enable_cueq',
        action='store_true',
        help='Enable cuequivariance acceleration.',
    )
    parser.add_argument(
        '--only-cueq',
        '--only_cueq',
        action='store_true',
        help='Force the cuequivariance backend for model operations.',
    )
    parser.add_argument(
        '--cueq-optimize-all',
        dest='cueq_optimize_all',
        action='store_true',
        default=None,
        help='Enable all cueq optimizations.',
    )
    parser.add_argument(
        '--no-cueq-optimize-all',
        dest='cueq_optimize_all',
        action='store_false',
        help='Disable cueq optimizations.',
    )
    parser.add_argument(
        '--cueq-conv-fusion',
        dest='cueq_conv_fusion',
        action='store_true',
        default=None,
        help='Enable cueq conv_fusion.',
    )
    parser.add_argument(
        '--no-cueq-conv-fusion',
        dest='cueq_conv_fusion',
        action='store_false',
        help='Disable cueq conv_fusion.',
    )
    parser.add_argument(
        '--cueq-layout',
        choices=['mul_ir', 'ir_mul'],
        default=None,
        help='Select cueq layout (mul_ir or ir_mul).',
    )
    parser.add_argument(
        '--cueq-group',
        choices=['O3', 'O3_e3nn'],
        default=None,
        help='Select cueq group (O3 or O3_e3nn).',
    )
    parser.add_argument(
        '--distributed',
        action='store_true',
        help='Initialize jax.distributed for multi-process training.',
    )
    parser.add_argument(
        '--launcher',
        choices=['none', 'local', 'auto'],
        default='auto',
        help=(
            'Launch strategy for distributed training. '
            'auto enables local multi-process when multiple GPUs are visible.'
        ),
    )
    parser.add_argument(
        '--process-count',
        '--process_count',
        type=int,
        default=None,
        help='Total number of JAX processes when using --distributed.',
    )
    parser.add_argument(
        '--process-index',
        '--process_index',
        type=int,
        default=None,
        help='Index of this JAX process when using --distributed.',
    )
    parser.add_argument(
        '--coordinator-address',
        '--coordinator_address',
        default=None,
        help='Coordinator address for jax.distributed initialization.',
    )
    parser.add_argument(
        '--coordinator-port',
        '--coordinator_port',
        type=int,
        default=None,
        help='Coordinator port for jax.distributed initialization.',
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
        '--atomic-numbers',
        '--atomic_numbers',
        dest='atomic_numbers',
        nargs='+',
        type=int,
        default=None,
        help='Override atomic numbers (binds both datasets.atomic_numbers and model.atomic_numbers).',
    )
    parser.add_argument(
        '--E0s',
        dest='E0s',
        default=None,
        help=(
            'Isolated atomic energies. Accepts "average", a JSON file path, or a dict literal '
            '(binds datasets.atomic_energies_override and model.atomic_energies).'
        ),
    )
    parser.add_argument(
        '--statistics-file',
        '--statistics_file',
        dest='statistics_file',
        default=None,
        help='Path to a statistics JSON file (populates atomic numbers, E0s, scaling, neighbors).',
    )
    parser.add_argument(
        '--scaling',
        choices=['std_scaling', 'rms_forces_scaling', 'no_scaling'],
        default=None,
        help=(
            'Scaling strategy for model outputs; overrides statistics-file scaling '
            '(binds mace_jax.tools.gin_model.model.scaling).'
        ),
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
        '--batch-max-edges',
        '--batch_max_edges',
        type=str,
        default=None,
        help=(
            'Maximum number of edges allowed in a padded batch. '
            "Pass 'auto' or 'None' to request automatic sizing."
        ),
    )
    parser.add_argument(
        '--stream-train-max-batches',
        '--stream_train_max_batches',
        type=int,
        default=None,
        help=(
            'Optional cap on the number of streaming batches per epoch '
            '(binds mace_jax.tools.gin_datasets.datasets.stream_train_max_batches).'
        ),
    )
    parser.add_argument(
        '--suffle',
        dest='stream_train_shuffle',
        type=str2bool,
        default=True,
        help=(
            'Shuffle streaming training samples each epoch '
            '(binds mace_jax.tools.gin_datasets.datasets.stream_train_shuffle).'
        ),
    )
    parser.add_argument(
        '--batch-node-precentile',
        '--stream-train-node-percentile',
        '--stream_train_node_percentile',
        dest='stream_train_node_percentile',
        type=float,
        default=90.0,
        help=(
            'Percentile (0-100, or 0-1 fraction) used to set the node padding cap; '
            'the cap is never below the largest single graph '
            '(binds mace_jax.tools.gin_datasets.datasets.stream_train_node_percentile).'
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
        '--heads-config',
        '--heads_config',
        help=(
            'Dictionary describing head-specific datasets (same syntax as the Torch CLI). '
            "Example: --heads-config \"{'Default':{'train_path':'data.xyz'}, 'Surface':{...}}\""
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
            'dataset overrides. Each head may override train/valid/test paths, '
            'loss weights, energy/force keys, etc. (binds mace_jax.tools.gin_datasets.datasets.head_configs).'
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
        choices=['adam', 'adamw', 'amsgrad', 'sgd', 'schedulefree'],
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
        '--schedule-free-b1',
        '--schedule_free_b1',
        type=float,
        default=None,
        help='Beta_1 parameter used by ScheduleFree (ignored for other optimizers).',
    )
    parser.add_argument(
        '--schedule-free-weight-lr-power',
        '--schedule_free_weight_lr_power',
        type=float,
        default=None,
        help='Weight exponent controlling ScheduleFree averaging (ignored otherwise).',
    )
    parser.add_argument(
        '--scheduler',
        type=str,
        default=None,
        choices=['constant', 'exponential', 'piecewise_constant', 'plateau'],
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
        '--lr-factor',
        '--lr_factor',
        type=float,
        default=None,
        help='Multiplicative factor used when the plateau scheduler steps.',
    )
    parser.add_argument(
        '--scheduler-patience',
        '--scheduler_patience',
        type=int,
        default=None,
        help='Number of evaluations without improvement before plateau scheduler steps.',
    )
    parser.add_argument(
        '--scheduler-threshold',
        '--scheduler_threshold',
        type=float,
        default=None,
        help='Minimum improvement needed to reset the plateau scheduler.',
    )
    parser.add_argument(
        '--max-epochs',
        '--max_epochs',
        '--max-num-epochs',
        '--max_num_epochs',
        '--max-num-intervals',
        '--max_num_intervals',
        dest='max_epochs',
        type=int,
        default=None,
        help='Maximum number of training epochs before stopping.',
    )
    parser.add_argument(
        '--patience',
        type=int,
        default=None,
        help='Intervals to wait for validation improvement before stopping early.',
    )
    parser.add_argument(
        '--eval-interval',
        '--eval_interval',
        type=int,
        default=None,
        help='How many intervals to wait between evaluations.',
    )
    parser.add_argument(
        '--eval-train',
        '--eval_train',
        dest='eval_train',
        action='store_const',
        const=True,
        default=None,
        help='Log training metrics each interval.',
    )
    parser.add_argument(
        '--no-eval-train',
        '--no_eval_train',
        dest='eval_train',
        action='store_const',
        const=False,
        help='Disable training-set evaluation.',
    )
    parser.add_argument(
        '--eval-train-fraction',
        '--eval_train_fraction',
        type=float,
        default=None,
        help='Evaluate only this fraction of the training set (0.0-1.0).',
    )
    parser.add_argument(
        '--eval-test',
        '--eval_test',
        dest='eval_test',
        action='store_const',
        const=True,
        default=None,
        help='Evaluate on the test set each interval.',
    )
    parser.add_argument(
        '--no-eval-test',
        '--no_eval_test',
        dest='eval_test',
        action='store_const',
        const=False,
        help='Disable test-set evaluation.',
    )
    parser.add_argument(
        '--log-errors',
        '--log_errors',
        choices=[
            'PerAtomRMSE',
            'rel_PerAtomRMSE',
            'TotalRMSE',
            'PerAtomMAE',
            'rel_PerAtomMAE',
            'TotalMAE',
            'PerAtomRMSEstressvirials',
            'PerAtomMAEstressvirials',
            'DipoleRMSE',
            'DipolePolarRMSE',
            'EnergyDipoleRMSE',
        ],
        default='PerAtomRMSE',
        help='Select which error summary to log each interval.',
    )
    parser.add_argument(
        '--checkpoint-dir',
        '--checkpoint_dir',
        help='Directory used to store training checkpoints (binds gin train.checkpoint_dir).',
        default=None,
    )
    parser.add_argument(
        '--checkpoint-every',
        '--checkpoint_every',
        type=int,
        default=None,
        help='Save a checkpoint every N epochs (binds gin train.checkpoint_every).',
    )
    parser.add_argument(
        '--checkpoint-keep',
        '--checkpoint_keep',
        type=int,
        default=None,
        help='Keep only the most recent N checkpoints (binds gin train.checkpoint_keep).',
    )
    parser.add_argument(
        '--checkpoint-best',
        '--checkpoint_best',
        action='store_true',
        help='Save a checkpoint whenever validation loss improves (binds gin train.checkpoint_best).',
    )
    parser.add_argument(
        '--resume-from',
        '--resume_from',
        help='Resume training from a previously saved checkpoint file.',
        default=None,
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
            'l1l2energyforces',
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
        '--swa-loss',
        '--swa_loss',
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
        help='Override the loss used once SWA/Stage Two begins.',
    )
    parser.add_argument(
        '--swa-energy-weight',
        '--swa_energy_weight',
        type=float,
        default=None,
        help='Stage Two energy weight.',
    )
    parser.add_argument(
        '--swa-forces-weight',
        '--swa_forces_weight',
        type=float,
        default=None,
        help='Stage Two forces weight.',
    )
    parser.add_argument(
        '--swa-stress-weight',
        '--swa_stress_weight',
        type=float,
        default=None,
        help='Stage Two stress weight.',
    )
    parser.add_argument(
        '--swa-virials-weight',
        '--swa_virials_weight',
        type=float,
        default=None,
        help='Stage Two virials weight.',
    )
    parser.add_argument(
        '--swa-dipole-weight',
        '--swa_dipole_weight',
        type=float,
        default=None,
        help='Stage Two dipole weight.',
    )
    parser.add_argument(
        '--swa-polarizability-weight',
        '--swa_polarizability_weight',
        type=float,
        default=None,
        help='Stage Two polarizability weight.',
    )
    parser.add_argument(
        '--swa-huber-delta',
        '--swa_huber_delta',
        type=float,
        default=None,
        help='Stage Two Huber delta.',
    )
    parser.add_argument(
        '--swa-lr',
        '--swa_lr',
        type=float,
        default=None,
        help='Learning rate to use during Stage Two/SWA.',
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


def build_preprocess_arg_parser() -> argparse.ArgumentParser:
    try:
        import configargparse

        parser = configargparse.ArgumentParser(
            config_file_parser_class=configargparse.YAMLConfigFileParser,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        parser.add(
            '--config',
            type=str,
            is_config_file=True,
            help='config file to aggregate options',
        )
    except ImportError:
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
    parser.add_argument(
        '--train_file',
        help='Training set xyz file',
        type=str,
        default=None,
        required=True,
    )
    parser.add_argument(
        '--valid_file',
        help='Validation set xyz file',
        type=str,
        default=None,
        required=False,
    )
    cpu_count = os.cpu_count() or 1
    parser.add_argument(
        '--num_process',
        help='Number of processes and output files to create.',
        type=int,
        default=max(1, int(cpu_count / 4)),
    )
    parser.add_argument(
        '--valid_fraction',
        help='Fraction of training set used for validation',
        type=float,
        default=0.1,
        required=False,
    )
    parser.add_argument(
        '--test_file',
        help='Test set xyz file',
        type=str,
        default=None,
        required=False,
    )
    parser.add_argument(
        '--work_dir',
        help='Directory for auxiliary outputs (e.g., split indices).',
        type=str,
        default='.',
    )
    parser.add_argument(
        '--h5_prefix',
        help='Prefix directory for output HDF5 files.',
        type=str,
        default='',
    )
    parser.add_argument(
        '--r_max', help='distance cutoff (in Ang)', type=float, default=5.0
    )
    parser.add_argument(
        '--config_type_weights',
        help='String of dictionary containing the weights for each config type',
        type=str,
        default='{"Default":1.0}',
    )
    parser.add_argument(
        '--energy_key',
        help='Key of reference energies in training xyz',
        type=str,
        default='energy',
    )
    parser.add_argument(
        '--forces_key',
        help='Key of reference forces in training xyz',
        type=str,
        default='forces',
    )
    parser.add_argument(
        '--virials_key',
        help='Key of reference virials in training xyz',
        type=str,
        default='virials',
    )
    parser.add_argument(
        '--stress_key',
        help='Key of reference stress in training xyz',
        type=str,
        default='stress',
    )
    parser.add_argument(
        '--dipole_key',
        help='Key of reference dipoles in training xyz',
        type=str,
        default='dipole',
    )
    parser.add_argument(
        '--polarizability_key',
        help='Key of polarizability in training xyz',
        type=str,
        default='polarizability',
    )
    parser.add_argument(
        '--charges_key',
        help='Key of atomic charges in training xyz',
        type=str,
        default='charges',
    )
    parser.add_argument(
        '--atomic_numbers',
        help='List of atomic numbers',
        type=str,
        default=None,
        required=False,
    )
    parser.add_argument(
        '--compute_statistics',
        help='Compute statistics for the dataset',
        action='store_true',
        default=False,
    )
    parser.add_argument(
        '--batch_size',
        help='Batch size used while computing statistics',
        type=int,
        default=16,
    )
    parser.add_argument(
        '--scaling',
        help='Type of scaling to the output',
        type=str,
        default='rms_forces_scaling',
        choices=['std_scaling', 'rms_forces_scaling', 'no_scaling'],
    )
    parser.add_argument(
        '--E0s',
        help='Dictionary of isolated atom energies',
        type=str,
        default=None,
        required=False,
    )
    parser.add_argument(
        '--shuffle',
        help='Shuffle the training dataset',
        type=str2bool,
        default=True,
    )
    parser.add_argument(
        '--seed',
        help='Random seed for splitting training and validation sets',
        type=int,
        default=123,
    )
    parser.add_argument(
        '--head_key',
        help='Key of head in training xyz',
        type=str,
        default='head',
    )
    parser.add_argument(
        '--heads',
        help='Dict of heads: containing individual files and E0s',
        type=str,
        default=None,
        required=False,
    )
    return parser


def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    if value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')
