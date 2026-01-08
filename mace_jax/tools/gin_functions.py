"""Gin-configurable training helpers and CLI glue.

This module bridges gin configuration to the JAX training stack. It wires up
losses, optimizers, schedules, logging, and evaluation so that training can be
launched from gin files or CLI flags. The training loop assumes fixed-shape
batches from the streaming loader so the model compiles once and runs at steady
throughput without shape-driven recompilation.
"""

import datetime
import inspect
import logging
import numbers
import pickle
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import gin
import jax
import jax.numpy as jnp
import jraph
import numpy as np
import optax

try:
    from optax.contrib import schedule_free as optax_schedule_free
    from optax.contrib import schedule_free_eval_params
except ImportError:  # pragma: no cover - optax versions without contrib support
    optax_schedule_free = None
    schedule_free_eval_params = None
from jax import config as jax_config
from jax.experimental import multihost_utils
from tqdm import tqdm
from unique_names_generator import get_random_name
from unique_names_generator.data import ADJECTIVES, NAMES

from mace_jax import modules, tools
from mace_jax.tools import device as device_utils

_PROFILE_HELPER = None
_PROFILE_HELPER_MISSING_WARNED = False


def _get_profile_helper(*, log_warning: bool) -> object | None:
    """Return the optional profiling helper module, caching the import.

    Args:
        log_warning: Whether to emit a warning if the profiler is unavailable.

    Returns:
        The imported `profile_nn_jax` module, or None if unavailable.

    This is used by `flags()` and the training loop to enable/trigger profiling.
    """
    global _PROFILE_HELPER, _PROFILE_HELPER_MISSING_WARNED
    if _PROFILE_HELPER is not None:
        return _PROFILE_HELPER
    if _PROFILE_HELPER_MISSING_WARNED and not log_warning:
        return None
    try:
        import profile_nn_jax  # noqa: PLC0415
    except ImportError:
        if log_warning and not _PROFILE_HELPER_MISSING_WARNED:
            logging.warning(
                'Requested profiler (--profile) but profile_nn_jax is not installed; '
                'continuing without profiling.'
            )
        _PROFILE_HELPER_MISSING_WARNED = True
        return None
    else:
        _PROFILE_HELPER = profile_nn_jax
        return _PROFILE_HELPER


def _split_config(params):
    """Split a parameter bundle into trainable params and optional config state.

    Args:
        params: Either a raw parameter tree or a dict containing 'params'/'config'.

    Returns:
        Tuple of (trainable_params, config_state_or_None).
    """
    if isinstance(params, Mapping) and 'params' in params:
        return params['params'], params.get('config')
    return params, None


def _attach_config(trainable, config):
    """Attach config state to trainable params for Flax modules.

    Args:
        trainable: Trainable parameter tree.
        config: Optional config/state tree to pair with parameters.

    Returns:
        Either `trainable` unchanged or a dict containing 'params' and 'config'.
    """
    if config is None:
        return trainable
    return {'params': trainable, 'config': config}


def _maybe_float(value):
    """Convert scalar-like values into Python floats for logging.

    Args:
        value: Numeric, NumPy, or JAX scalar-like value.

    Returns:
        A Python float if conversion is possible, otherwise None.
    """
    if isinstance(value, numbers.Number):
        return float(value)
    if isinstance(value, (np.ndarray, jnp.ndarray)) and value.shape == ():
        return float(np.asarray(value))
    return None


def _select_metrics_for_logging(
    metrics: Mapping[str, Any], log_errors: str | None
) -> list[str]:
    """Select which metric keys to log based on the configured error mode.

    Args:
        metrics: Metric dict produced by evaluation.
        log_errors: Named logging preset (e.g. 'PerAtomRMSE', 'PerAtomMAE').

    Returns:
        List of metric keys to format and print.
    """
    stress_rmse_key = (
        'rmse_stress'
        if metrics.get('rmse_stress') is not None
        else 'rmse_virials_per_atom'
    )
    stress_mae_key = (
        'mae_stress' if metrics.get('mae_stress') is not None else 'mae_virials'
    )
    log_selection = {
        'PerAtomRMSE': ['rmse_e_per_atom', 'rmse_f', stress_rmse_key],
        'rel_PerAtomRMSE': [
            'rmse_e_per_atom',
            'rel_rmse_f',
            'rel_rmse_stress',
        ],
        'TotalRMSE': ['rmse_e', 'rmse_f', stress_rmse_key],
        'PerAtomMAE': ['mae_e_per_atom', 'mae_f', stress_mae_key],
        'rel_PerAtomMAE': ['mae_e_per_atom', 'rel_mae_f', 'rel_mae_stress'],
        'TotalMAE': ['mae_e', 'mae_f', stress_mae_key],
        'PerAtomRMSEstressvirials': [
            'rmse_e_per_atom',
            'rmse_f',
            stress_rmse_key,
        ],
        'PerAtomMAEstressvirials': [
            'mae_e_per_atom',
            'mae_f',
            stress_mae_key,
        ],
        'DipoleRMSE': ['rmse_mu_per_atom'],
        'DipolePolarRMSE': [
            'rmse_mu_per_atom',
            'rmse_polarizability_per_atom',
        ],
        'EnergyDipoleRMSE': [
            'rmse_e_per_atom',
            'rmse_f',
            'rmse_mu_per_atom',
        ],
    }
    return log_selection.get(log_errors or 'PerAtomRMSE', log_selection['PerAtomRMSE'])


def _format_metric_value(metrics: Mapping[str, Any], key: str | None) -> str:
    """Format a metric value with units and human-readable precision.

    Args:
        metrics: Metric dict containing raw values.
        key: Metric key to format.

    Returns:
        Formatted string (with units if applicable) or 'N/A' if missing.
    """
    if not key:
        return 'N/A'
    value = metrics.get(key, None)
    if value is None:
        return 'N/A'
    maybe_value = _maybe_float(value)
    if maybe_value is not None:
        value = maybe_value
    elif not isinstance(value, numbers.Number):
        return 'N/A'
    if key.startswith('rel_'):
        return f'{100 * value:.1f}%'
    lower_key = key.lower()
    if 'mu' in lower_key:
        return f'{1e3 * value:.1f} mDebye'
    if 'polarizability' in lower_key:
        return f'{1e3 * value:.1f} me Å^2 / V'
    if 'virials' in lower_key or 'stress' in lower_key or lower_key.endswith('_s'):
        return f'{1e3 * value:.1f} meV/Å³'
    if lower_key.endswith('_f'):
        return f'{1e3 * value:.1f} meV/Å'
    if '_e' in key:
        return f'{1e3 * value:.1f} meV'
    return f'{value:.4e}'


def _instantiate_loss(loss_cls, overrides: dict):
    """Instantiate a loss class/function while validating keyword arguments.

    Args:
        loss_cls: Loss class or factory to construct.
        overrides: Keyword arguments to pass through.

    Returns:
        Instantiated loss object.

    Raises:
        ValueError: If an override is not accepted by the loss signature.
    """
    target = loss_cls.__init__ if inspect.isclass(loss_cls) else loss_cls
    try:
        signature = inspect.signature(target)
    except (TypeError, ValueError):
        signature = None

    if signature is None:
        return loss_cls(**overrides)

    accepts_kwargs = any(
        param.kind == inspect.Parameter.VAR_KEYWORD
        for param in signature.parameters.values()
    )
    if not accepts_kwargs:
        valid_params = {name for name in signature.parameters.keys() if name != 'self'}
        invalid = [key for key in overrides if key not in valid_params]
        if invalid:
            joined = ', '.join(sorted(invalid))
            raise ValueError(
                f"Loss '{loss_cls.__name__}' does not accept parameter(s): {joined}."
            )
    return loss_cls(**overrides)


def _required_targets_from_loss(loss_fn) -> set[str]:
    """Infer required targets based on loss weights and presence checks."""
    required: set[str] = set()
    if float(getattr(loss_fn, 'energy_weight', 0.0) or 0.0) != 0.0:
        required.add('energy')
    if float(getattr(loss_fn, 'forces_weight', 0.0) or 0.0) != 0.0:
        required.add('forces')
    if float(getattr(loss_fn, 'stress_weight', 0.0) or 0.0) != 0.0:
        required.add('stress')
    if float(getattr(loss_fn, 'virials_weight', 0.0) or 0.0) != 0.0:
        required.add('virials')
    if float(getattr(loss_fn, 'dipole_weight', 0.0) or 0.0) != 0.0:
        required.add('dipole')
    if float(getattr(loss_fn, 'polarizability_weight', 0.0) or 0.0) != 0.0:
        required.add('polarizability')
    return required


def _required_targets_from_log_errors(
    log_errors: str | None,
) -> tuple[set[str], list[tuple[str, set[str]]]]:
    """Infer required targets based on log_errors configuration."""
    selection = log_errors or 'PerAtomRMSE'
    required: set[str] = set()
    any_of: list[tuple[str, set[str]]] = []
    if selection in {
        'PerAtomRMSE',
        'rel_PerAtomRMSE',
        'TotalRMSE',
        'PerAtomMAE',
        'rel_PerAtomMAE',
        'TotalMAE',
        'PerAtomRMSEstressvirials',
        'PerAtomMAEstressvirials',
    }:
        required.update({'energy', 'forces'})
        any_of.append(('stress or virials', {'stress', 'virials'}))
    elif selection == 'DipoleRMSE':
        required.add('dipole')
    elif selection == 'DipolePolarRMSE':
        required.update({'dipole', 'polarizability'})
    elif selection == 'EnergyDipoleRMSE':
        required.update({'energy', 'forces', 'dipole'})
    return required, any_of


def _sample_graph_from_loader(loader):
    """Extract a representative graph from a loader without consuming state."""
    graphs = getattr(loader, 'graphs', None)
    if graphs:
        sample = graphs[0]
    else:
        try:
            sample = next(iter(loader))
        except StopIteration:
            return None
    if isinstance(sample, Sequence) and not isinstance(sample, jraph.GraphsTuple):
        for item in sample:
            if isinstance(item, jraph.GraphsTuple):
                return item
        return None
    return sample


def _assert_required_targets(
    loader,
    *,
    required: set[str],
    any_of: list[tuple[str, set[str]]],
    context: str,
) -> None:
    """Validate that requested targets are present in a loader."""
    if loader is None:
        return
    graph = _sample_graph_from_loader(loader)
    if graph is None:
        return
    available = {
        'energy': getattr(graph.globals, 'energy', None) is not None,
        'forces': getattr(graph.nodes, 'forces', None) is not None,
        'stress': getattr(graph.globals, 'stress', None) is not None,
        'virials': getattr(graph.globals, 'virials', None) is not None,
        'dipole': getattr(graph.globals, 'dipole', None) is not None,
        'polarizability': getattr(graph.globals, 'polarizability', None) is not None,
    }
    missing = sorted([name for name in required if not available.get(name, False)])
    if missing:
        raise ValueError(
            f'{context} requires targets {missing}, but they are missing from the dataset.'
        )
    for label, options in any_of:
        if not any(available.get(name, False) for name in options):
            raise ValueError(
                f'{context} requires {label}, but none of {sorted(options)} are present in the dataset.'
            )


@gin.configurable('loss')
def loss(
    loss_cls=modules.WeightedEnergyForcesStressLoss,
    energy_weight: float | None = None,
    forces_weight: float | None = None,
    stress_weight: float | None = None,
    virials_weight: float | None = None,
    dipole_weight: float | None = None,
    polarizability_weight: float | None = None,
    huber_delta: float | None = None,
):
    """Construct the configured loss object with optional weight overrides.

    Args:
        loss_cls: Loss class or factory to instantiate.
        energy_weight: Optional energy loss weight.
        forces_weight: Optional forces loss weight.
        stress_weight: Optional stress loss weight.
        virials_weight: Optional virials loss weight.
        dipole_weight: Optional dipole loss weight.
        polarizability_weight: Optional polarizability loss weight.
        huber_delta: Optional Huber delta for robust losses.

    Returns:
        Instantiated loss object used by the training/evaluation loops.
    """
    overrides = {
        'energy_weight': energy_weight,
        'forces_weight': forces_weight,
        'stress_weight': stress_weight,
        'virials_weight': virials_weight,
        'dipole_weight': dipole_weight,
        'polarizability_weight': polarizability_weight,
        'huber_delta': huber_delta,
    }
    filtered = {key: value for key, value in overrides.items() if value is not None}
    return _instantiate_loss(loss_cls, filtered)


@gin.configurable
def flags(
    debug: bool,
    dtype: str,
    seed: int,
    profile: bool = False,
    device: str | None = None,
    distributed: bool = False,
    process_count: int | None = None,
    process_index: int | None = None,
    coordinator_address: str | None = None,
    coordinator_port: int | None = None,
):
    """Configure JAX runtime, device selection, and RNG seeds.

    Args:
        debug: Enable JAX debug checks for NaNs/infs.
        dtype: Default dtype string ('float32', 'float64', etc.).
        seed: Base RNG seed; each process offsets by its index.
        profile: Whether to enable optional profiling support.
        device: Optional device platform override (e.g. 'cpu', 'gpu').
        distributed: Whether to initialize distributed JAX.
        process_count: Total number of processes when distributed.
        process_index: Index of the current process.
        coordinator_address: Coordinator host for multi-process setup.
        coordinator_port: Coordinator port for multi-process setup.

    Returns:
        The original seed value (for convenience in gin bindings).
    """
    jax_config.update('jax_debug_nans', debug)
    jax_config.update('jax_debug_infs', debug)
    tools.set_default_dtype(dtype)
    device_utils.initialize_jax_runtime(
        device=device,
        distributed=distributed,
        process_count=process_count,
        process_index=process_index,
        coordinator_address=coordinator_address,
        coordinator_port=coordinator_port,
    )
    process_rank = getattr(jax, 'process_index', lambda: 0)()
    effective_seed = seed + int(process_rank)
    tools.set_seeds(effective_seed)
    if profile:
        helper = _get_profile_helper(log_warning=True)
        if helper is not None:
            helper.enable()
    return seed


@gin.configurable
def logs(
    name: str = None,
    level=logging.INFO,
    directory: str = 'results',
):
    """Configure logging and return a metrics logger.

    Args:
        name: Optional run name; random name is generated if omitted.
        level: Logging level for the run.
        directory: Output directory for logs and metrics files.

    Returns:
        Tuple of (directory, tag, logger) used throughout training.
    """
    date = datetime.datetime.now().strftime('%Y%m%d_%H%M')

    if name is None:
        name = get_random_name(
            separator='-', style='lowercase', combo=[ADJECTIVES, NAMES]
        )

    tag = f'{date}_{name}'

    process_index = getattr(jax, 'process_index', lambda: 0)()
    suffix = f'.rank{process_index}' if process_index else ''

    tools.setup_logger(
        level,
        directory=directory,
        filename=f'{tag}{suffix}.log',
        name=name,
        stream=process_index == 0,
    )
    logger = tools.MetricsLogger(directory=directory, filename=f'{tag}{suffix}.metrics')

    return directory, tag, logger


@gin.configurable
def wandb_run(
    *,
    enabled: bool = False,
    project: str | None = None,
    entity: str | None = None,
    name: str | None = None,
    group: str | None = None,
    tags: Sequence[str] = (),
    notes: str | None = None,
    mode: str | None = None,
    resume: str | bool | None = None,
    dir: str | None = None,
    config: dict | None = None,
    anonymous: str | None = None,
):
    """Initialize a Weights & Biases run if enabled.

    Args:
        enabled: Whether to create a W&B run.
        project: W&B project name.
        entity: W&B entity/account.
        name: Run name.
        group: Run group name.
        tags: Tags to attach to the run.
        notes: Free-form notes for the run.
        mode: W&B mode (online/offline/disabled).
        resume: Resume setting for W&B.
        dir: Output directory for W&B files.
        config: Config dictionary to log to W&B.
        anonymous: Anonymous setting for W&B.

    Returns:
        The W&B run object, or None when disabled.
    """
    if not enabled:
        return None
    try:
        import wandb  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            'wandb is not installed; install it or disable wandb logging.'
        ) from exc

    init_kwargs = dict(
        project=project,
        entity=entity,
        name=name,
        group=group,
        notes=notes,
        mode=mode,
        resume=resume,
        dir=dir,
        config=config,
        anonymous=anonymous,
    )
    init_kwargs = {k: v for k, v in init_kwargs.items() if v is not None}
    if tags:
        init_kwargs['tags'] = list(tags)

    logging.info('Initializing Weights & Biases run %s', name or '')
    return wandb.init(**init_kwargs)


def finish_wandb(run) -> None:
    """Safely finalize a W&B run if one was created."""
    if run is None:
        return
    finish = getattr(run, 'finish', None)
    if finish is None:
        return
    try:
        finish()
    except Exception:  # pragma: no cover - best effort cleanup
        logging.debug('Failed to close wandb run cleanly', exc_info=True)


@gin.configurable
def reload(params, path=None):
    """Reload parameters from a checkpoint file.

    Args:
        params: Current parameter tree, used to check compatibility.
        path: Optional checkpoint path; if None, returns params unchanged.

    Returns:
        Reloaded parameter tree (or the original params if no path is provided).
    """
    if path is not None:
        logging.info(f"Reloading parameters from '{path}'")
        with open(path, 'rb') as f:
            _ = pickle.load(f)
            new_params = pickle.load(f)

        # check compatibility
        if jax.tree_util.tree_structure(params) != jax.tree_util.tree_structure(
            new_params
        ):
            logging.warning(
                f"Parameters from '{path}' are not compatible with current model"
            )

        return new_params
    return params


@gin.configurable
def checks(
    energy_forces_predictor, params, train_loader, *, enabled: bool = False
) -> bool:
    """Run a diagnostic pass to validate energy/force normalization.

    Args:
        energy_forces_predictor: Predictor callable returning energy/forces.
        params: Parameters to evaluate.
        train_loader: Loader providing graphs for the check.
        enabled: When False, this function is a no-op.

    Returns:
        True if checks ran (and requested early exit), otherwise False.
    """
    if not enabled:
        return False

    logging.info('We will check the normalization of the model and exit.')
    energies = []
    forces = []
    for graph in tqdm(train_loader):
        out = energy_forces_predictor(params, graph)
        node_mask = jraph.get_node_padding_mask(graph)
        graph_mask = jraph.get_graph_padding_mask(graph)
        energies += [out['energy'][graph_mask]]
        forces += [out['forces'][node_mask]]
    en = jnp.concatenate(energies)
    fo = jnp.concatenate(forces)
    fo = jnp.linalg.norm(fo, axis=1)

    logging.info(f'Energy: {jnp.mean(en):.3f} +/- {jnp.std(en):.3f}')
    logging.info(f'        min/max: {jnp.min(en):.3f}/{jnp.max(en):.3f}')
    logging.info(f'        median: {jnp.median(en):.3f}')
    logging.info(f'Forces: {jnp.mean(fo):.3f} +/- {jnp.std(fo):.3f}')
    logging.info(f'        min/max: {jnp.min(fo):.3f}/{jnp.max(fo):.3f}')
    logging.info(f'        median: {jnp.median(fo):.3f}')
    return True


@gin.configurable
def exponential_decay(
    lr: float,
    interval_length: int | None = None,
    *,
    transition_steps: float = 0.0,
    decay_rate: float = 0.5,
    transition_begin: float = 0.0,
    staircase: bool = True,
    end_value: float | None = None,
):
    """Create an exponential learning-rate schedule.

    Args:
        lr: Initial learning rate.
        interval_length: Optional steps-per-epoch scaling factor.
        transition_steps: Number of (scaled) steps between decays.
        decay_rate: Multiplicative decay factor.
        transition_begin: Step offset before decay starts.
        staircase: Whether to apply stepwise (staircase) decay.
        end_value: Optional minimum learning rate.

    Returns:
        Optax schedule callable taking a step index.
    """
    step_scale = int(interval_length) if interval_length and interval_length > 0 else 1
    return optax.exponential_decay(
        init_value=lr,
        transition_steps=int(transition_steps * step_scale),
        decay_rate=decay_rate,
        transition_begin=int(transition_begin * step_scale),
        staircase=staircase,
        end_value=end_value,
    )


@gin.configurable
def piecewise_constant_schedule(
    lr: float,
    interval_length: int | None = None,
    *,
    boundaries_and_scales: dict[float, float],
):
    """Create a piecewise constant learning-rate schedule.

    Args:
        lr: Initial learning rate.
        interval_length: Optional steps-per-epoch scaling factor.
        boundaries_and_scales: Dict mapping boundary epochs to scale factors.

    Returns:
        Optax schedule callable taking a step index.
    """
    step_scale = int(interval_length) if interval_length and interval_length > 0 else 1
    scaled = {
        boundary * step_scale: scale
        for boundary, scale in boundaries_and_scales.items()
    }
    return optax.piecewise_constant_schedule(
        init_value=lr, boundaries_and_scales=scaled
    )


@gin.register
def constant_schedule(lr, interval_length=None):
    """Return a constant learning-rate schedule."""
    return optax.constant_schedule(lr)


@gin.configurable
def reduce_on_plateau(
    lr: float,
    interval_length: int | None = None,
    *,
    factor: float = 0.8,
    patience: int = 10,
    min_lr: float = 0.0,
    threshold: float = 1e-4,
):
    """Create a schedule that decays the LR when validation loss plateaus.

    Args:
        lr: Initial learning rate.
        interval_length: Unused; kept for API consistency with other schedulers.
        factor: Multiplicative decay factor when plateau is detected.
        patience: Number of evaluations without improvement before decaying.
        min_lr: Lower bound for the learning rate.
        threshold: Minimum improvement to reset patience.

    Returns:
        A schedule callable with an attached `update(loss)` method.
    """
    state = {
        'best': np.inf,
        'num_bad': 0,
        'current_lr': lr,
        'min_lr': min_lr,
    }

    def schedule(step):
        """Return the current learning rate (step argument ignored)."""
        return state['current_lr']

    def update(loss_value):
        """Update plateau state and adjust the learning rate if needed."""
        if loss_value + threshold < state['best']:
            state['best'] = float(loss_value)
            state['num_bad'] = 0
            return
        state['num_bad'] += 1
        if state['num_bad'] >= patience:
            new_lr = max(state['current_lr'] * factor, state['min_lr'])
            if new_lr < state['current_lr'] - threshold:
                logging.info(
                    'Plateau scheduler triggered: lr %.3e -> %.3e',
                    state['current_lr'],
                    new_lr,
                )
                state['current_lr'] = new_lr
            state['num_bad'] = 0

    schedule.update = update
    return schedule


gin.configurable('adam')(optax.scale_by_adam)
gin.configurable('amsgrad')(tools.scale_by_amsgrad)
gin.register('sgd')(optax.identity)


@gin.configurable
def optimizer(
    max_epochs: int,
    interval_length: int | None = None,
    weight_decay=0.0,
    lr=0.01,
    algorithm: Callable = optax.scale_by_adam,
    scheduler: Callable = constant_schedule,
    stage_two_lr: float | None = None,
    stage_two_interval: int | None = None,
    decoupled_weight_decay: bool = True,
    schedule_free: bool = False,
    schedule_free_b1: float = 0.95,
    schedule_free_weight_lr_power: float = 2.0,
):
    """Build the Optax optimizer and schedule for training.

    Args:
        max_epochs: Number of training epochs to run.
        interval_length: Steps-per-epoch scaling for schedules.
        weight_decay: Weight decay coefficient.
        lr: Base learning rate.
        algorithm: Optax transformation factory (e.g. Adam, AMSGrad).
        scheduler: Schedule factory used to create the LR schedule.
        stage_two_lr: Optional LR for a second training stage.
        stage_two_interval: Epoch to switch to the stage-two LR.
        decoupled_weight_decay: Whether to use decoupled weight decay.
        schedule_free: Enable schedule-free optimizer wrapper (optax.contrib).
        schedule_free_b1: Momentum parameter for schedule-free optimizer.
        schedule_free_weight_lr_power: Weight LR power for schedule-free optimizer.

    Returns:
        Tuple of (gradient_transformation, max_epochs).
    """

    def weight_decay_mask(params):
        """Build a tree mask selecting parameters to regularize."""
        params = tools.flatten_dict(params)
        mask = {
            k: any(('linear_down' in ki) or ('symmetric_contraction' in ki) for ki in k)
            for k in params
        }
        assert any(any(('linear_down' in ki) for ki in k) for k in params)
        assert any(any(('symmetric_contraction' in ki) for ki in k) for k in params)
        return tools.unflatten_dict(mask)

    step_scale = int(interval_length) if interval_length and interval_length > 0 else 1
    schedule = scheduler(lr, step_scale)
    if (
        stage_two_lr is not None
        and stage_two_interval is not None
        and stage_two_interval >= 0
    ):
        boundary_step = int(stage_two_interval * step_scale)
        stage_schedule = optax.constant_schedule(stage_two_lr)
        schedule = optax.join_schedules(
            schedules=[schedule, stage_schedule],
            boundaries=[boundary_step],
        )

    transforms = []
    if weight_decay and weight_decay != 0.0:
        if decoupled_weight_decay:
            transforms.append(
                optax.add_decayed_weights(weight_decay, mask=weight_decay_mask)
            )
        else:
            transforms.append(
                _masked_additive_weight_decay(
                    weight_decay=weight_decay, mask_fn=weight_decay_mask
                )
            )
    transforms.extend(
        [
            algorithm(),
            optax.scale_by_schedule(schedule),
            optax.scale(-1.0),
        ]
    )

    gradient_chain = optax.chain(*transforms)
    plateau_update = getattr(schedule, 'update', None)
    if callable(plateau_update):
        setattr(gradient_chain, 'scheduler_update', plateau_update)

    if schedule_free:
        if optax_schedule_free is None or schedule_free_eval_params is None:
            raise ImportError(
                'optax.contrib.schedule_free is unavailable; upgrade optax to use ScheduleFree.'
            )
        gradient_chain = optax_schedule_free(
            base_optimizer=gradient_chain,
            learning_rate=schedule,
            b1=schedule_free_b1,
            weight_lr_power=schedule_free_weight_lr_power,
        )
        setattr(gradient_chain, 'schedule_free_eval_fn', schedule_free_eval_params)

    setattr(gradient_chain, 'lr_schedule', schedule)
    return gradient_chain, max_epochs


def _masked_additive_weight_decay(
    weight_decay: float, mask_fn: Callable[[dict], dict]
) -> optax.GradientTransformation:
    """Create additive weight decay that respects a parameter mask.

    Args:
        weight_decay: Coefficient applied to masked parameters.
        mask_fn: Callable returning a pytree mask matching the params.

    Returns:
        Optax GradientTransformation implementing additive weight decay.
    """

    def init_fn(params):
        """Initialize the weight-decay transformation state."""
        return ()

    def update_fn(updates, state, params=None):
        """Apply masked decay to updates using current parameters."""
        if params is None:
            raise ValueError('Additive weight decay requires current parameters.')
        mask = mask_fn(params)

        def apply(update, param, mask_entry):
            """Apply weight decay to a single parameter if masked."""
            if not mask_entry:
                return update
            return update + weight_decay * param

        updates = jax.tree_util.tree_map(apply, updates, params, mask)
        return updates, state

    return optax.GradientTransformation(init_fn, update_fn)


@gin.configurable
def train(
    predictor,
    params,
    optimizer_state,
    train_loader,
    valid_loader,
    test_loader,
    gradient_transform,
    max_epochs: int,
    logger,
    directory,
    tag,
    *,
    patience: int | None = None,
    eval_train: bool = True,
    eval_test: bool = False,
    eval_interval: int = 1,
    log_errors: str = 'PerAtomRMSE',
    ema_decay: float | None = None,
    max_grad_norm: float | None = None,
    wandb_run=None,
    swa_config=None,
    checkpoint_dir: str | None = None,
    checkpoint_every: int | None = None,
    checkpoint_keep: int | None = 1,
    checkpoint_best: bool = False,
    resume_from: str | None = None,
    data_seed: int | None = None,
    lr_scale_by_graphs: bool = True,
    model_config: dict | None = None,
    **kwargs,
):
    """Run the end-to-end training loop with evaluation and checkpointing.

    This wraps the lower-level `tools.train()` iterator, handling logging, validation,
    early stopping, optional EMA/SWA evaluation parameters, and W&B reporting.
    It expects fixed-shape batches from the streaming loader so JAX compilation
    happens once; changing batch shapes would otherwise trigger recompilation and
    stall the training loop.

    Args:
        predictor: Callable that maps (params, graph) -> predictions.
        params: Trainable params (and optional config state).
        optimizer_state: Optimizer state to update during training.
        train_loader: Training data loader.
        valid_loader: Validation data loader.
        test_loader: Test data loader.
        gradient_transform: Optax gradient transformation.
        max_epochs: Max number of epochs to run.
        logger: Metrics logger for persistent logs.
        directory: Output directory for logs/checkpoints.
        tag: Run tag used to name output files.
        patience: Optional early stopping patience (epochs).
        eval_train: Whether to log training metrics each interval.
        eval_test: Whether to run eval on test set during training.
        eval_interval: Epoch interval between validation runs.
        log_errors: Metric selection preset to log.
        ema_decay: Optional EMA decay for evaluation parameters.
        max_grad_norm: Optional gradient clipping threshold.
        wandb_run: Optional W&B run handle.
        swa_config: Optional SWA configuration.
        checkpoint_dir: Directory to store checkpoints.
        checkpoint_every: Save checkpoints every N epochs.
        checkpoint_keep: Number of checkpoints to keep.
        checkpoint_best: Save a checkpoint whenever validation loss improves.
        resume_from: Optional checkpoint path to resume from.
        data_seed: Optional seed for shuffling data loaders.
        lr_scale_by_graphs: Whether to scale LR based on graphs per batch.
        model_config: Optional serialized model configuration for checkpoints.
        **kwargs: Additional arguments forwarded to `tools.train()`.

    Returns:
        Tuple of (last_epoch, eval_params_with_config).
    """
    trainable_params, static_config = _split_config(params)
    lowest_loss = np.inf
    patience_counter = 0
    loss_fn = loss()
    if log_errors is None and isinstance(loss_fn, modules.WeightedEnergyForcesL1L2Loss):
        log_errors = 'PerAtomMAE'
    required_by_loss = _required_targets_from_loss(loss_fn)
    metric_required, metric_any_of = _required_targets_from_log_errors(log_errors)
    interval_start = time.perf_counter()
    total_time_per_interval = []
    eval_time_per_interval = []
    timing_active = False

    checkpoint_enabled = bool(checkpoint_every and checkpoint_every > 0)
    checkpoint_best = bool(checkpoint_best)
    process_index = getattr(jax, 'process_index', lambda: 0)()
    process_count = getattr(jax, 'process_count', lambda: 1)()
    is_primary = process_index == 0

    def _log_info(message, *args):
        """Log a formatted message from the primary process only."""
        tools.log_info_primary(message, *args)

    checkpoint_dir_path: Path | None = None
    checkpoint_history: deque[Path] = deque()
    resume_path = Path(resume_from).expanduser() if resume_from else None
    schedule_free_eval_fn = getattr(gradient_transform, 'schedule_free_eval_fn', None)
    if schedule_free_eval_fn is not None and (
        ema_decay is not None or swa_config is not None
    ):
        _log_info(
            'ScheduleFree evaluation overrides EMA/SWA averages; using ScheduleFree parameters for eval.'
        )

    def _shard_loader(loader):
        """Shard a data loader across processes when supported."""
        if loader is None:
            return None
        if process_count > 1:
            shard_fn = getattr(loader, 'shard', None)
            if callable(shard_fn):
                return shard_fn(process_count, process_index)
        return loader

    def _resolve_checkpoint_dir() -> Path:
        """Resolve the checkpoint directory based on config and resume path."""
        if checkpoint_dir:
            return Path(checkpoint_dir).expanduser()
        if resume_path is not None:
            return resume_path.parent
        return Path(directory) / 'checkpoints'

    start_interval = 0
    if resume_path is not None:
        if not resume_path.exists():
            raise FileNotFoundError(
                f"Requested resume checkpoint '{resume_from}' does not exist."
            )
        with resume_path.open('rb') as f:
            resume_state = pickle.load(f)
        params = resume_state.get(
            'params', _attach_config(trainable_params, static_config)
        )
        trainable_params, static_config = _split_config(params)
        optimizer_state = resume_state.get('optimizer_state', optimizer_state)
        lowest_loss = resume_state.get('lowest_loss', lowest_loss)
        patience_counter = resume_state.get('patience_counter', patience_counter)
        start_interval = resume_state.get('epoch', 0)
        _log_info(
            'Resuming training from checkpoint %s (starting at epoch %s).',
            resume_path,
            start_interval,
        )

    initial_epoch = start_interval

    if checkpoint_enabled or checkpoint_best:
        checkpoint_dir_path = _resolve_checkpoint_dir()

    def _checkpoint_state(
        epoch_idx, current_params, current_optimizer_state, eval_params_
    ):
        """Build a checkpoint payload for the current epoch."""
        state = {
            'epoch': epoch_idx,
            'params': _attach_config(current_params, static_config),
            'optimizer_state': current_optimizer_state,
            'eval_params': _attach_config(eval_params_, static_config),
            'lowest_loss': lowest_loss,
            'patience_counter': patience_counter,
            'checkpoint_format': 2,
        }
        if model_config is not None:
            state['model_config'] = model_config
        return state

    def _write_checkpoint(path: Path, state: dict):
        """Write a checkpoint payload to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('wb') as f:
            pickle.dump(state, f)

    def _save_checkpoint(
        epoch_idx, current_params, current_optimizer_state, eval_params_
    ):
        """Persist a checkpoint for the current epoch if enabled."""
        if not checkpoint_enabled or not is_primary:
            return None
        if epoch_idx < initial_epoch:
            return None
        if (
            checkpoint_every
            and checkpoint_every > 0
            and epoch_idx % checkpoint_every != 0
        ):
            return None
        filename = f'{tag}_epoch{epoch_idx:05d}.ckpt'
        path = checkpoint_dir_path / filename
        state = _checkpoint_state(
            epoch_idx, current_params, current_optimizer_state, eval_params_
        )
        _write_checkpoint(path, state)
        checkpoint_history.append(path)
        if checkpoint_keep and checkpoint_keep > 0:
            while len(checkpoint_history) > checkpoint_keep:
                old_path = checkpoint_history.popleft()
                try:
                    old_path.unlink(missing_ok=True)
                except FileNotFoundError:
                    pass
        _log_info('Saved checkpoint to %s', path)
        return path

    def _save_best_checkpoint(
        epoch_idx, current_params, current_optimizer_state, eval_params_
    ):
        """Persist a checkpoint when validation loss improves."""
        if not checkpoint_best or not is_primary:
            return None
        if checkpoint_dir_path is None:
            return None
        path = checkpoint_dir_path / f'{tag}_best.ckpt'
        state = _checkpoint_state(
            epoch_idx, current_params, current_optimizer_state, eval_params_
        )
        _write_checkpoint(path, state)
        _log_info('Saved best checkpoint to %s', path)
        return path

    def _loader_has_graphs(loader):
        """Return True if the loader has graph data to iterate over."""
        if loader is None:
            return False
        graphs = getattr(loader, 'graphs', None)
        if graphs is None:
            return True
        return len(graphs) > 0

    def _split_loader_by_heads(loader):
        """Split a loader into per-head sub-loaders if supported."""
        if loader is None:
            return {}
        splitter = getattr(loader, 'split_by_heads', None)
        if splitter is None:
            return {}
        return splitter()

    def _prepare_loader_for_eval(loader, epoch):
        """Prepare a loader for evaluation at the current epoch."""
        if loader is None:
            return None
        loader = _shard_loader(loader)
        if loader is None:
            return None
        setter = getattr(loader, 'set_epoch', None)
        if callable(setter):
            setter(epoch)
        return loader

    def _enumerate_eval_targets(loader, head_loaders, epoch):
        """Enumerate evaluation loaders (including per-head splits when available)."""
        if not _loader_has_graphs(loader):
            return []
        if head_loaders:
            items = []
            for head, sub_loader in head_loaders.items():
                prepped = _prepare_loader_for_eval(sub_loader, epoch)
                if _loader_has_graphs(prepped):
                    items.append((head, prepped))
            return items
        return [(None, _prepare_loader_for_eval(loader, epoch))]

    def _log_metrics(
        metrics_: Mapping[str, Any],
        mode: str,
        epoch: int,
        *,
        head_name: str | None = None,
    ) -> None:
        """Log metrics to file/stdout/W&B with consistent formatting."""
        if metrics_ is None:
            return
        metrics_ = dict(metrics_)
        metrics_['mode'] = mode
        if head_name is not None:
            metrics_['head'] = head_name
        metrics_['interval'] = epoch
        metrics_['epoch'] = epoch
        if is_primary:
            logger.log(metrics_)

        eval_mode = mode if head_name is None else f'{mode}:{head_name}'
        selected_metrics = _select_metrics_for_logging(metrics_, log_errors)
        metrics_blob = ', '.join(
            f'{metric}={_format_metric_value(metrics_, metric)}'
            for metric in selected_metrics
        )
        loss_value = metrics_.get('loss')
        if loss_value is None:
            loss_value = float('nan')
        _log_info(
            f'Epoch {epoch}: {eval_mode}: '
            f'loss={float(loss_value):.4e}'
            + (f', {metrics_blob}' if metrics_blob else '')
        )
        if wandb_run is not None and is_primary:
            wandb_payload = {
                'interval': int(epoch),
                'epoch': int(epoch),
                f'{eval_mode}/loss': float(loss_value),
            }
            for key, value in metrics_.items():
                maybe_value = _maybe_float(value)
                if maybe_value is not None:
                    wandb_payload[f'{eval_mode}/{key}'] = maybe_value
            wandb_run.log(wandb_payload, step=int(epoch))

    swa_loss_fn = None
    stage_two_active = False
    if swa_config and swa_config.stage_loss_factory is not None:
        swa_loss_fn = _instantiate_loss(
            swa_config.stage_loss_factory, swa_config.stage_loss_kwargs or {}
        )
    train_head_loaders = _split_loader_by_heads(train_loader)
    valid_head_loaders = _split_loader_by_heads(valid_loader)
    test_head_loaders = _split_loader_by_heads(test_loader)

    _assert_required_targets(
        train_loader,
        required=required_by_loss,
        any_of=[],
        context='Training loss',
    )
    if eval_train:
        _assert_required_targets(
            train_loader,
            required=metric_required,
            any_of=metric_any_of,
            context=f'Training metrics ({log_errors or "PerAtomRMSE"})',
        )
    _assert_required_targets(
        valid_loader,
        required=required_by_loss,
        any_of=[],
        context='Validation loss',
    )
    _assert_required_targets(
        valid_loader,
        required=metric_required,
        any_of=metric_any_of,
        context=f'Validation metrics ({log_errors or "PerAtomRMSE"})',
    )
    _assert_required_targets(
        test_loader,
        required=required_by_loss,
        any_of=[],
        context='Test loss',
    )
    _assert_required_targets(
        test_loader,
        required=metric_required,
        any_of=metric_any_of,
        context=f'Test metrics ({log_errors or "PerAtomRMSE"})',
    )
    for head_name, head_loader in valid_head_loaders.items():
        _assert_required_targets(
            head_loader,
            required=required_by_loss,
            any_of=[],
            context=f'Validation loss (head={head_name})',
        )
        _assert_required_targets(
            head_loader,
            required=metric_required,
            any_of=metric_any_of,
            context=f'Validation metrics ({log_errors or "PerAtomRMSE"}, head={head_name})',
        )
    for head_name, head_loader in test_head_loaders.items():
        _assert_required_targets(
            head_loader,
            required=required_by_loss,
            any_of=[],
            context=f'Test loss (head={head_name})',
        )
        _assert_required_targets(
            head_loader,
            required=metric_required,
            any_of=metric_any_of,
            context=f'Test metrics ({log_errors or "PerAtomRMSE"}, head={head_name})',
        )

    def _should_run_eval(current_epoch: int, is_last: bool) -> bool:
        """Decide whether to run evaluation for the current epoch."""
        if is_last:
            return True
        if current_epoch == initial_epoch:
            return True
        if eval_interval is None or eval_interval <= 0:
            return True
        return current_epoch % eval_interval == 0

    def with_config(params_):
        """Reattach static config state to params for model application."""
        return _attach_config(params_, static_config)

    def predictor_with_config(params_, graph_):
        """Apply the predictor with config state attached."""
        return predictor(with_config(params_), graph_)

    stop_after_epoch = False
    if not isinstance(eval_train, bool):
        raise ValueError('eval_train must be a boolean.')

    for train_item in tools.train(
        params=trainable_params,
        total_loss_fn=lambda params, graph: loss_fn(
            graph, predictor_with_config(params, graph)
        ),
        train_loader=train_loader,
        gradient_transform=gradient_transform,
        optimizer_state=optimizer_state,
        ema_decay=ema_decay,
        max_grad_norm=max_grad_norm,
        start_interval=start_interval,
        schedule_free_eval_fn=schedule_free_eval_fn,
        data_seed=data_seed,
        lr_scale_by_graphs=lr_scale_by_graphs,
        collect_metrics=eval_train,
        metrics_predictor=predictor_with_config,
        metrics_loss_fn=loss_fn,
        metrics_required_targets=required_by_loss if eval_train else None,
        **kwargs,
    ):
        if len(train_item) == 5:
            epoch, trainable_params, optimizer_state, eval_params, train_metrics = (
                train_item
            )
        else:
            epoch, trainable_params, optimizer_state, eval_params = train_item
            train_metrics = None
        stop_after_epoch = False
        now = time.perf_counter()
        if timing_active:
            total_time_per_interval.append(now - interval_start)
        eval_start = time.perf_counter()

        helper = _get_profile_helper(log_warning=False)
        if helper is not None:
            helper.restart_timer()

        last_epoch = max_epochs is not None and epoch >= max_epochs

        if is_primary:
            with open(f'{directory}/{tag}.pkl', 'wb') as f:
                pickle.dump(gin.operative_config_str(), f)
                pickle.dump(with_config(trainable_params), f)

        def eval_and_print(loader, mode: str, *, head_name: str | None = None):
            """Run evaluation on a loader and log metrics."""
            if loader is None:
                return
            loss_, metrics_ = tools.evaluate(
                predictor=predictor_with_config,
                params=eval_params,
                loss_fn=loss_fn,
                data_loader=loader,
                name=mode if head_name is None else f'{mode}:{head_name}',
                log_padding=False,
            )
            metrics_ = dict(metrics_)
            metrics_['loss'] = loss_
            _log_metrics(metrics_, mode, epoch, head_name=head_name)

            synced_loss = loss_
            if process_count > 1:
                loss_value = 0.0 if loss_ is None else float(loss_)
                synced = multihost_utils.broadcast_one_to_all(
                    jnp.array(loss_value, dtype=jnp.float32),
                    is_source=is_primary,
                )
                synced_loss = float(np.asarray(synced))
            return synced_loss

        evaluate_now = _should_run_eval(epoch, last_epoch)

        if eval_train and evaluate_now:
            if train_metrics is not None:
                head_metrics_map = train_metrics.get('head_metrics')
                if head_metrics_map:
                    metrics_ = dict(train_metrics)
                    metrics_.pop('head_metrics', None)
                    _log_metrics(metrics_, 'train', epoch)
                    for head_name, head_metrics in head_metrics_map.items():
                        _log_metrics(head_metrics, 'train', epoch, head_name=head_name)
                else:
                    metrics_ = dict(train_metrics)
                    metrics_.pop('head_metrics', None)
                    _log_metrics(metrics_, 'train', epoch)
            else:
                if epoch != initial_epoch:
                    for head_name, loader in _enumerate_eval_targets(
                        train_loader, train_head_loaders, epoch
                    ):
                        eval_and_print(loader, 'train', head_name=head_name)

        if (
            (eval_test or last_epoch)
            and _loader_has_graphs(test_loader)
            and evaluate_now
        ):
            for head_name, loader in _enumerate_eval_targets(
                test_loader, test_head_loaders, epoch
            ):
                eval_and_print(loader, 'eval_test', head_name=head_name)

        last_valid_loss = None
        if evaluate_now:
            for head_name, loader in _enumerate_eval_targets(
                valid_loader, valid_head_loaders, epoch
            ):
                last_valid_loss = eval_and_print(loader, 'valid', head_name=head_name)

        improved = False
        if last_valid_loss is not None and is_primary:
            if last_valid_loss >= lowest_loss:
                patience_counter += 1
                if patience is not None and patience_counter >= patience:
                    _log_info(
                        f'Stopping optimization after {patience_counter} epochs without improvement'
                    )
                    stop_after_epoch = True
            else:
                lowest_loss = last_valid_loss
                patience_counter = 0
                improved = True

        if (
            swa_loss_fn is not None
            and not stage_two_active
            and swa_config is not None
            and epoch >= swa_config.start_interval
        ):
            _log_info(
                'SWA stage starting at epoch %s: switching to Stage Two loss.',
                epoch,
            )
            loss_fn = swa_loss_fn
            lowest_loss = np.inf
            patience_counter = 0
            stage_two_active = True

        if timing_active:
            eval_time_per_interval.append(time.perf_counter() - eval_start)
            recent_train = total_time_per_interval[-3:]
            recent_eval = eval_time_per_interval[-3:]
            avg_train_time = np.mean(recent_train)
            avg_eval_time = np.mean(recent_eval)
            avg_interval_time = np.mean(
                [train + eval_ for train, eval_ in zip(recent_train, recent_eval)]
            )

            _log_info(
                f'Epoch {epoch}: Time per epoch: {avg_interval_time:.1f}s '
                f'(train {avg_train_time:.1f}s, eval {avg_eval_time:.1f}s).'
            )
            if wandb_run is not None:
                wandb_run.log(
                    {
                        'interval': int(epoch),
                        'timing/interval_seconds': float(
                            total_time_per_interval[-1] + eval_time_per_interval[-1]
                        ),
                        'timing/train_seconds': float(total_time_per_interval[-1]),
                        'timing/eval_seconds': float(eval_time_per_interval[-1]),
                    },
                    step=int(epoch),
                )
        interval_start = time.perf_counter()
        timing_active = True

        plateau_updater = getattr(gradient_transform, 'scheduler_update', None)
        if callable(plateau_updater) and last_valid_loss is not None:
            plateau_updater(last_valid_loss)

        if improved:
            _save_best_checkpoint(epoch, trainable_params, optimizer_state, eval_params)

        _save_checkpoint(epoch, trainable_params, optimizer_state, eval_params)
        _log_info('-' * 80)

        if stop_after_epoch or last_epoch:
            break

    if process_count > 1:
        try:
            multihost_utils.sync_global_devices('mace_jax_train_complete')
        except Exception as exc:  # pragma: no cover - best effort barrier
            _log_info('Failed to sync processes at shutdown: %s', exc)

    _log_info('Training complete')
    return epoch, _attach_config(eval_params, static_config)


def parse_argv(argv: list[str]):
    """Parse CLI-like arguments into gin configuration bindings.

    Args:
        argv: Argument vector, including the program name at index 0.

    This helper is used by CLI entry points to bind gin parameters from
    positional `.gin` files and `--key=value` overrides.
    """

    def gin_bind_parameter(key: str, value: str):
        """Bind a single key/value pair into the gin configuration."""
        # We need to guess if value is a string or not
        value = value.strip()
        if value[0] == value[-1] and value[0] in ('"', "'"):
            gin.parse_config(f'{key} = {value}')
        if value[0] == '@':
            gin.parse_config(f'{key} = {value}')
        if value in ['True', 'False', 'None']:
            gin.parse_config(f'{key} = {value}')
        if any(c.isalpha() for c in value):
            gin.parse_config(f'{key} = "{value}"')
        else:
            gin.parse_config(f'{key} = {value}')

    only_the_key = None
    for arg in argv[1:]:
        if only_the_key is None:
            if arg.endswith('.gin'):
                gin.parse_config_file(arg)
            elif arg.startswith('--'):
                if '=' in arg:
                    key, value = arg[2:].split('=')
                    gin_bind_parameter(key, value)
                else:
                    only_the_key = arg[2:]
            else:
                raise ValueError(
                    f'Unknown argument: \'{arg}\'. Expected a .gin file or a --key "some value" pair.'
                )
        else:
            gin_bind_parameter(only_the_key, arg)
            only_the_key = None
