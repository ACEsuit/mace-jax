import datetime
import inspect
import logging
import numbers
import pickle
import time
from collections.abc import Callable, Sequence
from typing import Dict, List, Optional

import gin
import jax
import jax.numpy as jnp
import jraph
import numpy as np
import optax
from jax import config as jax_config
from tqdm import tqdm
from unique_names_generator import get_random_name
from unique_names_generator.data import ADJECTIVES, NAMES

from mace_jax import modules, tools

_PROFILE_HELPER = None
_PROFILE_HELPER_MISSING_WARNED = False


def _get_profile_helper(*, log_warning: bool) -> object | None:
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


def _instantiate_loss(loss_cls, overrides: dict):
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
):
    jax_config.update('jax_debug_nans', debug)
    jax_config.update('jax_debug_infs', debug)
    tools.set_default_dtype(dtype)
    tools.set_seeds(seed)
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
    date = datetime.datetime.now().strftime('%Y%m%d_%H%M')

    if name is None:
        name = get_random_name(
            separator='-', style='lowercase', combo=[ADJECTIVES, NAMES]
        )

    tag = f'{date}_{name}'

    tools.setup_logger(level, directory=directory, filename=f'{tag}.log', name=name)
    logger = tools.MetricsLogger(directory=directory, filename=f'{tag}.metrics')

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
    steps_per_interval: int,
    *,
    transition_steps: float = 0.0,
    decay_rate: float = 0.5,
    transition_begin: float = 0.0,
    staircase: bool = True,
    end_value: float | None = None,
):
    return optax.exponential_decay(
        init_value=lr,
        transition_steps=transition_steps * steps_per_interval,
        decay_rate=decay_rate,
        transition_begin=transition_begin * steps_per_interval,
        staircase=staircase,
        end_value=end_value,
    )


@gin.configurable
def piecewise_constant_schedule(
    lr: float, steps_per_interval: int, *, boundaries_and_scales: dict[float, float]
):
    boundaries_and_scales = {
        boundary * steps_per_interval: scale
        for boundary, scale in boundaries_and_scales.items()
    }
    return optax.piecewise_constant_schedule(
        init_value=lr, boundaries_and_scales=boundaries_and_scales
    )


@gin.register
def constant_schedule(lr, steps_per_interval):
    return optax.constant_schedule(lr)


gin.configurable('adam')(optax.scale_by_adam)
gin.configurable('amsgrad')(tools.scale_by_amsgrad)
gin.register('sgd')(optax.identity)


@gin.configurable
def optimizer(
    steps_per_interval: int,
    max_num_intervals: int,
    weight_decay=0.0,
    lr=0.01,
    algorithm: Callable = optax.scale_by_adam,
    scheduler: Callable = constant_schedule,
):
    def weight_decay_mask(params):
        params = tools.flatten_dict(params)
        mask = {
            k: any(('linear_down' in ki) or ('symmetric_contraction' in ki) for ki in k)
            for k in params
        }
        assert any(any(('linear_down' in ki) for ki in k) for k in params)
        assert any(any(('symmetric_contraction' in ki) for ki in k) for k in params)
        return tools.unflatten_dict(mask)

    return (
        optax.chain(
            optax.add_decayed_weights(weight_decay, mask=weight_decay_mask),
            algorithm(),
            optax.scale_by_schedule(scheduler(lr, steps_per_interval)),
            optax.scale(-1.0),  # Gradient descent.
        ),
        steps_per_interval,
        max_num_intervals,
    )


@gin.configurable
def train(
    predictor,
    params,
    optimizer_state,
    train_loader,
    valid_loader,
    test_loader,
    gradient_transform,
    max_num_intervals: int,
    steps_per_interval: int,
    logger,
    directory,
    tag,
    *,
    patience: int | None = None,
    eval_train: bool = False,
    eval_test: bool = False,
    log_errors: str = 'PerAtomRMSE',
    ema_decay: float | None = None,
    max_grad_norm: float | None = None,
    wandb_run=None,
    **kwargs,
):
    lowest_loss = np.inf
    patience_counter = 0
    loss_fn = loss()
    start_time = time.perf_counter()
    total_time_per_interval = []
    eval_time_per_interval = []

    for interval, params, optimizer_state, eval_params in tools.train(
        params=params,
        total_loss_fn=lambda params, graph: loss_fn(graph, predictor(params, graph)),
        train_loader=train_loader,
        gradient_transform=gradient_transform,
        optimizer_state=optimizer_state,
        steps_per_interval=steps_per_interval,
        ema_decay=ema_decay,
        max_grad_norm=max_grad_norm,
        **kwargs,
    ):
        total_time_per_interval += [time.perf_counter() - start_time]
        start_time = time.perf_counter()

        helper = _get_profile_helper(log_warning=False)
        if helper is not None:
            helper.restart_timer()

        last_interval = interval == max_num_intervals

        with open(f'{directory}/{tag}.pkl', 'wb') as f:
            pickle.dump(gin.operative_config_str(), f)
            pickle.dump(params, f)

        def eval_and_print(loader, mode: str):
            loss_, metrics_ = tools.evaluate(
                predictor=predictor,
                params=eval_params,
                loss_fn=loss_fn,
                data_loader=loader,
                name=mode,
            )
            metrics_['mode'] = mode
            metrics_['interval'] = interval
            logger.log(metrics_)

            def _maybe_float(value):
                if isinstance(value, numbers.Number):
                    return float(value)
                if isinstance(value, (np.ndarray, jnp.ndarray)) and value.shape == ():
                    return float(np.asarray(value))
                return None

            if log_errors == 'PerAtomRMSE':
                error_e = 'rmse_e_per_atom'
                error_f = 'rmse_f'
                error_s = 'rmse_s'
            elif log_errors == 'rel_PerAtomRMSE':
                error_e = 'rmse_e_per_atom'
                error_f = 'rel_rmse_f'
                error_s = 'rel_rmse_s'
            elif log_errors == 'TotalRMSE':
                error_e = 'rmse_e'
                error_f = 'rmse_f'
                error_s = 'rmse_s'
            elif log_errors == 'PerAtomMAE':
                error_e = 'mae_e_per_atom'
                error_f = 'mae_f'
                error_s = 'mae_s'
            elif log_errors == 'rel_PerAtomMAE':
                error_e = 'mae_e_per_atom'
                error_f = 'rel_mae_f'
                error_s = 'rel_mae_s'
            elif log_errors == 'TotalMAE':
                error_e = 'mae_e'
                error_f = 'mae_f'
                error_s = 'mae_s'

            def _(x: str):
                v: float = metrics_.get(x, None)
                if v is None:
                    return 'N/A'
                if x.startswith('rel_'):
                    return f'{100 * v:.1f}%'
                if '_e' in x:
                    return f'{1e3 * v:.1f} meV'
                if '_f' in x:
                    return f'{1e3 * v:.1f} meV/Å'
                if '_s' in x:
                    return f'{1e3 * v:.1f} meV/Å³'
                raise NotImplementedError

            logging.info(
                f'Interval {interval}: {mode}: '
                f'loss={loss_:.4f}, '
                f'{error_e}={_(error_e)}, '
                f'{error_f}={_(error_f)}, '
                f'{error_s}={_(error_s)}'
            )
            if wandb_run is not None:
                wandb_payload = {
                    'interval': int(interval),
                    f'{mode}/loss': float(loss_),
                }
                for key, value in metrics_.items():
                    maybe_value = _maybe_float(value)
                    if maybe_value is not None:
                        wandb_payload[f'{mode}/{key}'] = maybe_value
                wandb_run.log(wandb_payload, step=int(interval))
            return loss_

        if eval_train or last_interval:
            if isinstance(eval_train, (int, float)):
                eval_and_print(train_loader.subset(eval_train), 'eval_train')
            else:
                eval_and_print(train_loader, 'eval_train')

        if (
            (eval_test or last_interval)
            and test_loader is not None
            and len(test_loader) > 0
        ):
            eval_and_print(test_loader, 'eval_test')

        if valid_loader is not None and len(valid_loader) > 0:
            loss_ = eval_and_print(valid_loader, 'eval_valid')

            if loss_ >= lowest_loss:
                patience_counter += 1
                if patience is not None and patience_counter >= patience:
                    logging.info(
                        f'Stopping optimization after {patience_counter} intervals without improvement'
                    )
                    break
            else:
                lowest_loss = loss_
                patience_counter = 0

        eval_time_per_interval += [time.perf_counter() - start_time]
        avg_time_per_interval = np.mean(total_time_per_interval[-3:])
        avg_eval_time_per_interval = np.mean(eval_time_per_interval[-3:])

        logging.info(
            f'Interval {interval}: Time per interval: {avg_time_per_interval:.1f}s, '
            f'among which {avg_eval_time_per_interval:.1f}s for evaluation.'
        )
        if wandb_run is not None and total_time_per_interval and eval_time_per_interval:
            wandb_run.log(
                {
                    'interval': int(interval),
                    'timing/interval_seconds': float(total_time_per_interval[-1]),
                    'timing/eval_seconds': float(eval_time_per_interval[-1]),
                },
                step=int(interval),
            )

        if last_interval:
            break

    logging.info('Training complete')
    return interval, eval_params


def parse_argv(argv: list[str]):
    def gin_bind_parameter(key: str, value: str):
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
