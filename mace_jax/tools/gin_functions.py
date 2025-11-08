import datetime
import inspect
import logging
import numbers
import pickle
import time
from collections import deque
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Dict, List, Optional

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
from tqdm import tqdm
from unique_names_generator import get_random_name
from unique_names_generator.data import ADJECTIVES, NAMES

from mace_jax import modules, tools
from mace_jax.tools import device as device_utils
from jax.experimental import multihost_utils

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
    device: str | None = None,
    distributed: bool = False,
    process_count: int | None = None,
    process_index: int | None = None,
    coordinator_address: str | None = None,
    coordinator_port: int | None = None,
):
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
    date = datetime.datetime.now().strftime('%Y%m%d_%H%M')

    if name is None:
        name = get_random_name(
            separator='-', style='lowercase', combo=[ADJECTIVES, NAMES]
        )

    tag = f'{date}_{name}'

    process_index = getattr(jax, 'process_index', lambda: 0)()
    suffix = f'.rank{process_index}' if process_index else ''

    tools.setup_logger(
        level, directory=directory, filename=f'{tag}{suffix}.log', name=name
    )
    logger = tools.MetricsLogger(
        directory=directory, filename=f'{tag}{suffix}.metrics'
    )

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


@gin.configurable
def reduce_on_plateau(
    lr: float,
    steps_per_interval: int,
    *,
    factor: float = 0.8,
    patience: int = 10,
    min_lr: float = 0.0,
    threshold: float = 1e-4,
):
    state = {
        'best': np.inf,
        'num_bad': 0,
        'current_lr': lr,
        'min_lr': min_lr,
    }

    def schedule(step):
        return state['current_lr']

    def update(loss_value):
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
    steps_per_interval: int,
    max_num_intervals: int,
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
    def weight_decay_mask(params):
        params = tools.flatten_dict(params)
        mask = {
            k: any(('linear_down' in ki) or ('symmetric_contraction' in ki) for ki in k)
            for k in params
        }
        assert any(any(('linear_down' in ki) for ki in k) for k in params)
        assert any(any(('symmetric_contraction' in ki) for ki in k) for k in params)
        return tools.unflatten_dict(mask)

    schedule = scheduler(lr, steps_per_interval)
    if (
        stage_two_lr is not None
        and stage_two_interval is not None
        and steps_per_interval is not None
        and stage_two_interval >= 0
    ):
        boundary_step = stage_two_interval * steps_per_interval
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
    transforms.extend([
        algorithm(),
        optax.scale_by_schedule(schedule),
        optax.scale(-1.0),
    ])

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

    return (
        gradient_chain,
        steps_per_interval,
        max_num_intervals,
    )


def _masked_additive_weight_decay(
    weight_decay: float, mask_fn: Callable[[dict], dict]
) -> optax.GradientTransformation:
    def init_fn(params):
        return ()

    def update_fn(updates, state, params=None):
        if params is None:
            raise ValueError('Additive weight decay requires current parameters.')
        mask = mask_fn(params)

        def apply(update, param, mask_entry):
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
    max_num_intervals: int,
    steps_per_interval: int,
    logger,
    directory,
    tag,
    *,
    patience: int | None = None,
    eval_train: bool = False,
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
    resume_from: str | None = None,
    data_seed: int | None = None,
    **kwargs,
):
    lowest_loss = np.inf
    patience_counter = 0
    loss_fn = loss()
    start_time = time.perf_counter()
    total_time_per_interval = []
    eval_time_per_interval = []

    checkpoint_enabled = bool(checkpoint_every and checkpoint_every > 0)
    process_index = getattr(jax, 'process_index', lambda: 0)()
    process_count = getattr(jax, 'process_count', lambda: 1)()
    is_primary = process_index == 0

    def _log_info(message, *args):
        if is_primary:
            logging.info(message, *args)

    checkpoint_dir_path: Path | None = None
    checkpoint_history: deque[Path] = deque()
    resume_path = Path(resume_from).expanduser() if resume_from else None
    schedule_free_eval_fn = getattr(gradient_transform, 'schedule_free_eval_fn', None)
    if schedule_free_eval_fn is not None and (ema_decay is not None or swa_config is not None):
        _log_info(
            'ScheduleFree evaluation overrides EMA/SWA averages; using ScheduleFree parameters for eval.'
        )

    def _shard_loader(loader):
        if loader is None:
            return None
        if process_count > 1:
            return loader.shard(process_count, process_index)
        return loader

    def _resolve_checkpoint_dir() -> Path:
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
        params = resume_state.get('params', params)
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

    if checkpoint_enabled:
        checkpoint_dir_path = _resolve_checkpoint_dir()

    def _save_checkpoint(
        epoch_idx, current_params, current_optimizer_state, eval_params_
    ):
        if not checkpoint_enabled or not is_primary:
            return None
        if epoch_idx < initial_epoch:
            return None
        if checkpoint_every and checkpoint_every > 0 and epoch_idx % checkpoint_every != 0:
            return None
        checkpoint_dir_path.mkdir(parents=True, exist_ok=True)
        filename = f'{tag}_epoch{epoch_idx:05d}.ckpt'
        path = checkpoint_dir_path / filename
        state = {
            'epoch': epoch_idx,
            'params': current_params,
            'optimizer_state': current_optimizer_state,
            'eval_params': eval_params_,
            'lowest_loss': lowest_loss,
            'patience_counter': patience_counter,
            'checkpoint_format': 2,
        }
        with path.open('wb') as f:
            pickle.dump(state, f)
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

    def _loader_has_graphs(loader):
        if loader is None:
            return False
        graphs = getattr(loader, 'graphs', None)
        if graphs is None:
            return True
        return len(graphs) > 0

    def _split_loader_by_heads(loader):
        if loader is None:
            return {}
        splitter = getattr(loader, 'split_by_heads', None)
        if splitter is None:
            return {}
        return splitter()

    def _prepare_loader_for_eval(loader, epoch):
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
        if process_count > 1 and not is_primary:
            return []
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

    swa_loss_fn = None
    stage_two_active = False
    if swa_config and swa_config.stage_loss_factory is not None:
        swa_loss_fn = _instantiate_loss(
            swa_config.stage_loss_factory, swa_config.stage_loss_kwargs or {}
        )
    valid_head_loaders = _split_loader_by_heads(valid_loader)
    test_head_loaders = _split_loader_by_heads(test_loader)

    def _should_run_eval(current_epoch: int, is_last: bool) -> bool:
        if is_last:
            return True
        if current_epoch == initial_epoch:
            return True
        if eval_interval is None or eval_interval <= 0:
            return True
        return current_epoch % eval_interval == 0

    stop_after_epoch = False
    for epoch, params, optimizer_state, eval_params in tools.train(
        params=params,
        total_loss_fn=lambda params, graph: loss_fn(graph, predictor(params, graph)),
        train_loader=train_loader,
        gradient_transform=gradient_transform,
        optimizer_state=optimizer_state,
        steps_per_interval=steps_per_interval,
        ema_decay=ema_decay,
        max_grad_norm=max_grad_norm,
        start_interval=start_interval,
        schedule_free_eval_fn=schedule_free_eval_fn,
        data_seed=data_seed,
        **kwargs,
    ):
        stop_after_epoch = False
        total_time_per_interval += [time.perf_counter() - start_time]
        start_time = time.perf_counter()

        helper = _get_profile_helper(log_warning=False)
        if helper is not None:
            helper.restart_timer()

        last_epoch = max_num_intervals is not None and epoch >= max_num_intervals

        if is_primary:
            with open(f'{directory}/{tag}.pkl', 'wb') as f:
                pickle.dump(gin.operative_config_str(), f)
                pickle.dump(params, f)

        def eval_and_print(loader, mode: str, *, head_name: str | None = None):
            eval_mode = mode if head_name is None else f'{mode}:{head_name}'
            loss_ = None
            metrics_: dict[str, Any] = {}
            if is_primary or process_count == 1:
                loss_, metrics_ = tools.evaluate(
                    predictor=predictor,
                    params=eval_params,
                    loss_fn=loss_fn,
                    data_loader=loader,
                    name=eval_mode,
                )
                metrics_['mode'] = mode
                if head_name is not None:
                    metrics_['head'] = head_name
                metrics_['interval'] = epoch
                metrics_['epoch'] = epoch
                if is_primary:
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

                _log_info(
                    f'Epoch {epoch}: {eval_mode}: '
                    f'loss={loss_:.4f}, '
                    f'{error_e}={_(error_e)}, '
                    f'{error_f}={_(error_f)}, '
                    f'{error_s}={_(error_s)}'
                )
                if wandb_run is not None and is_primary:
                    wandb_mode = eval_mode
                    wandb_payload = {
                        'interval': int(epoch),
                        'epoch': int(epoch),
                        f'{wandb_mode}/loss': float(loss_),
                    }
                    for key, value in metrics_.items():
                        maybe_value = _maybe_float(value)
                        if maybe_value is not None:
                            wandb_payload[f'{wandb_mode}/{key}'] = maybe_value
                    wandb_run.log(wandb_payload, step=int(epoch))

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

        if (eval_train or last_epoch) and evaluate_now and (
            is_primary or process_count == 1
        ):
            if isinstance(eval_train, (int, float)):
                subset_loader = train_loader.subset(eval_train)
                subset_loader = _prepare_loader_for_eval(subset_loader, epoch)
                eval_and_print(subset_loader, 'eval_train')
            else:
                eval_and_print(
                    _prepare_loader_for_eval(train_loader, epoch), 'eval_train'
                )

        if (
            (eval_test or last_epoch)
            and _loader_has_graphs(test_loader)
            and evaluate_now
            and (is_primary or process_count == 1)
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
                last_valid_loss = eval_and_print(
                    loader, 'eval_valid', head_name=head_name
                )

        if last_valid_loss is not None:
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

        eval_time_per_interval += [time.perf_counter() - start_time]
        avg_time_per_interval = np.mean(total_time_per_interval[-3:])
        avg_eval_time_per_interval = np.mean(eval_time_per_interval[-3:])

        _log_info(
            f'Epoch {epoch}: Time per epoch: {avg_time_per_interval:.1f}s, '
            f'among which {avg_eval_time_per_interval:.1f}s for evaluation.'
        )
        if wandb_run is not None and total_time_per_interval and eval_time_per_interval:
            wandb_run.log(
                {
                    'interval': int(epoch),
                    'timing/interval_seconds': float(total_time_per_interval[-1]),
                    'timing/eval_seconds': float(eval_time_per_interval[-1]),
                },
                step=int(epoch),
            )

        plateau_updater = getattr(gradient_transform, 'scheduler_update', None)
        if callable(plateau_updater) and last_valid_loss is not None:
            plateau_updater(last_valid_loss)

        _save_checkpoint(epoch, params, optimizer_state, eval_params)

        if stop_after_epoch or last_epoch:
            break

    _log_info('Training complete')
    return epoch, eval_params


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
