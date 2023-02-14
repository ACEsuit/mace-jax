import datetime
import logging
import pickle
import time
from typing import Callable, Dict, List, Optional

import gin
import jax
import jax.numpy as jnp
import jraph
import numpy as np
import optax
from tqdm import tqdm
from unique_names_generator import get_random_name
from unique_names_generator.data import ADJECTIVES, NAMES

from mace_jax import modules, tools

loss = gin.configurable("loss")(modules.WeightedEnergyFrocesStressLoss)


@gin.configurable
def flags(
    debug: bool,
    dtype: str,
    seed: int,
    profile: bool = False,
):
    jax.config.update("jax_debug_nans", debug)
    jax.config.update("jax_debug_infs", debug)
    tools.set_default_dtype(dtype)
    tools.set_seeds(seed)
    if profile:
        import profile_nn_jax

        profile_nn_jax.enable(timing=True, statistics=True)
    return seed


@gin.configurable
def logs(
    name: str = None,
    level=logging.INFO,
    directory: str = "results",
):
    date = datetime.datetime.now().strftime("%Y%m%d_%H%M")

    if name is None:
        name = get_random_name(
            separator="-", style="lowercase", combo=[ADJECTIVES, NAMES]
        )

    tag = f"{date}_{name}"

    tools.setup_logger(level, directory=directory, filename=f"{tag}.log", name=name)
    logger = tools.MetricsLogger(directory=directory, filename=f"{tag}.metrics")

    return directory, tag, logger


@gin.configurable
def reload(params, path=None):
    if path is not None:
        logging.info(f"Reloading parameters from '{path}'")
        with open(path, "rb") as f:
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

    logging.info("We will check the normalization of the model and exit.")
    energies = []
    forces = []
    for graph in tqdm(train_loader):
        out = energy_forces_predictor(params, graph)
        node_mask = jraph.get_node_padding_mask(graph)
        graph_mask = jraph.get_graph_padding_mask(graph)
        energies += [out["energy"][graph_mask]]
        forces += [out["forces"][node_mask]]
    en = jnp.concatenate(energies)
    fo = jnp.concatenate(forces)
    fo = jnp.linalg.norm(fo, axis=1)

    logging.info(f"Energy: {jnp.mean(en):.3f} +/- {jnp.std(en):.3f}")
    logging.info(f"        min/max: {jnp.min(en):.3f}/{jnp.max(en):.3f}")
    logging.info(f"        median: {jnp.median(en):.3f}")
    logging.info(f"Forces: {jnp.mean(fo):.3f} +/- {jnp.std(fo):.3f}")
    logging.info(f"        min/max: {jnp.min(fo):.3f}/{jnp.max(fo):.3f}")
    logging.info(f"        median: {jnp.median(fo):.3f}")
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
    end_value: Optional[float] = None,
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
    lr: float, steps_per_interval: int, *, boundaries_and_scales: Dict[float, float]
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


gin.configurable("adam")(optax.scale_by_adam)
gin.configurable("amsgrad")(tools.scale_by_amsgrad)
gin.register("sgd")(optax.identity)


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
            k: any(("linear_down" in ki) or ("symmetric_contraction" in ki) for ki in k)
            for k in params
        }
        assert any(any(("linear_down" in ki) for ki in k) for k in params)
        assert any(any(("symmetric_contraction" in ki) for ki in k) for k in params)
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
    model,
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
    patience: Optional[int] = None,
    eval_train: bool = False,
    eval_test: bool = False,
    log_errors: str = "PerAtomRMSE",
    **kwargs,
):
    lowest_loss = np.inf
    patience_counter = 0
    loss_fn = loss()
    start_time = time.perf_counter()
    total_time_per_interval = []
    eval_time_per_interval = []

    for interval, params, optimizer_state, ema_params in tools.train(
        model=model,
        params=params,
        loss_fn=loss_fn,
        train_loader=train_loader,
        gradient_transform=gradient_transform,
        optimizer_state=optimizer_state,
        steps_per_interval=steps_per_interval,
        **kwargs,
    ):
        total_time_per_interval += [time.perf_counter() - start_time]
        start_time = time.perf_counter()

        try:
            import profile_nn_jax
        except ImportError:
            pass
        else:
            profile_nn_jax.restart_timer()

        last_interval = interval == max_num_intervals

        with open(f"{directory}/{tag}.pkl", "wb") as f:
            pickle.dump(gin.operative_config_str(), f)
            pickle.dump(params, f)

        def eval_and_print(loader, mode: str):
            loss_, metrics_ = tools.evaluate(
                model=model,
                params=ema_params,
                loss_fn=loss_fn,
                data_loader=loader,
                name=mode,
            )
            metrics_["mode"] = mode
            metrics_["interval"] = interval
            logger.log(metrics_)

            if log_errors == "PerAtomRMSE":
                error_e = "rmse_e_per_atom"
                error_f = "rmse_f"
                error_s = "rmse_s"
            elif log_errors == "rel_PerAtomRMSE":
                error_e = "rmse_e_per_atom"
                error_f = "rel_rmse_f"
                error_s = "rel_rmse_s"
            elif log_errors == "TotalRMSE":
                error_e = "rmse_e"
                error_f = "rmse_f"
                error_s = "rmse_s"
            elif log_errors == "PerAtomMAE":
                error_e = "mae_e_per_atom"
                error_f = "mae_f"
                error_s = "mae_s"
            elif log_errors == "rel_PerAtomMAE":
                error_e = "mae_e_per_atom"
                error_f = "rel_mae_f"
                error_s = "rel_mae_s"
            elif log_errors == "TotalMAE":
                error_e = "mae_e"
                error_f = "mae_f"
                error_s = "mae_s"

            def _(x: str):
                v: float = metrics_.get(x, None)
                if v is None:
                    return "N/A"
                if x.startswith("rel_"):
                    return f"{100 * v:.1f}%"
                if "_e" in x:
                    return f"{1e3 * v:.1f} meV"
                if "_f" in x:
                    return f"{1e3 * v:.1f} meV/Å"
                if "_s" in x:
                    return f"{1e3 * v:.1f} meV/Å³"
                raise NotImplementedError

            logging.info(
                f"Interval {interval}: {mode}: "
                f"loss={loss_:.4f}, "
                f"{error_e}={_(error_e)}, "
                f"{error_f}={_(error_f)}, "
                f"{error_s}={_(error_s)}"
            )
            return loss_

        if eval_train or last_interval:
            if isinstance(eval_train, (int, float)):
                eval_and_print(train_loader.subset(eval_train), "eval_train")
            else:
                eval_and_print(train_loader, "eval_train")

        if (
            (eval_test or last_interval)
            and test_loader is not None
            and len(test_loader) > 0
        ):
            eval_and_print(test_loader, "eval_test")

        if valid_loader is not None and len(valid_loader) > 0:
            loss_ = eval_and_print(valid_loader, "eval_valid")

            if loss_ >= lowest_loss:
                patience_counter += 1
                if patience is not None and patience_counter >= patience:
                    logging.info(
                        f"Stopping optimization after {patience_counter} intervals without improvement"
                    )
                    break
            else:
                lowest_loss = loss_
                patience_counter = 0

        eval_time_per_interval += [time.perf_counter() - start_time]
        avg_time_per_interval = np.mean(total_time_per_interval[-3:])
        avg_eval_time_per_interval = np.mean(eval_time_per_interval[-3:])

        logging.info(
            f"Interval {interval}: Time per interval: {avg_time_per_interval:.1f}s, "
            f"among which {avg_eval_time_per_interval:.1f}s for evaluation."
        )

        if last_interval:
            break

    logging.info("Training complete")
    return interval, ema_params


def parse_argv(argv: List[str]):
    def gin_bind_parameter(key: str, value: str):
        # We need to guess if value is a string or not
        value = value.strip()
        if value[0] == value[-1] and value[0] in ('"', "'"):
            gin.parse_config(f"{key} = {value}")
        if value[0] == "@":
            gin.parse_config(f"{key} = {value}")
        if value in ["True", "False", "None"]:
            gin.parse_config(f"{key} = {value}")
        if any(c.isalpha() for c in value):
            gin.parse_config(f'{key} = "{value}"')
        else:
            gin.parse_config(f"{key} = {value}")

    only_the_key = None
    for arg in argv[1:]:
        if only_the_key is None:
            if arg.endswith(".gin"):
                gin.parse_config_file(arg)
            elif arg.startswith("--"):
                if "=" in arg:
                    key, value = arg[2:].split("=")
                    gin_bind_parameter(key, value)
                else:
                    only_the_key = arg[2:]
            else:
                raise ValueError(
                    f"Unknown argument: '{arg}'. Expected a .gin file or a --key \"some value\" pair."
                )
        else:
            gin_bind_parameter(only_the_key, arg)
            only_the_key = None
