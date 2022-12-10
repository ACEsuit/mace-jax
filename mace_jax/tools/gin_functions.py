import datetime
import logging
import pickle
import time
from typing import Callable, Dict, List, Optional, Union

import ase.data
import e3nn_jax as e3nn
import gin
import haiku as hk
import jax
import jax.numpy as jnp
import jraph
import numpy as np
import optax
from tqdm import tqdm
from unique_names_generator import get_random_name
from unique_names_generator.data import ADJECTIVES, NAMES

from mace_jax import data, modules, tools

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
def datasets(
    *,
    r_max: float,
    train_path: str,
    config_type_weights: Dict = None,
    num_train: int = None,
    valid_path: str = None,
    valid_fraction: float = None,
    test_path: str = None,
    seed: int = 1234,
    energy_key: str = "energy",
    forces_key: str = "forces",
    n_node: int = 1,
    n_edge: int = 1,
    n_graph: int = 1,
):
    """Load training and test dataset from xyz file"""

    atomic_energies_dict, all_train_configs = data.load_from_xyz(
        file_path=train_path,
        config_type_weights=config_type_weights,
        energy_key=energy_key,
        forces_key=forces_key,
        extract_atomic_energies=True,
        num_configs=num_train,
    )
    logging.info(
        f"Loaded {len(all_train_configs)} training configurations from '{train_path}'"
    )

    if valid_path is not None:
        _, valid_configs = data.load_from_xyz(
            file_path=valid_path,
            config_type_weights=config_type_weights,
            energy_key=energy_key,
            forces_key=forces_key,
            extract_atomic_energies=False,
        )
        logging.info(
            f"Loaded {len(valid_configs)} validation configurations from '{valid_path}'"
        )
        train_configs = all_train_configs
    else:
        logging.info(
            "Using random %s%% of training set for validation", 100 * valid_fraction
        )
        train_configs, valid_configs = data.random_train_valid_split(
            all_train_configs, valid_fraction, seed
        )

    if test_path is not None:
        _, test_configs = data.load_from_xyz(
            file_path=test_path,
            config_type_weights=config_type_weights,
            energy_key=energy_key,
            forces_key=forces_key,
            extract_atomic_energies=False,
        )
        logging.info(
            f"Loaded {len(test_configs)} test configurations from '{test_path}'"
        )
    else:
        test_configs = []

    z_table = data.get_atomic_number_table_from_zs(
        z
        for configs in (train_configs, valid_configs)
        for config in configs
        for z in config.atomic_numbers
    )
    logging.info(f"z_table= {z_table}")

    logging.info(
        f"Total number of configurations: "
        f"train={len(train_configs)}, "
        f"valid={len(valid_configs)}, "
        f"test={len(test_configs)}"
    )

    train_loader = data.GraphDataLoader(
        graphs=[
            data.graph_from_configuration(c, cutoff=r_max) for c in tqdm(train_configs)
        ],
        n_node=n_node,
        n_edge=n_edge,
        n_graph=n_graph,
        shuffle=True,
    )
    valid_loader = data.GraphDataLoader(
        graphs=[
            data.graph_from_configuration(c, cutoff=r_max) for c in tqdm(valid_configs)
        ],
        n_node=n_node,
        n_edge=n_edge,
        n_graph=n_graph,
        shuffle=False,
    )
    test_loader = data.GraphDataLoader(
        graphs=[
            data.graph_from_configuration(c, cutoff=r_max) for c in tqdm(test_configs)
        ],
        n_node=n_node,
        n_edge=n_edge,
        n_graph=n_graph,
        shuffle=False,
    )
    return dict(
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        atomic_energies_dict=atomic_energies_dict,
        r_max=r_max,
        z_table=z_table,
        train_configs=train_configs,
        valid_configs=valid_configs,
        test_configs=test_configs,
    )


gin.register(jax.nn.silu)
gin.register(jax.nn.relu)
gin.register(jax.nn.gelu)
gin.register(jnp.abs)
gin.register(jnp.tanh)
gin.register("identity")(lambda x: x)

gin.register("std_scaling")(tools.compute_mean_std_atomic_inter_energy)
gin.register("rms_forces_scaling")(tools.compute_mean_rms_energy_forces)


@gin.configurable
def constant_scaling(train_loader, atomic_energies, *, mean=0.0, std=1.0):
    return mean, std


@gin.configurable
def bessel_basis(length, max_length, number: int):
    return e3nn.bessel(length, number, max_length)


@gin.configurable
def soft_envelope(
    length, max_length, arg_multiplicator: float = 2.0, value_at_origin: float = 1.2
):
    return e3nn.soft_envelope(
        length,
        max_length,
        arg_multiplicator=arg_multiplicator,
        value_at_origin=value_at_origin,
    )


@gin.configurable
def polynomial_envelope(length, max_length, degree0: int, degree1: int):
    return e3nn.poly_envelope(degree0, degree1, max_length)(length)


@gin.configurable
def u_envelope(length, max_length, p: int):
    return e3nn.poly_envelope(p - 1, 2, max_length)(length)


gin.external_configurable(modules.LinearNodeEmbeddingBlock, "LinearEmbedding")


@gin.configurable
class LinearMassEmbedding(hk.Module):
    def __init__(self, num_species: int, irreps_out: e3nn.Irreps):
        super().__init__()
        self.num_species = num_species
        self.irreps_out = e3nn.Irreps(irreps_out).filter("0e").regroup()

    def __call__(self, node_specie: jnp.ndarray) -> e3nn.IrrepsArray:
        w = hk.get_parameter(
            f"embeddings",
            shape=(self.num_species, self.irreps_out.dim),
            dtype=jnp.float32,
            init=hk.initializers.RandomNormal(),
        )
        atomic_masses = jnp.asarray(ase.data.atomic_masses)[node_specie] / 90.0  # [...]
        return e3nn.IrrepsArray(
            self.irreps_out, w[node_specie] * atomic_masses[..., None]
        )


@gin.configurable
def model(
    seed: int,
    r_max: float,
    atomic_energies_dict: Dict[int, float] = None,
    train_loader: data.GraphDataLoader = None,
    train_configs=None,
    z_table=None,
    initialize: bool = True,
    *,
    scaling: Callable = None,
    atomic_energies: Union[str, np.ndarray, Dict[int, float]] = None,
    avg_num_neighbors: float = "average",
    avg_r_min: float = None,
    num_species: int = None,
    num_interactions=3,
    path_normalization="element",
    gradient_normalization="element",
    learnable_atomic_energies=False,
    radial_basis: Callable[[jnp.ndarray], jnp.ndarray] = bessel_basis,
    radial_envelope: Callable[[jnp.ndarray], jnp.ndarray] = soft_envelope,
    **kwargs,
):
    if avg_num_neighbors == "average":
        avg_num_neighbors = tools.compute_avg_num_neighbors(train_loader)
        logging.info(
            f"Compute the average number of neighbors: {avg_num_neighbors:.3f}"
        )
    else:
        logging.info(f"Use the average number of neighbors: {avg_num_neighbors:.3f}")

    if avg_r_min == "average":
        avg_r_min = tools.compute_avg_min_neighbor_distance(train_loader)
        logging.info(f"Compute the average min neighbor distance: {avg_r_min:.3f}")
    elif avg_r_min is None:
        logging.info(f"Do not normalize the radial basis (avg_r_min=None)")
    else:
        logging.info(f"Use the average min neighbor distance: {avg_r_min:.3f}")

    if atomic_energies is None:
        if atomic_energies_dict is None or len(atomic_energies_dict) == 0:
            atomic_energies = "average"
        else:
            atomic_energies = "isolated_atom"

    if atomic_energies == "average":
        atomic_energies_dict = data.compute_average_E0s(train_configs, z_table)
        logging.info(
            f"Computed average Atomic Energies using least squares: {atomic_energies_dict}"
        )
        atomic_energies = np.array(
            [atomic_energies_dict.get(z, 0.0) for z in range(num_species)]
        )
    elif atomic_energies == "isolated_atom":
        logging.info(
            f"Using atomic energies from isolated atoms in the dataset: {atomic_energies_dict}"
        )
        atomic_energies = np.array(
            [atomic_energies_dict.get(z, 0.0) for z in range(num_species)]
        )
    elif atomic_energies == "zero":
        logging.info("Not using atomic energies")
        atomic_energies = np.zeros(num_species)
    elif isinstance(atomic_energies, np.ndarray):
        logging.info(
            f"Use Atomic Energies that are provided: {atomic_energies.tolist()}"
        )
        if atomic_energies.shape != (num_species,):
            logging.error(
                f"atomic_energies.shape={atomic_energies.shape} != (num_species={num_species},)"
            )
            raise ValueError
    elif isinstance(atomic_energies, dict):
        atomic_energies_dict = atomic_energies
        logging.info(f"Use Atomic Energies that are provided: {atomic_energies_dict}")
        atomic_energies = np.array(
            [atomic_energies_dict.get(z, 0.0) for z in range(num_species)]
        )
    else:
        raise ValueError(f"atomic_energies={atomic_energies} is not supported")

    # check that num_species is consistent with the dataset
    if z_table is None:
        for graph in train_loader.graphs:
            if not np.all(graph.nodes.species < num_species):
                raise ValueError(
                    f"max(graph.nodes.species)={np.max(graph.nodes.species)} >= num_species={num_species}"
                )
    else:
        if max(z_table.zs) >= num_species:
            raise ValueError(
                f"max(z_table.zs)={max(z_table.zs)} >= num_species={num_species}"
            )

    if scaling is None:
        mean, std = 0.0, 1.0
    else:
        mean, std = scaling(train_loader, atomic_energies)
        logging.info(
            f"Scaling with {scaling.__qualname__}: mean={mean:.2f}, std={std:.2f}"
        )

    @hk.without_apply_rng
    @hk.transform
    def model_(
        vectors: jnp.ndarray,  # [n_edges, 3]
        node_z: jnp.ndarray,  # [n_nodes]
        senders: jnp.ndarray,  # [n_edges]
        receivers: jnp.ndarray,  # [n_edges]
    ) -> jnp.ndarray:
        e3nn.config("path_normalization", path_normalization)
        e3nn.config("gradient_normalization", gradient_normalization)

        mace = modules.GeneralMACE(
            output_irreps="0e",
            r_max=r_max,
            avg_num_neighbors=avg_num_neighbors,
            num_interactions=num_interactions,
            avg_r_min=avg_r_min,
            num_species=num_species,
            radial_basis=radial_basis,
            radial_envelope=radial_envelope,
            **kwargs,
        )

        if hk.running_init():
            logging.info(
                "model: "
                f"num_features={mace.num_features} "
                f"hidden_irreps={mace.hidden_irreps} "
                f"sh_irreps={mace.sh_irreps} "
                f"interaction_irreps={mace.interaction_irreps} ",
            )

        contributions = mace(
            vectors, node_z, senders, receivers
        )  # [n_nodes, num_interactions, 0e]
        contributions = contributions.array[:, :, 0]  # [n_nodes, num_interactions]
        node_energies = jnp.sum(contributions, axis=1)  # [n_nodes, ]

        node_energies = mean + std * node_energies

        if learnable_atomic_energies:
            atomic_energies_ = hk.get_parameter(
                "atomic_energies",
                shape=(num_species,),
                init=hk.initializers.Constant(atomic_energies),
            )
        else:
            atomic_energies_ = jnp.asarray(atomic_energies)
        node_energies += atomic_energies_[node_z]  # [n_nodes, ]

        return node_energies

    if initialize:
        params = jax.jit(model_.init)(
            jax.random.PRNGKey(seed),
            jnp.zeros((1, 3)),
            jnp.array([16]),
            jnp.array([0]),
            jnp.array([0]),
        )
    else:
        params = None

    return model_.apply, params, num_interactions


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
    steps_per_epoch: int,
    *,
    transition_steps: float = 0.0,
    decay_rate: float = 0.5,
    transition_begin: float = 0.0,
    staircase: bool = True,
    end_value: Optional[float] = None,
):
    return optax.exponential_decay(
        init_value=lr,
        transition_steps=transition_steps * steps_per_epoch,
        decay_rate=decay_rate,
        transition_begin=transition_begin * steps_per_epoch,
        staircase=staircase,
        end_value=end_value,
    )


@gin.configurable
def piecewise_constant_schedule(
    lr: float, steps_per_epoch: int, *, boundaries_and_scales: Dict[float, float]
):
    boundaries_and_scales = {
        boundary * steps_per_epoch: scale
        for boundary, scale in boundaries_and_scales.items()
    }
    return optax.piecewise_constant_schedule(
        init_value=lr, boundaries_and_scales=boundaries_and_scales
    )


@gin.register
def constant_schedule(lr, steps_per_epoch):
    return optax.constant_schedule(lr)


gin.configurable("adam")(optax.scale_by_adam)
gin.configurable("amsgrad")(tools.scale_by_amsgrad)
gin.register("sgd")(optax.identity)


@gin.configurable
def optimizer(
    steps_per_epoch: int,
    weight_decay=0.0,
    lr=0.01,
    max_num_epochs: int = 2048,
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
            optax.scale_by_schedule(scheduler(lr, steps_per_epoch)),
            optax.scale(-1.0),  # Gradient descent.
        ),
        max_num_epochs,
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
    max_num_epochs,
    logger,
    directory,
    tag,
    *,
    patience: int,
    eval_train: bool = False,
    eval_test: bool = False,
    eval_interval: int = 1,
    log_errors: str = "PerAtomRMSE",
    **kwargs,
):
    if log_errors == "PerAtomRMSE":
        error_e = "rmse_e_per_atom"
        error_f = "rmse_f"
        error_s = "rmse_s"
    elif log_errors == "TotalRMSE":
        error_e = "rmse_e"
        error_f = "rmse_f"
        error_s = "rmse_s"
    elif log_errors == "PerAtomMAE":
        error_e = "mae_e_per_atom"
        error_f = "mae_f"
        error_s = "mae_s"
    elif log_errors == "TotalMAE":
        error_e = "mae_e"
        error_f = "mae_f"
        error_s = "mae_s"

    lowest_loss = np.inf
    patience_counter = 0
    loss_fn = loss()
    start_time = time.perf_counter()
    total_time_per_epoch = []
    eval_time_per_epoch = []

    for epoch, params, optimizer_state, ema_params in tools.train(
        model=model,
        params=params,
        loss_fn=loss_fn,
        train_loader=train_loader,
        gradient_transform=gradient_transform,
        optimizer_state=optimizer_state,
        start_epoch=0,
        max_num_epochs=max_num_epochs,
        logger=logger,
        **kwargs,
    ):
        total_time_per_epoch += [time.perf_counter() - start_time]
        start_time = time.perf_counter()

        try:
            import profile_nn_jax
        except ImportError:
            pass
        else:
            profile_nn_jax.restart_timer()

        last_epoch = epoch == max_num_epochs - 1
        if epoch % eval_interval == 0 or last_epoch:
            with open(f"{directory}/{tag}.pkl", "wb") as f:
                pickle.dump(gin.operative_config_str(), f)
                pickle.dump(params, f)

            if eval_train or last_epoch:
                loss_, metrics_ = tools.evaluate(
                    model=model,
                    params=ema_params,
                    loss_fn=loss_fn,
                    data_loader=train_loader,
                )
                metrics_["mode"] = "eval_train"
                metrics_["epoch"] = epoch
                logger.log(metrics_)

                logging.info(
                    f"Epoch {epoch}: Train: "
                    f"loss={loss_:.4f}, "
                    f"{error_e}={1e3 * metrics_[error_e]:.1f} meV, "
                    f"{error_f}={1e3 * metrics_[error_f]:.1f} meV/A, "
                    f"{error_s}={1e3 * metrics_[error_s]:.1f} meV/A^3"
                )

            if (
                (eval_test or last_epoch)
                and test_loader is not None
                and len(test_loader) > 0
            ):
                loss_, metrics_ = tools.evaluate(
                    model=model,
                    params=ema_params,
                    loss_fn=loss_fn,
                    data_loader=test_loader,
                )
                metrics_["mode"] = "eval_test"
                metrics_["epoch"] = epoch
                logger.log(metrics_)

                logging.info(
                    f"Epoch {epoch}: Test: "
                    f"loss={loss_:.4f}, "
                    f"{error_e}={1e3 * metrics_[error_e]:.1f} meV, "
                    f"{error_f}={1e3 * metrics_[error_f]:.1f} meV/A, "
                    f"{error_s}={1e3 * metrics_[error_s]:.1f} meV/A^3"
                )

            loss_, metrics_ = tools.evaluate(
                model=model,
                params=ema_params,
                loss_fn=loss_fn,
                data_loader=valid_loader,
            )
            metrics_["mode"] = "eval"
            metrics_["epoch"] = epoch
            logger.log(metrics_)

            logging.info(
                f"Epoch {epoch}: Validation: "
                f"loss={loss_:.4f}, "
                f"{error_e}={1e3 * metrics_[error_e]:.1f} meV, "
                f"{error_f}={1e3 * metrics_[error_f]:.1f} meV/A, "
                f"{error_s}={1e3 * metrics_[error_s]:.1f} meV/A^3"
            )

            if loss_ >= lowest_loss:
                patience_counter += 1
                if patience_counter >= patience:
                    logging.info(
                        f"Stopping optimization after {patience_counter} epochs without improvement"
                    )
                    break
            else:
                lowest_loss = loss_
                patience_counter = 0

            eval_time_per_epoch += [time.perf_counter() - start_time]
            avg_time_per_epoch = np.mean(total_time_per_epoch[-eval_interval:])
            avg_eval_time_per_epoch = np.mean(eval_time_per_epoch[-eval_interval:])

            logging.info(
                f"Epoch {epoch}: Time per epoch: {avg_time_per_epoch:.1f}s, "
                f"among which {avg_eval_time_per_epoch:.1f}s for evaluation."
            )
        else:
            eval_time_per_epoch += [time.perf_counter() - start_time]  # basically 0

    logging.info("Training complete")
    return epoch, ema_params


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
