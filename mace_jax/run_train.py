import logging
import sys

import gin
import jax

import mace_jax
from mace_jax import tools
from mace_jax.tools.gin_datasets import datasets
from mace_jax.tools.gin_functions import (
    checks,
    flags,
    logs,
    optimizer,
    parse_argv,
    reload,
    train,
)
from mace_jax.tools.gin_model import model


def main():
    seed = flags()

    directory, tag, logger = logs()

    with open(f"{directory}/{tag}.gin", "wt") as f:
        f.write(gin.config_str())

    logging.info(f"MACE version: {mace_jax.__version__}")

    train_loader, valid_loader, test_loader, atomic_energies_dict, r_max = datasets()

    model_fn, params, num_message_passing = model(
        r_max=r_max,
        atomic_energies_dict=atomic_energies_dict,
        train_graphs=train_loader.graphs,
        initialize_seed=seed,
    )

    params = reload(params)

    predictor = jax.jit(
        lambda w, g: tools.predict_energy_forces_stress(lambda *x: model_fn(w, *x), g)
    )

    if checks(predictor, params, train_loader):
        return

    gradient_transform, steps_per_interval, max_num_intervals = optimizer()
    optimizer_state = gradient_transform.init(params)

    logging.info(f"Number of parameters: {tools.count_parameters(params)}")
    logging.info(
        f"Number of parameters in optimizer: {tools.count_parameters(optimizer_state)}"
    )

    train(
        predictor,
        params,
        optimizer_state,
        train_loader,
        valid_loader,
        test_loader,
        gradient_transform,
        max_num_intervals,
        steps_per_interval,
        logger,
        directory,
        tag,
    )


if __name__ == "__main__":
    parse_argv(sys.argv)
    main()
