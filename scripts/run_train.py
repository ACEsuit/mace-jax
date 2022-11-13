import logging
import pickle
import sys
from typing import Dict

import gin
import jax
import jax.numpy as jnp
import jraph

import mace_jax
from mace_jax import tools
from mace_jax.tools.gin_functions import (
    datasets,
    flags,
    logs,
    loss,
    model,
    optimizer,
    parse_argv,
    reload,
    train,
)


def main():
    seed = flags()

    directory, tag, logger = logs()

    with open(f"{directory}/{tag}.gin", "wt") as f:
        f.write(gin.config_str())

    logging.info(f"MACE version: {mace_jax.__version__}")

    dd = datasets()
    train_loader = dd["train_loader"]
    valid_loader = dd["valid_loader"]
    test_loader = dd["test_loader"]

    model_fn, params, num_message_passing = model(
        seed,
        dd["r_max"],
        dd["atomic_energies_dict"],
        dd["train_loader"],
        dd["train_configs"],
        dd["z_table"],
    )

    params = reload(params)

    gradient_transform, max_num_epochs = optimizer(steps_per_epoch=len(train_loader))
    optimizer_state = gradient_transform.init(params)

    logging.info(f"Number of parameters: {tools.count_parameters(params)}")
    logging.info(
        f"Number of parameters in optimizer: {tools.count_parameters(optimizer_state)}"
    )

    @jax.jit
    def energy_forces_predictor(w, graph: jraph.GraphsTuple) -> Dict[str, jnp.ndarray]:
        def energy_fn(positions):
            vectors = tools.get_edge_relative_vectors(
                positions=positions,
                senders=graph.senders,
                receivers=graph.receivers,
                shifts=graph.edges.shifts,
                cell=graph.globals.cell,
                n_edge=graph.n_edge,
            )
            node_energies = model_fn(
                w, vectors, graph.nodes.species, graph.senders, graph.receivers
            )  # [n_nodes, ]
            return jnp.sum(node_energies), node_energies

        minus_forces, node_energies = jax.grad(energy_fn, has_aux=True)(
            graph.nodes.positions
        )

        graph_energies = tools.sum_nodes_of_the_same_graph(
            graph, node_energies
        )  # [ n_graphs,]

        return {
            "energy": graph_energies,  # [n_graphs,]
            "forces": -minus_forces,  # [n_nodes, 3]
        }

    epoch, params = train(
        energy_forces_predictor,
        params,
        optimizer_state,
        train_loader,
        valid_loader,
        gradient_transform,
        max_num_epochs,
        logger,
    )

    with open(f"{directory}/{tag}.pkl", "wb") as f:
        pickle.dump(gin.operative_config_str(), f)
        pickle.dump(params, f)

    test_loss, eval_metrics = tools.evaluate(
        model=energy_forces_predictor,
        params=params,
        loss_fn=loss(),
        data_loader=test_loader,
    )
    eval_metrics["mode"] = "test"
    eval_metrics["epoch"] = epoch
    logger.log(eval_metrics)

    error_e = eval_metrics["rmse_e_per_atom"] * 1e3
    error_f = eval_metrics["rmse_f"] * 1e3
    logging.info(
        f"Final Test loss={test_loss:.4f}, RMSE_E_per_atom={error_e:.1f} meV, RMSE_F={error_f:.1f} meV / A"
    )


if __name__ == "__main__":
    parse_argv(sys.argv)
    main()
