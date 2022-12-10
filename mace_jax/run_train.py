import logging
import sys
from typing import Dict

import e3nn_jax as e3nn
import gin
import jax
import jax.numpy as jnp
import jraph

import mace_jax
from mace_jax import tools
from mace_jax.tools.gin_functions import (
    checks,
    datasets,
    flags,
    logs,
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

    @jax.jit
    def predictor(w, graph: jraph.GraphsTuple) -> Dict[str, jnp.ndarray]:
        def energy_fn(positions, cell):
            vectors = tools.get_edge_relative_vectors(
                positions=positions,
                senders=graph.senders,
                receivers=graph.receivers,
                shifts=graph.edges.shifts,
                cell=cell,
                n_edge=graph.n_edge,
            )
            node_energies = model_fn(
                w, vectors, graph.nodes.species, graph.senders, graph.receivers
            )  # [n_nodes, ]
            return jnp.sum(node_energies), node_energies

        (minus_forces, pseudo_stress), node_energies = jax.grad(
            energy_fn, (0, 1), has_aux=True
        )(graph.nodes.positions, graph.globals.cell)

        graph_energies = e3nn.scatter_sum(
            node_energies, nel=graph.n_node
        )  # [ n_graphs,]

        det = jnp.linalg.det(graph.globals.cell)[:, None, None]  # [n_graphs, 1, 1]
        det = jnp.where(det > 0.0, det, 1.0)  # dummy graphs have det = 0

        stress_cell = (
            jnp.transpose(pseudo_stress, (0, 2, 1)) @ graph.globals.cell
        )  # [n_graphs, 3, 3]
        stress_forces = e3nn.scatter_sum(
            jnp.einsum("iu,iv->iuv", minus_forces, graph.nodes.positions),
            nel=graph.n_node,
        )  # [n_graphs, 3, 3]
        stress = -1.0 / det * (stress_cell + stress_forces)

        return {
            "energy": graph_energies,  # [n_graphs,] energy per cell [eV]
            "forces": -minus_forces,  # [n_nodes, 3] forces on each atom [eV / A]
            "stress": stress,  # [n_graphs, 3, 3] stress tensor [eV / A^3]
        }

    if checks(predictor, params, train_loader):
        return

    gradient_transform, max_num_epochs = optimizer(
        steps_per_epoch=train_loader.approx_length()
    )
    optimizer_state = gradient_transform.init(params)

    logging.info(f"Number of parameters: {tools.count_parameters(params)}")
    logging.info(
        f"Number of parameters in optimizer: {tools.count_parameters(optimizer_state)}"
    )

    epoch, params = train(
        predictor,
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
    )


if __name__ == "__main__":
    parse_argv(sys.argv)
    main()
