from typing import Dict

import e3nn_jax as e3nn
import jax
import jax.numpy as jnp
import jraph

from mace_jax import tools


def predict_energy_forces_stress(
    model,  # model(relative_vectors, species, senders, receivers) -> [num_nodes]
    graph: jraph.GraphsTuple,
) -> Dict[str, jnp.ndarray]:
    """Predict energy, forces and stress tensor for a batch of graphs.

    Args:
        model: a model that takes as input the relative vectors, the species of the nodes,
            sender and receiver nodes and returns the energy for each node.
        graph: a batch of graphs.

    Returns:
        A dictionary with the following keys
            energy: [n_graphs,] energy per cell [eV]
            forces: [n_nodes, 3] forces on each atom [eV / A]
            stress: [n_graphs, 3, 3] stress tensor [eV / A^3]
            stress_cell: [n_graphs, 3, 3] stress tensor [eV / A^3]
            stress_forces: [n_graphs, 3, 3] stress tensor [eV / A^3]
            pressure: [n_graphs,] pressure [eV / A^3]
    """

    def energy_fn(positions, cell):
        vectors = tools.get_edge_relative_vectors(
            positions=positions,
            senders=graph.senders,
            receivers=graph.receivers,
            shifts=graph.edges.shifts,
            cell=cell,
            n_edge=graph.n_edge,
        )
        node_energies = model(
            vectors, graph.nodes.species, graph.senders, graph.receivers
        )  # [n_nodes, ]
        assert node_energies.shape == (
            len(positions),
        ), "model output needs to be an array of shape (n_nodes, )"
        return jnp.sum(node_energies), node_energies

    (minus_forces, pseudo_stress), node_energies = jax.grad(
        energy_fn, (0, 1), has_aux=True
    )(graph.nodes.positions, graph.globals.cell)

    graph_energies = e3nn.scatter_sum(node_energies, nel=graph.n_node)  # [ n_graphs,]

    det = jnp.linalg.det(graph.globals.cell)[:, None, None]  # [n_graphs, 1, 1]
    det = jnp.where(det > 0.0, det, 1.0)  # dummy graphs have det = 0

    stress_cell = (
        jnp.transpose(pseudo_stress, (0, 2, 1)) @ graph.globals.cell
    )  # [n_graphs, 3, 3]
    stress_forces = e3nn.scatter_sum(
        jnp.einsum("iu,iv->iuv", minus_forces, graph.nodes.positions),
        nel=graph.n_node,
    )  # [n_graphs, 3, 3]
    virial = stress_cell + stress_forces
    stress = virial / det

    # TODO(mario): fix this
    # make it traceless? because it seems that our formula is not valid for the trace
    pressure = jnp.trace(stress, axis1=1, axis2=2)  # [n_graphs,]
    # stress = stress - p[:, None, None] / 3.0 * jnp.eye(3)

    return {
        "energy": graph_energies,  # [n_graphs,] energy per cell [eV]
        "forces": -minus_forces,  # [n_nodes, 3] forces on each atom [eV / A]
        "stress": stress,  # [n_graphs, 3, 3] stress tensor [eV / A^3]
        "stress_cell": (
            -1.0 / det * stress_cell
        ),  # [n_graphs, 3, 3] stress tensor [eV / A^3]
        "stress_forces": (
            -1.0 / det * stress_forces
        ),  # [n_graphs, 3, 3] stress tensor [eV / A^3]
        "pressure": pressure,  # [n_graphs,] pressure [eV / A^3]
    }
