from typing import Dict

import e3nn_jax as e3nn
import jax
import jax.numpy as jnp
import jraph

from mace_jax import tools


def energy_forces_stress_predictor(model):
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
            node_energies = model(
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
        viriel = stress_cell + stress_forces  # NOTE: sign suggested by Ilyes Batatia
        stress = -1.0 / det * viriel  # NOTE: sign suggested by Ilyes Batatia

        # TODO(mario): fix this
        # make it traceless? because it seems that our formula is not valid for the trace
        # p = jnp.trace(stress, axis1=1, axis2=2)  # [n_graphs,]
        # stress = stress - p[:, None, None] / 3.0 * jnp.eye(3)

        return {
            "energy": graph_energies,  # [n_graphs,] energy per cell [eV]
            "forces": -minus_forces,  # [n_nodes, 3] forces on each atom [eV / A]
            "stress": stress,  # [n_graphs, 3, 3] stress tensor [eV / A^3]
            # "pressure": p,  # [n_graphs,] pressure [eV / A^3]
        }

    return predictor
