import jax.numpy as jnp
import jraph

from ..tools import sum_nodes_of_the_same_graph


def weighted_mean_squared_error_energy(graph, energy_pred) -> jnp.ndarray:
    energy_ref = graph.globals.energy  # [n_graphs, ]
    return graph.globals.weight * jnp.square(
        (energy_ref - energy_pred) / graph.n_node
    )  # [n_graphs, ]


def mean_squared_error_forces(graph, forces_pred) -> jnp.ndarray:
    forces_ref = graph.nodes.forces  # [n_nodes, 3]
    return (
        graph.globals.weight
        * sum_nodes_of_the_same_graph(
            graph, jnp.mean(jnp.square(forces_ref - forces_pred), axis=1)
        )
        / graph.n_node
    )  # [n_graphs, ]


class WeightedEnergyForcesLoss:
    def __init__(self, energy_weight=1.0, forces_weight=1.0) -> None:
        super().__init__()
        self.energy_weight = energy_weight
        self.forces_weight = forces_weight

    def __call__(self, graph: jraph.GraphsTuple, energy, forces) -> jnp.ndarray:
        loss_energy = weighted_mean_squared_error_energy(graph, energy)
        loss_forces = mean_squared_error_forces(graph, forces)
        return (
            self.energy_weight * loss_energy + self.forces_weight * loss_forces
        )  # [n_graphs, ]

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(energy_weight={self.energy_weight:.3f}, "
            f"forces_weight={self.forces_weight:.3f})"
        )
