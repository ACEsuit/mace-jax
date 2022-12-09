import jax.numpy as jnp
import jraph

from ..tools import sum_nodes_of_the_same_graph


def _safe_divide(x, y):
    return jnp.where(y == 0.0, 0.0, x / jnp.where(y == 0.0, 1.0, y))


def mean_squared_error_energy(graph, energy_pred) -> jnp.ndarray:
    energy_ref = graph.globals.energy  # [n_graphs, ]
    return graph.globals.weight * jnp.square(
        _safe_divide(energy_ref - energy_pred, graph.n_node)
    )  # [n_graphs, ]


def mean_squared_error_forces(graph, forces_pred) -> jnp.ndarray:
    forces_ref = graph.nodes.forces  # [n_nodes, 3]
    return graph.globals.weight * _safe_divide(
        sum_nodes_of_the_same_graph(
            graph, jnp.mean(jnp.square(forces_ref - forces_pred), axis=1)
        ),
        graph.n_node,
    )  # [n_graphs, ]


def mean_squared_error_stress(graph, stress_pred) -> jnp.ndarray:
    stress_ref = graph.globals.stress  # [n_graphs, 3, 3]
    return graph.globals.weight * jnp.mean(
        jnp.square(stress_ref - stress_pred), axis=(1, 2)
    )  # [n_graphs, ]


class WeightedEnergyFrocesStressLoss:
    def __init__(self, energy_weight=1.0, forces_weight=1.0, stress_weight=1.0) -> None:
        super().__init__()
        self.energy_weight = energy_weight
        self.forces_weight = forces_weight
        self.stress_weight = stress_weight

    def __call__(self, graph: jraph.GraphsTuple, predictions) -> jnp.ndarray:
        loss = 0

        if self.energy_weight > 0.0:
            energy = predictions["energy"]
            loss += self.energy_weight * mean_squared_error_energy(graph, energy)

        if self.forces_weight > 0.0:
            forces = predictions["forces"]
            loss += self.forces_weight * mean_squared_error_forces(graph, forces)

        if self.stress_weight > 0.0:
            stress = predictions["stress"]
            loss += self.stress_weight * mean_squared_error_stress(graph, stress)

        return loss  # [n_graphs, ]

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(energy_weight={self.energy_weight:.3f}, "
            f"forces_weight={self.forces_weight:.3f}, "
            f"stress_weight={self.stress_weight:.3f})"
        )
