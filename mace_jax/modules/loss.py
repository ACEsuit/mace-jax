from typing import Dict

import jax.numpy as jnp
import jraph
from jax import jit, vmap


def weighted_mean_squared_error_energy(
    configs_weight, num_atoms, energy_ref, energy_pred
) -> jnp.ndarray:
    # energy: [n_graphs, ]
    return configs_weight * jnp.square(
        (energy_ref - energy_pred) / num_atoms
    )  # [n_graphs, ]


def mean_squared_error_forces(configs_weight, forces_ref, forces_pred) -> jnp.ndarray:
    # forces: [n_atoms, 3]
    return jnp.mean(
        configs_weight * jnp.square(forces_ref - forces_pred), axis=1
    )  # [n_graphs, ]


class WeightedEnergyForcesLoss:
    def __init__(self, energy_weight=1.0, forces_weight=1.0) -> None:
        super().__init__()
        self.energy_weight = energy_weight
        self.forces_weight = forces_weight

    def __call__(
        self, graph: jraph.GraphsTuple, pred: Dict[str, jnp.ndarray]
    ) -> jnp.ndarray:
        configs_weight = graph.globals.weight  # [n_graphs, ]
        configs_weight_forces = jnp.repeat(
            graph.globals.weight, graph.globals.ptr[1:] - graph.globals.ptr[:-1]
        ).unsqueeze(
            -1
        )  # [n_atoms, 1]
        num_atoms = graph.ptr[1:] - graph.ptr[:-1]  # [n_graphs,]

        return self.energy_weight * weighted_mean_squared_error_energy(
            configs_weight, num_atoms, graph.globals.energy, pred["energy"]
        ) + self.forces_weight * mean_squared_error_forces(
            configs_weight_forces, graph.globals.forces, pred["forces"]
        )  # [n_graphs,]

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(energy_weight={self.energy_weight:.3f}, "
            f"forces_weight={self.forces_weight:.3f})"
        )
