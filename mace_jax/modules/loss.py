import jax.numpy as jnp
from jax import vmap
from jax import jit
import haiku as hk


@jit
def weighted_mean_squared_error_energy(
    configs_weight, num_atoms, energy_ref, energy_pred
) -> jnp.ndarray:
    # energy: [n_graphs, ]
    return jnp.mean(
        configs_weight * jnp.square((energy_ref - energy_pred) / num_atoms)
    )  # []


@jit
def mean_squared_error_forces(configs_weight, forces_ref, forces_pred) -> jnp.ndarray:
    # forces: [n_atoms, 3]
    return jnp.mean(configs_weight * jnp.square(forces_ref - forces_pred))  # []


class WeightedEnergyForcesLoss(hk.Module):
    def __init__(self, energy_weight=1.0, forces_weight=1.0) -> None:
        super().__init__()
        self.energy_weight = energy_weight
        self.forces_weight = forces_weight

    def forward(self, graph, model, params, ref) -> jnp.ndarray:
        pred = model.apply(graph, params)
        configs_weight = ref.weight  # [n_graphs, ]
        configs_weight_forces = jnp.repeat(
            ref.weight, ref.ptr[1:] - ref.ptr[:-1]
        ).unsqueeze(
            -1
        )  # [n_atoms, 1]
        num_atoms = ref.ptr[1:] - ref.ptr[:-1]  # [n_graphs,]
        energy_loss_fn = vmap(weighted_mean_squared_error_energy, (None, None, 0, 0))
        forces_loss_fn = vmap(weighted_mean_squared_error_energy, (None, 0, 0))
        return self.energy_weight * energy_loss_fn(
            configs_weight, num_atoms, ref.energy, pred.energy
        ) + self.forces_weight * forces_loss_fn(
            configs_weight_forces, ref.forces, pred.forces
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(energy_weight={self.energy_weight:.3f}, "
            f"forces_weight={self.forces_weight:.3f})"
        )
