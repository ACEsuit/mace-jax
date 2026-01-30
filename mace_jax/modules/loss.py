from __future__ import annotations

import gin
import jax.numpy as jnp
import jraph

from ..tools import sum_nodes_of_the_same_graph


def _safe_divide(x, y):
    return jnp.where(y == 0.0, 0.0, x / jnp.where(y == 0.0, 1.0, y))


def _graph_mask(graph: jraph.GraphsTuple) -> jnp.ndarray:
    mask = jraph.get_graph_padding_mask(graph).astype(jnp.float32)
    if mask.shape == graph.n_node.shape:
        non_empty = jnp.asarray(graph.n_node > 0, dtype=jnp.float32)
        mask = mask * non_empty
    weights = getattr(graph.globals, 'weight', None)
    if weights is not None:
        weight_mask = jnp.asarray(weights > 0, dtype=jnp.float32)
        if weight_mask.shape == mask.shape:
            mask = mask * weight_mask
    return mask


def _node_mask(graph: jraph.GraphsTuple) -> jnp.ndarray:
    return jraph.get_node_padding_mask(graph).astype(jnp.float32)


def _num_atoms(graph: jraph.GraphsTuple) -> jnp.ndarray:
    return jnp.maximum(graph.n_node.astype(jnp.float32), 1.0)


def _graph_attribute(
    graph: jraph.GraphsTuple, name: str, default: float, dtype: jnp.dtype | None = None
) -> jnp.ndarray:
    value = getattr(graph.globals, name, None)
    if value is None:
        base = getattr(graph.globals, 'weight', None)
        if base is not None:
            base = jnp.asarray(base)
            return jnp.full_like(base, default, dtype=base.dtype)
        shape = graph.n_node.shape
        dtype = dtype or jnp.result_type(
            jnp.asarray(default, dtype=jnp.float32), jnp.float32
        )
        return jnp.full(shape, default, dtype=dtype)
    return jnp.asarray(value)


def _graph_tensor(
    graph: jraph.GraphsTuple,
    name: str,
    default_shape: tuple[int, ...],
    dtype: jnp.dtype,
) -> jnp.ndarray:
    value = getattr(graph.globals, name, None)
    if value is None:
        shape = (graph.n_node.shape[0],) + default_shape
        return jnp.zeros(shape, dtype=dtype)
    return jnp.asarray(value)


def _broadcast_to_nodes(
    graph: jraph.GraphsTuple, per_graph_values: jnp.ndarray
) -> jnp.ndarray:
    per_graph_values = jnp.asarray(per_graph_values)
    total_nodes = graph.nodes.positions.shape[0]
    graph_indices = jnp.repeat(
        jnp.arange(per_graph_values.shape[0], dtype=jnp.int32),
        graph.n_node,
        total_repeat_length=total_nodes,
        axis=0,
    )
    return per_graph_values[graph_indices]


def mean_squared_error_energy(
    graph: jraph.GraphsTuple, energy_pred: jnp.ndarray
) -> jnp.ndarray:
    mask = _graph_mask(graph)
    mask_bool = mask > 0.0
    energy_ref = jnp.asarray(graph.globals.energy)
    diff = energy_ref - jnp.asarray(energy_pred)
    diff = jnp.where(mask_bool, diff, 0.0)
    per_atom = _safe_divide(diff, _num_atoms(graph))
    loss = _graph_attribute(graph, 'weight', 1.0) * per_atom**2
    return loss * mask


def weighted_mean_squared_error_energy(
    graph: jraph.GraphsTuple, energy_pred: jnp.ndarray
) -> jnp.ndarray:
    energy_weight = _graph_attribute(graph, 'energy_weight', 1.0)
    loss = energy_weight * mean_squared_error_energy(graph, energy_pred)
    return loss


def weighted_mean_absolute_error_energy(
    graph: jraph.GraphsTuple, energy_pred: jnp.ndarray
) -> jnp.ndarray:
    mask = _graph_mask(graph)
    mask_bool = mask > 0.0
    energy_ref = jnp.asarray(graph.globals.energy)
    diff = jnp.abs(energy_ref - jnp.asarray(energy_pred))
    diff = jnp.where(mask_bool, diff, 0.0)
    per_atom = _safe_divide(diff, _num_atoms(graph))
    weight = _graph_attribute(graph, 'weight', 1.0)
    energy_weight = _graph_attribute(graph, 'energy_weight', 1.0, dtype=per_atom.dtype)
    loss = weight * energy_weight * per_atom
    return loss * mask


def mean_squared_error_forces(
    graph: jraph.GraphsTuple, forces_pred: jnp.ndarray
) -> jnp.ndarray:
    if not hasattr(graph.nodes, 'forces') or graph.nodes.forces is None:
        return jnp.zeros(graph.n_node.shape, dtype=jnp.asarray(forces_pred).dtype)
    forces_ref = jnp.asarray(graph.nodes.forces)
    forces_pred = jnp.asarray(forces_pred)
    node_mask = _node_mask(graph)[:, None]
    mask_bool = node_mask > 0.0
    diff = jnp.where(mask_bool, forces_ref - forces_pred, 0.0)
    diff_sq = jnp.mean(jnp.square(diff), axis=-1)

    per_graph_weight = _graph_attribute(graph, 'weight', 1.0, dtype=diff_sq.dtype)
    per_node_weight = _broadcast_to_nodes(graph, per_graph_weight)
    per_node_loss = per_node_weight * diff_sq
    per_graph = _safe_divide(
        sum_nodes_of_the_same_graph(graph, per_node_loss), _num_atoms(graph)
    )
    return per_graph * _graph_mask(graph)


def mean_normed_error_forces(
    graph: jraph.GraphsTuple, forces_pred: jnp.ndarray
) -> jnp.ndarray:
    if not hasattr(graph.nodes, 'forces') or graph.nodes.forces is None:
        return jnp.zeros(graph.n_node.shape, dtype=jnp.asarray(forces_pred).dtype)
    forces_ref = jnp.asarray(graph.nodes.forces)
    forces_pred = jnp.asarray(forces_pred)
    node_mask = _node_mask(graph)[:, None]
    mask_bool = node_mask > 0.0
    diff = jnp.where(mask_bool, forces_ref - forces_pred, 0.0)
    diff_norm = jnp.linalg.norm(diff, axis=-1)

    per_graph_weight = _graph_attribute(graph, 'weight', 1.0, dtype=diff_norm.dtype)
    per_node_weight = _broadcast_to_nodes(graph, per_graph_weight)
    per_node_loss = per_node_weight * diff_norm
    per_graph = _safe_divide(
        sum_nodes_of_the_same_graph(graph, per_node_loss), _num_atoms(graph)
    )
    return per_graph * _graph_mask(graph)


def weighted_mean_squared_stress(
    graph: jraph.GraphsTuple, stress_pred: jnp.ndarray
) -> jnp.ndarray:
    mask = _graph_mask(graph)
    mask_bool = mask[:, None, None] > 0.0
    stress_ref = _graph_tensor(
        graph, 'stress', default_shape=(3, 3), dtype=jnp.asarray(stress_pred).dtype
    )
    stress_pred = jnp.asarray(stress_pred)
    diff = stress_ref - stress_pred
    diff = jnp.where(mask_bool, diff, 0.0)
    diff_sq = jnp.mean(jnp.square(diff), axis=(1, 2))
    weight = _graph_attribute(graph, 'weight', 1.0, dtype=diff_sq.dtype)
    stress_weight = _graph_attribute(graph, 'stress_weight', 1.0, dtype=diff_sq.dtype)
    loss = weight * stress_weight * diff_sq
    return loss * mask


def weighted_mean_squared_virials(
    graph: jraph.GraphsTuple, virials_pred: jnp.ndarray
) -> jnp.ndarray:
    mask = _graph_mask(graph)
    mask_bool = mask[:, None, None] > 0.0
    virials_ref = _graph_tensor(
        graph, 'virials', default_shape=(3, 3), dtype=jnp.asarray(virials_pred).dtype
    )
    virials_pred = jnp.asarray(virials_pred)
    num_atoms = _num_atoms(graph)[:, None, None]
    diff = virials_ref - virials_pred
    diff = jnp.where(mask_bool, diff, 0.0)
    per_atom_diff = _safe_divide(diff, num_atoms)
    diff_sq = jnp.mean(jnp.square(per_atom_diff), axis=(1, 2))
    weight = _graph_attribute(graph, 'weight', 1.0, dtype=diff_sq.dtype)
    virials_weight = _graph_attribute(graph, 'virials_weight', 1.0, dtype=diff_sq.dtype)
    loss = weight * virials_weight * diff_sq
    return loss * mask


def weighted_mean_squared_error_dipole(
    graph: jraph.GraphsTuple, dipole_pred: jnp.ndarray
) -> jnp.ndarray:
    mask = _graph_mask(graph)
    mask_bool = mask[:, None] > 0.0
    dipole_ref = _graph_tensor(
        graph, 'dipole', default_shape=(3,), dtype=jnp.asarray(dipole_pred).dtype
    )
    dipole_pred = jnp.asarray(dipole_pred)
    num_atoms = _num_atoms(graph)[:, None]
    diff = dipole_ref - dipole_pred
    diff = jnp.where(mask_bool, diff, 0.0)
    per_atom_diff = _safe_divide(diff, num_atoms)
    diff_sq = jnp.mean(jnp.square(per_atom_diff), axis=-1)
    weight = _graph_attribute(graph, 'weight', 1.0, dtype=diff_sq.dtype)
    dipole_weight = _graph_attribute(graph, 'dipole_weight', 1.0, dtype=diff_sq.dtype)
    loss = weight * dipole_weight * diff_sq
    return loss * mask


def weighted_mean_squared_error_polarizability(
    graph: jraph.GraphsTuple, polarizability_pred: jnp.ndarray
) -> jnp.ndarray:
    mask = _graph_mask(graph)
    mask_bool = mask[:, None, None] > 0.0
    polar_ref = _graph_tensor(
        graph,
        'polarizability',
        default_shape=(3, 3),
        dtype=jnp.asarray(polarizability_pred).dtype,
    )
    polarizability_pred = jnp.asarray(polarizability_pred)
    num_atoms = _num_atoms(graph)[:, None, None]
    diff = polar_ref - polarizability_pred
    diff = jnp.where(mask_bool, diff, 0.0)
    per_atom_diff = _safe_divide(diff, num_atoms)
    diff_sq = jnp.mean(jnp.square(per_atom_diff), axis=(1, 2))
    weight = _graph_attribute(graph, 'weight', 1.0, dtype=diff_sq.dtype)
    polar_weight = _graph_attribute(
        graph, 'polarizability_weight', 1.0, dtype=diff_sq.dtype
    )
    loss = weight * polar_weight * diff_sq
    return loss * mask


def conditional_mse_forces(
    graph: jraph.GraphsTuple, forces_pred: jnp.ndarray
) -> jnp.ndarray:
    if not hasattr(graph.nodes, 'forces') or graph.nodes.forces is None:
        return jnp.zeros(graph.n_node.shape, dtype=jnp.asarray(forces_pred).dtype)
    forces_ref = jnp.asarray(graph.nodes.forces)
    forces_pred = jnp.asarray(forces_pred)
    node_mask = _node_mask(graph)[:, None]
    mask_bool = node_mask > 0.0
    diff = jnp.where(mask_bool, forces_ref - forces_pred, 0.0)

    norm_forces = jnp.linalg.norm(forces_ref * node_mask, axis=-1)
    factors = jnp.array([1.0, 0.7, 0.4, 0.1], dtype=forces_ref.dtype)
    factor_per_node = jnp.select(
        [
            norm_forces < 100.0,
            (norm_forces >= 100.0) & (norm_forces < 200.0),
            (norm_forces >= 200.0) & (norm_forces < 300.0),
        ],
        factors[:3],
        default=factors[3],
    )

    diff_sq = jnp.mean(jnp.square(diff), axis=-1)

    per_graph_weight = _graph_attribute(graph, 'weight', 1.0, dtype=diff_sq.dtype)
    forces_weight = _graph_attribute(graph, 'forces_weight', 1.0, dtype=diff_sq.dtype)
    per_node_weight = _broadcast_to_nodes(graph, per_graph_weight * forces_weight)

    per_node_loss = per_node_weight * factor_per_node * diff_sq
    per_graph = _safe_divide(
        sum_nodes_of_the_same_graph(graph, per_node_loss), _num_atoms(graph)
    )
    return per_graph * _graph_mask(graph)


def conditional_huber_forces(
    graph: jraph.GraphsTuple,
    forces_pred: jnp.ndarray,
    huber_delta: float,
) -> jnp.ndarray:
    if not hasattr(graph.nodes, 'forces') or graph.nodes.forces is None:
        return jnp.zeros(graph.n_node.shape, dtype=jnp.asarray(forces_pred).dtype)
    forces_ref = jnp.asarray(graph.nodes.forces)
    forces_pred = jnp.asarray(forces_pred)
    node_mask = _node_mask(graph)[:, None]
    mask_bool = node_mask > 0.0
    diff = jnp.where(mask_bool, forces_ref - forces_pred, 0.0)

    norm_forces = jnp.linalg.norm(forces_ref * node_mask, axis=-1)
    factors = jnp.array([1.0, 0.7, 0.4, 0.1], dtype=forces_ref.dtype) * huber_delta
    delta_per_node = jnp.select(
        [
            norm_forces < 100.0,
            (norm_forces >= 100.0) & (norm_forces < 200.0),
            (norm_forces >= 200.0) & (norm_forces < 300.0),
        ],
        factors[:3],
        default=factors[3],
    )
    delta_per_node = jnp.maximum(delta_per_node, 1e-12)
    delta_expanded = delta_per_node[:, None]
    loss_components = jnp.where(
        jnp.abs(diff) <= delta_expanded,
        0.5 * diff**2 / delta_expanded,
        delta_expanded * (jnp.abs(diff) - 0.5 * delta_expanded),
    )
    loss_components *= node_mask

    per_component_loss = jnp.mean(loss_components, axis=-1)
    per_graph_weight = _graph_attribute(
        graph, 'weight', 1.0, dtype=per_component_loss.dtype
    )
    forces_weight = _graph_attribute(
        graph, 'forces_weight', 1.0, dtype=per_component_loss.dtype
    )
    per_node_weight = _broadcast_to_nodes(graph, per_graph_weight * forces_weight)
    per_node_loss = per_node_weight * per_component_loss
    per_graph = _safe_divide(
        sum_nodes_of_the_same_graph(graph, per_node_loss), _num_atoms(graph)
    )
    return per_graph * _graph_mask(graph)


@gin.register
class WeightedEnergyForcesLoss:
    def __init__(self, energy_weight=1.0, forces_weight=1.0) -> None:
        self.energy_weight = energy_weight
        self.forces_weight = forces_weight

    def __call__(self, graph: jraph.GraphsTuple, predictions) -> jnp.ndarray:
        loss = jnp.zeros(graph.n_node.shape, dtype=jnp.float32)

        if self.energy_weight != 0.0 and 'energy' in predictions:
            loss += self.energy_weight * weighted_mean_squared_error_energy(
                graph, predictions['energy']
            )

        if self.forces_weight != 0.0 and 'forces' in predictions:
            loss += self.forces_weight * mean_squared_error_forces(
                graph, predictions['forces']
            )

        return loss

    def __repr__(self):
        return (
            f'{self.__class__.__name__}(energy_weight={self.energy_weight:.3f}, '
            f'forces_weight={self.forces_weight:.3f})'
        )


@gin.register
class WeightedForcesLoss:
    def __init__(self, forces_weight=1.0) -> None:
        self.forces_weight = forces_weight

    def __call__(self, graph: jraph.GraphsTuple, predictions) -> jnp.ndarray:
        if self.forces_weight == 0.0 or 'forces' not in predictions:
            return jnp.zeros(graph.n_node.shape, dtype=jnp.float32)
        return self.forces_weight * mean_squared_error_forces(
            graph, predictions['forces']
        )

    def __repr__(self):
        return f'{self.__class__.__name__}(forces_weight={self.forces_weight:.3f})'


@gin.register
class WeightedEnergyForcesStressLoss:
    def __init__(self, energy_weight=1.0, forces_weight=1.0, stress_weight=1.0) -> None:
        self.energy_weight = energy_weight
        self.forces_weight = forces_weight
        self.stress_weight = stress_weight

    def __call__(self, graph: jraph.GraphsTuple, predictions) -> jnp.ndarray:
        loss = jnp.zeros(graph.n_node.shape, dtype=jnp.float32)

        if self.energy_weight != 0.0 and 'energy' in predictions:
            loss += self.energy_weight * weighted_mean_squared_error_energy(
                graph, predictions['energy']
            )

        if self.forces_weight != 0.0 and 'forces' in predictions:
            loss += self.forces_weight * mean_squared_error_forces(
                graph, predictions['forces']
            )

        if self.stress_weight != 0.0 and 'stress' in predictions:
            loss += self.stress_weight * weighted_mean_squared_stress(
                graph, predictions['stress']
            )

        return loss

    def __repr__(self):
        return (
            f'{self.__class__.__name__}(energy_weight={self.energy_weight:.3f}, '
            f'forces_weight={self.forces_weight:.3f}, '
            f'stress_weight={self.stress_weight:.3f})'
        )


@gin.register
class WeightedHuberEnergyForcesStressLoss:
    def __init__(
        self, energy_weight=1.0, forces_weight=1.0, stress_weight=1.0, huber_delta=0.01
    ) -> None:
        self.energy_weight = energy_weight
        self.forces_weight = forces_weight
        self.stress_weight = stress_weight
        self.huber_delta = huber_delta

    def __call__(self, graph: jraph.GraphsTuple, predictions) -> jnp.ndarray:
        loss = jnp.zeros(graph.n_node.shape, dtype=jnp.float32)
        num_atoms = _num_atoms(graph)

        if self.energy_weight != 0.0 and 'energy' in predictions:
            energy_ref = jnp.asarray(graph.globals.energy)
            energy_pred = jnp.asarray(predictions['energy'])
            diff = _safe_divide(energy_ref - energy_pred, num_atoms)
            huber = jnp.where(
                jnp.abs(diff) <= self.huber_delta,
                0.5 * diff**2 / self.huber_delta,
                self.huber_delta * (jnp.abs(diff) - 0.5 * self.huber_delta),
            )
            weight = _graph_attribute(graph, 'weight', 1.0, dtype=huber.dtype)
            energy_weight = _graph_attribute(
                graph, 'energy_weight', 1.0, dtype=huber.dtype
            )
            loss += (
                self.energy_weight * weight * energy_weight * huber * _graph_mask(graph)
            )

        if self.forces_weight != 0.0 and 'forces' in predictions:
            loss += self.forces_weight * conditional_huber_forces(
                graph, predictions['forces'], self.huber_delta
            )

        if self.stress_weight != 0.0 and 'stress' in predictions:
            stress_ref = _graph_tensor(
                graph, 'stress', default_shape=(3, 3), dtype=jnp.float32
            )
            stress_pred = jnp.asarray(predictions['stress'])
            diff = stress_ref - stress_pred
            delta = jnp.asarray(self.huber_delta, dtype=diff.dtype)
            loss_stress = jnp.where(
                jnp.abs(diff) <= delta,
                0.5 * diff**2 / delta,
                delta * (jnp.abs(diff) - 0.5 * delta),
            )
            loss_stress = jnp.mean(loss_stress, axis=(1, 2))
            weight = _graph_attribute(graph, 'weight', 1.0, dtype=loss_stress.dtype)
            stress_weight = _graph_attribute(
                graph, 'stress_weight', 1.0, dtype=loss_stress.dtype
            )
            loss += (
                self.stress_weight
                * weight
                * stress_weight
                * loss_stress
                * _graph_mask(graph)
            )

        return loss

    def __repr__(self):
        return (
            f'{self.__class__.__name__}(energy_weight={self.energy_weight:.3f}, '
            f'forces_weight={self.forces_weight:.3f}, '
            f'stress_weight={self.stress_weight:.3f})'
        )


@gin.register
class UniversalLoss:
    def __init__(
        self, energy_weight=1.0, forces_weight=1.0, stress_weight=1.0, huber_delta=0.01
    ) -> None:
        self.energy_weight = energy_weight
        self.forces_weight = forces_weight
        self.stress_weight = stress_weight
        self.huber_delta = huber_delta

    def __call__(self, graph: jraph.GraphsTuple, predictions) -> jnp.ndarray:
        loss = jnp.zeros(graph.n_node.shape, dtype=jnp.float32)
        num_atoms = _num_atoms(graph)

        if self.energy_weight != 0.0 and 'energy' in predictions:
            energy_ref = jnp.asarray(graph.globals.energy)
            energy_pred = jnp.asarray(predictions['energy'])
            energy_weight = _graph_attribute(
                graph, 'energy_weight', 1.0, dtype=energy_ref.dtype
            )
            diff = energy_weight * _safe_divide(energy_ref - energy_pred, num_atoms)
            huber = jnp.where(
                jnp.abs(diff) <= self.huber_delta,
                0.5 * diff**2 / self.huber_delta,
                self.huber_delta * (jnp.abs(diff) - 0.5 * self.huber_delta),
            )
            loss += (
                self.energy_weight
                * _graph_attribute(graph, 'weight', 1.0, dtype=huber.dtype)
                * huber
                * _graph_mask(graph)
            )

        if self.forces_weight != 0.0 and 'forces' in predictions:
            forces_weight = _graph_attribute(
                graph, 'forces_weight', 1.0, dtype=jnp.float32
            )
            per_node_weight = _broadcast_to_nodes(
                graph,
                _graph_attribute(graph, 'weight', 1.0, dtype=jnp.float32)
                * forces_weight,
            )
            scaled_pred = predictions['forces'] * per_node_weight[:, None]
            loss += self.forces_weight * conditional_huber_forces(
                graph, scaled_pred, self.huber_delta
            )

        if self.stress_weight != 0.0 and 'stress' in predictions:
            stress_ref = _graph_tensor(
                graph, 'stress', default_shape=(3, 3), dtype=jnp.float32
            )
            stress_pred = jnp.asarray(predictions['stress'])
            stress_weight = _graph_attribute(
                graph, 'stress_weight', 1.0, dtype=stress_ref.dtype
            )
            diff = stress_weight[:, None, None] * (stress_ref - stress_pred)
            huber = jnp.where(
                jnp.abs(diff) <= self.huber_delta,
                0.5 * diff**2 / self.huber_delta,
                self.huber_delta * (jnp.abs(diff) - 0.5 * self.huber_delta),
            )
            huber = jnp.mean(huber, axis=(1, 2))
            loss += (
                self.stress_weight
                * _graph_attribute(graph, 'weight', 1.0, dtype=huber.dtype)
                * huber
                * _graph_mask(graph)
            )

        return loss

    def __repr__(self):
        return (
            f'{self.__class__.__name__}(energy_weight={self.energy_weight:.3f}, '
            f'forces_weight={self.forces_weight:.3f}, '
            f'stress_weight={self.stress_weight:.3f})'
        )


@gin.register
class WeightedEnergyForcesVirialsLoss:
    def __init__(
        self, energy_weight=1.0, forces_weight=1.0, virials_weight=1.0
    ) -> None:
        self.energy_weight = energy_weight
        self.forces_weight = forces_weight
        self.virials_weight = virials_weight

    def __call__(self, graph: jraph.GraphsTuple, predictions) -> jnp.ndarray:
        loss = jnp.zeros(graph.n_node.shape, dtype=jnp.float32)

        if self.energy_weight != 0.0 and 'energy' in predictions:
            loss += self.energy_weight * weighted_mean_squared_error_energy(
                graph, predictions['energy']
            )

        if self.forces_weight != 0.0 and 'forces' in predictions:
            loss += self.forces_weight * mean_squared_error_forces(
                graph, predictions['forces']
            )

        if self.virials_weight != 0.0 and 'virials' in predictions:
            loss += self.virials_weight * weighted_mean_squared_virials(
                graph, predictions['virials']
            )

        return loss

    def __repr__(self):
        return (
            f'{self.__class__.__name__}(energy_weight={self.energy_weight:.3f}, '
            f'forces_weight={self.forces_weight:.3f}, '
            f'virials_weight={self.virials_weight:.3f})'
        )


@gin.register
class DipoleSingleLoss:
    def __init__(self, dipole_weight=1.0) -> None:
        self.dipole_weight = dipole_weight

    def __call__(self, graph: jraph.GraphsTuple, predictions) -> jnp.ndarray:
        if self.dipole_weight == 0.0 or 'dipole' not in predictions:
            return jnp.zeros(graph.n_node.shape, dtype=jnp.float32)
        return self.dipole_weight * weighted_mean_squared_error_dipole(
            graph, predictions['dipole']
        )

    def __repr__(self):
        return f'{self.__class__.__name__}(dipole_weight={self.dipole_weight:.3f})'


@gin.register
class DipolePolarLoss:
    def __init__(self, dipole_weight=1.0, polarizability_weight=1.0) -> None:
        self.dipole_weight = dipole_weight
        self.polarizability_weight = polarizability_weight

    def __call__(self, graph: jraph.GraphsTuple, predictions) -> jnp.ndarray:
        loss = jnp.zeros(graph.n_node.shape, dtype=jnp.float32)

        if self.dipole_weight != 0.0 and 'dipole' in predictions:
            loss += self.dipole_weight * weighted_mean_squared_error_dipole(
                graph, predictions['dipole']
            )

        if self.polarizability_weight != 0.0 and 'polarizability' in predictions:
            loss += (
                self.polarizability_weight
                * weighted_mean_squared_error_polarizability(
                    graph, predictions['polarizability']
                )
            )

        return loss

    def __repr__(self):
        return (
            f'{self.__class__.__name__}(dipole_weight={self.dipole_weight:.3f}, '
            f'polarizability_weight={self.polarizability_weight:.3f})'
        )


@gin.register
class WeightedEnergyForcesDipoleLoss:
    def __init__(self, energy_weight=1.0, forces_weight=1.0, dipole_weight=1.0) -> None:
        self.energy_weight = energy_weight
        self.forces_weight = forces_weight
        self.dipole_weight = dipole_weight

    def __call__(self, graph: jraph.GraphsTuple, predictions) -> jnp.ndarray:
        loss = jnp.zeros(graph.n_node.shape, dtype=jnp.float32)

        if self.energy_weight != 0.0 and 'energy' in predictions:
            loss += self.energy_weight * weighted_mean_squared_error_energy(
                graph, predictions['energy']
            )

        if self.forces_weight != 0.0 and 'forces' in predictions:
            loss += self.forces_weight * mean_squared_error_forces(
                graph, predictions['forces']
            )

        if self.dipole_weight != 0.0 and 'dipole' in predictions:
            loss += self.dipole_weight * weighted_mean_squared_error_dipole(
                graph, predictions['dipole']
            )

        return loss

    def __repr__(self):
        return (
            f'{self.__class__.__name__}(energy_weight={self.energy_weight:.3f}, '
            f'forces_weight={self.forces_weight:.3f}, '
            f'dipole_weight={self.dipole_weight:.3f})'
        )


@gin.register
class WeightedEnergyForcesL1L2Loss:
    def __init__(self, energy_weight=1.0, forces_weight=1.0) -> None:
        self.energy_weight = energy_weight
        self.forces_weight = forces_weight

    def __call__(self, graph: jraph.GraphsTuple, predictions) -> jnp.ndarray:
        loss = jnp.zeros(graph.n_node.shape, dtype=jnp.float32)

        if self.energy_weight != 0.0 and 'energy' in predictions:
            loss += self.energy_weight * weighted_mean_absolute_error_energy(
                graph, predictions['energy']
            )

        if self.forces_weight != 0.0 and 'forces' in predictions:
            loss += self.forces_weight * mean_normed_error_forces(
                graph, predictions['forces']
            )

        return loss

    def __repr__(self):
        return (
            f'{self.__class__.__name__}(energy_weight={self.energy_weight:.3f}, '
            f'forces_weight={self.forces_weight:.3f})'
        )


def uber_loss(x, t=1.0):
    x_center = jnp.where(jnp.abs(x) <= 1.01 * t, x, 0.0)
    center = x_center**2 / (2 * t)
    sides = jnp.abs(x) - t / 2
    return jnp.where(jnp.abs(x) <= t, center, sides)
