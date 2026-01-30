import jax.numpy as jnp
import jraph
import numpy as np

from mace_jax.data import GraphEdges, GraphGlobals, GraphNodes
from mace_jax.modules import loss as loss_mod
from mace_jax.tools import sum_nodes_of_the_same_graph


def build_test_graph():
    graph1 = jraph.GraphsTuple(
        nodes=GraphNodes(
            positions=jnp.zeros((2, 3), dtype=jnp.float32),
            forces=jnp.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=jnp.float32),
            species=jnp.array([1, 6], dtype=jnp.int32),
        ),
        edges=GraphEdges(
            shifts=jnp.zeros((0, 3), dtype=jnp.float32),
            unit_shifts=jnp.zeros((0, 3), dtype=jnp.float32),
        ),
        globals=GraphGlobals(
            cell=jnp.zeros((1, 3, 3), dtype=jnp.float32),
            energy=jnp.array([3.0], dtype=jnp.float32),
            stress=jnp.zeros((1, 3, 3), dtype=jnp.float32),
            weight=jnp.array([1.0], dtype=jnp.float32),
        ),
        receivers=jnp.zeros((0,), dtype=jnp.int32),
        senders=jnp.zeros((0,), dtype=jnp.int32),
        n_node=jnp.array([2], dtype=jnp.int32),
        n_edge=jnp.array([0], dtype=jnp.int32),
    )

    graph2 = jraph.GraphsTuple(
        nodes=GraphNodes(
            positions=jnp.zeros((1, 3), dtype=jnp.float32),
            forces=jnp.array([[0.0, 0.0, 1.0]], dtype=jnp.float32),
            species=jnp.array([8], dtype=jnp.int32),
        ),
        edges=GraphEdges(
            shifts=jnp.zeros((0, 3), dtype=jnp.float32),
            unit_shifts=jnp.zeros((0, 3), dtype=jnp.float32),
        ),
        globals=GraphGlobals(
            cell=jnp.zeros((1, 3, 3), dtype=jnp.float32),
            energy=jnp.array([-1.0], dtype=jnp.float32),
            stress=jnp.zeros((1, 3, 3), dtype=jnp.float32),
            weight=jnp.array([2.0], dtype=jnp.float32),
        ),
        receivers=jnp.zeros((0,), dtype=jnp.int32),
        senders=jnp.zeros((0,), dtype=jnp.int32),
        n_node=jnp.array([1], dtype=jnp.int32),
        n_edge=jnp.array([0], dtype=jnp.int32),
    )

    batched = jraph.batch([graph1, graph2])
    return jraph.pad_with_graphs(batched, n_node=4, n_edge=0, n_graph=3)


def build_predictions():
    return {
        'energy': jnp.array([2.5, -0.4, 0.0], dtype=jnp.float32),
        'forces': jnp.array(
            [
                [0.6, 0.1, 0.0],
                [-0.1, 2.2, 0.0],
                [0.0, 0.0, 0.2],
                [0.0, 0.0, 0.0],
            ],
            dtype=jnp.float32,
        ),
        'stress': jnp.array(
            [
                [[0.1, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.2, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                jnp.zeros((3, 3), dtype=jnp.float32),
            ],
            dtype=jnp.float32,
        ),
    }


def test_weighted_energy_and_forces_losses():
    graph = build_test_graph()
    preds = build_predictions()

    energy_loss = loss_mod.weighted_mean_squared_error_energy(graph, preds['energy'])
    np.testing.assert_allclose(
        np.asarray(energy_loss),
        np.array([0.0625, 0.72, 0.0], dtype=np.float32),
        rtol=1e-6,
        atol=1e-6,
    )

    forces_loss = loss_mod.mean_squared_error_forces(graph, preds['forces'])
    np.testing.assert_allclose(
        np.asarray(forces_loss),
        np.array([0.03666667, 0.42666668, 0.0], dtype=np.float32),
        rtol=1e-6,
        atol=1e-6,
    )

    loss_fn = loss_mod.WeightedEnergyForcesLoss(energy_weight=2.0, forces_weight=0.5)
    combined = loss_fn(graph, preds)
    np.testing.assert_allclose(
        np.asarray(combined),
        np.array([0.14333335, 1.6533334, 0.0], dtype=np.float32),
        rtol=1e-6,
        atol=1e-6,
    )


def test_weighted_energy_forces_stress_loss():
    graph = build_test_graph()
    preds = build_predictions()

    stress_loss = loss_mod.weighted_mean_squared_stress(graph, preds['stress'])
    np.testing.assert_allclose(
        np.asarray(stress_loss),
        np.array([0.00111111, 0.00888889, 0.0], dtype=np.float32),
        rtol=1e-6,
        atol=1e-6,
    )

    loss_fn = loss_mod.WeightedEnergyForcesStressLoss(
        energy_weight=2.0, forces_weight=0.5, stress_weight=3.0
    )
    combined = loss_fn(graph, preds)
    np.testing.assert_allclose(
        np.asarray(combined),
        np.array([0.14666666, 1.6800001, 0.0], dtype=np.float32),
        rtol=1e-6,
        atol=1e-6,
    )


def test_weighted_energy_forces_l1_l2_loss():
    graph = build_test_graph()
    preds = build_predictions()

    loss_fn = loss_mod.WeightedEnergyForcesL1L2Loss(
        energy_weight=1.5, forces_weight=0.25
    )
    combined = loss_fn(graph, preds)
    np.testing.assert_allclose(
        np.asarray(combined),
        np.array([0.4544895, 2.2, 0.0], dtype=np.float32),
        rtol=1e-6,
        atol=1e-6,
    )


def test_conditional_losses_match_baseline_for_small_forces():
    graph = build_test_graph()
    preds = build_predictions()

    mse = loss_mod.mean_squared_error_forces(graph, preds['forces'])
    conditional_mse = loss_mod.conditional_mse_forces(graph, preds['forces'])
    conditional_huber = loss_mod.conditional_huber_forces(
        graph, preds['forces'], huber_delta=0.5
    )

    node_mask = jraph.get_node_padding_mask(graph).astype(jnp.float32)
    diff = (graph.nodes.forces - preds['forces']) * node_mask[:, None]
    delta_per_node = jnp.full(diff.shape[0], 0.5, dtype=diff.dtype)
    delta_expanded = delta_per_node[:, None]
    loss_components = jnp.where(
        jnp.abs(diff) <= delta_expanded,
        0.5 * diff**2 / delta_expanded,
        delta_expanded * (jnp.abs(diff) - 0.5 * delta_expanded),
    )
    loss_components *= node_mask[:, None]

    per_node_loss = jnp.mean(loss_components, axis=-1)
    per_graph_weight = graph.globals.weight
    per_node_weight = jnp.repeat(
        per_graph_weight, graph.n_node, total_repeat_length=per_node_loss.shape[0]
    )
    per_node_loss *= per_node_weight
    expected_huber = sum_nodes_of_the_same_graph(graph, per_node_loss)
    expected_huber = expected_huber / jnp.maximum(graph.n_node.astype(jnp.float32), 1.0)

    np.testing.assert_allclose(
        np.asarray(conditional_mse), np.asarray(mse), rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(
        np.asarray(conditional_huber), np.asarray(expected_huber), rtol=1e-6, atol=1e-6
    )
