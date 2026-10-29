import jax
import jax.numpy as jnp
import jraph
import numpy as np
import pytest
import torch
from mace.modules.loss import conditional_huber_forces as torch_conditional_huber

from mace_jax.data import GraphEdges, GraphGlobals, GraphNodes
from mace_jax.modules import loss as loss_mod
from mace_jax.tools import sum_nodes_of_the_same_graph
from mace_jax.tools.train import (
    _checked_loss,
    _make_eval_batch_metrics,
    _prepare_pred_graph,
)


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


def test_missing_predictions_do_not_create_perfect_metrics():
    graph = build_test_graph()
    output = {'energy': build_predictions()['energy']}
    pred_graph, predictions = _prepare_pred_graph(graph, output)
    loss_fn = loss_mod.WeightedEnergyForcesLoss(forces_weight=0.0)
    metrics = _make_eval_batch_metrics(loss_fn)(graph, pred_graph, predictions, None)
    assert metrics['energy'] is not None
    assert metrics['forces'] is None
    assert metrics['stress'] is None


@pytest.mark.parametrize('missing', ['energy', 'forces', 'virials'])
def test_training_rejects_missing_required_predictions(missing):
    graph = build_test_graph()
    output = build_predictions() | {'virials': jnp.zeros((3, 3, 3))}
    output.pop(missing)
    with pytest.raises(ValueError, match=f'missing model predictions.*{missing}'):
        _checked_loss(loss_mod.WeightedEnergyForcesVirialsLoss(), graph, output)


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
    loss_components = jnp.asarray(
        torch.nn.functional.huber_loss(
            torch.tensor(np.asarray(diff)),
            torch.zeros(diff.shape),
            reduction='none',
            delta=0.5,
        ).numpy()
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


def test_force_weights_disable_missing_labels_and_scale_labeled_graphs():
    graph = build_test_graph()
    preds = build_predictions()
    baseline = loss_mod.mean_squared_error_forces(graph, preds['forces'])
    graph = graph._replace(
        globals=graph.globals._replace(forces_weight=jnp.array([0.0, 0.25, 0.0]))
    )
    actual = loss_mod.mean_squared_error_forces(graph, preds['forces'])
    np.testing.assert_allclose(actual, baseline * jnp.array([0.0, 0.25, 0.0]))
    grad = jax.grad(
        lambda forces: loss_mod.WeightedForcesLoss()(graph, {'forces': forces}).sum()
    )(preds['forces'])
    np.testing.assert_array_equal(grad[:2], 0.0)


@pytest.mark.parametrize(
    'loss_cls',
    [
        loss_mod.WeightedHuberEnergyForcesStressLoss,
        loss_mod.UniversalLoss,
    ],
)
@pytest.mark.parametrize('property_name', ['energy', 'forces', 'stress'])
def test_huber_matches_torch_values_and_gradients(loss_cls, property_name):
    graph = build_test_graph()
    graph = graph._replace(globals=graph.globals._replace(weight=jnp.ones(3)))
    # Include errors on both sides of delta=0.5 in each property.
    reference = (
        graph.nodes.forces
        if property_name == 'forces'
        else getattr(graph.globals, property_name)
    )
    reference = reference.astype(jnp.float64)
    pred = reference + jnp.linspace(0.1, 0.9, reference.size).reshape(reference.shape)
    weights = {
        f'{name}_weight': float(name == property_name)
        for name in ('energy', 'forces', 'stress')
    }
    loss_fn = loss_cls(**weights, huber_delta=0.5)
    actual, actual_grad = jax.value_and_grad(
        lambda value: loss_fn(graph, {property_name: value}).sum()
    )(pred)
    pred_torch = torch.tensor(np.asarray(pred), requires_grad=True)
    ref_torch = torch.tensor(np.asarray(reference))
    if property_name == 'energy':
        expected = torch.nn.functional.huber_loss(
            pred_torch[:2] / torch.tensor([2.0, 1.0]),
            ref_torch[:2] / torch.tensor([2.0, 1.0]),
            delta=0.5,
            reduction='sum',
        )
    elif property_name == 'forces':
        expected = sum(
            torch.nn.functional.huber_loss(
                pred_torch[start:end], ref_torch[start:end], delta=0.5
            )
            for start, end in [(0, 2), (2, 3)]
        )
    else:
        expected = sum(
            torch.nn.functional.huber_loss(pred_torch[i], ref_torch[i], delta=0.5)
            for i in range(2)
        )
    expected.backward()
    np.testing.assert_allclose(actual, expected.detach().numpy(), rtol=1e-6)
    np.testing.assert_allclose(actual_grad, pred_torch.grad.numpy(), atol=1e-7)


def test_universal_loss_is_zero_for_exact_forces_with_nonunit_weights():
    graph = build_test_graph()
    graph = graph._replace(
        globals=graph.globals._replace(forces_weight=jnp.array([0.5, 3.0, 0.0]))
    )
    loss_fn = loss_mod.UniversalLoss(energy_weight=0.0, stress_weight=0.0)
    value, grad = jax.value_and_grad(
        lambda forces: loss_fn(graph, {'forces': forces}).sum()
    )(graph.nodes.forces)
    assert value == 0.0
    np.testing.assert_array_equal(grad, 0.0)


@pytest.mark.parametrize('force_norm', [50.0, 150.0, 250.0, 350.0])
def test_conditional_huber_matches_torch(force_norm):
    graph = build_test_graph()
    reference = jnp.zeros_like(graph.nodes.forces).at[:, 0].set(force_norm)
    graph = graph._replace(nodes=graph.nodes._replace(forces=reference))
    pred = reference + jnp.array([0.01, 0.1, 0.6], dtype=reference.dtype)
    actual = loss_mod.conditional_huber_forces(graph, pred, huber_delta=0.5)
    expected = torch_conditional_huber(
        torch.tensor(np.asarray(reference[:2])), torch.tensor(np.asarray(pred[:2])), 0.5
    )
    np.testing.assert_allclose(
        actual[:2], [expected.item(), 2 * expected.item()], rtol=1e-5
    )
    assert actual[2] == 0.0


def test_weighted_huber_uses_fixed_threshold_for_large_forces():
    graph = build_test_graph()
    reference = jnp.full_like(graph.nodes.forces, 350.0)
    graph = graph._replace(nodes=graph.nodes._replace(forces=reference))
    pred = reference + 0.2
    loss_fn = loss_mod.WeightedHuberEnergyForcesStressLoss(
        energy_weight=0.0, stress_weight=0.0, huber_delta=0.5
    )
    expected = torch.nn.functional.huber_loss(
        torch.tensor(np.asarray(reference)), torch.tensor(np.asarray(pred)), delta=0.5
    ).item()
    np.testing.assert_allclose(
        loss_fn(graph, {'forces': pred})[:2], [expected, 2 * expected], rtol=1e-6
    )
