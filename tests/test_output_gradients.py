import jax
import jax.numpy as jnp
import jraph
import numpy as np
import pytest

from mace_jax.data.utils import (
    AtomicNumberTable,
    Configuration,
    graph_from_configuration,
)
from mace_jax.modules.utils import compute_forces_and_stress, get_outputs
from mace_jax.tools import gin_model


@pytest.mark.parametrize('use_output_interface', [False, True])
@pytest.mark.parametrize('volume', [0.0, 8.0])
def test_stress_gradients_are_finite_with_zero_volume_cells(
    use_output_interface, volume
):
    positions = jnp.array([[1.0, 2.0, 3.0]])
    cell = jnp.eye(3)[None] * volume ** (1 / 3)
    data = dict(
        positions=positions,
        cell=cell,
        unit_shifts=jnp.zeros((0, 3)),
        edge_index=jnp.zeros((2, 0), dtype=jnp.int32),
        batch=jnp.zeros(1, dtype=jnp.int32),
        ptr=jnp.array([0, 1]),
    )

    def loss(weight):
        def energy(pos, shifts=None):
            return jnp.reshape(weight * jnp.sum(pos**2), (1,))

        if use_output_interface:
            stress = get_outputs(energy, data, compute_stress=True).stress
        else:
            _, stress = compute_forces_and_stress(
                energy,
                positions,
                cell,
                data['unit_shifts'],
                data['edge_index'],
                data['batch'],
                1,
            )
        return jnp.sum(stress**2)

    value, grad = jax.jit(jax.value_and_grad(loss))(1.0)
    expected = (
        0.0
        if volume == 0
        else np.sum((2 * np.outer(positions[0], positions[0]) / volume) ** 2)
    )
    np.testing.assert_allclose(value, expected, rtol=1e-6)
    np.testing.assert_allclose(grad, 2 * expected, rtol=1e-6)


@pytest.mark.parametrize('interface', ['direct', 'outputs', 'traced_flags'])
def test_mixed_periodicity_masks_only_molecular_stress(interface):
    positions = jnp.tile(jnp.array([[1.0, 2.0, 3.0]]), (4, 1))
    cell = jnp.tile((2 * jnp.eye(3))[None], (4, 1, 1)).at[3].set(0)
    pbc = jnp.array(
        [
            [False, False, False],
            [True, True, False],
            [True, True, True],
            [True, True, True],
        ]
    )
    data = dict(
        positions=positions,
        cell=cell,
        pbc=pbc,
        unit_shifts=jnp.zeros((0, 3)),
        edge_index=jnp.zeros((2, 0), jnp.int32),
        batch=jnp.arange(4),
        ptr=jnp.arange(5),
    )

    def outputs(weight, data):
        def energy(pos, shifts=None):
            return weight * jnp.sum(pos**2, axis=1)

        if interface == 'direct':
            return compute_forces_and_stress(
                energy,
                data['positions'],
                data['cell'],
                data['unit_shifts'],
                data['edge_index'],
                data['batch'],
                4,
                pbc=data['pbc'],
            )
        flags = dict(compute_force=True, compute_stress=True)
        if interface == 'traced_flags':
            flags = dict(
                compute_force=jnp.asarray(True), compute_stress=jnp.asarray(True)
            )
        result = get_outputs(energy, data, **flags)
        return result.forces, result.stress

    forces, stress = jax.jit(outputs)(1.0, data)
    expected_stress = np.tile(
        (2 * np.outer(positions[0], positions[0]) / 8)[None], (4, 1, 1)
    )
    expected_stress[[0, 3]] = 0
    np.testing.assert_allclose(stress, expected_stress)
    np.testing.assert_allclose(forces, -2 * positions)
    value, grad = jax.jit(
        jax.value_and_grad(lambda w, data: jnp.sum(outputs(w, data)[1] ** 2))
    )(1.0, data)
    np.testing.assert_allclose(value, np.sum(expected_stress**2))
    np.testing.assert_allclose(grad, 2 * value)


def test_native_mace_propagates_periodicity_through_padding():
    periodicities = [(False, False, False), (True, True, False), (True, True, True)]
    graphs = [
        graph_from_configuration(
            Configuration(
                atomic_numbers=np.ones(2, dtype=int),
                positions=np.array([[0.0, 0.0, 0.0], [1.2, 0.1, 0.0]]),
                forces=np.zeros((2, 3)),
                energy=0.0,
                stress=np.zeros((3, 3)),
                cell=np.eye(3) * 8,
                pbc=pbc,
            ),
            cutoff=3.0,
            z_table=AtomicNumberTable([1]),
        )
        for pbc in periodicities
    ]
    graph = jraph.pad_with_graphs(jraph.batch(graphs), n_node=7, n_edge=7, n_graph=4)
    np.testing.assert_array_equal(
        graph.globals.pbc, [*periodicities, (False, False, False)]
    )
    predictor, params, _ = gin_model.model(
        r_max=3.0,
        num_species=1,
        atomic_numbers=(1,),
        atomic_energies='zero',
        avg_num_neighbors=1.0,
        num_interactions=2,
        num_bessel=2,
        num_polynomial_cutoff=5,
        max_ell=1,
        correlation=2,
        gate=jax.nn.silu,
        hidden_irreps='2x0e+2x1o',
        MLP_irreps='2x0e',
    )
    output = jax.jit(lambda p, g: predictor(p, g, compute_stress=True))(params, graph)
    np.testing.assert_array_equal(
        np.asarray(output['stress'])[[0, 3]], np.zeros((2, 3, 3))
    )
    assert np.linalg.norm(output['stress'][1]) > 0
    np.testing.assert_allclose(output['stress'][1], output['stress'][2], atol=1e-12)
    np.testing.assert_allclose(
        output['energy'][:3], np.repeat(output['energy'][0], 3), atol=1e-12
    )
    assert np.all(np.isfinite(output['forces']))
