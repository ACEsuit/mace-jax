import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mace_jax.modules.utils import compute_forces_and_stress, get_outputs


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
