import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from mace_jax.modules.blocks import ScaleShiftBlock
from mace_jax.nnx_utils import state_to_pure_dict


class TestScaleShiftBlock:
    def test_gradients_are_zero(self):
        block = ScaleShiftBlock(scale=1.5, shift=-0.2)
        x = jnp.asarray([1.0, 2.0], dtype=jnp.float64)
        head = jnp.asarray([0, 0], dtype=jnp.int32)

        graphdef, state = nnx.split(block)
        params = state_to_pure_dict(state)

        def loss_fn(p):
            y, _ = graphdef.apply(p)(x, head)
            return jnp.sum(y**2)

        grads = jax.grad(loss_fn)(params)
        scale_grad = grads['scale']
        shift_grad = grads['shift']

        assert jnp.all(scale_grad == 0.0)
        assert jnp.all(shift_grad == 0.0)

    def test_values_do_not_update(self):
        block = ScaleShiftBlock(scale=np.array([1.0, 2.0]), shift=np.array([0.5, -0.5]))
        x = jnp.asarray([3.0, 4.0], dtype=jnp.float64)
        head = jnp.asarray([1, 1], dtype=jnp.int32)

        graphdef, state = nnx.split(block)
        params = state_to_pure_dict(state)
        original_scale = params['scale']
        original_shift = params['shift']

        def loss_fn(p):
            y, _ = graphdef.apply(p)(x, head)
            return jnp.sum(y**2)

        grads = jax.grad(loss_fn)(params)
        updates = jax.tree_util.tree_map(lambda g: -0.1 * g, grads)
        updated = jax.tree_util.tree_map(lambda p, u: p + u, params, updates)

        assert jnp.array_equal(updated['scale'], original_scale)
        assert jnp.array_equal(updated['shift'], original_shift)
