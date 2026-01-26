import torch.serialization

from mace_jax.nnx_utils import state_to_pure_dict

# Allowlist slice objects for e3nn constants when importing modules.
if hasattr(torch.serialization, 'add_safe_globals'):
    torch.serialization.add_safe_globals([slice])

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from mace_jax.modules.blocks import AtomicEnergiesBlock


class TestAtomicEnergiesBlock:
    def test_gradient_is_zero(self):
        block = AtomicEnergiesBlock(np.array([1.0, 2.0, 3.0], dtype=np.float64))
        x = jnp.eye(3, dtype=jnp.float64)

        graphdef, state = nnx.split(block)
        params = state_to_pure_dict(state)

        def loss_fn(p):
            y, _ = graphdef.apply(p)(x)
            return jnp.sum((y - 1.0) ** 2)

        grads = jax.grad(loss_fn)(params)
        atomic_grad = grads['atomic_energies']

        assert jnp.all(atomic_grad == 0.0)

    def test_values_remain_constant_after_update(self):
        block = AtomicEnergiesBlock(np.array([1.0, 2.0, 3.0], dtype=np.float64))
        x = jnp.eye(3, dtype=jnp.float64)

        graphdef, state = nnx.split(block)
        params = state_to_pure_dict(state)
        original = params['atomic_energies']

        def loss_fn(p):
            y, _ = graphdef.apply(p)(x)
            return jnp.sum((y - 1.0) ** 2)

        grads = jax.grad(loss_fn)(params)
        updates = jax.tree_util.tree_map(lambda g: -0.1 * g, grads)
        updated = jax.tree_util.tree_map(lambda p, u: p + u, params, updates)

        assert jnp.all(updated['atomic_energies'] == original)
