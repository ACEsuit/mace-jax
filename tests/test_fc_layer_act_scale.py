import jax
import jax.numpy as jnp
from flax import nnx

from mace_jax.adapters.e3nn.nn._fc import Layer
from mace_jax.nnx_utils import state_to_pure_dict


class TestFullyConnectedLayerActScale:
    def test_act_scale_not_trainable(self):
        layer = Layer(
            h_in=4, h_out=3, act=jnp.tanh, var_in=1.0, var_out=1.0, rngs=nnx.Rngs(0)
        )
        x = jnp.ones((2, 4), dtype=jnp.float64)

        graphdef, state = nnx.split(layer)
        params = state_to_pure_dict(state)

        def loss_fn(p):
            y, _ = graphdef.apply(p)(x)
            return jnp.sum(y**2)

        grads = jax.grad(loss_fn)(params)
        act_scale_grad = grads['act_scale']

        assert jnp.all(act_scale_grad == 0.0)
