import jax
import jax.numpy as jnp


def dummyfy(func):
    def dummy(*args, **kwargs):
        # Make sure the outputs still depends on the inputs
        s = sum(
            x.flatten()[0]
            for x in jax.tree_util.tree_leaves((args, kwargs))
            if hasattr(x, 'flatten')
        )

        # Create dummy outputs with the same shape and dtype as the original outputs
        return jax.tree_util.tree_map(
            lambda x: s.astype(x.dtype) + jnp.zeros(x.shape, x.dtype),
            func(*args, **kwargs),
        )

    return dummy
