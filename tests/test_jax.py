import e3nn_jax as e3nn
import haiku as hk
import jax
import jax.numpy as jnp
from e3nn_jax.util import assert_equivariant

from mace_jax.modules import SymmetricContraction


def test_symmetric_contraction():
    x = e3nn.normal("0e + 0o + 1o + 1e + 2e + 2o", jax.random.PRNGKey(0), (32, 128))
    y = jax.random.normal(jax.random.PRNGKey(1), (32, 4))

    model = hk.without_apply_rng(
        hk.transform(lambda x, y: SymmetricContraction(3, ["0e", "1o", "2e"])(x, y))
    )
    w = model.init(jax.random.PRNGKey(2), x, y)
    out = model.apply(w, x, y)

    assert_equivariant(lambda x: model.apply(w, x, y), jax.random.PRNGKey(3), (x,))


if __name__ == "__main__":
    test_symmetric_contraction()
