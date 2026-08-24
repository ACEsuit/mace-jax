import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from mace_jax.modules.radial import RadialMLP


@pytest.mark.parametrize('dtype', [jnp.float32, jnp.float64])
def test_split_first_matches_forward_gradient_and_hvp(dtype):
    if dtype == jnp.float64 and not jax.config.x64_enabled:
        pytest.skip('JAX x64 is disabled')

    node_count, edge_count = 7, 13
    edge_dim, source_dim, target_dim = 5, 11, 9
    model = RadialMLP(
        [edge_dim + source_dim + target_dim, 17, 19, 8],
        output_permutation=(4, 0, 7, 2, 5, 1, 6, 3),
        rngs=nnx.Rngs(17),
    )
    edge = jax.random.normal(jax.random.key(1), (edge_count, edge_dim), dtype=dtype)
    source = jax.random.normal(
        jax.random.key(2), (node_count, source_dim), dtype=dtype
    )
    target = jax.random.normal(
        jax.random.key(3), (node_count, target_dim), dtype=dtype
    )
    senders = jnp.asarray([0, 1, 5, 2, 6, 0, 4, 3, 1, 2, 5, 6, 3])
    receivers = jnp.asarray([1, 0, 2, 6, 3, 4, 0, 5, 6, 1, 3, 2, 4])

    def reference(edge_value, source_value, target_value):
        joined = jnp.concatenate(
            (edge_value, source_value[senders], target_value[receivers]), axis=-1
        )
        return model(joined)

    def split(edge_value, source_value, target_value):
        return model.apply_with_split_first_linear(
            edge_value, source_value, target_value, senders, receivers
        )

    expected = reference(edge, source, target)
    actual = split(edge, source, target)
    if dtype == jnp.float32:
        assert float(jnp.max(jnp.abs(actual - expected))) < 1.0e-3
    else:
        np.testing.assert_allclose(actual, expected, rtol=3e-12, atol=3e-12)

    def gradient(function):
        return jax.grad(
            lambda e, s, t: jnp.sum(jnp.sin(function(e, s, t))),
            argnums=(0, 1, 2),
        )

    tangents = tuple(jnp.full_like(value, 0.125) for value in (edge, source, target))
    expected_gradient = gradient(reference)(edge, source, target)
    actual_gradient = gradient(split)(edge, source, target)
    for actual_leaf, expected_leaf in zip(
        actual_gradient, expected_gradient, strict=True
    ):
        if dtype == jnp.float32:
            assert float(jnp.max(jnp.abs(actual_leaf - expected_leaf))) < 2.0e-3
        else:
            np.testing.assert_allclose(
                actual_leaf, expected_leaf, rtol=3e-12, atol=3e-12
            )

    _, expected_hvp = jax.jvp(gradient(reference), (edge, source, target), tangents)
    _, actual_hvp = jax.jvp(gradient(split), (edge, source, target), tangents)
    for actual_leaf, expected_leaf in zip(actual_hvp, expected_hvp, strict=True):
        if dtype == jnp.float32:
            assert float(jnp.max(jnp.abs(actual_leaf - expected_leaf))) < 1.0e-3
        else:
            np.testing.assert_allclose(
                actual_leaf, expected_leaf, rtol=3e-12, atol=3e-12
            )


def test_split_first_preserves_checkpoint_state():
    model = RadialMLP([9, 10, 4], rngs=nnx.Rngs(23))
    before = jax.tree.map(np.asarray, nnx.state(model))
    model.apply_with_split_first_linear(
        jnp.ones((3, 3)),
        jnp.ones((4, 2)),
        jnp.ones((4, 4)),
        jnp.asarray([0, 1, 2]),
        jnp.asarray([1, 2, 3]),
    )
    after = jax.tree.map(np.asarray, nnx.state(model))

    assert jax.tree.structure(before) == jax.tree.structure(after)
    for old, new in zip(jax.tree.leaves(before), jax.tree.leaves(after), strict=True):
        np.testing.assert_array_equal(old, new)


def test_split_first_validates_augmented_input_width():
    model = RadialMLP([9, 4, 2], rngs=nnx.Rngs(0))
    with pytest.raises(ValueError, match='input dimension mismatch'):
        model.apply_with_split_first_linear(
            jnp.ones((3, 2)),
            jnp.ones((4, 3)),
            jnp.ones((4, 3)),
            jnp.asarray([0, 1, 2]),
            jnp.asarray([1, 2, 3]),
        )
