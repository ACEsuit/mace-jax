import sys
import types

import jax.numpy as jnp
import numpy as np
from e3nn_jax import Irreps
from flax import nnx

from mace_jax.adapters.e3nn.nn._fc import FullyConnectedNet
from mace_jax.adapters.openequivariance import (
    FullyConnectedTensorProduct,
    TensorProduct,
)


def _install_fake_backend(monkeypatch):
    package = types.ModuleType('openequivariance')
    package.__path__ = []
    jax_module = types.ModuleType('openequivariance.jax')

    class Problem:
        weight_numel = 2

        def __init__(self, *args, **kwargs):
            self.shared_weights = kwargs['shared_weights']
            self.instructions = args[3]

    class Operator:
        def __init__(self, problem, **kwargs):
            self.problem = problem

        def reorder_weights_from_e3nn(self, weights, has_batch_dim=True):
            return weights[..., ::-1]

        def forward(self, x1, x2, weights):
            return x1 * x2 * weights

    class Conv(Operator):
        def forward(self, x1, x2, weights, rows, cols):
            values = x1[cols] * x2 * weights
            return jnp.zeros_like(x1).at[rows].add(values)

    package.Irreps = lambda value: value
    package.TPProblem = Problem
    package.jax = jax_module
    jax_module.TensorProduct = Operator
    jax_module.TensorProductConv = Conv
    monkeypatch.setitem(sys.modules, 'openequivariance', package)
    monkeypatch.setitem(sys.modules, 'openequivariance.jax', jax_module)


def test_standalone_tp_reorders_canonical_shared_weights(monkeypatch):
    _install_fake_backend(monkeypatch)
    tp = TensorProduct(
        Irreps('2x0e'),
        Irreps('2x0e'),
        Irreps('2x0e'),
        instructions=[(0, 0, 0, 'uvw', True)],
        shared_weights=True,
        conv_fusion=False,
        rngs=nnx.Rngs(0),
    )
    dtype = tp.dtype
    actual = tp(
        jnp.ones((3, 2), dtype=dtype),
        jnp.ones((3, 2), dtype=dtype),
        jnp.asarray([2.0, 5.0], dtype=dtype),
    )
    np.testing.assert_allclose(actual, np.tile([5.0, 2.0], (3, 1)))


def test_fctp_builds_weighted_uvw_paths_and_restores(monkeypatch):
    _install_fake_backend(monkeypatch)
    fctp = FullyConnectedTensorProduct(
        Irreps('2x0e'),
        Irreps('2x0e'),
        Irreps('2x0e'),
        internal_weights=True,
        rngs=nnx.Rngs(0),
    )
    assert all(path[3:5] == ('uvw', True) for path in fctp.instructions)
    graphdef, state = nnx.split(fctp)
    restored = nnx.merge(graphdef, state)
    assert restored(jnp.empty((0, 2)), jnp.empty((0, 2))).shape == (0, 2)


def test_projection_reorders_parameter_columns_not_batch_output():
    network = FullyConnectedNet(
        [3, 2], output_permutation=(1, 0), rngs=nnx.Rngs(0)
    )
    canonical_weight = np.asarray(network.layers[0].weight)
    x = jnp.arange(12.0).reshape(4, 3)
    actual = network(x)
    scale = np.sqrt(3.0)
    expected = np.asarray(x) @ canonical_weight[:, ::-1] / scale
    np.testing.assert_allclose(actual, expected, rtol=1e-6)
    np.testing.assert_array_equal(np.asarray(network.layers[0].weight), canonical_weight)
