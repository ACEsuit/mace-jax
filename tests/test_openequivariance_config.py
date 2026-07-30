import sys
import types

import gin
import jax.numpy as jnp
import numpy as np
import pytest
from e3nn_jax import Irreps
from flax import nnx

from mace_jax.adapters.openequivariance import TensorProduct as OpenEqTensorProduct
from mace_jax.cli import mace_jax_train as train_cli
from mace_jax.modules.wrapper_ops import (
    EquivarianceConfig,
    FullyConnectedTensorProduct,
    TensorProduct,
)


def _enabled(**kwargs):
    config = {
        "enabled": True,
        "optimize_all": True,
        "conv_fusion": True,
    }
    config.update(kwargs)
    return config


def _enabled_equivariance(**kwargs):
    config = {
        "backend": "openeq",
        "optimize_all": True,
        "conv_fusion": True,
    }
    config.update(kwargs)
    return EquivarianceConfig(**config)


@pytest.mark.parametrize(
    ('kwargs', 'match'),
    [
        ({"group": "O3"}, "group O3_e3nn"),
        ({'optimize_linear': True}, 'optimize_linear'),
        ({'optimize_symmetric': True}, 'optimize_symmetric'),
        ({'optimize_fctp': True}, 'not safe for general FCTP'),
    ],
)
def test_openeq_rejects_unsupported_options(kwargs, match):
    with pytest.raises(ValueError, match=match):
        _enabled_equivariance(**kwargs)


def test_openeq_rejects_fctp_from_serialized_config():
    with pytest.raises(ValueError, match="not safe for general FCTP"):
        EquivarianceConfig(backend="openeq", optimize_fctp=True)


def test_openeq_adapter_rejects_standalone_tensor_products():
    with pytest.raises(ValueError, match='Standalone OpenEquivariance'):
        OpenEqTensorProduct(
            Irreps('1x0e'),
            Irreps('1x0e'),
            Irreps('1x0e'),
            instructions=[(0, 0, 0, 'uvw', True)],
            shared_weights=True,
            internal_weights=False,
            conv_fusion=False,
            rngs=nnx.Rngs(0),
        )


def test_optimize_all_does_not_select_openeq_fctp():
    module = FullyConnectedTensorProduct(
        Irreps('1x0e'),
        Irreps('1x0e'),
        Irreps('1x0e'),
        shared_weights=True,
        internal_weights=True,
        equivariance_config=_enabled_equivariance(),
        rngs=nnx.Rngs(0),
    )
    assert not type(module).__module__.startswith(
        'mace_jax.adapters.openequivariance'
    )


def test_openeq_selects_only_fused_channelwise_configuration():
    assert not (
        _enabled_equivariance(
            optimize_all=False, conv_fusion=False
        ).openeq_config.channelwise_fusion
    )
    assert not (
        _enabled_equivariance(conv_fusion=False).openeq_config.channelwise_fusion
    )
    assert _enabled_equivariance().openeq_config.channelwise_fusion


def test_cueq_rejects_same_kernel_conflict():
    with pytest.deprecated_call(match="Nested cueq_config"):
        with pytest.raises(ValueError, match="multiple acceleration backends"):
            EquivarianceConfig(
                cueq_config={
                    "enabled": True,
                    "optimize_channelwise": True,
                },
                openeq_config=_enabled(),
            )


def test_config_round_trip_is_canonical_and_legacy_is_preserved():
    with pytest.deprecated_call(match="Nested cueq_config"):
        config = EquivarianceConfig(openeq_config=_enabled())
    payload = config.to_dict()
    restored = EquivarianceConfig(**payload)
    assert restored.to_dict() == payload
    assert payload["backend"] == "openeq"
    assert payload["layout"] == "mul_ir"
    assert payload["group"] == "O3_e3nn"
    assert payload["optimize_all"] is True
    assert payload["conv_fusion"] is True
    assert "cueq_config" not in payload
    assert "openeq_config" not in payload


def test_cli_binds_openeq_config():
    gin.clear_config()
    args, _ = train_cli.parse_args(['--enable-openeq'])
    train_cli.apply_cli_overrides(args)
    config = gin.query_parameter(
        'mace_jax.tools.gin_model.model.equivariance_config'
    )
    assert isinstance(config, EquivarianceConfig)
    assert config.openeq_config.enabled
    assert config.openeq_config.optimize_all
    assert config.openeq_config.conv_fusion
    gin.clear_config()


def test_cli_explicit_openeq_flags_select_backend():
    gin.clear_config()
    args, _ = train_cli.parse_args(
        ["--openeq-optimize-all", "--openeq-conv-fusion"]
    )
    train_cli.apply_cli_overrides(args)
    config = gin.query_parameter(
        "mace_jax.tools.gin_model.model.equivariance_config"
    )
    assert config.backend == "openeq"
    assert config.optimize_all
    assert config.conv_fusion
    assert config.openeq_config.channelwise_fusion
    gin.clear_config()


def test_openeq_import_is_lazy_and_missing_dependency_is_actionable(monkeypatch):
    for name in list(sys.modules):
        if name.startswith('openequivariance'):
            monkeypatch.delitem(sys.modules, name, raising=False)
    config = _enabled_equivariance()
    assert 'openequivariance' not in sys.modules
    import builtins

    real_import = builtins.__import__

    def missing_openeq(name, *args, **kwargs):
        if name.startswith('openequivariance'):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', missing_openeq)
    with pytest.raises(RuntimeError, match=r'openequivariance\[jax\]==0.6.8'):
        TensorProduct(
            Irreps('1x0e'),
            Irreps('1x0e'),
            Irreps('1x0e'),
            instructions=[(0, 0, 0, 'uvu', True)],
            shared_weights=False,
            internal_weights=False,
            equivariance_config=config,
            rngs=nnx.Rngs(0),
        )


def test_adapter_uses_receiver_rows_sender_cols_and_permuted_weights(monkeypatch):
    package = types.ModuleType('openequivariance')
    package.__path__ = []
    jax_module = types.ModuleType('openequivariance.jax')

    class FakeProblem:
        weight_numel = 1

        def __init__(self, *args, **kwargs):
            assert kwargs['irrep_normalization'] == 'component'
            assert kwargs['path_normalization'] == 'element'
            assert kwargs['layout'] == 'mul_ir'

    class FakeConv:
        def __init__(self, problem, **kwargs):
            assert kwargs == {'deterministic': False, 'requires_jvp': True}

        def reorder_weights_from_e3nn(self, weights, has_batch_dim):
            assert has_batch_dim
            return weights

        def forward(self, nodes, edges, weights, rows, cols):
            assert rows.dtype == jnp.int32
            assert cols.dtype == jnp.int32
            values = nodes[cols, :1] * edges[:, :1] * weights[:, :1]
            return jnp.zeros((nodes.shape[0], 1), nodes.dtype).at[rows].add(values)

    package.Irreps = lambda value: value
    package.TPProblem = FakeProblem
    package.jax = jax_module
    jax_module.TensorProductConv = FakeConv
    monkeypatch.setitem(sys.modules, 'openequivariance', package)
    monkeypatch.setitem(sys.modules, 'openequivariance.jax', jax_module)

    module = TensorProduct(
        Irreps('1x0e'),
        Irreps('1x0e'),
        Irreps('1x0e'),
        instructions=[(0, 0, 0, 'uvu', True)],
        shared_weights=False,
        internal_weights=False,
        equivariance_config=_enabled_equivariance(),
        rngs=nnx.Rngs(0),
    )
    dtype = module.dtype
    edge_index = jnp.asarray([[2, 0, 2], [1, 1, 0]], dtype=jnp.int64)
    actual = module(
        jnp.asarray([[1.0], [2.0], [3.0]], dtype=dtype),
        jnp.ones((3, 1), dtype=dtype),
        jnp.ones((3, 1), dtype=dtype),
        edge_index,
    )
    np.testing.assert_allclose(actual, np.asarray([[3.0], [4.0], [0.0]]))

    graphdef, state = nnx.split(module)
    restored = nnx.merge(graphdef, state)
    empty = restored(
        jnp.ones((2, 1), dtype=dtype),
        jnp.empty((0, 1), dtype=dtype),
        jnp.empty((0, 1), dtype=dtype),
        jnp.empty((2, 0), dtype=jnp.int32),
    )
    assert empty.shape == (2, 1)
