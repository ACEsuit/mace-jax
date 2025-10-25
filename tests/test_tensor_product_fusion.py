import importlib

import cuequivariance as cue
import jax
import jax.numpy as jnp
import pytest
from e3nn_jax import Irreps  # type: ignore

from mace_jax.modules.blocks import RealAgnosticInteractionBlock
from mace_jax.modules.wrapper_ops import CuEquivarianceConfig, TensorProduct
from mace_jax.tools.cg import O3_e3nn
from mace_jax.tools.scatter import scatter_sum

_cue_jax_available = importlib.util.find_spec('cuequivariance_jax') is not None
try:
    importlib.import_module('cuequivariance_ops_jax')
except ModuleNotFoundError:
    _cue_jax_available = False

_cue_torch_available = False
try:
    importlib.import_module('cuequivariance_ops_torch')
    import torch

    cue_torch_ops = getattr(torch.ops, 'cuequivariance', None)
    _cue_torch_available = hasattr(cue_torch_ops, 'segmented_transpose')
except ModuleNotFoundError:
    _cue_torch_available = False


pytestmark = pytest.mark.skipif(
    not (_cue_jax_available and _cue_torch_available),
    reason='conv_fusion tests require cuequivariance backends with fused support',
)


def _make_module(conv_fusion: bool) -> TensorProduct:
    config = CuEquivarianceConfig(
        enabled=True,
        optimize_channelwise=True,
        conv_fusion=conv_fusion,
        layout='mul_ir',
    )
    return TensorProduct(
        Irreps('1x0e'),
        Irreps('1x0e'),
        Irreps('1x0e'),
        shared_weights=True,
        internal_weights=False,
        cueq_config=config,
    )


def _weight_numel() -> int:
    descriptor = cue.descriptors.channelwise_tensor_product(
        cue.Irreps(O3_e3nn, '1x0e'),
        cue.Irreps(O3_e3nn, '1x0e'),
        cue.Irreps(O3_e3nn, '1x0e'),
    )
    return descriptor.polynomial.operands[0].size


_WEIGHT_NUMEL = _weight_numel()


def _conv_method(module: TensorProduct, *, edge_index=None) -> str:
    dim = module.irreps_in1.dim  # type: ignore[attr-defined]
    x = jnp.ones((1, dim), dtype=jnp.float32)
    weights = jnp.ones((1, _WEIGHT_NUMEL), dtype=jnp.float32)
    variables = module.init(
        jax.random.PRNGKey(0),
        x,
        x,
        weights=weights,
        edge_index=edge_index,
    )
    return module.apply(variables, method=lambda mdl: mdl._conv_method)


class TestCueTensorProductFusion:
    def test_conv_fusion_requires_edge_index(self):
        module = _make_module(conv_fusion=True)
        dim = module.irreps_in1.dim  # type: ignore[attr-defined]
        x = jnp.ones((1, dim), dtype=jnp.float32)
        with pytest.raises(ValueError):
            module.init(
                jax.random.PRNGKey(0),
                x,
                x,
                weights=jnp.ones((1, _WEIGHT_NUMEL), dtype=jnp.float32),
            )

    def test_conv_fusion_reports_method(self):
        module = _make_module(conv_fusion=True)
        edge_index = jnp.array([[0], [0]], dtype=jnp.int32)
        method = _conv_method(module, edge_index=edge_index)
        if method not in {'uniform_1d', 'naive'}:
            pytest.fail(f'Unexpected conv method: {method}')

    def test_conv_fusion_matches_non_fused_when_unavailable(self):
        edge_index = jnp.array([[0], [0]], dtype=jnp.int32)
        fused_module = _make_module(conv_fusion=True)
        baseline_module = _make_module(conv_fusion=False)

        fused_method = _conv_method(fused_module, edge_index=edge_index)
        baseline_method = _conv_method(baseline_module)

        if fused_method == 'uniform_1d':
            assert baseline_method == 'naive'
        else:
            assert fused_method == baseline_method == 'naive'

    def test_conv_fusion_matches_scatter_sum(self):
        key = jax.random.PRNGKey(42)
        edge_index = jnp.array([[0, 1, 1], [1, 0, 1]], dtype=jnp.int32)
        fused_module = _make_module(conv_fusion=True)
        baseline_module = _make_module(conv_fusion=False)

        dim_node = fused_module.irreps_in1.dim  # type: ignore[attr-defined]
        dim_edge = fused_module.irreps_in2.dim  # type: ignore[attr-defined]
        num_edges = edge_index.shape[1]
        num_nodes = 2

        key, node_key, attr_key, weight_key = jax.random.split(key, 4)
        node_feats = jax.random.normal(node_key, (num_nodes, dim_node))
        edge_attrs = jax.random.normal(attr_key, (num_edges, dim_edge))
        weights = jax.random.normal(weight_key, (num_edges, _WEIGHT_NUMEL))

        fused_variables = fused_module.init(
            key,
            node_feats,
            edge_attrs,
            weights,
            edge_index=edge_index,
        )
        fused_out = fused_module.apply(
            fused_variables,
            node_feats,
            edge_attrs,
            weights,
            edge_index=edge_index,
        )

        sender = edge_index[0]
        receiver = edge_index[1]

        baseline_variables = baseline_module.init(
            key,
            node_feats[sender],
            edge_attrs,
            weights,
        )
        mji = baseline_module.apply(
            baseline_variables,
            node_feats[sender],
            edge_attrs,
            weights,
        )
        baseline_out = scatter_sum(mji, receiver, dim=0, dim_size=num_nodes)

        assert fused_out.shape == baseline_out.shape
        assert jnp.allclose(fused_out, baseline_out, rtol=1e-5, atol=1e-6)


class TestWrapperBlockConvFusion:
    def test_matches_baseline(self):
        irreps = Irreps('1x0e')
        num_nodes = 3
        num_edges = 4

        edge_index = jnp.array([[0, 1, 2, 1], [1, 0, 1, 2]], dtype=jnp.int32)

        key = jax.random.PRNGKey(123)
        key, attrs_key, feats_key, edge_attr_key, edge_feats_key = jax.random.split(
            key, 5
        )

        node_attrs = jax.random.normal(attrs_key, (num_nodes, irreps.dim))
        node_feats = jax.random.normal(feats_key, (num_nodes, irreps.dim))
        edge_attrs = jax.random.normal(edge_attr_key, (num_edges, irreps.dim))
        edge_feats = jax.random.normal(edge_feats_key, (num_edges, irreps.num_irreps))

        fused_config = CuEquivarianceConfig(
            enabled=True,
            optimize_channelwise=True,
            conv_fusion=True,
            layout='mul_ir',
        )
        baseline_config = CuEquivarianceConfig(
            enabled=True,
            optimize_channelwise=True,
            conv_fusion=False,
            layout='mul_ir',
        )

        fused_block = RealAgnosticInteractionBlock(
            node_attrs_irreps=irreps,
            node_feats_irreps=irreps,
            edge_attrs_irreps=irreps,
            edge_feats_irreps=irreps,
            target_irreps=irreps,
            hidden_irreps=irreps,
            avg_num_neighbors=1.5,
            radial_MLP=[8],
            cueq_config=fused_config,
        )
        baseline_block = RealAgnosticInteractionBlock(
            node_attrs_irreps=irreps,
            node_feats_irreps=irreps,
            edge_attrs_irreps=irreps,
            edge_feats_irreps=irreps,
            target_irreps=irreps,
            hidden_irreps=irreps,
            avg_num_neighbors=1.5,
            radial_MLP=[8],
            cueq_config=baseline_config,
        )

        init_args = (node_attrs, node_feats, edge_attrs, edge_feats, edge_index)

        fused_variables = fused_block.init(key, *init_args)
        baseline_variables = baseline_block.init(key, *init_args)

        fused_out, _ = fused_block.apply(fused_variables, *init_args)
        baseline_out, _ = baseline_block.apply(baseline_variables, *init_args)

        fused_method = fused_block.apply(
            fused_variables, method=lambda mdl: mdl.conv_tp._conv_method
        )
        baseline_method = baseline_block.apply(
            baseline_variables, method=lambda mdl: mdl.conv_tp._conv_method
        )

        if fused_method == 'uniform_1d':
            assert baseline_method == 'naive'
        else:
            assert fused_method == baseline_method == 'naive'
        assert fused_out.shape == baseline_out.shape
        assert jnp.allclose(fused_out, baseline_out, rtol=1e-5, atol=1e-6)
