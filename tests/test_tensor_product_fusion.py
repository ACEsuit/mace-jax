import importlib

import cuequivariance as cue
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from e3nn_jax import Irreps  # type: ignore
from flax import nnx

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
        rngs=nnx.Rngs(0),
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
    graphdef, state = nnx.split(module)
    _, (graphdef, state) = graphdef.apply(state)(
        x, x, weights=weights, edge_index=edge_index
    )
    module_after = nnx.merge(graphdef, state)
    return module_after._conv_method


class TestCueTensorProductFusion:
    def test_conv_fusion_requires_edge_index(self):
        module = _make_module(conv_fusion=True)
        dim = module.irreps_in1.dim  # type: ignore[attr-defined]
        x = jnp.ones((1, dim), dtype=jnp.float32)
        graphdef, state = nnx.split(module)
        with pytest.raises(ValueError):
            graphdef.apply(state)(
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

        fused_graphdef, fused_state = nnx.split(fused_module)
        fused_out, _ = fused_graphdef.apply(fused_state)(
            node_feats,
            edge_attrs,
            weights,
            edge_index=edge_index,
        )

        sender = edge_index[0]
        receiver = edge_index[1]

        baseline_graphdef, baseline_state = nnx.split(baseline_module)
        mji, _ = baseline_graphdef.apply(baseline_state)(
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
            rngs=nnx.Rngs(0),
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
            rngs=nnx.Rngs(0),
        )

        init_args = (node_attrs, node_feats, edge_attrs, edge_feats, edge_index)

        fused_graphdef, fused_state = nnx.split(fused_block)
        fused_out, (fused_graphdef, fused_state) = fused_graphdef.apply(fused_state)(
            *init_args
        )
        baseline_graphdef, baseline_state = nnx.split(baseline_block)
        baseline_out, (baseline_graphdef, baseline_state) = baseline_graphdef.apply(
            baseline_state
        )(*init_args)
        if isinstance(fused_out, tuple):
            fused_out = fused_out[0]
        if isinstance(baseline_out, tuple):
            baseline_out = baseline_out[0]

        fused_method = nnx.merge(fused_graphdef, fused_state).conv_tp._conv_method
        baseline_method = nnx.merge(
            baseline_graphdef, baseline_state
        ).conv_tp._conv_method

        if fused_method == 'uniform_1d':
            assert baseline_method == 'naive'
        else:
            assert fused_method == baseline_method == 'naive'
        assert fused_out.shape == baseline_out.shape
        assert jnp.allclose(fused_out, baseline_out, rtol=1e-5, atol=1e-6)


class TestConvFusionTorchParity:
    def test_tensor_product_matches_torch(self):
        import torch
        from mace.modules.wrapper_ops import (
            CuEquivarianceConfig as TorchCuEquivarianceConfig,
        )
        from mace.modules.wrapper_ops import TensorProduct as TorchTensorProduct

        if not torch.cuda.is_available():
            pytest.skip('Torch CUDA is required for conv_fusion parity.')
        if not any(device.platform == 'gpu' for device in jax.devices()):
            pytest.skip('JAX GPU backend is required for conv_fusion parity.')

        irreps = Irreps('1x0e')
        num_nodes = 3
        num_edges = 4
        edge_index = np.array([[0, 1, 2, 1], [1, 2, 0, 1]], dtype=np.int32)

        torch_config = TorchCuEquivarianceConfig(
            enabled=True,
            optimize_channelwise=True,
            conv_fusion=True,
            layout='mul_ir',
        )
        torch_module = TorchTensorProduct(
            irreps,
            irreps,
            irreps,
            shared_weights=True,
            internal_weights=False,
            cueq_config=torch_config,
        )
        torch_module.eval()

        weight_numel = int(getattr(torch_module, 'weight_numel', _WEIGHT_NUMEL))
        assert weight_numel == _WEIGHT_NUMEL

        rng = np.random.default_rng(0)
        node_feats = rng.normal(size=(num_nodes, irreps.dim)).astype(np.float32)
        edge_attrs = rng.normal(size=(num_edges, irreps.dim)).astype(np.float32)
        weights = rng.normal(size=(num_edges, weight_numel)).astype(np.float32)

        torch_out = torch_module(
            torch.tensor(node_feats),
            torch.tensor(edge_attrs),
            torch.tensor(weights),
            torch.tensor(edge_index, dtype=torch.int64),
        )
        torch_out = torch_out.detach().cpu().numpy()

        jax_config = CuEquivarianceConfig(
            enabled=True,
            optimize_channelwise=True,
            conv_fusion=True,
            layout='mul_ir',
        )
        jax_module = TensorProduct(
            irreps,
            irreps,
            irreps,
            shared_weights=True,
            internal_weights=False,
            cueq_config=jax_config,
            rngs=nnx.Rngs(0),
        )

        node_feats_j = jnp.asarray(node_feats)
        edge_attrs_j = jnp.asarray(edge_attrs)
        weights_j = jnp.asarray(weights)
        edge_index_j = jnp.asarray(edge_index)

        graphdef, state = nnx.split(jax_module)
        jax_out, (graphdef, state) = graphdef.apply(state)(
            node_feats_j,
            edge_attrs_j,
            weights_j,
            edge_index=edge_index_j,
        )
        method = nnx.merge(graphdef, state)._conv_method

        if method != 'uniform_1d':
            pytest.skip('JAX conv_fusion was not selected for this configuration.')

        assert torch_out.shape == jax_out.shape
        np.testing.assert_allclose(
            np.array(jax_out),
            torch_out,
            rtol=1e-5,
            atol=1e-6,
        )
