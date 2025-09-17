import re

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from e3nn import o3
from e3nn_jax import Irreps
from mace.modules.blocks import (
    RealAgnosticAttResidualInteractionBlock as RealAgnosticAttResidualInteractionBlockTorch,
)
from mace.modules.blocks import (
    RealAgnosticDensityInteractionBlock as RealAgnosticDensityInteractionBlockTorch,
)
from mace.modules.blocks import (
    RealAgnosticDensityResidualInteractionBlock as RealAgnosticDensityResidualInteractionBlockTorch,
)
from mace.modules.blocks import (
    RealAgnosticInteractionBlock as RealAgnosticInteractionBlockTorch,
)
from mace.modules.blocks import (
    RealAgnosticResidualInteractionBlock as RealAgnosticResidualInteractionBlockTorch,
)
from mace.modules.blocks import (
    RealAgnosticResidualNonLinearInteractionBlock as RealAgnosticResidualNonLinearInteractionBlockTorch,
)

from mace_jax.modules.blocks import (
    RealAgnosticAttResidualInteractionBlock as RealAgnosticAttResidualInteractionBlockJAX,
)
from mace_jax.modules.blocks import (
    RealAgnosticDensityInteractionBlock as RealAgnosticDensityInteractionBlockJAX,
)
from mace_jax.modules.blocks import (
    RealAgnosticDensityResidualInteractionBlock as RealAgnosticDensityResidualInteractionBlockJAX,
)
from mace_jax.modules.blocks import (
    RealAgnosticInteractionBlock as RealAgnosticInteractionBlockJAX,
)
from mace_jax.modules.blocks import (
    RealAgnosticResidualInteractionBlock as RealAgnosticResidualInteractionBlockJAX,
)
from mace_jax.modules.blocks import (
    RealAgnosticResidualNonLinearInteractionBlock as RealAgnosticResidualNonLinearInteractionBlockJAX,
)


def map_keys(jax_params):
    result = {}
    for k1, v1 in jax_params.items():
        for k2 in v1.keys():
            key = f'{k1.split("~_setup/")[-1]}.{k2}'
            key = re.sub('/~/', '.', key)
            key = re.sub(r'net_(\d+)', r'net.\1', key)

            # Map haiku param names → torch names
            if k2 == 'w':
                key = key.replace('.w', '.weight')
            elif k2 == 'b':
                key = key.replace('.b', '.bias')
            elif k2 == 'scale':
                key = key.replace('.scale', '.weight')  # LayerNorm
            elif k2 == 'offset':
                key = key.replace('.offset', '.bias')  # LayerNorm

            if k2 in ('alpha', 'beta'):
                key = k2

            result[key] = (k1, k2)

    return result


def copy_jax_to_torch(torch_module, jax_params):
    """Copy parameters from JAX/Haiku to PyTorch module."""
    torch_state = torch_module.state_dict()
    key_mapping = map_keys(jax_params)

    new_state = {}
    for k in torch_state.keys():
        if k.endswith('.output_mask'):
            new_state[k] = torch_state[k]  # leave unchanged
            continue
        if torch_state[k].numel() == 0:  # empty bias tensor
            new_state[k] = torch_state[k]
            continue

        if k not in key_mapping:
            raise KeyError(f'Missing mapping for Torch key {k}')

        k1, k2 = key_mapping[k]
        jax_arr = jax_params[k1][k2]
        jax_tensor = torch.from_numpy(np.array(jax_arr))

        if torch_state[k].shape != jax_tensor.shape:
            if (
                'weight' in k
                and torch_state[k].ndim == 2
                and torch_state[k].shape == jax_tensor.T.shape
            ):
                jax_tensor = jax_tensor.T
            else:
                raise ValueError(
                    f'Shape mismatch for {k}: '
                    f'Torch {torch_state[k].shape}, JAX {jax_tensor.shape}'
                )
        new_state[k] = jax_tensor

    torch_module.load_state_dict(new_state)


def run_jax_forward(jax_module_cls, inputs, **kwargs):
    """Initialize and run Haiku module once."""

    def forward_fn(*args):
        mod = jax_module_cls(**kwargs)
        return mod(*args)

    transformed = hk.transform(forward_fn)
    rng = jax.random.PRNGKey(42)
    params = transformed.init(rng, *inputs)
    out = transformed.apply(params, rng, *inputs)
    return out, params


# === Fixtures ===
@pytest.fixture
def dummy_data():
    n_nodes, n_edges, feat_dim = 5, 8, 2
    node_attrs = np.random.randn(n_nodes, feat_dim).astype(np.float64)
    node_feats = np.random.randn(n_nodes, feat_dim).astype(np.float64)
    edge_attrs = np.random.randn(n_edges, feat_dim).astype(np.float64)
    edge_feats = np.random.randn(n_edges, feat_dim).astype(np.float64)
    edge_index = np.random.randint(0, n_nodes, size=(2, n_edges)).astype(np.int64)
    return node_attrs, node_feats, edge_attrs, edge_feats, edge_index


# === Parametrized Tests for All Blocks ===
@pytest.mark.parametrize(
    'jax_cls, torch_cls, multi_output',
    [
        (RealAgnosticInteractionBlockJAX, RealAgnosticInteractionBlockTorch, 1),
        (
            RealAgnosticResidualInteractionBlockJAX,
            RealAgnosticResidualInteractionBlockTorch,
            2,
        ),
        (
            RealAgnosticDensityInteractionBlockJAX,
            RealAgnosticDensityInteractionBlockTorch,
            1,
        ),
        (
            RealAgnosticResidualNonLinearInteractionBlockJAX,
            RealAgnosticResidualNonLinearInteractionBlockTorch,
            2,
        ),
        (
            RealAgnosticAttResidualInteractionBlockJAX,
            RealAgnosticAttResidualInteractionBlockTorch,
            2,
        ),
        (
            RealAgnosticDensityResidualInteractionBlockJAX,
            RealAgnosticDensityResidualInteractionBlockTorch,
            2,
        ),
    ],
)
def test_torch_vs_jax(dummy_data, jax_cls, torch_cls, multi_output):
    node_attrs, node_feats, edge_attrs, edge_feats, edge_index = dummy_data
    irreps = Irreps('2x0e')

    # === Run JAX ===
    jax_inputs = (
        jnp.array(node_attrs),
        jnp.array(node_feats),
        jnp.array(edge_attrs),
        jnp.array(edge_feats),
        jnp.array(edge_index),
    )
    jax_out, jax_params = run_jax_forward(
        jax_cls,
        jax_inputs,
        node_attrs_irreps=irreps,
        node_feats_irreps=irreps,
        edge_attrs_irreps=irreps,
        edge_feats_irreps=irreps,
        target_irreps=irreps,
        hidden_irreps=irreps,
        avg_num_neighbors=3.0,
    )

    # === Run Torch ===
    torch_module = torch_cls(
        node_attrs_irreps=o3.Irreps('2x0e'),
        node_feats_irreps=o3.Irreps('2x0e'),
        edge_attrs_irreps=o3.Irreps('2x0e'),
        edge_feats_irreps=o3.Irreps('2x0e'),
        target_irreps=o3.Irreps('2x0e'),
        hidden_irreps=o3.Irreps('2x0e'),
        avg_num_neighbors=3.0,
    )
    copy_jax_to_torch(torch_module, jax_params)

    torch_inputs = (
        torch.from_numpy(node_attrs),
        torch.from_numpy(node_feats),
        torch.from_numpy(edge_attrs),
        torch.from_numpy(edge_feats),
        torch.from_numpy(edge_index),
    )
    torch_out = torch_module(*torch_inputs)

    # === Compare Outputs ===
    if multi_output == 1:
        torch_arr = torch_out[0].detach().cpu().numpy()
        jax_arr = np.array(jax_out[0])
        np.testing.assert_allclose(torch_arr, jax_arr, rtol=0.01, atol=0.001)
    else:
        for i in range(multi_output):
            torch_arr = torch_out[i].detach().cpu().numpy()
            jax_arr = np.array(jax_out[i])
            np.testing.assert_allclose(torch_arr, jax_arr, rtol=0.01, atol=0.001)
