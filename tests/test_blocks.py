import re
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

warnings.filterwarnings("ignore", category=DeprecationWarning, module="haiku")
torch.serialization.add_safe_globals([slice])

import haiku as hk  # noqa: E402
from e3nn import o3  # noqa: E402
from e3nn_jax import Irreps  # noqa: E402
from jax import config as jax_config  # noqa: E402
from mace.modules.blocks import (
    RealAgnosticDensityInteractionBlock as RealAgnosticDensityInteractionBlockTorch,
)
from mace.modules.blocks import (  # noqa: E402
    RealAgnosticInteractionBlock as RealAgnosticInteractionBlockTorch,
)
from mace.modules.blocks import (  # noqa: E402
    RealAgnosticResidualInteractionBlock as RealAgnosticResidualInteractionBlockTorch,
)

from mace_jax.modules.blocks import (
    RealAgnosticDensityInteractionBlock as RealAgnosticDensityInteractionBlockJAX,
)
from mace_jax.modules.blocks import (  # noqa: E402
    RealAgnosticInteractionBlock as RealAgnosticInteractionBlockJAX,
)
from mace_jax.modules.blocks import (  # noqa: E402
    RealAgnosticResidualInteractionBlock as RealAgnosticResidualInteractionBlockJAX,
)


# === Helpers ===
def map_keys(jax_params):
    result = {}

    for k1, v1 in jax_params.items():
        for k2, _ in v1.items():
            key = f"{k1.split('~_setup/')[-1]}.{k2}"
            key = re.sub("/~/", ".", key)
            result[key] = (k1, k2)

    return result


def copy_jax_to_torch(torch_module, jax_params):
    """
    Copy parameters from JAX/Haiku to PyTorch module.
    Assumes names align via mapping dict or identical hierarchy.
    """
    torch_state = torch_module.state_dict()

    key_mapping = map_keys(jax_params)

    for k in torch_state.keys():
        # Skip output_mask in linear layes, which seems to be dead code
        if k.endswith(".output_mask"):
            continue

        # In Torch we have an explicit bias, which is a tensor of size 0
        if len(torch_state[k]) == 0:
            assert k not in jax_params.keys()
            continue

        k1, k2 = key_mapping[k]

        jax_arr = jax_params[k1][k2]
        jax_tensor = torch.from_numpy(np.array(jax_arr))

        if torch_state[k].shape != jax_tensor.shape:
            raise ValueError(
                f"Shape mismatch for {k}: "
                f"torch {torch_state[k].shape}, jax {jax_tensor.shape}"
            )
        torch_state[k] = jax_tensor

    torch_module.load_state_dict(torch_state)


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


# === PyTest class ===
class TestRealAgnosticInteractionBlock:
    @pytest.fixture
    def dummy_data(self):
        n_nodes, n_edges, feat_dim = 5, 8, 2
        node_attrs = np.random.randn(n_nodes, feat_dim).astype(np.float64)
        node_feats = np.random.randn(n_nodes, feat_dim).astype(np.float64)
        edge_attrs = np.random.randn(n_edges, feat_dim).astype(np.float64)
        edge_feats = np.random.randn(n_edges, feat_dim).astype(np.float64)
        edge_index = np.random.randint(0, n_nodes, size=(2, n_edges)).astype(np.int64)
        return node_attrs, node_feats, edge_attrs, edge_feats, edge_index

    def test_torch_vs_jax(self, dummy_data):
        node_attrs, node_feats, edge_attrs, edge_feats, edge_index = dummy_data

        assert node_attrs.shape[1] == o3.Irreps("2x0e").dim
        assert node_feats.shape[1] == o3.Irreps("2x0e").dim
        assert edge_attrs.shape[1] == o3.Irreps("2x0e").dim
        assert edge_feats.shape[1] == o3.Irreps("2x0e").dim

        # === Set dtype ===
        torch.set_default_dtype(torch.float64)
        jax_config.update("jax_enable_x64", True)

        # === Run JAX version ===
        jax_inputs = (
            jnp.array(node_attrs),
            jnp.array(node_feats),
            jnp.array(edge_attrs),
            jnp.array(edge_feats),
            jnp.array(edge_index),
        )
        jax_out, jax_params = run_jax_forward(
            RealAgnosticInteractionBlockJAX,
            jax_inputs,
            node_attrs_irreps=Irreps("2x0e"),
            node_feats_irreps=Irreps("2x0e"),
            edge_attrs_irreps=Irreps("2x0e"),
            edge_feats_irreps=Irreps("2x0e"),
            target_irreps=Irreps("2x0e"),
            hidden_irreps=Irreps("2x0e"),
            avg_num_neighbors=3.0,
        )

        # === Torch version ===
        torch_module = RealAgnosticInteractionBlockTorch(
            node_attrs_irreps=o3.Irreps("2x0e"),
            node_feats_irreps=o3.Irreps("2x0e"),
            edge_attrs_irreps=o3.Irreps("2x0e"),
            edge_feats_irreps=o3.Irreps("2x0e"),
            target_irreps=o3.Irreps("2x0e"),
            hidden_irreps=o3.Irreps("2x0e"),
            avg_num_neighbors=3.0,
        )

        # Copy weights JAX → Torch
        copy_jax_to_torch(torch_module, jax_params)

        # Run forward in Torch
        torch_inputs = (
            torch.from_numpy(node_attrs),
            torch.from_numpy(node_feats),
            torch.from_numpy(edge_attrs),
            torch.from_numpy(edge_feats),
            torch.from_numpy(edge_index),
        )
        torch_out = torch_module(*torch_inputs)

        # === Compare outputs ===
        torch_arr = torch_out[0].detach().cpu().numpy()
        jax_arr = np.array(jax_out[0])

        np.testing.assert_allclose(
            torch_arr,
            jax_arr,
            rtol=0.01,
            atol=0.001,
            err_msg="Torch and JAX RealAgnosticInteractionBlock outputs differ!",
        )


class TestRealAgnosticResidualInteractionBlock:
    @pytest.fixture
    def dummy_data(self):
        n_nodes, n_edges, feat_dim = 5, 8, 2
        node_attrs = np.random.randn(n_nodes, feat_dim).astype(np.float64)
        node_feats = np.random.randn(n_nodes, feat_dim).astype(np.float64)
        edge_attrs = np.random.randn(n_edges, feat_dim).astype(np.float64)
        edge_feats = np.random.randn(n_edges, feat_dim).astype(np.float64)
        edge_index = np.random.randint(0, n_nodes, size=(2, n_edges)).astype(np.int64)
        return node_attrs, node_feats, edge_attrs, edge_feats, edge_index

    def test_torch_vs_jax(self, dummy_data):
        node_attrs, node_feats, edge_attrs, edge_feats, edge_index = dummy_data

        assert node_attrs.shape[1] == o3.Irreps("2x0e").dim
        assert node_feats.shape[1] == o3.Irreps("2x0e").dim
        assert edge_attrs.shape[1] == o3.Irreps("2x0e").dim
        assert edge_feats.shape[1] == o3.Irreps("2x0e").dim

        # === Set dtype ===
        torch.set_default_dtype(torch.float64)
        jax_config.update("jax_enable_x64", True)

        # === Run JAX version ===
        jax_inputs = (
            jnp.array(node_attrs),
            jnp.array(node_feats),
            jnp.array(edge_attrs),
            jnp.array(edge_feats),
            jnp.array(edge_index),
        )
        jax_out, jax_params = run_jax_forward(
            RealAgnosticResidualInteractionBlockJAX,
            jax_inputs,
            node_attrs_irreps=Irreps("2x0e"),
            node_feats_irreps=Irreps("2x0e"),
            edge_attrs_irreps=Irreps("2x0e"),
            edge_feats_irreps=Irreps("2x0e"),
            target_irreps=Irreps("2x0e"),
            hidden_irreps=Irreps("2x0e"),
            avg_num_neighbors=3.0,
        )

        # === Torch version ===
        torch_module = RealAgnosticResidualInteractionBlockTorch(
            node_attrs_irreps=o3.Irreps("2x0e"),
            node_feats_irreps=o3.Irreps("2x0e"),
            edge_attrs_irreps=o3.Irreps("2x0e"),
            edge_feats_irreps=o3.Irreps("2x0e"),
            target_irreps=o3.Irreps("2x0e"),
            hidden_irreps=o3.Irreps("2x0e"),
            avg_num_neighbors=3.0,
        )

        # Copy weights JAX → Torch
        copy_jax_to_torch(torch_module, jax_params)

        # Run forward in Torch
        torch_inputs = (
            torch.from_numpy(node_attrs),
            torch.from_numpy(node_feats),
            torch.from_numpy(edge_attrs),
            torch.from_numpy(edge_feats),
            torch.from_numpy(edge_index),
        )
        torch_out = torch_module(*torch_inputs)

        # === Compare outputs (both message and skip connection) ===
        for i in range(2):
            torch_arr = torch_out[i].detach().cpu().numpy()
            jax_arr = np.array(jax_out[i])
            np.testing.assert_allclose(
                torch_arr,
                jax_arr,
                rtol=0.01,
                atol=0.001,
                err_msg=f"Torch and JAX RealAgnosticResidualInteractionBlock output[{i}] differ!",
            )


class TestRealAgnosticDensityInteractionBlock:
    @pytest.fixture
    def dummy_data(self):
        n_nodes, n_edges, feat_dim = 5, 8, 2
        node_attrs = np.random.randn(n_nodes, feat_dim).astype(np.float64)
        node_feats = np.random.randn(n_nodes, feat_dim).astype(np.float64)
        edge_attrs = np.random.randn(n_edges, feat_dim).astype(np.float64)
        edge_feats = np.random.randn(n_edges, feat_dim).astype(np.float64)
        edge_index = np.random.randint(0, n_nodes, size=(2, n_edges)).astype(np.int64)
        return node_attrs, node_feats, edge_attrs, edge_feats, edge_index

    def test_torch_vs_jax(self, dummy_data):
        node_attrs, node_feats, edge_attrs, edge_feats, edge_index = dummy_data

        assert node_attrs.shape[1] == o3.Irreps("2x0e").dim
        assert node_feats.shape[1] == o3.Irreps("2x0e").dim
        assert edge_attrs.shape[1] == o3.Irreps("2x0e").dim
        assert edge_feats.shape[1] == o3.Irreps("2x0e").dim

        # === Set dtype ===
        torch.set_default_dtype(torch.float64)
        jax_config.update("jax_enable_x64", True)

        # === Run JAX version ===
        jax_inputs = (
            jnp.array(node_attrs),
            jnp.array(node_feats),
            jnp.array(edge_attrs),
            jnp.array(edge_feats),
            jnp.array(edge_index),
        )
        jax_out, jax_params = run_jax_forward(
            RealAgnosticDensityInteractionBlockJAX,
            jax_inputs,
            node_attrs_irreps=Irreps("2x0e"),
            node_feats_irreps=Irreps("2x0e"),
            edge_attrs_irreps=Irreps("2x0e"),
            edge_feats_irreps=Irreps("2x0e"),
            target_irreps=Irreps("2x0e"),
            hidden_irreps=Irreps("2x0e"),
            avg_num_neighbors=3.0,
        )

        # === Torch version ===
        torch_module = RealAgnosticDensityInteractionBlockTorch(
            node_attrs_irreps=o3.Irreps("2x0e"),
            node_feats_irreps=o3.Irreps("2x0e"),
            edge_attrs_irreps=o3.Irreps("2x0e"),
            edge_feats_irreps=o3.Irreps("2x0e"),
            target_irreps=o3.Irreps("2x0e"),
            hidden_irreps=o3.Irreps("2x0e"),
            avg_num_neighbors=3.0,
        )

        # Copy weights JAX → Torch
        copy_jax_to_torch(torch_module, jax_params)

        # Run forward in Torch
        torch_inputs = (
            torch.from_numpy(node_attrs),
            torch.from_numpy(node_feats),
            torch.from_numpy(edge_attrs),
            torch.from_numpy(edge_feats),
            torch.from_numpy(edge_index),
        )
        torch_out = torch_module(*torch_inputs)

        # === Compare outputs (only message, since second return is None) ===
        torch_arr = torch_out[0].detach().cpu().numpy()
        jax_arr = np.array(jax_out[0])
        np.testing.assert_allclose(
            torch_arr,
            jax_arr,
            rtol=0.01,
            atol=0.001,
            err_msg="Torch and JAX RealAgnosticDensityInteractionBlock outputs differ!",
        )
