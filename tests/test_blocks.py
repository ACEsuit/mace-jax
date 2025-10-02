import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from e3nn import o3
from e3nn_jax import Irreps
from mace.modules.blocks import (
    LinearNodeEmbeddingBlock as LinearNodeEmbeddingBlockTorch,
)
from mace.modules.blocks import (
    LinearReadoutBlock as LinearReadoutBlockTorch,
)
from mace.modules.blocks import (
    NonLinearReadoutBlock as NonLinearReadoutBlockTorch,
)
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

from mace_jax.haiku.torch import copy_torch_to_jax
from mace_jax.modules.blocks import (
    LinearNodeEmbeddingBlock as LinearNodeEmbeddingBlockJAX,
)
from mace_jax.modules.blocks import (
    LinearReadoutBlock as LinearReadoutBlockJAX,
)
from mace_jax.modules.blocks import (
    NonLinearReadoutBlock as NonLinearReadoutBlockJAX,
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


class TestLinearNodeEmbeddingBlock:
    """Compare LinearNodeEmbeddingBlock in Haiku vs PyTorch."""

    @pytest.mark.parametrize(
        'irreps_in, irreps_out',
        [
            ('3x0e', '2x0e'),  # scalars → scalars
            ('1x0e + 1x1o', '1x0e + 1x1o'),  # mixed scalar + vector → same
        ],
    )
    def test_forward_match(self, irreps_in, irreps_out):
        """Check JAX and Torch implementations match on forward pass."""

        irreps_in_obj = o3.Irreps(irreps_in)
        irreps_out_obj = o3.Irreps(irreps_out)

        n_nodes = 5

        # --- Create random input ---
        np.random.seed(0)
        x_np = np.random.randn(n_nodes, irreps_in_obj.dim).astype(np.float64)
        x_jax = jnp.array(x_np)
        x_torch = torch.tensor(x_np)

        # --- Torch model ---
        torch_model = LinearNodeEmbeddingBlockTorch(irreps_in_obj, irreps_out_obj)

        # --- Torch output ---
        out_torch = torch_model(x_torch)

        # --- JAX model ---
        def forward_fn(x):
            model = LinearNodeEmbeddingBlockJAX(
                irreps_in=irreps_in_obj,
                irreps_out=irreps_out_obj,
            )
            return model(x)

        transformed = hk.transform(forward_fn)
        rng = jax.random.PRNGKey(42)
        params = transformed.init(rng, x_jax)

        # Copy Torch → JAX
        params = copy_torch_to_jax(torch_model, params)

        out_jax = transformed.apply(params, None, x_jax)

        # --- Compare ---
        np.testing.assert_allclose(
            np.array(out_jax),
            out_torch.detach().numpy(),
            rtol=1e-5,
            atol=1e-6,
        )


class TestLinearReadoutBlock:
    """Compare LinearReadoutBlock in Haiku vs PyTorch."""

    @pytest.mark.parametrize(
        'irreps_in, irrep_out',
        [
            ('3x0e', '1x0e'),  # scalars → scalar
            ('1x0e + 1x1o', '2x0e'),  # mixed input → scalar outputs
        ],
    )
    def test_forward_match(self, irreps_in, irrep_out):
        """Check JAX and Torch implementations match on forward pass."""

        irreps_in_obj = o3.Irreps(irreps_in)
        irrep_out_obj = o3.Irreps(irrep_out)

        n_nodes = 5

        # --- Create random input ---
        np.random.seed(0)
        x_np = np.random.randn(n_nodes, irreps_in_obj.dim).astype(np.float64)
        x_jax = jnp.array(x_np)
        x_torch = torch.tensor(x_np)

        # --- Torch model ---
        torch_model = LinearReadoutBlockTorch(irreps_in_obj, irrep_out_obj)

        # --- Torch output ---
        out_torch = torch_model(x_torch)

        # --- JAX model ---
        def forward_fn(x):
            model = LinearReadoutBlockJAX(
                irreps_in=irreps_in_obj,
                irrep_out=irrep_out_obj,
            )
            return model(x)

        transformed = hk.transform(forward_fn)
        rng = jax.random.PRNGKey(42)
        params = transformed.init(rng, x_jax)

        # Copy Torch → JAX
        params = copy_torch_to_jax(torch_model, params)

        out_jax = transformed.apply(params, None, x_jax)

        # --- Compare ---
        np.testing.assert_allclose(
            np.array(out_jax),
            out_torch.detach().numpy(),
            rtol=1e-5,
            atol=1e-6,
        )


class TestNonLinearReadoutBlock:
    """Compare NonLinearReadoutBlock in Haiku vs PyTorch."""

    @pytest.mark.parametrize(
        'irreps_in, MLP_irreps, irrep_out, num_heads',
        [
            ('3x0e', '4x0e', '1x0e', 1),  # simple scalar case
            ('10x0e', '2x0e', '1x0e', 1),  # mixed scalars + vector
        ],
    )
    def test_forward_match(self, irreps_in, MLP_irreps, irrep_out, num_heads):
        """Check forward pass matches between JAX and Torch."""

        irreps_in_obj = o3.Irreps(irreps_in)
        MLP_irreps_obj = o3.Irreps(MLP_irreps)
        irrep_out_obj = o3.Irreps(irrep_out)

        n_nodes = 5

        # --- Create random input ---
        np.random.seed(0)
        x_np = np.random.randn(n_nodes, irreps_in_obj.dim).astype(np.float64)
        x_jax = jnp.array(x_np)
        x_torch = torch.tensor(x_np)

        # --- Torch model ---
        torch_model = NonLinearReadoutBlockTorch(
            irreps_in=irreps_in_obj,
            MLP_irreps=MLP_irreps_obj,
            gate=torch.nn.SiLU(),
            irrep_out=irrep_out_obj,
            num_heads=num_heads,
        )

        # --- Torch output ---
        out_torch = torch_model(x_torch)

        # --- JAX model ---
        def forward_fn(x):
            model = NonLinearReadoutBlockJAX(
                irreps_in=irreps_in_obj,
                MLP_irreps=MLP_irreps_obj,
                gate=jax.nn.silu,
                irrep_out=irrep_out_obj,
                num_heads=num_heads,
            )
            return model(x)

        transformed = hk.transform(forward_fn)
        rng = jax.random.PRNGKey(42)
        params = transformed.init(rng, x_jax)
        params = copy_torch_to_jax(torch_model, params)

        out_jax = transformed.apply(params, None, x_jax)

        # --- Compare outputs ---
        np.testing.assert_allclose(
            np.array(out_jax),
            out_torch.detach().numpy(),
            rtol=5e-4,
            atol=5e-3,
        )


class TestRealAgnosticBlocks:
    # === Fixtures ===
    @pytest.fixture
    def dummy_data(self):
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
    def test_torch_vs_jax(self, dummy_data, jax_cls, torch_cls, multi_output):
        node_attrs, node_feats, edge_attrs, edge_feats, edge_index = dummy_data
        irreps = Irreps('2x0e')

        # === Define inputs ===
        jax_inputs = (
            jnp.array(node_attrs),
            jnp.array(node_feats),
            jnp.array(edge_attrs),
            jnp.array(edge_feats),
            jnp.array(edge_index),
        )

        torch_inputs = (
            torch.from_numpy(node_attrs),
            torch.from_numpy(node_feats),
            torch.from_numpy(edge_attrs),
            torch.from_numpy(edge_feats),
            torch.from_numpy(edge_index),
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
        torch_out = torch_module(*torch_inputs)

        # === Run JAX ===
        def run_jax_forward(jax_module_cls, inputs, **kwargs):
            """Initialize and run Haiku module once."""

            def forward_fn(*args):
                mod = jax_module_cls(**kwargs)
                return mod(*args)

            transformed = hk.transform(forward_fn)
            rng = jax.random.PRNGKey(42)
            params = transformed.init(rng, *inputs)
            params = copy_torch_to_jax(torch_module, params)
            out = transformed.apply(params, rng, *inputs)
            return out

        jax_out = run_jax_forward(
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
