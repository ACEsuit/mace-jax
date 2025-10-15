import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from e3nn import o3
from e3nn_jax import Irreps, IrrepsArray
from mace.modules.blocks import (
    EquivariantProductBasisBlock as EquivariantProductBasisBlockTorch,
)
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
from mace.modules.wrapper_ops import (
    CuEquivarianceConfig as CuEquivarianceConfigTorch,
)

from mace_jax.adapters.flax.torch import init_from_torch
from mace_jax.modules.blocks import (
    EquivariantProductBasisBlock as EquivariantProductBasisBlockJAX,
)
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
from mace_jax.modules.wrapper_ops import CuEquivarianceConfig as CuEquivarianceConfigJAX


def _to_numpy(x):
    array = x.array if hasattr(x, 'array') else x
    return np.asarray(array)


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
        rng = np.random.default_rng(0)
        x_np = rng.standard_normal((n_nodes, irreps_in_obj.dim)).astype(np.float32)
        x_jax = jnp.asarray(x_np)
        x_torch = torch.tensor(x_np, dtype=torch.float32)

        # --- Torch model ---
        torch_model = LinearNodeEmbeddingBlockTorch(
            irreps_in_obj, irreps_out_obj
        ).float()

        # --- Torch output ---
        out_torch = torch_model(x_torch)

        # --- JAX model ---
        module = LinearNodeEmbeddingBlockJAX(
            irreps_in=irreps_in_obj,
            irreps_out=irreps_out_obj,
        )
        module, variables = init_from_torch(
            module,
            torch_model,
            jax.random.PRNGKey(42),
            x_jax,
        )

        out_jax = module.apply(variables, x_jax)

        # --- Compare ---
        np.testing.assert_allclose(
            _to_numpy(out_jax),
            out_torch.detach().cpu().numpy(),
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
        rng = np.random.default_rng(0)
        x_np = rng.standard_normal((n_nodes, irreps_in_obj.dim)).astype(np.float32)
        x_jax = jnp.asarray(x_np)
        x_torch = torch.tensor(x_np, dtype=torch.float32)

        # --- Torch model ---
        torch_model = LinearReadoutBlockTorch(irreps_in_obj, irrep_out_obj).float()

        # --- Torch output ---
        out_torch = torch_model(x_torch)

        # --- JAX model ---
        module = LinearReadoutBlockJAX(
            irreps_in=irreps_in_obj,
            irrep_out=irrep_out_obj,
        )
        module, variables = init_from_torch(
            module,
            torch_model,
            jax.random.PRNGKey(42),
            x_jax,
        )

        out_jax = module.apply(variables, x_jax)

        # --- Compare ---
        np.testing.assert_allclose(
            _to_numpy(out_jax),
            out_torch.detach().cpu().numpy(),
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
        rng = np.random.default_rng(0)
        x_np = rng.standard_normal((n_nodes, irreps_in_obj.dim)).astype(np.float32)
        x_jax = jnp.asarray(x_np)
        x_ir = IrrepsArray(irreps_in_obj, x_jax)
        x_torch = torch.tensor(x_np, dtype=torch.float32)

        # --- Torch model ---
        torch_model = NonLinearReadoutBlockTorch(
            irreps_in=irreps_in_obj,
            MLP_irreps=MLP_irreps_obj,
            gate=torch.nn.SiLU(),
            irrep_out=irrep_out_obj,
            num_heads=num_heads,
        ).float()

        # --- Torch output ---
        out_torch = torch_model(x_torch)

        # --- JAX model ---
        module = NonLinearReadoutBlockJAX(
            irreps_in=irreps_in_obj,
            MLP_irreps=MLP_irreps_obj,
            gate=jax.nn.silu,
            irrep_out=irrep_out_obj,
            num_heads=num_heads,
        )
        module, variables = init_from_torch(
            module,
            torch_model,
            jax.random.PRNGKey(42),
            x_ir,
        )

        out_jax = module.apply(variables, x_ir)

        # --- Compare outputs ---
        np.testing.assert_allclose(
            _to_numpy(out_jax),
            out_torch.detach().cpu().numpy(),
            rtol=5e-4,
            atol=5e-3,
        )


class TestEquivariantProductBasisBlock:
    """Compare EquivariantProductBasisBlock in Haiku (JAX) vs PyTorch."""

    @pytest.mark.parametrize(
        'cue_config_kwargs',
        [
            dict(enabled=True, optimize_symmetric=True),
            dict(enabled=False, optimize_symmetric=False),
        ],
    )
    @pytest.mark.parametrize(
        'node_feats_irreps,target_irreps,correlation,use_sc,num_elements',
        [
            # Simple scalar contraction
            ('128x0e+128x1o+128x2e+128x3o', '128x0e+128x1o+128x2e', 1, False, 10),
            ('128x0e+128x1o+128x2e+128x3o', '128x0e+128x1o+128x2e', 1, True, 10),
            ('128x0e+128x1o+128x2e+128x3o', '128x0e+128x1o+128x2e', 2, False, 10),
            ('128x0e+128x1o+128x2e+128x3o', '128x0e+128x1o+128x2e', 2, True, 10),
            ('128x0e+128x1o+128x2e+128x3o', '128x0e+128x1o+128x2e', 3, False, 10),
            ('128x0e+128x1o+128x2e+128x3o', '128x0e+128x1o+128x2e', 3, True, 10),
        ],
    )
    def test_forward_match(
        self,
        node_feats_irreps,
        target_irreps,
        correlation,
        use_sc,
        num_elements,
        cue_config_kwargs,
    ):
        """Check forward pass matches between JAX and Torch."""

        node_feats_irreps = Irreps(node_feats_irreps)
        target_irreps = Irreps(target_irreps)

        n_nodes = 6
        rng = np.random.default_rng(0)

        # --- Inputs ---
        mul = node_feats_irreps[0].mul
        ell_dim_sum = sum(ir.ir.dim for ir in node_feats_irreps)

        x_np = rng.standard_normal((n_nodes, mul, ell_dim_sum)).astype(np.float32)
        sc_np = (
            rng.standard_normal((n_nodes, target_irreps.dim)).astype(np.float32)
            if use_sc
            else None
        )
        attr_indices = rng.integers(0, num_elements, size=n_nodes)
        attrs_np = np.eye(num_elements, dtype=np.float32)[attr_indices]

        x_jax = jnp.asarray(x_np)
        x_torch = torch.tensor(x_np, dtype=torch.float32)
        sc_jax = jnp.asarray(sc_np) if sc_np is not None else None
        sc_torch = (
            torch.tensor(sc_np, dtype=torch.float32) if sc_np is not None else None
        )
        attrs_jax = jnp.asarray(attrs_np)
        attrs_torch = torch.tensor(attrs_np, dtype=torch.float32)

        # --- Torch model ---
        cue_config_torch = CuEquivarianceConfigTorch(**cue_config_kwargs)

        torch_model = EquivariantProductBasisBlockTorch(
            node_feats_irreps=str(node_feats_irreps),
            target_irreps=str(target_irreps),
            correlation=correlation,
            use_sc=use_sc,
            num_elements=num_elements,
            use_reduced_cg=True,
            cueq_config=cue_config_torch,
        ).float()

        # Torch forward pass
        out_torch = torch_model(x_torch, sc_torch, attrs_torch)

        # --- JAX model ---
        module = EquivariantProductBasisBlockJAX(
            node_feats_irreps=node_feats_irreps,
            target_irreps=target_irreps,
            correlation=correlation,
            use_sc=use_sc,
            num_elements=num_elements,
            use_reduced_cg=True,
            cueq_config=CuEquivarianceConfigJAX(**cue_config_kwargs),
        )
        module, variables = init_from_torch(
            module,
            torch_model,
            jax.random.PRNGKey(42),
            x_jax,
            sc_jax,
            attrs_jax,
        )

        out_jax = module.apply(variables, x_jax, sc_jax, attrs_jax)

        # --- Compare ---
        np.testing.assert_allclose(
            _to_numpy(out_jax),
            out_torch.detach().cpu().numpy(),
            rtol=5e-4,
            atol=5e-3,
        )


class TestRealAgnosticBlocks:
    # === Fixtures ===
    @pytest.fixture
    def dummy_data(self):
        rng = np.random.default_rng(0)
        n_nodes, n_edges, feat_dim = 5, 8, 2
        attr_indices = rng.integers(0, feat_dim, size=n_nodes)
        node_attrs = np.eye(feat_dim, dtype=np.float32)[attr_indices]
        node_feats = rng.standard_normal((n_nodes, feat_dim)).astype(np.float32)
        edge_attrs = rng.standard_normal((n_edges, feat_dim)).astype(np.float32)
        edge_feats = rng.standard_normal((n_edges, feat_dim)).astype(np.float32)
        edge_index = rng.integers(0, n_nodes, size=(2, n_edges)).astype(np.int32)
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
            jnp.asarray(node_attrs),
            jnp.asarray(node_feats),
            jnp.asarray(edge_attrs),
            jnp.asarray(edge_feats),
            jnp.asarray(edge_index),
        )

        torch_inputs = (
            torch.from_numpy(node_attrs).float(),
            torch.from_numpy(node_feats).float(),
            torch.from_numpy(edge_attrs).float(),
            torch.from_numpy(edge_feats).float(),
            torch.from_numpy(edge_index).long(),
        )

        # === Run Torch ===
        cue_config_torch = CuEquivarianceConfigTorch(
            enabled=True,
            optimize_symmetric=True,
        )

        torch_module = torch_cls(
            node_attrs_irreps=o3.Irreps('2x0e'),
            node_feats_irreps=o3.Irreps('2x0e'),
            edge_attrs_irreps=o3.Irreps('2x0e'),
            edge_feats_irreps=o3.Irreps('2x0e'),
            target_irreps=o3.Irreps('2x0e'),
            hidden_irreps=o3.Irreps('2x0e'),
            avg_num_neighbors=3.0,
            cueq_config=cue_config_torch,
        ).float()
        torch_out = torch_module(*torch_inputs)

        cue_config_jax = CuEquivarianceConfigJAX(
            enabled=True,
            optimize_symmetric=True,
        )

        # === Run JAX ===
        module = jax_cls(
            node_attrs_irreps=irreps,
            node_feats_irreps=irreps,
            edge_attrs_irreps=irreps,
            edge_feats_irreps=irreps,
            target_irreps=irreps,
            hidden_irreps=irreps,
            avg_num_neighbors=3.0,
            cueq_config=cue_config_jax,
        )
        module, variables = init_from_torch(
            module,
            torch_module,
            jax.random.PRNGKey(42),
            *jax_inputs,
        )

        jax_out = module.apply(variables, *jax_inputs)

        # === Compare Outputs ===
        if multi_output == 1:
            torch_arr = torch_out[0].detach().cpu().numpy()
            jax_arr = _to_numpy(jax_out[0])
            np.testing.assert_allclose(torch_arr, jax_arr, rtol=0.01, atol=0.001)
        else:
            for i in range(multi_output):
                torch_arr = torch_out[i].detach().cpu().numpy()
                jax_arr = _to_numpy(jax_out[i])
                np.testing.assert_allclose(torch_arr, jax_arr, rtol=0.01, atol=0.001)
