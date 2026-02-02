import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from e3nn import o3
from e3nn_jax import Irreps, IrrepsArray
from flax import nnx
from mace.modules.blocks import (
    EquivariantProductBasisBlock as EquivariantProductBasisBlockTorch,
)
from mace.modules.blocks import (
    LinearNodeEmbeddingBlock as LinearNodeEmbeddingBlockTorch,
)
from mace.modules.blocks import LinearReadoutBlock as LinearReadoutBlockTorch
from mace.modules.blocks import NonLinearReadoutBlock as NonLinearReadoutBlockTorch
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
from mace.modules.wrapper_ops import CuEquivarianceConfig as CuEquivarianceConfigTorch

from mace_jax.adapters.cuequivariance.utility import mul_ir_to_ir_mul
from mace_jax.adapters.nnx.torch import init_from_torch
from mace_jax.modules.blocks import (
    EquivariantProductBasisBlock as EquivariantProductBasisBlockJAX,
)
from mace_jax.modules.blocks import (
    LinearNodeEmbeddingBlock as LinearNodeEmbeddingBlockJAX,
)
from mace_jax.modules.blocks import LinearReadoutBlock as LinearReadoutBlockJAX
from mace_jax.modules.blocks import NonLinearReadoutBlock as NonLinearReadoutBlockJAX
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
from mace_jax.tools.device import configure_torch_runtime


def _to_numpy(x):
    array = x.array if hasattr(x, 'array') else x
    return np.asarray(array)


def _module_dtype(module: torch.nn.Module) -> torch.dtype:
    for param in module.parameters():
        return param.dtype
    for buf in module.buffers():
        return buf.dtype
    return torch.float32


def _node_feats_for_layout(node_feats: np.ndarray, layout: str) -> np.ndarray:
    if layout == 'ir_mul':
        return np.transpose(node_feats, (0, 2, 1))
    return node_feats


def _flat_for_layout(
    features: np.ndarray | None, irreps: Irreps, layout: str
) -> np.ndarray | None:
    if features is None:
        return None
    if layout == 'ir_mul':
        return np.asarray(mul_ir_to_ir_mul(jnp.asarray(features), irreps))
    return features


def _features_from_xyz(
    simple_xyz_features: np.ndarray, irreps_in: o3.Irreps
) -> np.ndarray:
    """Expand XYZ-derived scalar/vector channels to match an irreps' total dim."""
    base_scalar = simple_xyz_features[:, :1]
    base_vector = simple_xyz_features[:, 1:]
    parts = []
    for mul, ir in irreps_in:
        dim = ir.dim * mul
        if ir.l == 0:
            scalars = np.tile(base_scalar, (1, dim))
            parts.append(scalars)
        else:
            repeats = math.ceil(dim / base_vector.shape[1])
            vecs = np.tile(base_vector, (1, repeats))[:, :dim]
            parts.append(vecs)
    return np.concatenate(parts, axis=1).astype(np.float32)


def _cue_device_or_skip() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip('Cue-equivariance kernels require CUDA.')
    device = configure_torch_runtime('cuda')

    cfg = CuEquivarianceConfigTorch(enabled=True, optimize_symmetric=True)
    block = (
        EquivariantProductBasisBlockTorch(
            node_feats_irreps='1x0e',
            target_irreps='1x0e',
            correlation=1,
            use_sc=False,
            num_elements=1,
            use_reduced_cg=True,
            cueq_config=cfg,
        )
        .float()
        .to(device)
    )

    dtype = _module_dtype(block)
    x = torch.zeros((1, 1, 1), dtype=dtype, device=device)
    attrs = torch.ones((1, 1), dtype=dtype, device=device)

    with torch.no_grad():
        try:
            block(x, None, attrs)
        except Exception as exc:  # pragma: no cover - backend dependent
            pytest.skip(f'Cue-equivariance kernels unavailable: {exc}')

    return device


class TestLinearNodeEmbeddingBlock:
    """Compare LinearNodeEmbeddingBlock in Haiku vs PyTorch."""

    @pytest.mark.parametrize(
        'irreps_in, irreps_out',
        [
            ('3x0e', '2x0e'),  # pure scalar inputs
            ('1x0e + 1x1o', '1x0e + 1x1o'),  # scalar + vector inputs
        ],
    )
    def test_forward_match_real_xyz(self, irreps_in, irreps_out, simple_xyz_features):
        """Check JAX and Torch implementations match on real XYZ-derived inputs."""

        irreps_in_obj = o3.Irreps(irreps_in)
        irreps_out_obj = o3.Irreps(irreps_out)

        x_np = _features_from_xyz(simple_xyz_features, irreps_in_obj)
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
            rngs=nnx.Rngs(42),
        )
        module, _ = init_from_torch(module, torch_model)
        graphdef, state = nnx.split(module)
        out_jax, _ = graphdef.apply(state)(x_jax)

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
    def test_forward_match_real_xyz(self, irreps_in, irrep_out, simple_xyz_features):
        """Check parity using real XYZ-derived scalar+vector inputs."""

        irreps_in_obj = o3.Irreps(irreps_in)
        irrep_out_obj = o3.Irreps(irrep_out)

        x_np = _features_from_xyz(simple_xyz_features, irreps_in_obj)
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
            rngs=nnx.Rngs(42),
        )
        module, _ = init_from_torch(module, torch_model)
        graphdef, state = nnx.split(module)
        out_jax, _ = graphdef.apply(state)(x_jax)

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
            ('3x0e', '4x0e', '1x0e', 1),  # scalar-only case
            ('10x0e', '2x0e', '1x0e', 1),  # wider scalar input
            ('1x0e + 1x1o', '2x0e', '1x0e', 2),  # mix with multi-head
        ],
    )
    def test_forward_match_real_xyz(
        self, irreps_in, MLP_irreps, irrep_out, num_heads, simple_xyz_features
    ):
        """Parity check on real XYZ-derived inputs to catch ordering issues."""
        irreps_in_obj = o3.Irreps(irreps_in)
        MLP_irreps_obj = o3.Irreps(MLP_irreps)
        irrep_out_obj = o3.Irreps(irrep_out)

        x_np = _features_from_xyz(simple_xyz_features, irreps_in_obj)
        x_jax = jnp.asarray(x_np)
        x_ir = IrrepsArray(irreps_in_obj, x_jax)
        x_torch = torch.tensor(x_np, dtype=torch.float32)

        torch_model = NonLinearReadoutBlockTorch(
            irreps_in=irreps_in_obj,
            MLP_irreps=MLP_irreps_obj,
            gate=torch.nn.SiLU(),
            irrep_out=irrep_out_obj,
            num_heads=num_heads,
        ).float()
        out_torch = torch_model(x_torch)

        module = NonLinearReadoutBlockJAX(
            irreps_in=irreps_in_obj,
            MLP_irreps=MLP_irreps_obj,
            gate=jax.nn.silu,
            irrep_out=irrep_out_obj,
            num_heads=num_heads,
            rngs=nnx.Rngs(42),
        )
        module, _ = init_from_torch(module, torch_model)
        graphdef, state = nnx.split(module)
        out_jax, _ = graphdef.apply(state)(x_ir)

        np.testing.assert_allclose(
            _to_numpy(out_jax),
            out_torch.detach().cpu().numpy(),
            rtol=5e-7,
            atol=5e-7,
        )


class TestEquivariantProductBasisBlock:
    """Compare EquivariantProductBasisBlock in Haiku (JAX) vs PyTorch."""

    @pytest.mark.parametrize(
        'enabled,layout',
        [
            (True, 'mul_ir'),
            (False, 'mul_ir'),
            (True, 'ir_mul'),
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
        enabled,
        layout,
    ):
        """Check forward pass matches between JAX and Torch."""

        cue_config_kwargs = dict(
            enabled=enabled,
            optimize_symmetric=enabled,
            layout=layout,
        )
        if layout == 'ir_mul':
            cue_config_kwargs['optimize_linear'] = True

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
        layout = cue_config_kwargs.get('layout', 'mul_ir')
        x_np = _node_feats_for_layout(x_np, layout)
        sc_np = _flat_for_layout(sc_np, target_irreps, layout)
        attr_indices = rng.integers(0, num_elements, size=n_nodes)
        attrs_np = np.eye(num_elements, dtype=np.float32)[attr_indices]

        device = torch.device('cpu')
        if cue_config_kwargs.get('enabled', False):
            device = _cue_device_or_skip()

        x_jax = jnp.asarray(x_np)
        x_torch = torch.tensor(x_np, dtype=torch.float32, device=device)
        sc_jax = jnp.asarray(sc_np) if sc_np is not None else None
        sc_torch = (
            torch.tensor(sc_np, dtype=torch.float32, device=device)
            if sc_np is not None
            else None
        )
        attrs_jax = jnp.asarray(attrs_np)
        attrs_torch = torch.tensor(attrs_np, dtype=torch.float32, device=device)

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
        torch_model = torch_model.to(device=device)

        # Torch forward pass
        try:
            out_torch = torch_model(x_torch, sc_torch, attrs_torch)
        except (NotImplementedError, RuntimeError) as exc:
            if cue_config_kwargs.get('enabled', False):
                pytest.skip(
                    'cuequivariance_torch backend unavailable during forward pass: '
                    f'{exc}'
                )
            raise

        # --- JAX model ---
        module = EquivariantProductBasisBlockJAX(
            node_feats_irreps=node_feats_irreps,
            target_irreps=target_irreps,
            correlation=correlation,
            use_sc=use_sc,
            num_elements=num_elements,
            use_reduced_cg=True,
            cueq_config=CuEquivarianceConfigJAX(**cue_config_kwargs),
            rngs=nnx.Rngs(42),
        )
        torch_model = torch_model.to('cpu')
        x_torch = x_torch.cpu()
        if sc_torch is not None:
            sc_torch = sc_torch.cpu()
        attrs_torch = attrs_torch.cpu()

        module, _ = init_from_torch(module, torch_model)
        graphdef, state = nnx.split(module)
        out_jax, _ = graphdef.apply(state)(x_jax, sc_jax, attrs_jax)

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

        device = torch.device('cpu')
        if torch.cuda.is_available():
            device = _cue_device_or_skip()

        torch_inputs = (
            torch.from_numpy(node_attrs).float().to(device),
            torch.from_numpy(node_feats).float().to(device),
            torch.from_numpy(edge_attrs).float().to(device),
            torch.from_numpy(edge_feats).float().to(device),
            torch.from_numpy(edge_index).long().to(device),
        )

        # === Run Torch ===
        layout = 'mul_ir'
        cue_config_torch = CuEquivarianceConfigTorch(
            enabled=True,
            optimize_symmetric=True,
            layout=layout,
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
        torch_module = torch_module.to(device)
        try:
            torch_out = torch_module(*torch_inputs)
        except (NotImplementedError, RuntimeError) as exc:
            if 'cuequivariance::' in str(exc) or isinstance(exc, NotImplementedError):
                pytest.skip(
                    'cuequivariance_torch backend unavailable during forward pass: '
                    f'{exc}'
                )
            raise
        torch_module = torch_module.to('cpu')
        torch_inputs = tuple(t.cpu() for t in torch_inputs)

        cue_config_jax = CuEquivarianceConfigJAX(
            enabled=True,
            optimize_symmetric=True,
            layout=layout,
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
            rngs=nnx.Rngs(42),
        )
        module, _ = init_from_torch(module, torch_module)
        graphdef, state = nnx.split(module)
        jax_out, _ = graphdef.apply(state)(*jax_inputs)

        # === Compare Outputs ===
        if multi_output == 1:
            torch_arr = torch_out[0].detach().cpu().numpy()
            jax_arr = _to_numpy(jax_out[0])
            np.testing.assert_allclose(torch_arr, jax_arr, rtol=1e-3, atol=1e-4)
        else:
            for i in range(multi_output):
                torch_arr = torch_out[i].detach().cpu().numpy()
                jax_arr = _to_numpy(jax_out[i])
                np.testing.assert_allclose(torch_arr, jax_arr, rtol=1e-3, atol=1e-4)
