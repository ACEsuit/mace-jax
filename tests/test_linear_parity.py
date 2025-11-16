import math
import numpy as np
import pytest
import torch
from torch.serialization import add_safe_globals

add_safe_globals([slice])

import jax
import jax.numpy as jnp
from e3nn import o3
from e3nn_jax import Irreps, IrrepsArray

from mace.modules.wrapper_ops import CuEquivarianceConfig as CuEquivarianceConfigTorch
from mace.modules.wrapper_ops import Linear as TorchLinearWrapper
from mace_jax.adapters.cuequivariance.linear import Linear as JaxLinear

import cuequivariance as cue
from cuequivariance_torch.operations.linear import Linear as CueLinearTorch
from mace_jax.adapters.cuequivariance.utility import (
    ir_mul_to_mul_ir as cue_ir_mul_to_mul_ir,
)
from mace_jax.adapters.cuequivariance.utility import (
    mul_ir_to_ir_mul as cue_mul_ir_to_ir_mul,
)


def _as_irreps_array(irreps: Irreps, array: jnp.ndarray) -> IrrepsArray:
    if hasattr(IrrepsArray, 'from_array'):
        return IrrepsArray.from_array(irreps, array)
    return IrrepsArray(irreps, array)


def _features_from_xyz(simple_xyz_features: np.ndarray, irreps_in: Irreps) -> np.ndarray:
    """Expand XYZ-derived scalar/vector channels to match an irreps' total dim."""
    base_scalar = simple_xyz_features[:, :1]
    base_vector = simple_xyz_features[:, 1:]
    parts = []
    for mul, ir in irreps_in:
        dim = ir.dim * mul
        if ir.l == 0:
            parts.append(np.tile(base_scalar, (1, dim)))
        else:
            repeats = math.ceil(dim / base_vector.shape[1])
            vecs = np.tile(base_vector, (1, repeats))[:, :dim]
            parts.append(vecs)
    return np.concatenate(parts, axis=1).astype(np.float32)


class TestLinearParity:
    def test_high_dim_linear_parity(self):
        """Cue-backed Linear vs e3nn o3.Linear on high-dimensional irreps."""

        rng = np.random.default_rng(123)

        irreps_in = o3.Irreps('128x0e + 128x1o')
        irreps_out = o3.Irreps('16x0e')

        torch_linear = o3.Linear(irreps_in, irreps_out, shared_weights=True)
        torch_linear = torch_linear.float().eval()

        features_np = rng.standard_normal((4, irreps_in.dim), dtype=np.float32)
        features_torch = torch.tensor(features_np, dtype=torch.float32)

        with torch.no_grad():
            out_torch = torch_linear(features_torch).cpu().numpy()

        jax_linear = JaxLinear(
            irreps_in=Irreps(irreps_in),
            irreps_out=Irreps(irreps_out),
            shared_weights=True,
        )
        features_jax = _as_irreps_array(Irreps(irreps_in), jnp.asarray(features_np))

        variables = jax_linear.init(jax.random.PRNGKey(0), features_jax)
        variables = JaxLinear.import_from_torch(torch_linear, variables)

        out_jax = jax_linear.apply(variables, features_jax)
        out_jax_arr = np.asarray(out_jax.array if hasattr(out_jax, 'array') else out_jax)

        np.testing.assert_allclose(out_jax_arr, out_torch, rtol=1e-6, atol=1e-6)

    def test_linear_parity_real_xyz(self, simple_xyz_features):
        """Parity check on real scalar+vector inputs derived from simple.xyz."""
        irreps_in = o3.Irreps('1x0e + 1x1o')
        irreps_out = o3.Irreps('1x0e')

        features_np = simple_xyz_features
        features_torch = torch.tensor(features_np, dtype=torch.float32)

        torch_linear = o3.Linear(irreps_in, irreps_out, shared_weights=True)
        torch_linear = torch_linear.float().eval()
        with torch.no_grad():
            out_torch = torch_linear(features_torch).cpu().numpy()

        jax_linear = JaxLinear(
            irreps_in=Irreps(irreps_in),
            irreps_out=Irreps(irreps_out),
            shared_weights=True,
        )
        features_jax = _as_irreps_array(Irreps(irreps_in), jnp.asarray(features_np))
        variables = jax_linear.init(jax.random.PRNGKey(0), features_jax)
        variables = JaxLinear.import_from_torch(torch_linear, variables)

        out_jax = jax_linear.apply(variables, features_jax)
        out_jax_arr = np.asarray(out_jax.array if hasattr(out_jax, 'array') else out_jax)

        np.testing.assert_allclose(out_jax_arr, out_torch, rtol=1e-6, atol=1e-6)

    def test_linear_parity_torch_backends(self, simple_xyz_features):
        """JAX Linear should import weights from both e3nn and cue torch backends."""
        irreps_in = o3.Irreps('1x0e + 1x1o')
        irreps_out = o3.Irreps('1x0e')
        features_np = simple_xyz_features.astype(np.float32)
        features_torch = torch.tensor(features_np, dtype=torch.float32)
        features_jax = _as_irreps_array(Irreps(irreps_in), jnp.asarray(features_np))

        # --- e3nn backend via wrapper (cue disabled) ---
        torch_linear_e3nn = TorchLinearWrapper(
            irreps_in,
            irreps_out,
            shared_weights=True,
            cueq_config=CuEquivarianceConfigTorch(enabled=False),
        ).float().eval()
        with torch.no_grad():
            out_torch_e3nn = torch_linear_e3nn(features_torch).cpu().numpy()

        jax_linear = JaxLinear(
            irreps_in=Irreps(irreps_in),
            irreps_out=Irreps(irreps_out),
            shared_weights=True,
        )
        variables = jax_linear.init(jax.random.PRNGKey(0), features_jax)
        variables_e3nn = JaxLinear.import_from_torch(torch_linear_e3nn, variables)
        out_jax = jax_linear.apply(variables_e3nn, features_jax)
        out_jax_arr = np.asarray(out_jax.array if hasattr(out_jax, 'array') else out_jax)
        np.testing.assert_allclose(out_jax_arr, out_torch_e3nn, rtol=1e-6, atol=1e-6)

        # --- cue-backed torch backend (enable optimize_linear to trigger cue) ---
        cue_cfg = CuEquivarianceConfigTorch(enabled=True, optimize_linear=True)
        try:
            torch_linear_cue = TorchLinearWrapper(
                irreps_in,
                irreps_out,
                shared_weights=True,
                cueq_config=cue_cfg,
            )
            torch_linear_cue = torch_linear_cue.float().eval()
        except Exception as exc:
            pytest.skip(f'cuequivariance torch backend unavailable: {exc}')

        with torch.no_grad():
            out_torch_cue = torch_linear_cue(features_torch).cpu().numpy()

        variables_cue = JaxLinear.import_from_torch(torch_linear_cue, variables)
        out_jax = jax_linear.apply(variables_cue, features_jax)
        out_jax_arr = np.asarray(out_jax.array if hasattr(out_jax, 'array') else out_jax)

        np.testing.assert_allclose(out_jax_arr, out_torch_cue, rtol=1e-6, atol=1e-6)

    def test_linear_parity_torch_backends_complex_irreps(self):
        """Parity on higher-dimensional irreps to mirror foundation-style setups."""
        rng = np.random.default_rng(1234)
        irreps_in = o3.Irreps('16x0e + 16x1o + 10x2e + 6x3o')
        irreps_out = o3.Irreps('8x0e + 4x1o')

        features_np = rng.standard_normal((5, irreps_in.dim), dtype=np.float32)
        features_torch = torch.tensor(features_np, dtype=torch.float32)
        features_jax = _as_irreps_array(Irreps(irreps_in), jnp.asarray(features_np))

        # --- e3nn backend via wrapper (cue disabled) ---
        torch_linear_e3nn = TorchLinearWrapper(
            irreps_in,
            irreps_out,
            shared_weights=True,
            cueq_config=CuEquivarianceConfigTorch(enabled=False),
        ).float().eval()
        with torch.no_grad():
            out_torch_e3nn = torch_linear_e3nn(features_torch).cpu().numpy()

        jax_linear = JaxLinear(
            irreps_in=Irreps(irreps_in),
            irreps_out=Irreps(irreps_out),
            shared_weights=True,
        )
        variables = jax_linear.init(jax.random.PRNGKey(1), features_jax)
        variables_e3nn = JaxLinear.import_from_torch(torch_linear_e3nn, variables)
        out_jax = jax_linear.apply(variables_e3nn, features_jax)
        out_jax_arr = np.asarray(out_jax.array if hasattr(out_jax, 'array') else out_jax)
        np.testing.assert_allclose(out_jax_arr, out_torch_e3nn, rtol=1e-6, atol=1e-6)

        # --- cue-backed torch backend (enable optimize_linear to trigger cue) ---
        cue_cfg = CuEquivarianceConfigTorch(enabled=True, optimize_linear=True)
        try:
            torch_linear_cue = TorchLinearWrapper(
                irreps_in,
                irreps_out,
                shared_weights=True,
                cueq_config=cue_cfg,
            )
            torch_linear_cue = torch_linear_cue.float().eval()
        except Exception as exc:
            pytest.skip(f'cuequivariance torch backend unavailable: {exc}')

        with torch.no_grad():
            out_torch_cue = torch_linear_cue(features_torch).cpu().numpy()

        variables_cue = JaxLinear.import_from_torch(torch_linear_cue, variables)
        out_jax = jax_linear.apply(variables_cue, features_jax)
        out_jax_arr = np.asarray(out_jax.array if hasattr(out_jax, 'array') else out_jax)

        np.testing.assert_allclose(out_jax_arr, out_torch_cue, rtol=1e-6, atol=1e-6)

    def test_cue_linear_matches_torch_mul_ir(self, simple_xyz_features):
        """Cue Linear adapter should mirror torch cuet.Linear in mul_ir layout."""
        irreps_in = '1x0e + 1x1o'
        irreps_out = '2x0e'

        torch_layer = CueLinearTorch(
            cue.Irreps(cue.O3, irreps_in),
            cue.Irreps(cue.O3, irreps_out),
            layout=cue.mul_ir,
        ).float()

        features_np = _features_from_xyz(simple_xyz_features, Irreps(irreps_in))
        features_jax = jnp.asarray(features_np)
        features_torch = torch.tensor(features_np, dtype=torch.float32)

        with torch.no_grad():
            out_torch = torch_layer(features_torch).cpu().numpy()

        module = JaxLinear(
            irreps_in=Irreps(irreps_in),
            irreps_out=Irreps(irreps_out),
        )
        variables = module.init(jax.random.PRNGKey(5), features_jax)
        variables = JaxLinear.import_from_torch(torch_layer, variables)
        out_jax = module.apply(variables, features_jax)

        np.testing.assert_allclose(
            np.asarray(out_jax),
            out_torch,
            rtol=1e-5,
            atol=1e-6,
        )

    def test_cue_linear_matches_torch_ir_mul(self, simple_xyz_features):
        """Cue Linear adapter should mirror torch cuet.Linear in ir_mul layout."""
        irreps_in = '2x0e + 1x1o'
        irreps_out = '1x0e + 1x1o'
        irreps_in_obj = Irreps(irreps_in)

        torch_layer = CueLinearTorch(
            cue.Irreps(cue.O3, irreps_in),
            cue.Irreps(cue.O3, irreps_out),
            layout=cue.ir_mul,
        ).float()

        # Build mul_ir features from the XYZ sample, then convert to ir_mul for torch.
        features_mul_ir = _features_from_xyz(simple_xyz_features, irreps_in_obj)
        features_ir_mul = cue_mul_ir_to_ir_mul(
            jnp.asarray(features_mul_ir), irreps_in_obj
        )
        features_torch = torch.tensor(np.asarray(features_ir_mul), dtype=torch.float32)

        with torch.no_grad():
            out_torch_ir_mul = torch_layer(features_torch).cpu().numpy()
        out_torch = cue_ir_mul_to_mul_ir(
            jnp.asarray(out_torch_ir_mul), Irreps(irreps_out)
        )

        module = JaxLinear(
            irreps_in=Irreps(irreps_in),
            irreps_out=Irreps(irreps_out),
            layout='ir_mul',
        )
        features_jax = jnp.asarray(features_mul_ir)
        variables = module.init(jax.random.PRNGKey(2), features_jax)
        variables = JaxLinear.import_from_torch(torch_layer, variables)
        out_jax = module.apply(variables, features_jax)

        np.testing.assert_allclose(
            np.asarray(out_jax),
            np.asarray(out_torch),
            rtol=1e-5,
            atol=1e-6,
        )
