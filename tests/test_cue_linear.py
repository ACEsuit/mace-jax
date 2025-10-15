import cuequivariance as cue
import jax
import jax.numpy as jnp
import numpy as np
import torch
from cuequivariance_torch.operations.linear import Linear as CueLinearTorch
from e3nn import o3
from e3nn_jax import Irreps

from mace_jax.adapters.cuequivariance.linear import Linear as CueLinearJAX
from mace_jax.adapters.cuequivariance.utility import (
    ir_mul_to_mul_ir as cue_ir_mul_to_mul_ir,
)
from mace_jax.adapters.cuequivariance.utility import (
    mul_ir_to_ir_mul as cue_mul_ir_to_ir_mul,
)
from mace_jax.modules.wrapper_ops import Linear as WrapperLinear


class TestCueLinear:
    """Ensure the cue Linear adapter mirrors the Torch implementation."""

    def test_forward_matches_torch(self):
        irreps_in_spec = '1x0e + 1x1o'
        irreps_out_spec = '2x0e'

        irreps_in = o3.Irreps(irreps_in_spec)
        irreps_out = o3.Irreps(irreps_out_spec)

        torch_layer = CueLinearTorch(
            cue.Irreps(cue.O3, irreps_in_spec),
            cue.Irreps(cue.O3, irreps_out_spec),
            layout=cue.mul_ir,
        )

        rng = np.random.default_rng(0)
        features_np = rng.standard_normal((3, irreps_in.dim))
        features_jax = jnp.array(features_np)
        features_torch = torch.tensor(features_np)

        with torch.no_grad():
            out_torch = torch_layer(features_torch).cpu().numpy()

        module = CueLinearJAX(
            irreps_in=Irreps(str(irreps_in)),
            irreps_out=Irreps(str(irreps_out)),
        )
        variables = module.init(jax.random.PRNGKey(5), features_jax)
        variables = CueLinearJAX.import_from_torch(torch_layer, variables)
        out_jax = module.apply(variables, features_jax)

        np.testing.assert_allclose(
            np.array(out_jax),
            out_torch,
            rtol=1e-5,
            atol=1e-6,
        )

    def test_forward_matches_torch_ir_mul_layout(self):
        irreps_in_spec = '2x0e + 1x1o'
        irreps_out_spec = '1x0e + 1x1o'

        irreps_in = o3.Irreps(irreps_in_spec)
        irreps_out = o3.Irreps(irreps_out_spec)

        torch_layer = CueLinearTorch(
            cue.Irreps(cue.O3, irreps_in_spec),
            cue.Irreps(cue.O3, irreps_out_spec),
            layout=cue.ir_mul,
        ).float()

        rng = np.random.default_rng(3)
        features_np = rng.standard_normal((4, irreps_in.dim), dtype=np.float32)

        # Torch module expects ir_mul layout when configured accordingly.
        features_ir_mul = cue_mul_ir_to_ir_mul(
            jnp.asarray(features_np), Irreps(str(irreps_in))
        )
        features_torch = torch.tensor(np.asarray(features_ir_mul), dtype=torch.float32)

        with torch.no_grad():
            out_torch_ir_mul = torch_layer(features_torch).cpu().numpy()

        out_torch = cue_ir_mul_to_mul_ir(
            jnp.asarray(out_torch_ir_mul), Irreps(str(irreps_out))
        )

        module = CueLinearJAX(
            irreps_in=Irreps(str(irreps_in)),
            irreps_out=Irreps(str(irreps_out)),
            layout='ir_mul',
        )
        features_jax = jnp.asarray(features_np)
        variables = module.init(jax.random.PRNGKey(2), features_jax)
        variables = CueLinearJAX.import_from_torch(torch_layer, variables)
        out_jax = module.apply(variables, features_jax)

        np.testing.assert_allclose(
            np.asarray(out_jax),
            np.asarray(out_torch),
            rtol=1e-5,
            atol=1e-6,
        )

    def test_wrapper_matches_o3_linear_without_cue(self):
        """When cue acceleration is disabled, wrapper should behave like o3.Linear."""

        irreps_in_spec = '4x0e + 4x1o + 2x2e'
        irreps_out_spec = '3x0e + 3x1o'

        irreps_in = o3.Irreps(irreps_in_spec)
        irreps_out = o3.Irreps(irreps_out_spec)

        torch_layer = o3.Linear(irreps_in, irreps_out).float()

        rng = np.random.default_rng(123)
        features_np = rng.standard_normal((6, irreps_in.dim)).astype(np.float32)
        features_torch = torch.tensor(features_np)
        with torch.no_grad():
            out_torch = torch_layer(features_torch).cpu().numpy()

        module = WrapperLinear(
            Irreps(irreps_in_spec),
            Irreps(irreps_out_spec),
        )
        features_jax = jnp.array(features_np)
        variables = module.init(jax.random.PRNGKey(0), features_jax)
        variables = module.import_from_torch(torch_layer, variables)
        out_jax = np.asarray(module.apply(variables, features_jax))

        np.testing.assert_allclose(
            out_jax,
            out_torch,
            rtol=1e-5,
            atol=1e-6,
        )
