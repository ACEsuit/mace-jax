import cuequivariance as cue
import jax
import jax.numpy as jnp
import numpy as np
import torch
from cuequivariance_torch.operations.linear import Linear as CueLinearTorch
from e3nn import o3
from e3nn_jax import Irreps

from mace_jax.adapters.cuequivariance.linear import Linear as CueLinearJAX


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
