import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import torch
from cuequivariance_torch.operations.linear import Linear as CueLinearTorch
from e3nn import o3

from mace_jax.adapters.cuequivariance.linear import Linear as CueLinearJAX
from mace_jax.haiku.torch import copy_torch_to_jax


class TestCueLinear:
    """Ensure the cue Linear adapter mirrors the Torch implementation."""

    def test_forward_matches_torch(self):
        irreps_in = o3.Irreps('1x0e + 1x1o')
        irreps_out = o3.Irreps('2x0e')

        torch_layer = CueLinearTorch(irreps_in, irreps_out)

        def forward_fn(x):
            layer = CueLinearJAX(
                irreps_in=irreps_in,
                irreps_out=irreps_out,
            )
            return layer(x)

        transformed = hk.transform(forward_fn)

        rng = np.random.default_rng(0)
        features_np = rng.standard_normal((3, irreps_in.dim)).astype(np.float32)
        features_jax = jnp.array(features_np)
        features_torch = torch.tensor(features_np, dtype=torch.float32)

        with torch.no_grad():
            out_torch = torch_layer(features_torch).cpu().numpy()

        params = transformed.init(jax.random.PRNGKey(5), features_jax)
        params = copy_torch_to_jax(torch_layer, params)
        out_jax = transformed.apply(params, None, features_jax)

        np.testing.assert_allclose(
            np.array(out_jax),
            out_torch,
            rtol=1e-5,
            atol=1e-6,
        )
