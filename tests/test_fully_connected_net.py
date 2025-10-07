import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import torch
from e3nn.nn import FullyConnectedNet as FullyConnectedNetTorch

from mace_jax.adapters.e3nn.nn._fc import FullyConnectedNet as FullyConnectedNetJAX
from mace_jax.haiku.torch import copy_torch_to_jax


class TestFullyConnectedNet:
    """Compare the Haiku FullyConnectedNet with the e3nn reference implementation."""

    def test_forward_matches_e3nn(self):
        hs = [6, 5, 4]
        variance_in = 1
        variance_out = 1
        out_act = False

        torch_net = FullyConnectedNetTorch(
            hs=hs,
            act=None,
            variance_in=variance_in,
            variance_out=variance_out,
            out_act=out_act,
        ).double()

        def forward_fn(x):
            net = FullyConnectedNetJAX(
                hs=hs,
                act=None,
                variance_in=variance_in,
                variance_out=variance_out,
                out_act=out_act,
            )
            return net(x)

        transformed = hk.transform(forward_fn)

        rng = np.random.default_rng(0)
        features_np = rng.standard_normal((4, hs[0])).astype(np.float64)
        features_jax = jnp.array(features_np, dtype=jnp.float32)
        features_torch = torch.tensor(features_np, dtype=torch.float64)

        with torch.no_grad():
            out_torch = torch_net(features_torch).cpu().numpy().astype(np.float32)

        params = transformed.init(jax.random.PRNGKey(7), features_jax)
        params = copy_torch_to_jax(torch_net, params)
        out_jax = transformed.apply(params, None, features_jax)
        out_jax_array = np.asarray(out_jax)

        np.testing.assert_allclose(
            out_jax_array,
            out_torch,
            rtol=1e-5,
            atol=1e-6,
        )
