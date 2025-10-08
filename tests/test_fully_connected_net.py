import jax
import jax.numpy as jnp
import numpy as np
import torch
from e3nn.nn import FullyConnectedNet as FullyConnectedNetTorch

from mace_jax.adapters.e3nn.nn._fc import FullyConnectedNet as FullyConnectedNetJAX


class TestFullyConnectedNet:
    """Compare the Flax FullyConnectedNet with the e3nn reference implementation."""

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
        ).float()

        rng = np.random.default_rng(0)
        features_np = rng.standard_normal((4, hs[0])).astype(np.float32)
        features_jax = jnp.asarray(features_np)
        features_torch = torch.tensor(features_np, dtype=torch.float32)

        with torch.no_grad():
            out_torch = torch_net(features_torch).cpu().numpy()

        flax_net = FullyConnectedNetJAX(
            hs=hs,
            act=None,
            variance_in=variance_in,
            variance_out=variance_out,
            out_act=out_act,
        )
        variables = flax_net.init(jax.random.PRNGKey(7), features_jax)
        variables = FullyConnectedNetJAX.import_from_torch(torch_net, variables)
        out_jax = flax_net.apply(variables, features_jax)
        out_jax_array = np.asarray(out_jax)

        np.testing.assert_allclose(
            out_jax_array,
            out_torch,
            rtol=1e-5,
            atol=1e-6,
        )
