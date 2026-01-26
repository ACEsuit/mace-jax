import jax
import jax.nn as jnn
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from e3nn.nn import FullyConnectedNet as FullyConnectedNetTorch
from flax import nnx

from mace_jax.adapters.e3nn.nn._fc import FullyConnectedNet as FullyConnectedNetJAX
from mace_jax.nnx_utils import state_to_pure_dict


class TestFullyConnectedNet:
    """Compare the Flax FullyConnectedNet with the e3nn reference implementation."""

    @pytest.mark.parametrize(
        ('torch_act', 'jax_act', 'out_act'),
        [
            (None, None, False),
            (torch.tanh, jnp.tanh, False),
            (torch.tanh, jnp.tanh, True),
            (torch.nn.SiLU(), jnn.silu, False),
            (torch.nn.SiLU(), jnn.silu, True),
        ],
    )
    def test_forward_matches_e3nn(self, torch_act, jax_act, out_act):
        hs = [6, 5, 4]
        variance_in = 1
        variance_out = 1

        torch_net = FullyConnectedNetTorch(
            hs=hs,
            act=torch_act,
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
            act=jax_act,
            variance_in=variance_in,
            variance_out=variance_out,
            out_act=out_act,
            rngs=nnx.Rngs(7),
        )
        graphdef, state = nnx.split(flax_net)
        variables = state_to_pure_dict(state)
        variables = FullyConnectedNetJAX.import_from_torch(torch_net, variables)
        out_jax, _ = graphdef.apply(variables)(features_jax)
        out_jax_array = np.asarray(out_jax)

        np.testing.assert_allclose(
            out_jax_array,
            out_torch,
            rtol=1e-5,
            atol=1e-6,
        )
