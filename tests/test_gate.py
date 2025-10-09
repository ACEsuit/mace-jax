import jax
import jax.numpy as jnp
import numpy as np
import torch
from e3nn import o3
from e3nn.nn import Gate as GateTorch
from e3nn_jax import Irreps, IrrepsArray

from mace_jax.adapters.e3nn.nn._gate import Gate as GateJAX


class TestGate:
    """Compare the Flax Gate module against the reference e3nn implementation."""

    def test_forward_matches_e3nn(self, monkeypatch):
        """Check the JAX gate matches e3nn.nn.Gate on the same parameters."""

        if not hasattr(IrrepsArray, 'from_array'):
            monkeypatch.setattr(
                IrrepsArray,
                'from_array',
                staticmethod(lambda irreps, array: IrrepsArray(irreps, array)),
                raising=False,
            )

        irreps_scalars = '2x0e + 1x0o'
        irreps_gates = '1x0e + 1x0o'
        irreps_gated = '1x1e + 1x2o'

        act_scalars = [None] * len(Irreps(irreps_scalars))
        act_gates = [None] * len(Irreps(irreps_gates))

        gate_torch = GateTorch(
            o3.Irreps(irreps_scalars),
            act_scalars,
            o3.Irreps(irreps_gates),
            act_gates,
            o3.Irreps(irreps_gated),
        )

        rng = np.random.default_rng(0)
        features_np = rng.standard_normal((4, gate_torch.irreps_in.dim))
        features_jax = jnp.array(features_np)
        features_torch = torch.tensor(features_np)

        with torch.no_grad():
            out_torch = gate_torch(features_torch).cpu().numpy()

        gate_flax = GateJAX(
            irreps_scalars=Irreps(irreps_scalars),
            act_scalars=act_scalars,
            irreps_gates=Irreps(irreps_gates),
            act_gates=act_gates,
            irreps_gated=Irreps(irreps_gated),
        )

        variables = gate_flax.init(jax.random.PRNGKey(42), features_jax)
        variables = GateJAX.import_from_torch(gate_torch, variables)
        out_jax = gate_flax.apply(variables, features_jax)
        out_jax_array = np.asarray(
            out_jax.array if hasattr(out_jax, 'array') else out_jax
        )

        np.testing.assert_allclose(
            out_jax_array,
            out_torch,
            rtol=1e-5,
            atol=1e-6,
        )
