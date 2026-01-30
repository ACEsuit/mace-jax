import jax
import jax.numpy as jnp
import numpy as np
import torch
from e3nn import o3
from e3nn_jax import Irreps, IrrepsArray
from flax import nnx
from mace.modules.blocks import NonLinearReadoutBlock as TorchReadout
from torch.serialization import add_safe_globals

from mace_jax.adapters.nnx.torch import init_from_torch
from mace_jax.modules.blocks import NonLinearReadoutBlock as JaxReadout
from mace_jax.nnx_utils import state_to_pure_dict

add_safe_globals([slice])


def _as_irreps_array(irreps: Irreps, array: jnp.ndarray) -> IrrepsArray:
    if hasattr(IrrepsArray, 'from_array'):
        return IrrepsArray.from_array(irreps, array)
    return IrrepsArray(irreps, array)


class TestReadoutFixture:
    def test_non_linear_readout_components_parity(self):
        """Step-by-step parity for linear_1 -> activation -> linear_2."""
        rng = np.random.default_rng(1)
        prod = rng.normal(size=(3, 128)).astype(np.float32)

        irreps_in = Irreps('128x0e')
        irreps_out = Irreps('16x0e')

        torch_readout = TorchReadout(
            irreps_in=o3.Irreps(str(irreps_in)),
            MLP_irreps=o3.Irreps(str(irreps_out)),
            gate=torch.nn.functional.silu,
            num_heads=1,
        ).float()
        torch_readout.eval()

        prod_t = torch.tensor(prod)
        torch_lin1 = torch_readout.linear_1(prod_t).detach().cpu().numpy()
        torch_act = (
            torch_readout.non_linearity(torch_readout.linear_1(prod_t))
            .detach()
            .cpu()
            .numpy()
        )
        torch_out = torch_readout(prod_t).detach().cpu().numpy()

        jax_readout = JaxReadout(
            irreps_in=irreps_in,
            MLP_irreps=irreps_out,
            gate=jax.nn.silu,
            num_heads=1,
            rngs=nnx.Rngs(0),
        )
        prod_jax = _as_irreps_array(irreps_in, jnp.asarray(prod))
        jax_readout, _ = init_from_torch(jax_readout, torch_readout)
        graphdef, state = nnx.split(jax_readout)
        module = nnx.merge(graphdef, state)

        lin1 = module.linear_1(prod_jax)
        act = module.non_linearity(lin1)
        out = module.linear_2(act)

        lin1_np = np.asarray(lin1.array if hasattr(lin1, 'array') else lin1)
        act_np = np.asarray(act.array if hasattr(act, 'array') else act)
        out_np = np.asarray(out.array if hasattr(out, 'array') else out)

        np.testing.assert_allclose(lin1_np, torch_lin1, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(act_np, torch_act, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(out_np, torch_out, rtol=1e-6, atol=1e-6)

    def test_non_linear_readout_scalar_parity_foundation_shapes(self):
        """Parity on foundation-style scalar readout weights."""

        rng = np.random.default_rng(0)
        # Match foundation product_1 scalar shape: (batch_nodes, 128) scalars
        prod = rng.normal(size=(8, 128)).astype(np.float32)

        irreps_in = Irreps('128x0e')
        irreps_out = Irreps('16x0e')

        torch_readout = TorchReadout(
            irreps_in=o3.Irreps(str(irreps_in)),
            MLP_irreps=o3.Irreps(str(irreps_out)),
            gate=torch.nn.functional.silu,
            num_heads=1,
        ).float()

        # Generate fixed weights for determinism.
        w1 = rng.normal(size=tuple(torch_readout.linear_1.weight.shape)).astype(
            np.float32
        )
        w2 = rng.normal(size=tuple(torch_readout.linear_2.weight.shape)).astype(
            np.float32
        )
        with torch.no_grad():
            torch_readout.linear_1.weight.copy_(torch.tensor(w1))
            torch_readout.linear_2.weight.copy_(torch.tensor(w2))
        torch_readout.eval()
        out_torch = torch_readout(torch.tensor(prod)).detach().cpu().numpy()

        jax_readout = JaxReadout(
            irreps_in=irreps_in,
            MLP_irreps=irreps_out,
            gate=jax.nn.silu,
            num_heads=1,
            rngs=nnx.Rngs(0),
        )
        prod_jax = _as_irreps_array(irreps_in, jnp.asarray(prod))
        graphdef, state = nnx.split(jax_readout)
        params = state_to_pure_dict(state)
        params['linear_1']['weight'] = jnp.asarray(w1).reshape(
            params['linear_1']['weight'].shape
        )
        params['linear_2']['weight'] = jnp.asarray(w2).reshape(
            params['linear_2']['weight'].shape
        )

        out_jax, _ = graphdef.apply(params)(prod_jax)
        out_jax_arr = np.asarray(
            out_jax.array if hasattr(out_jax, 'array') else out_jax
        )

        # Parity check; bias here would mirror the foundation delta if present.
        np.testing.assert_allclose(out_jax_arr, out_torch, rtol=1e-6, atol=1e-6)

    def test_non_linear_readout_scalar_parity(self):
        """Force parity on scalar-only NonLinearReadoutBlock using random weights."""
        rng = np.random.default_rng(0)
        prod = rng.normal(size=(4, 128)).astype(np.float32)

        irreps_in = Irreps('128x0e')
        irreps_out = Irreps('16x0e')

        torch_readout = TorchReadout(
            irreps_in=o3.Irreps(str(irreps_in)),
            MLP_irreps=o3.Irreps(str(irreps_out)),
            gate=torch.nn.functional.silu,
            num_heads=1,
        ).float()

        # Initial torch output
        out_torch = torch_readout(torch.tensor(prod)).detach().cpu().numpy()

        jax_readout = JaxReadout(
            irreps_in=irreps_in,
            MLP_irreps=irreps_out,
            gate=jax.nn.silu,
            num_heads=1,
            rngs=nnx.Rngs(0),
        )
        prod_jax = _as_irreps_array(irreps_in, jnp.asarray(prod))
        jax_readout, _ = init_from_torch(jax_readout, torch_readout)
        graphdef, state = nnx.split(jax_readout)
        out_jax, _ = graphdef.apply(state)(prod_jax)
        out_jax_arr = np.asarray(
            out_jax.array if hasattr(out_jax, 'array') else out_jax
        )

        np.testing.assert_allclose(out_jax_arr, out_torch, rtol=1e-6, atol=1e-6)

    def test_fixture_mismatch(self):
        # Synthetic but deterministic fixture (4 nodes x 128 scalars)
        rng = np.random.default_rng(0)
        prod = rng.normal(size=(4, 128)).astype(np.float32)

        irreps_in = Irreps('128x0e')
        irreps_out = Irreps('16x0e')

        torch_readout = TorchReadout(
            irreps_in=o3.Irreps(str(irreps_in)),
            MLP_irreps=o3.Irreps(str(irreps_out)),
            gate=torch.nn.functional.silu,
            num_heads=1,
        ).float()

        # Generate weights that match the actual parameter shapes of the Torch block
        w1 = rng.normal(size=tuple(torch_readout.linear_1.weight.shape)).astype(
            np.float32
        )
        w2 = rng.normal(size=tuple(torch_readout.linear_2.weight.shape)).astype(
            np.float32
        )

        with torch.no_grad():
            torch_readout.linear_1.weight.copy_(torch.tensor(w1))
            torch_readout.linear_2.weight.copy_(torch.tensor(w2))
        torch_readout.eval()
        out_torch = torch_readout(torch.tensor(prod)).detach().cpu().numpy()

        jax_readout = JaxReadout(
            irreps_in=irreps_in,
            MLP_irreps=irreps_out,
            gate=jax.nn.silu,
            num_heads=1,
            rngs=nnx.Rngs(0),
        )
        prod_jax = _as_irreps_array(irreps_in, jnp.asarray(prod))
        graphdef, state = nnx.split(jax_readout)
        params = state_to_pure_dict(state)
        params['linear_1']['weight'] = jnp.asarray(w1).reshape(
            params['linear_1']['weight'].shape
        )
        params['linear_2']['weight'] = jnp.asarray(w2).reshape(
            params['linear_2']['weight'].shape
        )

        out_jax, _ = graphdef.apply(params)(prod_jax)
        out_jax_arr = np.asarray(
            out_jax.array if hasattr(out_jax, 'array') else out_jax
        )

        # Parity check; if there is a discrepancy it will surface here.
        np.testing.assert_allclose(out_jax_arr, out_torch, rtol=1e-6, atol=1e-6)
