import cuequivariance as cue
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from cuequivariance_torch.operations.tp_fully_connected import (
    FullyConnectedTensorProduct as CueFullyConnectedTensorProductTorch,
)
from e3nn import o3
from e3nn.o3._tensor_product._sub import (
    FullyConnectedTensorProduct as FullyConnectedTensorProductTorch,
)
from e3nn_jax import Irreps

from mace_jax.adapters.cuequivariance.fully_connected_tensor_product import (
    FullyConnectedTensorProduct as FullyConnectedTensorProductJAX,
)


class TestFullyConnectedTensorProductImport:
    rng = np.random.default_rng(0)
    batch = 4

    @pytest.mark.parametrize(
        'shared_weights, internal_weights',
        [
            pytest.param(True, True, id='shared_internal'),
            pytest.param(True, False, id='shared_external'),
            pytest.param(False, False, id='unshared_external'),
        ],
    )
    def test_e3nn_fully_connected_forward(self, shared_weights, internal_weights):
        irreps_in1 = o3.Irreps('1x0e + 1x1o')
        irreps_in2 = o3.Irreps('1x0e')
        irreps_out = o3.Irreps('1x0e + 1x1o')

        if internal_weights and not shared_weights:
            pytest.skip('internal weights require shared weights')

        torch_module = FullyConnectedTensorProductTorch(
            irreps_in1,
            irreps_in2,
            irreps_out,
            shared_weights=shared_weights,
            internal_weights=internal_weights,
        )

        self._assert_forward_matches(
            torch_module=torch_module,
            irreps_in1=str(irreps_in1),
            irreps_in2=str(irreps_in2),
            irreps_out=str(irreps_out),
            shared_weights=shared_weights,
            internal_weights=internal_weights,
            backend='e3nn',
        )

    @pytest.mark.parametrize(
        'shared_weights, internal_weights',
        [
            pytest.param(True, True, id='shared_internal'),
            pytest.param(True, False, id='shared_external'),
            pytest.param(False, False, id='unshared_external'),
        ],
    )
    def test_cue_fully_connected_forward(self, shared_weights, internal_weights):
        irreps_in1 = '1x0e + 1x1o'
        irreps_in2 = '1x0e'
        irreps_out = '1x0e + 1x1o'

        if internal_weights and not shared_weights:
            pytest.skip('internal weights require shared weights')

        cue_irreps_in1 = cue.Irreps(cue.O3, irreps_in1)
        cue_irreps_in2 = cue.Irreps(cue.O3, irreps_in2)
        cue_irreps_out = cue.Irreps(cue.O3, irreps_out)

        torch_module = CueFullyConnectedTensorProductTorch(
            cue_irreps_in1,
            cue_irreps_in2,
            cue_irreps_out,
            shared_weights=shared_weights,
            internal_weights=internal_weights,
            method='naive',
            layout=cue.mul_ir,
        )

        self._assert_forward_matches(
            torch_module=torch_module,
            irreps_in1=irreps_in1,
            irreps_in2=irreps_in2,
            irreps_out=irreps_out,
            shared_weights=shared_weights,
            internal_weights=internal_weights,
            backend='cue',
        )

    def _assert_forward_matches(
        self,
        *,
        torch_module,
        irreps_in1: str,
        irreps_in2: str,
        irreps_out: str,
        shared_weights: bool,
        internal_weights: bool,
        backend: str,
    ) -> None:
        key = jax.random.PRNGKey(123)

        ir_in1 = Irreps(irreps_in1)
        ir_in2 = Irreps(irreps_in2)
        ir_out = Irreps(irreps_out)

        x1_np = self.rng.standard_normal((self.batch, ir_in1.dim))
        x2_np = self.rng.standard_normal((self.batch, ir_in2.dim))
        x1_jax = jnp.array(x1_np)
        x2_jax = jnp.array(x2_np)

        weight_numel = torch_module.weight_numel

        module = FullyConnectedTensorProductJAX(
            ir_in1,
            ir_in2,
            ir_out,
            shared_weights=shared_weights,
            internal_weights=internal_weights,
        )

        if internal_weights:
            variables = module.init(key, x1_jax, x2_jax)
            variables = FullyConnectedTensorProductJAX.import_from_torch(
                torch_module, variables
            )
            out_jax = module.apply(variables, x1_jax, x2_jax)
            weights_torch = None
        else:
            if shared_weights:
                if backend == 'cue':
                    weights_np = self.rng.standard_normal((1, weight_numel))
                else:
                    weights_np = self.rng.standard_normal((weight_numel,))
            else:
                weights_np = self.rng.standard_normal((self.batch, weight_numel))

            weights_jax = jnp.array(weights_np)
            variables = module.init(key, x1_jax, x2_jax, weights=weights_jax)
            out_jax = module.apply(variables, x1_jax, x2_jax, weights=weights_jax)
            weights_torch = torch.tensor(weights_np)

        x1_torch = torch.tensor(x1_np)
        x2_torch = torch.tensor(x2_np)
        with torch.no_grad():
            if internal_weights:
                out_torch = torch_module(x1_torch, x2_torch).cpu().numpy()
            else:
                out_torch = (
                    torch_module(x1_torch, x2_torch, weights_torch).cpu().numpy()
                )

        np.testing.assert_allclose(
            np.array(out_jax),
            out_torch,
            rtol=1e-5,
            atol=1e-6,
        )
