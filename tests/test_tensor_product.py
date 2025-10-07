import cuequivariance as cue
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from cuequivariance_torch.operations.tp_channel_wise import (
    ChannelWiseTensorProduct as CueTensorProductTorch,
)
from e3nn import o3
from e3nn.o3 import TensorProduct as TensorProductTorch
from e3nn_jax import Irreps

from mace_jax.adapters.cuequivariance.tensor_product import (
    TensorProduct as TensorProductJAX,
)
from mace_jax.haiku.torch import copy_torch_to_jax


class TestTensorProductImport:
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
    def test_e3nn_tensor_product_forward(self, shared_weights, internal_weights):
        irreps_in1 = o3.Irreps('1x0e + 1x1o')
        irreps_in2 = o3.Irreps('1x0e')
        irreps_out = o3.Irreps('1x0e + 1x1o')
        instructions = [
            (0, 0, 0, 'uvu', True),
            (1, 0, 1, 'uvu', True),
        ]

        torch_module = TensorProductTorch(
            irreps_in1,
            irreps_in2,
            irreps_out,
            instructions,
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
            instructions=instructions,
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
    def test_cue_tensor_product_forward(self, shared_weights, internal_weights):
        irreps_in1 = '1x0e + 1x1o'
        irreps_in2 = '1x0e'
        irreps_out = '1x0e + 1x1o'

        cue_irreps_in1 = cue.Irreps(cue.O3, irreps_in1)
        cue_irreps_in2 = cue.Irreps(cue.O3, irreps_in2)
        cue_irreps_out = cue.Irreps(cue.O3, irreps_out)

        torch_module = CueTensorProductTorch(
            cue_irreps_in1,
            cue_irreps_in2,
            [entry.ir for entry in cue_irreps_out],
            shared_weights=shared_weights,
            internal_weights=internal_weights,
            method='naive',
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
        instructions=None,
    ) -> None:
        key = jax.random.PRNGKey(42)

        jax_irreps_in1 = Irreps(irreps_in1)
        jax_irreps_in2 = Irreps(irreps_in2)
        jax_irreps_out = Irreps(irreps_out)

        dim_in1 = jax_irreps_in1.dim
        dim_in2 = jax_irreps_in2.dim

        x1_np = self.rng.standard_normal((self.batch, dim_in1))
        x2_np = self.rng.standard_normal((self.batch, dim_in2))

        x1_jax = jnp.array(x1_np)
        x2_jax = jnp.array(x2_np)

        weight_numel = torch_module.weight_numel

        if internal_weights:
            weights_jax = None
            weights_torch = None

            def forward_fn(x1, x2):
                module = TensorProductJAX(
                    irreps_in1=jax_irreps_in1,
                    irreps_in2=jax_irreps_in2,
                    irreps_out=jax_irreps_out,
                    shared_weights=shared_weights,
                    internal_weights=internal_weights,
                    instructions=instructions,
                )
                return module(x1, x2)

            transformed = hk.transform(forward_fn)
            params = transformed.init(key, x1_jax, x2_jax)
            params = copy_torch_to_jax(torch_module, params, scope='tensor_product')
            out_jax = transformed.apply(params, None, x1_jax, x2_jax)
        else:
            if shared_weights:
                if backend == 'cue':
                    weights_np = self.rng.standard_normal((1, weight_numel))
                else:
                    weights_np = self.rng.standard_normal((weight_numel,))
            else:
                weights_np = self.rng.standard_normal((self.batch, weight_numel))

            weights_jax = jnp.array(weights_np)
            weights_torch = torch.tensor(weights_np)

            def forward_fn(x1, x2):
                module = TensorProductJAX(
                    irreps_in1=jax_irreps_in1,
                    irreps_in2=jax_irreps_in2,
                    irreps_out=jax_irreps_out,
                    shared_weights=shared_weights,
                    internal_weights=internal_weights,
                    instructions=instructions,
                )
                return module(x1, x2, weights=weights_jax)

            transformed = hk.transform(forward_fn)
            params = transformed.init(key, x1_jax, x2_jax)
            out_jax = transformed.apply(params, None, x1_jax, x2_jax)

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
