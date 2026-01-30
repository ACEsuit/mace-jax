import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
import cuequivariance as cue
from e3nn import o3
from e3nn.o3 import TensorProduct as TensorProductTorch
from e3nn_jax import Irreps
from flax import nnx
from mace.modules.irreps_tools import tp_out_irreps_with_instructions

from mace_jax.adapters.cuequivariance.tensor_product import (
    TensorProduct as TensorProductJAX,
)
from mace_jax.adapters.nnx.torch import init_from_torch


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
    @pytest.mark.parametrize(
        'backend, irreps_in1, irreps_in2, irreps_out',
        [
            pytest.param(
                'e3nn',
                '1x0e + 1x1o',
                '1x0e',
                '1x0e + 1x1o',
                id='e3nn_low_l',
            ),
            pytest.param(
                'e3nn',
                '3x1e',
                '1x0e + 1x1e + 1x2e',
                '3x0e + 6x1e + 3x2e',
                id='e3nn_high_l',
            ),
            pytest.param(
                'cue',
                '1x0e + 1x1o',
                '1x0e',
                '1x0e + 1x1o',
                id='cue_low_l',
            ),
        ],
    )
    def test_tensor_product_forward(
        self,
        backend,
        irreps_in1,
        irreps_in2,
        irreps_out,
        shared_weights,
        internal_weights,
    ):
        if backend == 'e3nn':
            irreps_in1_obj = o3.Irreps(irreps_in1)
            irreps_in2_obj = o3.Irreps(irreps_in2)
            irreps_out_obj = o3.Irreps(irreps_out)
            target_irreps, instructions = tp_out_irreps_with_instructions(
                irreps_in1_obj, irreps_in2_obj, irreps_out_obj
            )

            torch_module = TensorProductTorch(
                irreps_in1_obj,
                irreps_in2_obj,
                target_irreps,
                instructions,
                shared_weights=shared_weights,
                internal_weights=internal_weights,
            )

            irreps_in1_str = str(irreps_in1_obj)
            irreps_in2_str = str(irreps_in2_obj)
            irreps_out_str = str(target_irreps)
        elif backend == 'cue':
            try:
                from cuequivariance_torch.operations.tp_channel_wise import (
                    ChannelWiseTensorProduct as CueTensorProductTorch,
                )
            except Exception as exc:  # pragma: no cover - optional backend
                pytest.skip(f'cuequivariance_torch unavailable: {exc}')

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
                layout=cue.mul_ir,
            )

            irreps_in1_str = irreps_in1
            irreps_in2_str = irreps_in2
            irreps_out_str = irreps_out
            instructions = None
        else:
            raise ValueError(f'Unknown backend: {backend}')

        self._assert_forward_matches(
            torch_module=torch_module,
            irreps_in1=irreps_in1_str,
            irreps_in2=irreps_in2_str,
            irreps_out=irreps_out_str,
            shared_weights=shared_weights,
            internal_weights=internal_weights,
            instructions=instructions,
            backend=backend,
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

        module = TensorProductJAX(
            irreps_in1=jax_irreps_in1,
            irreps_in2=jax_irreps_in2,
            irreps_out=jax_irreps_out,
            shared_weights=shared_weights,
            internal_weights=internal_weights,
            instructions=instructions,
            rngs=nnx.Rngs(key),
        )

        if internal_weights:
            module, _ = init_from_torch(module, torch_module)
            graphdef, state = nnx.split(module)
            out_jax, _ = graphdef.apply(state)(x1_jax, x2_jax)
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
            graphdef, state = nnx.split(module)
            out_jax, _ = graphdef.apply(state)(x1_jax, x2_jax, weights=weights_jax)
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
