import jax.numpy as jnp
import numpy as np
import pytest
import torch
from e3nn.o3 import Irreps
from e3nn.o3._tensor_product._codegen import (
    _sum_tensors as sum_tensors_torch,
)
from e3nn.o3._tensor_product._codegen import (
    codegen_tensor_product_left_right as codegen_tensor_product_left_right_torch,
)

from mace_jax.e3nn._tensor_product._codegen import (
    _sum_tensors as sum_tensors_jax,
)
from mace_jax.e3nn._tensor_product._codegen import (
    codegen_tensor_product_left_right as codegen_tensor_product_left_right_jax,
)
from mace_jax.e3nn._tensor_product._instruction import Instruction


class TestSumTensors:
    @pytest.mark.parametrize("shape", [(2, 2), (3, 1), (2, 3, 4)])
    def test_multiple_tensors(self, shape):
        # Random tensors
        torch_tensors = [torch.randn(shape) for _ in range(3)]
        jax_tensors = [jnp.array(t.numpy()) for t in torch_tensors]

        # Compute results
        result_torch = sum_tensors_torch(torch_tensors, shape, torch_tensors[0])
        result_jax = sum_tensors_jax(jax_tensors, shape)

        # Compare
        np.testing.assert_allclose(
            result_torch.numpy(), np.array(result_jax), rtol=1e-6, atol=1e-6
        )

    def test_single_tensor(self):
        shape = (2, 3)
        torch_t = torch.randn(shape)
        jax_t = jnp.array(torch_t.numpy())

        result_torch = sum_tensors_torch([torch_t], shape, torch_t)
        result_jax = sum_tensors_jax([jax_t], shape)

        np.testing.assert_allclose(
            result_torch.numpy(), np.array(result_jax), rtol=1e-6, atol=1e-6
        )

    def test_empty_list(self):
        shape = (2, 2)
        like = torch.zeros(shape)
        result_torch = sum_tensors_torch([], shape, like)
        result_jax = sum_tensors_jax([], shape)
        np.testing.assert_allclose(
            result_torch.numpy(), np.array(result_jax), rtol=1e-6, atol=1e-6
        )


class TestTensorProduct:
    @pytest.fixture(autouse=True)
    def setup_irreps_instructions(self):
        # Setup real irreps
        self.irreps_in1 = Irreps("1x0e + 1x1o")
        self.irreps_in2 = Irreps("1x0e + 1x1o")
        self.irreps_out = Irreps("1x0e + 1x1o")

        # Setup example instructions
        # For simplicity, one instruction connecting first irrep of each input to first output
        self.instructions = [
            Instruction(
                i_in1=0,
                i_in2=0,
                i_out=0,
                has_weight=True,
                path_weight=1.0,
                connection_mode="uvw",
                path_shape=(
                    self.irreps_in1[0].dim,
                    self.irreps_in2[0].dim,
                    self.irreps_out[0].dim,
                ),
            )
        ]

        # Random weights and inputs
        self.batch_size = 4
        self.x1 = torch.randn(self.batch_size, self.irreps_in1.dim)
        self.x2 = torch.randn(self.batch_size, self.irreps_in2.dim)
        total_weight_numel = (
            self.irreps_in1[0].dim * self.irreps_in2[0].dim * self.irreps_out[0].dim
        )
        self.w = torch.randn(self.batch_size, total_weight_numel)

    def test_jax_matches_torch(self):
        # ------------------------
        # PyTorch FX version
        graphmod = codegen_tensor_product_left_right_torch(
            irreps_in1=self.irreps_in1,
            irreps_in2=self.irreps_in2,
            irreps_out=self.irreps_out,
            instructions=self.instructions,
            shared_weights=False,
            specialized_code=True,
            optimize_einsums=False,  # disable optimization for stable comparison
        )
        out_torch = graphmod(self.x1, self.x2, self.w).detach().cpu().numpy()

        # ------------------------
        # JAX version
        x1_j = jnp.array(self.x1.detach().cpu().numpy())
        x2_j = jnp.array(self.x2.detach().cpu().numpy())
        w_j = jnp.array(self.w.detach().cpu().numpy())

        out_jax = codegen_tensor_product_left_right_jax(
            irreps_in1=self.irreps_in1,
            irreps_in2=self.irreps_in2,
            irreps_out=self.irreps_out,
            instructions=self.instructions,
            x1=x1_j,
            x2=x2_j,
            weights=w_j,
            shared_weights=False,
            specialized_code=True,
        )
        out_jax_np = np.array(out_jax)

        # ------------------------
        # Compare outputs
        assert np.allclose(out_torch, out_jax_np, rtol=1e-5, atol=1e-6), (
            "JAX output differs from PyTorch output"
        )
