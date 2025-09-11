import jax.numpy as jnp
import numpy as np
import pytest
import torch
from e3nn.o3 import Irreps
from e3nn.o3._tensor_product._codegen import (
    codegen_tensor_product_left_right as codegen_tensor_product_left_right_torch,
)

from mace_jax.e3nn._tensor_product._codegen import (
    codegen_tensor_product_left_right as codegen_tensor_product_left_right_jax,
)
from mace_jax.e3nn._tensor_product._instruction import Instruction


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
        self.w = torch.randn(
            self.irreps_in1[0].dim * self.irreps_in2[0].dim * self.irreps_out[0].dim
        )

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
