import pytest
import torch
import jax
import jax.numpy as jnp
import numpy as np
import haiku as hk

from e3nn.o3 import Irreps
from e3nn.o3 import TensorProduct as TensorProductTorch

from mace_jax.e3nn.tensor_product import TensorProduct as TensorProductJax


class TestTensorProductParity:
    @pytest.mark.parametrize("batch", [1, 3])
    def test_forward_parity(self, batch):
        irreps_in1 = Irreps("1x0e + 1x1o")
        irreps_in2 = Irreps("1x0e + 1x1o")
        irreps_out = Irreps("1x0e + 1x1o")

        instructions = [
            (0, 0, 0, "uuu", True, 1.0),  # 0e ⊗ 0e -> 0e ✓
            (0, 1, 1, "uuu", True, 1.0),  # 0e ⊗ 1o -> 1o ✓
            (1, 0, 1, "uuu", True, 1.0),  # 1o ⊗ 0e -> 1o ✓
        ]

        x_np = np.random.randn(batch, irreps_in1.dim).astype(np.float32)
        y_np = np.random.randn(batch, irreps_in2.dim).astype(np.float32)

        x_t = torch.tensor(x_np)
        y_t = torch.tensor(y_np)

        x_j = jnp.array(x_np)
        y_j = jnp.array(y_np)

        # --- JAX TensorProduct inside hk.transform ---
        def forward_fn(x, y):
            model = TensorProductJax(
                irreps_in1=irreps_in1,
                irreps_in2=irreps_in2,
                irreps_out=irreps_out,
                instructions=instructions,
                shared_weights=True,
            )
            return model(x, y)

        forward = hk.transform(forward_fn)
        key = jax.random.PRNGKey(0)
        params_jax = forward.init(key, x_j, y_j)
        out_j = forward.apply(params_jax, None, x_j, y_j)

        # --- PyTorch TensorProduct ---
        model_torch = TensorProductTorch(
            irreps_in1=irreps_in1,
            irreps_in2=irreps_in2,
            irreps_out=irreps_out,
            instructions=instructions,
            shared_weights=True,
        )
        model_torch.eval()

        # Copy weights from JAX -> PyTorch
        if model_torch.weight_numel > 0:
            model_torch.weight.data = torch.tensor(
                np.array(params_jax["tensor_product"]["weight"]), dtype=torch.float32
            )

        out_t = model_torch(x_t, y_t).detach().numpy()

        # --- Compare ---
        assert out_j.shape == out_t.shape
        assert np.allclose(out_j, out_t, atol=1e-5, rtol=1e-5)
