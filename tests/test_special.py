import pytest
import torch
import jax.numpy as jnp
import numpy as np

from mace_jax.modules.special import chebyshev_polynomial_t

class TestChebyshevPolynomialT:
    @pytest.mark.parametrize("n", [0, 1, 2, 3, 5, 10])
    @pytest.mark.parametrize("shape", [(5,), (2, 3), (4, 2, 2)])
    def test_parity(self, n, shape):
        # random x in [-2, 2] (to test inside and outside [-1, 1])
        rng = np.random.default_rng(0)
        x_np = rng.uniform(-2, 2, size=shape).astype(np.float32)

        # torch
        x_torch = torch.tensor(x_np, dtype=torch.float32)
        out_torch = torch.special.chebyshev_polynomial_t(x_torch, n).numpy()

        # jax
        x_jax = jnp.array(x_np)
        out_jax = np.array(chebyshev_polynomial_t(x_jax, n))

        # compare
        assert out_torch.shape == out_jax.shape
        np.testing.assert_allclose(out_jax, out_torch, rtol=1e-5, atol=1e-6)
