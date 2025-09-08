import pytest
import numpy as np
import torch
import jax
import jax.numpy as jnp
import haiku as hk

from mace.modules.radial import (
    BesselBasis as BesselBasisTorch,
    ChebychevBasis as ChebychevBasisTorch,
)
from mace_jax.modules.radial import (
    BesselBasis as BesselBasisJAX,
    ChebychevBasis as ChebychevBasisJAX,
)


class TestBesselBasisParity:
    @pytest.mark.parametrize("trainable", [False, True])
    @pytest.mark.parametrize("num_basis", [4, 8])
    def test_forward(self, trainable, num_basis):
        r_max = 5.0
        batch = 6

        # --------------------
        # JAX version
        # --------------------
        def forward_fn(x):
            model = BesselBasisJAX(
                r_max=r_max, num_basis=num_basis, trainable=trainable
            )
            return model(x)

        forward = hk.transform(forward_fn)

        key = jax.random.PRNGKey(0)
        x_j = jax.random.uniform(
            key, (batch, 1), minval=0.1, maxval=r_max
        )  # avoid division by 0

        params = forward.init(key, x_j)
        out_j = forward.apply(params, None, x_j)

        # --------------------
        # Torch version
        # --------------------
        model_torch = BesselBasisTorch(
            r_max=r_max, num_basis=num_basis, trainable=trainable
        )
        model_torch.eval()

        # Copy weights from JAX -> Torch if trainable
        if trainable:
            w_jax = params["bessel_basis"]["bessel_weights"]
            model_torch.bessel_weights.data = torch.tensor(
                np.array(w_jax), dtype=torch.float32
            )
        else:
            # For non-trainable, Torch has buffer already set, but we force sync
            w_jax = (
                np.pi / r_max * np.linspace(1.0, num_basis, num_basis, dtype=np.float32)
            )
            model_torch.bessel_weights.copy_(torch.tensor(w_jax, dtype=torch.float32))

        # Torch input
        x_t = torch.tensor(np.array(x_j), dtype=torch.float32)

        out_t = model_torch(x_t).detach().numpy()

        # --------------------
        # Compare outputs
        # --------------------
        assert out_t.shape == out_j.shape
        np.testing.assert_allclose(out_t, np.array(out_j), rtol=1e-6, atol=1e-6)


class TestChebychevBasisParity:
    @pytest.mark.parametrize("num_basis", [4, 8])
    @pytest.mark.parametrize("r_max", [3.0, 5.0])
    def test_forward(self, num_basis, r_max):
        batch = 5

        # --- Torch version ---
        model_torch = ChebychevBasisTorch(r_max=r_max, num_basis=num_basis)
        model_torch.eval()

        # Torch input
        x_torch = torch.rand(batch, 1, dtype=torch.get_default_dtype())
        out_torch = model_torch(x_torch).detach().numpy()

        # --- JAX version ---
        def forward_fn(x):
            model = ChebychevBasisJAX(r_max=r_max, num_basis=num_basis)
            return model(x)

        forward = hk.transform(forward_fn)

        key = jax.random.PRNGKey(0)
        x_jax = jnp.array(x_torch.detach().numpy())
        params = forward.init(key, x_jax)

        out_jax = np.array(forward.apply(params, None, x_jax))

        # --- Compare ---
        assert out_torch.shape == out_jax.shape
        np.testing.assert_allclose(out_jax, out_torch, rtol=1e-5, atol=1e-6)
