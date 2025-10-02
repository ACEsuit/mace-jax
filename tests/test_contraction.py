import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from e3nn import o3
from e3nn_jax import Irreps
from mace.modules.symmetric_contraction import (
    Contraction as ContractionTorch,
)
from mace.modules.symmetric_contraction import (
    SymmetricContraction as SymmetricContractionTorch,
)

from mace_jax.haiku.torch import copy_torch_to_jax
from mace_jax.modules.symmetric_contraction import (
    Contraction as ContractionJax,
)
from mace_jax.modules.symmetric_contraction import (
    SymmetricContraction as SymmetricContractionJax,
)


class TestContractionParity:
    @pytest.mark.parametrize('correlation', [1, 2])
    @pytest.mark.parametrize('lmax', [0, 1])
    def test_forward(self, correlation, lmax):
        # === Set inputs ===
        irreps_in = o3.Irreps('2x0e + 1x1o')
        irrep_out = o3.Irreps(f'{lmax}e')

        # Torch version
        torch_model = ContractionTorch(
            irreps_in=irreps_in,
            irrep_out=irrep_out,
            correlation=correlation,
            internal_weights=True,
            use_reduced_cg=False,
            num_elements=3,
        )
        torch_model.eval()

        batch = 4
        num_feats = torch_model.num_features  # from irreps_in
        num_elements = 3
        num_ell = torch_model.U_tensors(correlation).shape[-2]

        # Torch inputs
        x_t = torch.randn(batch, num_feats, num_ell)
        y_t = torch.randn(batch, num_elements)

        # JAX inputs
        x_j = jnp.array(x_t.detach().numpy())
        y_j = jnp.array(y_t.detach().numpy())

        # --- JAX model ---
        def forward_fn(x, y):
            model = ContractionJax(
                irreps_in=Irreps(str(irreps_in)),
                irrep_out=Irreps(str(irrep_out)),
                correlation=correlation,
                internal_weights=True,
                use_reduced_cg=False,
                num_elements=3,
            )
            return model(x, y)

        forward = hk.transform(forward_fn)
        key = jax.random.PRNGKey(0)
        params = forward.init(key, x_j, y_j)
        params = copy_torch_to_jax(torch_model, params)

        # --- Forward passes ---
        out_t = torch_model(x_t, y_t).detach().numpy()
        out_j = forward.apply(params, None, x_j, y_j)

        # --- Compare ---
        assert out_t.shape == out_j.shape
        assert jnp.allclose(out_j, out_t, atol=1e-5, rtol=1e-5)


class TestSymmetricContractionParity:
    @pytest.mark.parametrize('correlation', [1, 2])
    @pytest.mark.parametrize('lmax', [0, 1, 2])
    def test_forward(self, correlation, lmax):
        # === Set inputs ===
        irreps_in = o3.Irreps('2x0e + 1x1o')
        irreps_out = o3.Irreps(f'{lmax}e + 1x1o')  # multi-output test

        batch = 4
        num_elements = 3
        num_feats = irreps_in.count((0, 1))

        # --- PyTorch version ---
        torch_model = SymmetricContractionTorch(
            irreps_in=irreps_in,
            irreps_out=irreps_out,
            correlation=correlation,
            shared_weights=False,
            use_reduced_cg=False,
            num_elements=num_elements,
        )
        torch_model.eval()
        num_ell = torch_model.contractions[0].U_tensors(correlation).shape[-2]

        x_j = jax.random.normal(jax.random.PRNGKey(0), (batch, num_feats, num_ell))
        y_j = jax.random.normal(jax.random.PRNGKey(1), (batch, num_elements))

        def forward_fn(x, y):
            model = SymmetricContractionJax(
                irreps_in=Irreps(str(irreps_in)),
                irreps_out=Irreps(str(irreps_out)),
                correlation=correlation,
                shared_weights=False,
                use_reduced_cg=False,
                num_elements=num_elements,
            )
            return model(x, y)

        forward = hk.transform(forward_fn)
        key = jax.random.PRNGKey(42)
        params = forward.init(key, x_j, y_j)
        params = copy_torch_to_jax(torch_model, params)

        # Torch inputs
        x_t = torch.tensor(np.array(x_j))
        y_t = torch.tensor(np.array(y_j))

        # Forward pass
        out_t = torch_model(x_t, y_t).detach().numpy()

        # Forward JAX with same params
        out_j = forward.apply(params, None, x_j, y_j)

        # Compare
        assert out_t.shape == out_j.shape
        assert np.allclose(out_t, np.array(out_j), atol=1e-5, rtol=1e-5)
