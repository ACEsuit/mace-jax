import pytest
import torch
import jax.numpy as jnp
import jax
import haiku as hk

from e3nn import o3
from e3nn_jax import Irreps

from mace.modules.symmetric_contraction import Contraction as ContractionTorch
from mace_jax.modules.symmetric_contraction import Contraction as ContractionJax


class TestContractionParity:
    @pytest.mark.parametrize("correlation", [1, 2])
    @pytest.mark.parametrize("lmax", [0, 1])
    def test_forward(self, correlation, lmax):
        irreps_in = o3.Irreps("2x0e + 1x1o")
        irrep_out = o3.Irreps(f"{lmax}e")

        # Torch version
        model_torch = ContractionTorch(
            irreps_in=irreps_in,
            irrep_out=irrep_out,
            correlation=correlation,
            internal_weights=True,
            use_reduced_cg=False,
            num_elements=3,
        )
        model_torch.eval()

        batch = 4
        num_feats = model_torch.num_features  # from irreps_in
        num_elements = 3
        num_ell = model_torch.U_tensors(correlation).shape[-2]

        # Torch inputs
        x_t = torch.randn(batch, num_feats, num_ell)  # 3D input
        y_t = torch.randn(batch, num_elements)  # 2D input

        # JAX inputs
        x_j = jnp.array(x_t.detach().numpy())
        y_j = jnp.array(y_t.detach().numpy())

        # Torch forward
        out_t = model_torch(x_t, y_t).detach().numpy()

        # Haiku forward: wrap in hk.transform
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
        out_j = forward.apply(params, None, x_j, y_j)

        # Compare
        assert out_t.shape == out_j.shape
        assert jnp.allclose(out_j, out_t, atol=1e-5, rtol=1e-5)
