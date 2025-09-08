import numpy as np
import pytest
import torch
import jax.numpy as jnp
import jax
import haiku as hk

from e3nn import o3
from e3nn_jax import Irreps

from mace.modules.symmetric_contraction import (
    Contraction as ContractionTorch,
    SymmetricContraction as SymmetricContractionTorch,
)
from mace_jax.modules.symmetric_contraction import (
    Contraction as ContractionJax,
    SymmetricContraction as SymmetricContractionJax,
)


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

        # --- Transfer weights from JAX → Torch ---
        jax_wmax = np.array(params["contraction"]["weights_max"])
        jax_ws = [
            np.array(params["contraction"][f"weights_{i + 1}"])
            for i in range(len(model_torch.weights))
        ]

        with torch.no_grad():
            model_torch.weights_max.copy_(torch.tensor(jax_wmax))
            for tw, jw in zip(model_torch.weights, jax_ws):
                tw.copy_(torch.tensor(jw))

        # --- Forward passes ---
        out_t = model_torch(x_t, y_t).detach().numpy()
        out_j = forward.apply(params, None, x_j, y_j)

        # --- Compare ---
        assert out_t.shape == out_j.shape
        assert jnp.allclose(out_j, out_t, atol=1e-5, rtol=1e-5)


class TestSymmetricContractionParity:
    @pytest.mark.parametrize("correlation", [1, 2])
    @pytest.mark.parametrize("lmax", [0, 1, 2])
    def test_forward(self, correlation, lmax):
        irreps_in = o3.Irreps("2x0e + 1x1o")
        irreps_out = o3.Irreps(f"{lmax}e + 1x1o")  # multi-output test

        batch = 4
        num_elements = 3
        num_feats = irreps_in.count((0, 1))

        # --- PyTorch version ---
        model_torch = SymmetricContractionTorch(
            irreps_in=irreps_in,
            irreps_out=irreps_out,
            correlation=correlation,
            shared_weights=False,
            use_reduced_cg=False,
            num_elements=num_elements,
        )
        model_torch.eval()
        num_ell = model_torch.contractions[0].U_tensors(correlation).shape[-2]

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

        # --- Copy JAX params -> PyTorch ---
        for idx, contraction_torch in enumerate(model_torch.contractions):
            contraction_name = f"contraction_{idx}" if idx > 0 else "contraction"
            contraction_jax = params[f"symmetric_contraction/{contraction_name}"]

            # Copy weights_max
            w_max = contraction_jax["weights_max"]
            contraction_torch.weights_max.data = torch.tensor(
                np.array(w_max), dtype=torch.float32
            )

            # Copy lower-correlation weights
            for key_name, w_jax in contraction_jax.items():
                if key_name == "weights_max":
                    continue
                if key_name.startswith("weights_"):
                    i = int(key_name.split("_")[1])
                    contraction_torch.weights[i - 1].data = torch.tensor(
                        np.array(w_jax), dtype=torch.float32
                    )

        # Torch inputs
        x_t = torch.tensor(np.array(x_j), dtype=torch.float32)
        y_t = torch.tensor(np.array(y_j), dtype=torch.float32)

        # Forward pass
        out_t = model_torch(x_t, y_t).detach().numpy()

        # Forward JAX with same params
        out_j = forward.apply(params, None, x_j, y_j)

        # Compare
        assert out_t.shape == out_j.shape
        assert np.allclose(out_t, np.array(out_j), atol=1e-5, rtol=1e-5)
