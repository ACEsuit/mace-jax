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
    SymmetricContraction as SymmetricContractionJax,
)


class TestSymmetricContractionParity:
    @pytest.mark.parametrize(
        'irreps_in', [o3.Irreps('2x0e + 1x1o'), o3.Irreps('3x0e + 2x1o + 1x2e')]
    )
    @pytest.mark.parametrize(
        'irreps_out',
        [
            o3.Irreps('0e + 1x1o'),
            o3.Irreps('1e + 2o + 1x2e'),
            o3.Irreps('2e + 1o + 1x3e'),
        ],
    )
    @pytest.mark.parametrize('correlation', [1, 2])
    def test_forward(self, irreps_in, irreps_out, correlation):
        # === Set inputs ===
        irreps_in = o3.Irreps(irreps_in)
        irreps_out = o3.Irreps(irreps_out)

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
                num_elements=num_elements,
                method='naive',
            )
            outputs = model(x, y)
            block_params = jnp.array(
                [block.total_params for block in model.blocks], dtype=jnp.int32
            )
            block_active = jnp.array(
                [
                    1 if (block.total_params > 0 and block.poly is not None) else 0
                    for block in model.blocks
                ],
                dtype=jnp.int32,
            )
            return outputs, block_params, block_active

        forward = hk.transform(forward_fn)
        key = jax.random.PRNGKey(42)
        params = forward.init(key, x_j, y_j)
        params_dict = hk.data_structures.to_mutable_dict(params)
        if 'symmetric_contraction' in params_dict:
            params = copy_torch_to_jax(torch_model, params)
            params_dict = hk.data_structures.to_mutable_dict(params)

        # Torch inputs
        x_t = torch.tensor(np.array(x_j))
        y_t = torch.tensor(np.array(y_j))

        # Forward pass
        out_t = torch_model(x_t, y_t).detach().numpy()

        # Forward JAX with same params
        out_j, block_params, block_active = forward.apply(params, None, x_j, y_j)

        # Compare
        assert out_t.shape == out_j.shape
        assert np.allclose(out_t, np.array(out_j), atol=1e-5, rtol=1e-5)

        # Validate internal parameter allocation matches block metadata
        params_dict = hk.data_structures.to_mutable_dict(params)
        sc_params = params_dict.get('symmetric_contraction', {})
        for idx, (total, active) in enumerate(
            zip(np.array(block_params), np.array(block_active))  # type: ignore[arg-type]
        ):
            key = f'weights_{idx}'
            if active == 0 or total == 0:
                assert key not in sc_params
            else:
                assert sc_params[key].shape == (num_elements, total, num_feats)
