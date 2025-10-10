import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from e3nn import o3
from e3nn_jax import Irreps
from mace.modules.wrapper_ops import CuEquivarianceConfig, SymmetricContractionWrapper

from mace_jax.adapters.cuequivariance.symmetric_contraction import SymmetricContraction


@pytest.fixture
def module_and_params():
    module = SymmetricContraction(
        irreps_in=Irreps('1x0e'),
        irreps_out=Irreps('1x0e'),
        correlation=1,
        num_elements=2,
    )
    inputs = jnp.zeros((1, 1, 1))
    indices = jnp.zeros((1,), dtype=jnp.int32)
    params = module.init(jax.random.PRNGKey(0), inputs, indices)
    return module, params


class TestSymmetricContractionImport:
    def test_import_cue_equivariant_weights(self, module_and_params):
        module, params = module_and_params
        torch_module = SymmetricContractionWrapper(
            irreps_in=o3.Irreps('1x0e'),
            irreps_out=o3.Irreps('1x0e'),
            correlation=1,
            num_elements=2,
            cueq_config=CuEquivarianceConfig(
                enabled=True,
                layout='mul_ir',
                optimize_symmetric=True,
            ),
        )

        imported = module.import_from_torch(torch_module, params)

        torch_weight = torch_module.weight.detach().cpu().numpy()
        jax_weight = np.asarray(imported['params']['weight'])
        np.testing.assert_allclose(jax_weight, torch_weight, rtol=1e-6, atol=1e-7)

    def test_forward_matches_cue_torch(self, module_and_params):
        module, params = module_and_params

        torch_module = SymmetricContractionWrapper(
            irreps_in=o3.Irreps('1x0e'),
            irreps_out=o3.Irreps('1x0e'),
            correlation=1,
            num_elements=2,
            cueq_config=CuEquivarianceConfig(
                enabled=True,
                layout='mul_ir',
                optimize_symmetric=True,
            ),
        ).eval()

        variables = module.import_from_torch(torch_module, params)

        irreps_in = Irreps(module.irreps_in)
        mul = irreps_in[0].mul
        feature_dim = sum(term.ir.dim for term in irreps_in)

        batch = 5
        inputs = jax.random.normal(
            jax.random.PRNGKey(1),
            (batch, mul, feature_dim),
            dtype=jnp.float32,
        )
        indices = jnp.array([0, 1, 0, 1, 0], dtype=jnp.int32)

        jax_output = module.apply(variables, inputs, indices)
        jax_output_np = np.asarray(jax_output).reshape(batch, -1)

        torch_inputs = torch.tensor(
            np.asarray(inputs).reshape(batch, -1),
            dtype=torch.float32,
        )
        torch_indices = torch.tensor(np.asarray(indices), dtype=torch.long)
        with torch.no_grad():
            torch_output = torch_module(torch_inputs, torch_indices).cpu().numpy()

        np.testing.assert_allclose(jax_output_np, torch_output, rtol=1e-6, atol=1e-6)

    def test_import_native_sym_contraction_raises(self, module_and_params):
        module, params = module_and_params
        native_module = SymmetricContractionWrapper(
            irreps_in=o3.Irreps('1x0e'),
            irreps_out=o3.Irreps('1x0e'),
            correlation=1,
            num_elements=2,
        )

        with pytest.raises(NotImplementedError, match='--only_cueq=True'):
            module.import_from_torch(native_module, params)
