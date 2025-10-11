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

        inputs_ir_mul = np.swapaxes(inputs, 1, 2)
        torch_inputs = torch.tensor(
            np.asarray(inputs_ir_mul).reshape(batch, -1),
            dtype=torch.float32,
        )
        torch_indices = torch.tensor(np.asarray(indices), dtype=torch.long)
        with torch.no_grad():
            torch_output = torch_module(torch_inputs, torch_indices).cpu().numpy()

        np.testing.assert_allclose(jax_output_np, torch_output, rtol=1e-6, atol=1e-6)

    def test_import_native_sym_contraction_weights(self, module_and_params):
        module, params = module_and_params
        native_module = SymmetricContractionWrapper(
            irreps_in=o3.Irreps('1x0e'),
            irreps_out=o3.Irreps('1x0e'),
            correlation=1,
            num_elements=2,
        )

        imported = module.import_from_torch(native_module, params)

        torch_weight = native_module.weight.detach().cpu().numpy()
        jax_weight = np.asarray(imported['params']['weight'])
        np.testing.assert_allclose(jax_weight, torch_weight, rtol=1e-6, atol=1e-7)

    def test_forward_matches_cue_torch_large_irreps(self):
        irreps_in = Irreps('128x0e + 128x1o + 128x2e + 128x3o')
        irreps_out = Irreps('128x0e + 128x1o + 128x2e')
        correlation = 3
        num_elements = 10

        module = SymmetricContraction(
            irreps_in=irreps_in,
            irreps_out=irreps_out,
            correlation=correlation,
            num_elements=num_elements,
        )

        mul = irreps_in[0].mul
        feature_dim = sum(term.ir.dim for term in irreps_in)

        init_inputs = jnp.zeros((1, mul, feature_dim), dtype=jnp.float32)
        init_indices = jnp.zeros((1,), dtype=jnp.int32)
        params = module.init(jax.random.PRNGKey(0), init_inputs, init_indices)

        torch_module = SymmetricContractionWrapper(
            irreps_in=o3.Irreps(str(irreps_in)),
            irreps_out=o3.Irreps(str(irreps_out)),
            correlation=correlation,
            num_elements=num_elements,
            cueq_config=CuEquivarianceConfig(
                enabled=True,
                layout='mul_ir',
                optimize_symmetric=True,
            ),
        ).eval()

        variables = module.import_from_torch(torch_module, params)

        batch = 3
        rng_feats, rng_indices = jax.random.split(jax.random.PRNGKey(1))
        inputs = jax.random.normal(
            rng_feats,
            (batch, mul, feature_dim),
            dtype=jnp.float32,
        )
        indices = jax.random.randint(
            rng_indices,
            (batch,),
            0,
            num_elements,
        ).astype(jnp.int32)

        jax_output = module.apply(variables, inputs, indices)
        jax_output_np = np.asarray(jax_output).reshape(batch, -1)

        inputs_ir_mul = np.swapaxes(inputs, 1, 2)
        torch_inputs = torch.tensor(
            np.asarray(inputs_ir_mul).reshape(batch, -1),
            dtype=torch.float32,
        )
        torch_indices = torch.tensor(np.asarray(indices), dtype=torch.long)

        with torch.no_grad():
            torch_output = (
                torch_module(torch_inputs, torch_indices)
                .cpu()
                .numpy()
                .reshape(batch, -1)
            )

        np.testing.assert_allclose(jax_output_np, torch_output, rtol=1e-5, atol=1e-5)
