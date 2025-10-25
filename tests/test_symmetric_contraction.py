import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from e3nn import o3
from e3nn_jax import Irreps
from mace.modules.wrapper_ops import CuEquivarianceConfig, SymmetricContractionWrapper

from mace_jax.adapters.cuequivariance.symmetric_contraction import SymmetricContraction
from mace_jax.tools.device import configure_torch_runtime


def _jax_float_dtype() -> jnp.dtype:
    return jnp.asarray(0.0).dtype


def _module_dtype(module: torch.nn.Module) -> torch.dtype:
    for param in module.parameters():
        return param.dtype
    for buf in module.buffers():
        return buf.dtype
    return torch.get_default_dtype()


def _available_torch_devices() -> list[str]:
    devices = ['cpu']
    if torch.cuda.is_available():
        devices.append('cuda')
    return devices


def _validate_cue_on_device(device: torch.device) -> None:
    cfg = CuEquivarianceConfig(enabled=True, optimize_symmetric=True)
    module = SymmetricContractionWrapper(
        irreps_in=o3.Irreps('1x0e'),
        irreps_out=o3.Irreps('1x0e'),
        correlation=1,
        num_elements=2,
        cueq_config=cfg,
    ).to(device).eval()

    dtype = _module_dtype(module)
    x = torch.zeros((1, 1), dtype=dtype, device=device)
    indices = torch.zeros((1,), dtype=torch.long, device=device)
    with torch.no_grad():
        module(x, indices)


@pytest.fixture(params=_available_torch_devices())
def torch_device(request):
    device_str: str = request.param
    try:
        torch_device = configure_torch_runtime(device_str)
    except RuntimeError:
        pytest.skip('Requested Torch device is not available.')

    if torch_device.type == 'cuda':
        try:
            _validate_cue_on_device(torch_device)
        except Exception as exc:  # pragma: no cover - environment dependent
            pytest.skip(f'cuequivariance kernels unavailable on CUDA: {exc}')
    return torch_device


@pytest.fixture
def module_and_params():
    module = SymmetricContraction(
        irreps_in=Irreps('1x0e'),
        irreps_out=Irreps('1x0e'),
        correlation=1,
        num_elements=2,
    )
    inputs = jnp.zeros((1, 1, 1), dtype=_jax_float_dtype())
    indices = jnp.zeros((1,), dtype=jnp.int32)
    params = module.init(jax.random.PRNGKey(0), inputs, indices)
    return module, params


class TestSymmetricContractionImport:
    def test_forward_matches_cue_torch(self, module_and_params, torch_device):
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
        ).to(torch_device).eval()
        dtype = _module_dtype(torch_module)

        variables = module.import_from_torch(torch_module, params)

        irreps_in = Irreps(module.irreps_in)
        mul = irreps_in[0].mul
        feature_dim = sum(term.ir.dim for term in irreps_in)

        batch = 5
        inputs = jax.random.normal(
            jax.random.PRNGKey(1),
            (batch, mul, feature_dim),
            dtype=_jax_float_dtype(),
        )
        indices = jnp.array([0, 1, 0, 1, 0], dtype=jnp.int32)

        jax_output = module.apply(variables, inputs, indices)
        jax_output_np = np.asarray(jax_output).reshape(batch, -1)

        inputs_ir_mul = np.swapaxes(inputs, 1, 2)
        torch_inputs = torch.tensor(
            np.asarray(inputs_ir_mul).reshape(batch, -1),
            dtype=dtype,
            device=torch_device,
        )
        torch_indices = torch.tensor(
            np.asarray(indices),
            dtype=torch.long,
            device=torch_device,
        )
        with torch.no_grad():
            torch_output = torch_module(torch_inputs, torch_indices).cpu().numpy()

        np.testing.assert_allclose(jax_output_np, torch_output, rtol=1e-6, atol=1e-6)

    def test_import_native_sym_contraction_weights_reduced_cg(self, torch_device):
        irreps_in = Irreps('2x0e + 2x1o + 2x2e')
        irreps_out = Irreps('2x0e + 2x1o')
        correlation = 3
        num_elements = 3

        module = SymmetricContraction(
            irreps_in=irreps_in,
            irreps_out=irreps_out,
            correlation=correlation,
            num_elements=num_elements,
            use_reduced_cg=True,
        )

        mul = irreps_in[0].mul
        feature_dim = sum(term.ir.dim for term in irreps_in)
        init_inputs = jnp.zeros((1, mul, feature_dim), dtype=_jax_float_dtype())
        init_indices = jnp.zeros((1,), dtype=jnp.int32)
        params = module.init(jax.random.PRNGKey(0), init_inputs, init_indices)

        torch_module = SymmetricContractionWrapper(
            irreps_in=o3.Irreps(str(irreps_in)),
            irreps_out=o3.Irreps(str(irreps_out)),
            correlation=correlation,
            num_elements=num_elements,
            use_reduced_cg=True,
        ).to(torch_device).eval()
        dtype = _module_dtype(torch_module)

        variables = module.import_from_torch(torch_module, params)

        batch = 4
        rng_feats, rng_indices = jax.random.split(jax.random.PRNGKey(2))
        inputs = jax.random.normal(
            rng_feats,
            (batch, mul, feature_dim),
            dtype=_jax_float_dtype(),
        )
        indices = jax.random.randint(
            rng_indices,
            (batch,),
            0,
            num_elements,
        ).astype(jnp.int32)

        jax_output = module.apply(variables, inputs, indices)
        jax_output_np = np.asarray(jax_output).reshape(batch, -1)

        torch_inputs = torch.tensor(
            np.asarray(inputs),
            dtype=dtype,
            device=torch_device,
        )
        torch_attrs = torch.tensor(
            np.eye(num_elements, dtype=np.float32)[np.asarray(indices)],
            dtype=dtype,
            device=torch_device,
        )
        with torch.no_grad():
            torch_output = (
                torch_module(torch_inputs, torch_attrs).cpu().numpy().reshape(batch, -1)
            )

        np.testing.assert_allclose(
            jax_output_np,
            torch_output,
            rtol=5e-4,
            atol=5e-3,
        )

    def test_forward_matches_cue_torch_large_irreps(self, torch_device):
        if torch_device.type != 'cuda':
            pytest.skip('cue-equivariance large irreps test requires CUDA backend.')
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

        init_inputs = jnp.zeros((1, mul, feature_dim), dtype=_jax_float_dtype())
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
        ).to(torch_device).eval()
        dtype = _module_dtype(torch_module)

        variables = module.import_from_torch(torch_module, params)

        batch = 3
        rng_feats, rng_indices = jax.random.split(jax.random.PRNGKey(1))
        inputs = jax.random.normal(
            rng_feats,
            (batch, mul, feature_dim),
            dtype=_jax_float_dtype(),
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
            dtype=dtype,
            device=torch_device,
        )
        torch_indices = torch.tensor(
            np.asarray(indices),
            dtype=torch.long,
            device=torch_device,
        )

        with torch.no_grad():
            torch_output = (
                torch_module(torch_inputs, torch_indices)
                .cpu()
                .numpy()
                .reshape(batch, -1)
            )

        np.testing.assert_allclose(jax_output_np, torch_output, rtol=1e-5, atol=1e-5)

    def test_forward_matches_native_low_correlation(self, torch_device):
        irreps_in = Irreps('1x0e + 1x1o')
        irreps_out = Irreps('1x0e + 1x1o')
        correlation = 1
        num_elements = 3

        module = SymmetricContraction(
            irreps_in=irreps_in,
            irreps_out=irreps_out,
            correlation=correlation,
            num_elements=num_elements,
        )

        mul = irreps_in[0].mul
        feature_dim = sum(term.ir.dim for term in irreps_in)

        init_inputs = jnp.zeros((1, mul, feature_dim), dtype=_jax_float_dtype())
        init_indices = jnp.zeros((1,), dtype=jnp.int32)
        params = module.init(jax.random.PRNGKey(0), init_inputs, init_indices)

        torch_module = SymmetricContractionWrapper(
            irreps_in=o3.Irreps(str(irreps_in)),
            irreps_out=o3.Irreps(str(irreps_out)),
            correlation=correlation,
            num_elements=num_elements,
        ).to(torch_device).eval()
        dtype = _module_dtype(torch_module)

        variables = module.import_from_torch(torch_module, params)

        batch = 4
        rng_feats, rng_indices = jax.random.split(jax.random.PRNGKey(1))
        inputs = jax.random.normal(
            rng_feats,
            (batch, mul, feature_dim),
            dtype=_jax_float_dtype(),
        )
        indices = jax.random.randint(
            rng_indices,
            (batch,),
            0,
            num_elements,
        ).astype(jnp.int32)

        jax_output = module.apply(variables, inputs, indices)
        jax_output_np = np.asarray(jax_output).reshape(batch, -1)

        torch_inputs = torch.tensor(
            np.asarray(inputs),
            dtype=dtype,
            device=torch_device,
        )
        torch_attrs = torch.tensor(
            np.eye(num_elements, dtype=np.float32)[np.asarray(indices)],
            dtype=dtype,
            device=torch_device,
        )

        with torch.no_grad():
            torch_output = (
                torch_module(torch_inputs, torch_attrs).cpu().numpy().reshape(batch, -1)
            )

        np.testing.assert_allclose(jax_output_np, torch_output, rtol=1e-6, atol=1e-6)

    def test_forward_matches_native_high_correlation(self, torch_device):
        """Expected to fail until native high-order import is harmonised."""

        irreps_in = Irreps('2x0e + 2x1o + 2x2e')
        irreps_out = Irreps('2x0e + 2x1o')
        correlation = 2
        num_elements = 4

        module = SymmetricContraction(
            irreps_in=irreps_in,
            irreps_out=irreps_out,
            correlation=correlation,
            num_elements=num_elements,
            use_reduced_cg=False,
        )

        mul = irreps_in[0].mul
        feature_dim = sum(term.ir.dim for term in irreps_in)

        init_inputs = jnp.zeros((1, mul, feature_dim), dtype=_jax_float_dtype())
        init_indices = jnp.zeros((1,), dtype=jnp.int32)
        params = module.init(jax.random.PRNGKey(0), init_inputs, init_indices)

        torch_module = SymmetricContractionWrapper(
            irreps_in=o3.Irreps(str(irreps_in)),
            irreps_out=o3.Irreps(str(irreps_out)),
            correlation=correlation,
            num_elements=num_elements,
            use_reduced_cg=False,
        ).to(torch_device).eval()
        dtype = _module_dtype(torch_module)

        variables = module.import_from_torch(torch_module, params)

        batch = 3
        rng_feats, rng_indices = jax.random.split(jax.random.PRNGKey(1))
        inputs = jax.random.normal(
            rng_feats,
            (batch, mul, feature_dim),
            dtype=_jax_float_dtype(),
        )
        indices = jax.random.randint(
            rng_indices,
            (batch,),
            0,
            num_elements,
        ).astype(jnp.int32)

        jax_output = module.apply(variables, inputs, indices)
        jax_output_np = np.asarray(jax_output).reshape(batch, -1)

        torch_inputs = torch.tensor(
            np.asarray(inputs),
            dtype=dtype,
            device=torch_device,
        )
        torch_attrs = torch.tensor(
            np.eye(num_elements, dtype=np.float32)[np.asarray(indices)],
            dtype=dtype,
            device=torch_device,
        )

        with torch.no_grad():
            torch_output = (
                torch_module(torch_inputs, torch_attrs).cpu().numpy().reshape(batch, -1)
            )

        np.testing.assert_allclose(jax_output_np, torch_output, rtol=1e-5, atol=1e-4)
