import importlib
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import torch
from e3nn import o3
from e3nn_jax import Irreps, IrrepsArray
from mace.modules.irreps_tools import (
    CuEquivarianceConfig as TorchCuEquivarianceConfig,
)
from mace.modules.irreps_tools import (
    reshape_irreps as TorchReshapeIrreps,
)


def _load_jax_irreps_tools():
    module_name = 'mace_jax.modules.irreps_tools'
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError:
        pkg_name = 'mace_jax.modules'
        if pkg_name not in sys.modules:
            modules_pkg = types.ModuleType(pkg_name)
            modules_pkg.__path__ = [
                str(Path(__file__).resolve().parents[1] / 'mace_jax/modules')
            ]
            sys.modules[pkg_name] = modules_pkg

        spec = importlib.util.spec_from_file_location(
            module_name,
            Path(__file__).resolve().parents[1] / 'mace_jax/modules/irreps_tools.py',
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module


_jax_irreps_tools = _load_jax_irreps_tools()
CuEquivarianceConfig = _jax_irreps_tools.CuEquivarianceConfig
reshape_irreps = _jax_irreps_tools.reshape_irreps


def _random_flat(batch: int, irreps: Irreps, rng: np.random.Generator) -> np.ndarray:
    flat_dim = sum(mul * ir.dim for mul, ir in irreps)
    return rng.standard_normal((batch, flat_dim), dtype=np.float32)


class TestReshapeIrreps:
    def test_matches_torch_default_layout(self):
        rng = np.random.default_rng(0)
        irreps = Irreps('1x0e + 1x1o + 1x2e')

        tensor_np = _random_flat(batch=4, irreps=irreps, rng=rng)

        torch_module = TorchReshapeIrreps(o3.Irreps(str(irreps)))
        torch_out = torch_module(torch.tensor(tensor_np, dtype=torch.float32))
        torch_np = torch_out.detach().cpu().numpy()

        jax_module = reshape_irreps(irreps)
        jax_np = np.asarray(jax_module(tensor_np))

        np.testing.assert_allclose(jax_np, torch_np, rtol=1e-6, atol=1e-7)

        # Ensure IrrepsArray input is also handled
        array = IrrepsArray(irreps, tensor_np)
        array_np = np.asarray(jax_module(array))
        np.testing.assert_allclose(array_np, torch_np, rtol=1e-6, atol=1e-7)

    def test_matches_torch_ir_mul_layout(self):
        rng = np.random.default_rng(1)
        irreps = Irreps('1x0e + 1x1o')

        tensor_np = _random_flat(batch=3, irreps=irreps, rng=rng)

        torch_cfg = TorchCuEquivarianceConfig(layout_str='ir_mul')
        torch_module = TorchReshapeIrreps(o3.Irreps(str(irreps)), cueq_config=torch_cfg)
        torch_out = torch_module(torch.tensor(tensor_np, dtype=torch.float32))
        torch_np = torch_out.detach().cpu().numpy()

        jax_cfg = CuEquivarianceConfig(layout_str='ir_mul')
        jax_module = reshape_irreps(irreps, cueq_config=jax_cfg)
        jax_np = np.asarray(jax_module(tensor_np))

        np.testing.assert_allclose(jax_np, torch_np, rtol=1e-6, atol=1e-7)
