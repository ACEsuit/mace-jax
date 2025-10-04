from math import prod

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from e3nn.o3 import FullyConnectedTensorProduct as FullyConnectedTensorProductTorch
from e3nn.o3 import Irreps
from e3nn.o3 import Irreps as IrrepsTorch
from e3nn.o3 import SphericalHarmonics as SphericalHarmonicsTorch
from e3nn.o3 import TensorProduct as TensorProductTorch
from e3nn.o3._linear import Linear as LinearTorch
from e3nn_jax import Irreps as IrrepsJAX
from e3nn_jax import spherical_harmonics

from mace_jax.e3nn._linear import Linear as LinearJax
from mace_jax.e3nn._tensor_product._tensor_product import (
    TensorProduct as TensorProductJAX,
)
from mace_jax.e3nn.o3 import SphericalHarmonics as SphericalHarmonicsJAX
from mace_jax.modules.irreps_tools import tp_out_irreps_with_instructions
from mace_jax.modules.wrapper_ops import (
    CuEquivarianceConfig,
    FullyConnectedTensorProduct,
)
from mace_jax.modules.wrapper_ops import (
    TensorProduct as WrapperTensorProduct,
)


class TestFullyConnectedTensorProduct:
    _CASES = [
        ('1x0e + 1x1o', '1x0e + 1x1o', '1x0e + 1x1o + 1x2e', 4, 0),
        ('1x0e + 2x1o', '1x0e + 2x1o', '1x0e + 2x1o + 1x2e', 3, 1),
        ('2x0e + 1x1o', '1x0e + 1x1o', '2x0e + 2x1o', 2, 2),
    ]

    @pytest.mark.parametrize(
        'irreps_in1_str,irreps_in2_str,irreps_out_str,batch,seed', _CASES
    )
    def test_torch_vs_jax(
        self,
        irreps_in1_str: str,
        irreps_in2_str: str,
        irreps_out_str: str,
        batch: int,
        seed: int,
    ) -> None:
        irreps_in1 = Irreps(irreps_in1_str)
        irreps_in2 = Irreps(irreps_in2_str)
        irreps_out = Irreps(irreps_out_str)

        rng = np.random.default_rng(seed)
        x1_np = rng.standard_normal((batch, irreps_in1.dim))
        x2_np = rng.standard_normal((batch, irreps_in2.dim))

        x1_torch = torch.tensor(x1_np, dtype=torch.float64)
        x2_torch = torch.tensor(x2_np, dtype=torch.float64)

        tp_torch = FullyConnectedTensorProductTorch(
            irreps_in1,
            irreps_in2,
            irreps_out,
            shared_weights=True,
            internal_weights=False,
        )
        weight_np = rng.standard_normal(tp_torch.weight_numel)
        weight_torch = torch.tensor(weight_np, dtype=torch.float64)

        x1_jax = jnp.array(x1_np, dtype=jnp.float64)
        x2_jax = jnp.array(x2_np, dtype=jnp.float64)
        weight_jax = jnp.array(weight_np, dtype=jnp.float64)

        def forward_fn(x1_, x2_, w_):
            tp = FullyConnectedTensorProduct(
                irreps_in1,
                irreps_in2,
                irreps_out,
                shared_weights=True,
                internal_weights=False,
            )
            return tp(x1_, x2_, w_)

        hk_tp = hk.without_apply_rng(hk.transform(forward_fn))
        params = hk_tp.init(jax.random.PRNGKey(seed + 7), x1_jax, x2_jax, weight_jax)

        out_torch = tp_torch(x1_torch, x2_torch, weight_torch).detach().cpu().numpy()
        out_jax = hk_tp.apply(params, x1_jax, x2_jax, weight_jax)
        np.testing.assert_allclose(out_torch, np.array(out_jax), rtol=1e-5, atol=1e-5)


class TestTensorProductWithIrrepsTool:
    _CASES = [
        ('1x0e + 1x1o', '1x0e + 1x1o', '1x0e + 1x1o + 1x2e', 4, 0),
        ('1x0e + 2x1o', '1x0e + 2x1o', '1x0e + 2x1o + 1x2e', 3, 1),
        ('2x0e + 1x1o', '1x0e + 1x1o', '2x0e + 2x1o', 2, 2),
    ]

    @staticmethod
    def _prepare_case(
        irreps_in1_str: str,
        irreps_in2_str: str,
        target_irreps_str: str,
        *,
        batch: int,
        seed: int,
    ) -> dict:
        irreps_in1 = Irreps(irreps_in1_str)
        irreps_in2 = Irreps(irreps_in2_str)
        target_irreps = Irreps(target_irreps_str)

        irreps_out, instructions = tp_out_irreps_with_instructions(
            irreps_in1, irreps_in2, target_irreps
        )
        instructions = list(instructions)

        rng = np.random.default_rng(seed)
        x1_np = rng.standard_normal((batch, irreps_in1.dim))
        x2_np = rng.standard_normal((batch, irreps_in2.dim))

        weight_numel = 0
        for i_in1, i_in2, i_out, mode, has_weight in instructions:
            if not has_weight:
                continue
            if mode != 'uvu':
                raise AssertionError(
                    'tp_out_irreps_with_instructions returned an unsupported mode.'
                )
            weight_numel += irreps_in1[i_in1].mul * irreps_in2[i_in2].mul

        weight_np = rng.standard_normal(weight_numel)

        x1_torch = torch.tensor(x1_np, dtype=torch.float64)
        x2_torch = torch.tensor(x2_np, dtype=torch.float64)
        weight_torch = torch.tensor(weight_np, dtype=torch.float64)

        x1_jax = jnp.array(x1_np, dtype=jnp.float64)
        x2_jax = jnp.array(x2_np, dtype=jnp.float64)
        weight_jax = jnp.array(weight_np, dtype=jnp.float64)

        cue_config = CuEquivarianceConfig(enabled=True)

        tp_torch = TensorProductTorch(
            IrrepsTorch(str(irreps_in1)),
            IrrepsTorch(str(irreps_in2)),
            IrrepsTorch(str(irreps_out)),
            instructions,
            shared_weights=True,
            internal_weights=False,
        )

        def jax_forward(x1_, x2_, w_):
            tp = TensorProductJAX(
                irreps_in1,
                irreps_in2,
                irreps_out,
                instructions=instructions,
                shared_weights=True,
                internal_weights=False,
            )
            return tp(x1_, x2_, w_)

        def cue_forward(x1_, x2_, w_):
            tp = WrapperTensorProduct(
                irreps_in1,
                irreps_in2,
                irreps_out,
                instructions=instructions,
                shared_weights=True,
                internal_weights=False,
                cueq_config=cue_config,
            )
            return tp(x1_, x2_, w_)

        hk_tp = hk.without_apply_rng(hk.transform(jax_forward))
        hk_cue = hk.without_apply_rng(hk.transform(cue_forward))
        key = jax.random.PRNGKey(seed + 11)
        key_jax, key_cue = jax.random.split(key)
        params_jax = hk_tp.init(key_jax, x1_jax, x2_jax, weight_jax)
        params_cue = hk_cue.init(key_cue, x1_jax, x2_jax, weight_jax)

        return {
            'tp_torch': tp_torch,
            'x1_torch': x1_torch,
            'x2_torch': x2_torch,
            'weight_torch': weight_torch,
            'hk_tp': hk_tp,
            'params_jax': params_jax,
            'hk_cue': hk_cue,
            'params_cue': params_cue,
            'x1_jax': x1_jax,
            'x2_jax': x2_jax,
            'weight_jax': weight_jax,
        }

    @pytest.mark.parametrize(
        'irreps_in1_str,irreps_in2_str,target_irreps_str,batch,seed', _CASES
    )
    def test_torch_vs_jax(
        self,
        irreps_in1_str: str,
        irreps_in2_str: str,
        target_irreps_str: str,
        batch: int,
        seed: int,
    ) -> None:
        setup = self._prepare_case(
            irreps_in1_str,
            irreps_in2_str,
            target_irreps_str,
            batch=batch,
            seed=seed,
        )

        out_torch = (
            setup['tp_torch'](
                setup['x1_torch'], setup['x2_torch'], setup['weight_torch']
            )
            .detach()
            .cpu()
            .numpy()
        )
        out_jax = setup['hk_tp'].apply(
            setup['params_jax'],
            setup['x1_jax'],
            setup['x2_jax'],
            setup['weight_jax'],
        )
        np.testing.assert_allclose(out_torch, np.array(out_jax), rtol=1e-5, atol=1e-5)

    @pytest.mark.parametrize(
        'irreps_in1_str,irreps_in2_str,target_irreps_str,batch,seed', _CASES
    )
    def test_torch_vs_cue_wrapper(
        self,
        irreps_in1_str: str,
        irreps_in2_str: str,
        target_irreps_str: str,
        batch: int,
        seed: int,
    ) -> None:
        setup = self._prepare_case(
            irreps_in1_str,
            irreps_in2_str,
            target_irreps_str,
            batch=batch,
            seed=seed,
        )

        out_torch = (
            setup['tp_torch'](
                setup['x1_torch'], setup['x2_torch'], setup['weight_torch']
            )
            .detach()
            .cpu()
            .numpy()
        )
        out_cue = setup['hk_cue'].apply(
            setup['params_cue'],
            setup['x1_jax'],
            setup['x2_jax'],
            setup['weight_jax'],
        )
        np.testing.assert_allclose(out_torch, np.array(out_cue), rtol=1e-5, atol=1e-5)


class TestCodegenLinear:
    @staticmethod
    def compute_weight_bias_numel(instructions, irreps_in, irreps_out):
        weight_numel = sum(
            prod(ins.path_shape) for ins in instructions if ins.i_in != -1
        )
        bias_numel = sum(prod(ins.path_shape) for ins in instructions if ins.i_in == -1)
        return weight_numel, bias_numel

    @pytest.mark.parametrize('batch_size,f_in,f_out', [(2, 3, 4)])
    def test_block_sparse_linear(self, batch_size, f_in, f_out):
        # Define irreps
        irreps_in = Irreps('4x0e + 3x1o')
        irreps_out = Irreps('4x0e + 3x1o')

        # ========== PyTorch Linear ==========
        linear_torch = LinearTorch(
            irreps_in,
            irreps_out,
        )
        weight_numel = linear_torch.weight_numel
        bias_numel = linear_torch.bias_numel

        # Create random inputs (PyTorch)
        x_torch = torch.randn(batch_size, f_in, irreps_in.dim, dtype=torch.float32)
        ws_torch = torch.randn(weight_numel, dtype=torch.float32)
        bs_torch = torch.randn(bias_numel, dtype=torch.float32)

        # Use the same weights and biases
        linear_torch.weight = torch.nn.Parameter(ws_torch)
        linear_torch.bias = torch.nn.Parameter(bs_torch)

        out_torch = linear_torch(x_torch)
        out_torch_np = out_torch.detach().numpy()

        # ========== JAX/Haiku Linear ==========
        # Convert to JAX
        x_jax = jnp.array(x_torch.numpy())
        ws_jax = jnp.array(ws_torch.numpy())
        bs_jax = jnp.array(bs_torch.numpy())

        def forward_fn(x, w, b):
            linear = LinearJax(irreps_in, irreps_out, shared_weights=True)
            return linear(x, w, b)

        linear_transformed = hk.without_apply_rng(hk.transform(forward_fn))

        # Initialize parameters
        params = linear_transformed.init(jax.random.PRNGKey(42), x_jax, ws_jax, bs_jax)

        # Apply the Linear layer
        out_jax = linear_transformed.apply(params, x_jax, ws_jax, bs_jax)

        # Compare outputs
        assert jnp.allclose(out_jax, out_torch_np, atol=1e-5), (
            f'JAX and PyTorch outputs differ:\nJAX: {out_jax}\nPyTorch: {out_torch_np}'
        )


class TestSphericalHarmonicsParity:
    def setup_method(self):
        self.max_ell = 2
        self.irreps_torch = IrrepsTorch.spherical_harmonics(self.max_ell)
        self.irreps_jax = IrrepsJAX.spherical_harmonics(self.max_ell)

    @pytest.mark.parametrize('normalize', [False, True])
    @pytest.mark.parametrize(
        'vec',
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
        ],
    )
    def test_jax_vs_torch(self, vec, normalize):
        vec_np = np.array(vec, dtype=np.float32)[None, :]  # [1,3]

        # Torch version (module interface)
        sh_torch = SphericalHarmonicsTorch(
            self.irreps_torch, normalize=normalize, normalization='component'
        )
        x_torch = torch.tensor(vec_np)
        y_torch = sh_torch(x_torch).detach().numpy()

        # JAX version (functional)
        x_jax = jnp.array(vec_np)
        y_jax = spherical_harmonics(
            self.irreps_jax, x_jax, normalize=normalize, normalization='component'
        ).array

        # Compare
        np.testing.assert_allclose(
            np.array(y_jax),
            y_torch,
            rtol=1e-5,
            atol=1e-6,
        )


class TestSphericalHarmonicsComparison:
    def setup_method(self):
        self.max_ell = 2
        self.sh_irreps = Irreps.spherical_harmonics(self.max_ell)
        self.sh_irreps_jax = IrrepsJAX.spherical_harmonics(self.max_ell)

        # Torch modules
        self.torch_sh_normed = SphericalHarmonicsTorch(
            self.sh_irreps, normalize=True, normalization='component'
        )
        self.torch_sh_raw = SphericalHarmonicsTorch(
            self.sh_irreps, normalize=False, normalization='component'
        )

        # Batch of input vectors
        self.x = np.random.randn(10, 3).astype(np.float32)

        # Haiku transformed functions
        def forward_normed(x):
            sh = SphericalHarmonicsJAX(
                self.sh_irreps_jax, normalize=True, normalization='component'
            )
            return sh(x)

        def forward_raw(x):
            sh = SphericalHarmonicsJAX(
                self.sh_irreps_jax, normalize=False, normalization='component'
            )
            return sh(x)

        self.hk_normed = hk.without_apply_rng(hk.transform(forward_normed))
        self.hk_raw = hk.without_apply_rng(hk.transform(forward_raw))

        # Initialize Haiku modules
        self.rng = jax.random.PRNGKey(42)
        self.params_normed = self.hk_normed.init(self.rng, jnp.array(self.x))
        self.params_raw = self.hk_raw.init(self.rng, jnp.array(self.x))

    def test_output_shapes(self):
        # Haiku
        hk_out = self.hk_raw.apply(self.params_raw, jnp.array(self.x))
        assert hk_out.shape == (self.x.shape[0], self.sh_irreps.dim)

        # Torch
        torch_out = self.torch_sh_raw(torch.tensor(self.x))
        assert torch_out.shape == (self.x.shape[0], self.sh_irreps.dim)

    def test_compare_raw(self):
        hk_out = self.hk_raw.apply(self.params_raw, jnp.array(self.x))
        torch_out = self.torch_sh_raw(torch.tensor(self.x)).detach().numpy()

        np.testing.assert_allclose(hk_out, torch_out, rtol=1e-5, atol=1e-5)

    def test_compare_normalized(self):
        torch_out = self.torch_sh_normed(torch.tensor(self.x)).detach().numpy()
        hk_out = np.array(self.hk_normed.apply(self.params_normed, jnp.array(self.x)))

        np.testing.assert_allclose(hk_out, torch_out, rtol=1e-5, atol=1e-5)
