from collections.abc import Sequence
from math import prod

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from e3nn.o3 import Irreps
from e3nn.o3 import Irreps as IrrepsTorch
from e3nn.o3 import SphericalHarmonics as SphericalHarmonicsTorch
from e3nn.o3 import TensorProduct as TensorProductTorch
from e3nn.o3._linear import Linear as LinearTorch
from e3nn_jax import Irreps as IrrepsJAX
from e3nn_jax import spherical_harmonics

from mace_jax.cuequivariance.tensor_product import _infer_path_shape
from mace_jax.e3nn._linear import Linear as LinearJax
from mace_jax.e3nn._tensor_product._tensor_product import (
    TensorProduct as TensorProductJAX,
)
from mace_jax.e3nn.o3 import SphericalHarmonics as SphericalHarmonicsJAX
from mace_jax.modules.irreps_tools import tp_out_irreps_with_instructions
from mace_jax.modules.wrapper_ops import CuEquivarianceConfig
from mace_jax.modules.wrapper_ops import TensorProduct as WrapperTensorProduct


class TestTensorProductAllExamples:
    @pytest.mark.parametrize('example_idx', list(range(6)))  # we have 6 examples
    def test_tensor_product_examples(self, example_idx):
        # ------------------------
        # Define examples
        if example_idx == 0:
            irreps_in1 = irreps_in2 = Irreps('16x1o')
            irreps_out = Irreps('16x1e')
            instructions = [(0, 0, 0, 'uuu', False)]
        elif example_idx == 1:
            irreps_in1 = Irreps('16x1o')
            irreps_in2 = Irreps('16x1o')
            irreps_out = Irreps('16x1e')
            instructions = [(0, 0, 0, 'uvw', True)]
        elif example_idx == 2:
            irreps_in1 = Irreps('8x0o + 8x1o')
            irreps_in2 = Irreps('16x1o')
            irreps_out = Irreps('16x1e')
            instructions = [
                (0, 0, 0, 'uvw', True, 3),
                (1, 0, 0, 'uvw', True, 1),
            ]
        elif example_idx == 3:
            irreps = Irreps('3x0e + 4x0o + 1e + 2o + 3o')
            irreps_in1 = irreps_in2 = irreps
            irreps_out = Irreps('0e')
            instructions = [
                (i, i, 0, 'uuw', False) for i, (mul, ir) in enumerate(irreps)
            ]
        elif example_idx == 4:
            irreps_in1 = Irreps('8x0o + 7x1o + 3x2e')
            irreps_in2 = Irreps('10x0e + 10x1e + 10x2e')
            irreps_out = Irreps('8x0o + 7x1o + 3x2e')
            instructions = [
                (0, 0, 0, 'uvu', True),
                (1, 0, 1, 'uvu', True),
                (1, 1, 1, 'uvu', True),
                (1, 2, 1, 'uvu', True),
                (2, 0, 2, 'uvu', True),
                (2, 1, 2, 'uvu', True),
                (2, 2, 2, 'uvu', True),
            ]
        elif example_idx == 5:
            irreps_in1 = Irreps('5x0e + 10x1o + 1x2e')
            irreps_in2 = Irreps('5x0e + 10x1o + 1x2e')
            irreps_out = Irreps('5x0e + 10x1o + 1x2e')
            instructions = [
                (i1, i2, i_out, 'uvw', True, mul1 * mul2)
                for i1, (mul1, ir1) in enumerate(irreps_in1)
                for i2, (mul2, ir2) in enumerate(irreps_in2)
                for i_out, (mul_out, ir_out) in enumerate(irreps_out)
                if ir_out in ir1 * ir2
            ]
        else:
            raise ValueError(f'Unexpected example_idx {example_idx}')

        # ------------------------
        # Random inputs
        batch_size = 4
        x1 = torch.randn(batch_size, irreps_in1.dim)
        x2 = torch.randn(batch_size, irreps_in2.dim)

        total_weight_numel = sum(
            np.prod((irreps_in1[i1].mul, irreps_in2[i2].mul, irreps_out[i_out].mul))
            if mode == 'uvw'
            else np.prod((irreps_in1[i1].mul, irreps_in2[i2].mul))
            if mode in ['uvu', 'uvv']
            else np.prod((irreps_in1[i1].mul, irreps_out[i_out].mul))
            if mode == 'uuw'
            else 1
            for i1, i2, i_out, mode, has_weight, *rest in instructions
            if has_weight
        )
        w = torch.randn(total_weight_numel) if total_weight_numel > 0 else None

        # ------------------------
        # PyTorch
        tp_torch = TensorProductTorch(irreps_in1, irreps_in2, irreps_out, instructions)
        out_torch = tp_torch(x1, x2, w).detach().cpu().numpy()

        # ------------------------
        # JAX
        def forward_fn(x1_, x2_, w_):
            tp_jax = TensorProductJAX(irreps_in1, irreps_in2, irreps_out, instructions)
            return tp_jax(x1_, x2_, w_)

        forward_transformed = hk.transform(forward_fn)
        rng = jax.random.PRNGKey(42)
        x1_j = jnp.array(x1.detach().cpu().numpy())
        x2_j = jnp.array(x2.detach().cpu().numpy())
        w_j = jnp.array(w.detach().cpu().numpy()) if w is not None else None
        params = forward_transformed.init(rng, x1_j, x2_j, w_j)
        out_jax = forward_transformed.apply(params, None, x1_j, x2_j, w_j)
        out_jax_np = np.array(out_jax)

        # ------------------------
        # Compare outputs
        np.testing.assert_allclose(out_torch, out_jax_np, rtol=1e-5, atol=1e-6)


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


class TestCueTensorProduct:
    @classmethod
    def setup_class(cls):
        cls.irreps_in1 = Irreps('1x0e + 1x1o')
        cls.irreps_in2 = Irreps('1x0e + 1x1o')
        cls.irreps_out, instructions = tp_out_irreps_with_instructions(
            cls.irreps_in1, cls.irreps_in2, Irreps('1x0e + 1x1o')
        )
        cls.instructions = list(instructions)

        base_weights = []
        cue_weights = []
        keys = jax.random.split(jax.random.PRNGKey(123), len(cls.instructions))

        for key_inst, ins in zip(keys, cls.instructions):
            i_in1, i_in2, i_out, mode, has_weight = ins[:5]
            if not has_weight:
                continue
            mul_in1 = cls.irreps_in1[i_in1].mul
            mul_in2 = cls.irreps_in2[i_in2].mul
            mul_out = cls.irreps_out[i_out].mul

            if mode == 'uvw':
                shape_ref = (mul_in1, mul_in2, mul_out)
                weight_ref = jax.random.normal(key_inst, shape_ref)
                base_weights.append(weight_ref.reshape(-1))
                cue_weights.append(weight_ref.reshape(-1))
            elif mode == 'uvu':
                shape_ref = (mul_in1, mul_in2)
                weight_ref = jax.random.normal(key_inst, shape_ref)
                base_weights.append(weight_ref.reshape(-1))

                expanded = jnp.broadcast_to(
                    weight_ref[..., None], (mul_in1, mul_in2, mul_out)
                )
                cue_weights.append(expanded.reshape(-1))
            else:
                raise NotImplementedError(mode)

        cls.weight_ref = (
            jnp.concatenate(base_weights) if base_weights else jnp.zeros((0,))
        )
        cls.weight_cue = (
            jnp.concatenate(cue_weights) if cue_weights else jnp.zeros((0,))
        )

        key = jax.random.PRNGKey(0)
        key_x1, key_x2, key_ref, key_cue = jax.random.split(key, 4)
        cls.x1 = jax.random.normal(key_x1, (3, cls.irreps_in1.dim))
        cls.x2 = jax.random.normal(key_x2, (3, cls.irreps_in2.dim))
        cls.config = CuEquivarianceConfig(enabled=True)

        def ref_forward(x1_, x2_, w_):
            tp = TensorProductJAX(
                cls.irreps_in1,
                cls.irreps_in2,
                cls.irreps_out,
                instructions=cls.instructions,
                shared_weights=True,
                internal_weights=False,
            )
            return tp(x1_, x2_, w_)

        def cue_forward(x1_, x2_, w_):
            tp = WrapperTensorProduct(
                cls.irreps_in1,
                cls.irreps_in2,
                cls.irreps_out,
                instructions=cls.instructions,
                shared_weights=True,
                internal_weights=False,
                cueq_config=cls.config,
            )
            return tp(x1_, x2_, w_)

        cls.hk_ref = hk.without_apply_rng(hk.transform(ref_forward))
        cls.hk_cue = hk.without_apply_rng(hk.transform(cue_forward))

        cls.params_ref = cls.hk_ref.init(key_ref, cls.x1, cls.x2, cls.weight_ref)
        cls.params_cue = cls.hk_cue.init(key_cue, cls.x1, cls.x2, cls.weight_cue)

    def test_outputs_match(self):
        out_ref = self.hk_ref.apply(self.params_ref, self.x1, self.x2, self.weight_ref)
        out_cue = self.hk_cue.apply(self.params_cue, self.x1, self.x2, self.weight_cue)
        np.testing.assert_allclose(out_ref, out_cue, rtol=1e-5, atol=1e-5)


class TestCueTensorProductAdditionalModes:
    _CASES = [
        (
            Irreps('1x0e'),
            Irreps('1x0e'),
            Irreps('1x0e'),
            [(0, 0, 0, 'uvw', True)],
        ),
        (
            Irreps('1x0e'),
            Irreps('2x0e'),
            Irreps('2x0e'),
            [(0, 0, 0, 'uvv', True)],
        ),
        (
            Irreps('2x0e'),
            Irreps('2x0e'),
            Irreps('1x0e'),
            [(0, 0, 0, 'uuw', True)],
        ),
        (
            Irreps('2x0e'),
            Irreps('2x0e'),
            Irreps('2x0e'),
            [(0, 0, 0, 'uuu', True)],
        ),
        (
            Irreps('2x0e'),
            Irreps('2x0e'),
            Irreps('4x0e'),
            [(0, 0, 0, 'uvuv', True)],
        ),
    ]

    @pytest.mark.parametrize('irreps_in1,irreps_in2,irreps_out,instructions', _CASES)
    def test_modes(self, irreps_in1, irreps_in2, irreps_out, instructions):
        self._compare_tensor_product(irreps_in1, irreps_in2, irreps_out, instructions)

    @staticmethod
    def _compare_tensor_product(
        irreps_in1: Irreps,
        irreps_in2: Irreps,
        irreps_out: Irreps,
        instructions: Sequence[tuple],
        *,
        seed: int = 0,
    ) -> None:
        instructions = list(instructions)

        weight_shapes = []
        for ins in instructions:
            i_in1, i_in2, i_out, mode, has_weight, *rest = ins
            if not has_weight:
                continue
            mul_in1 = irreps_in1[i_in1].mul
            mul_in2 = irreps_in2[i_in2].mul
            mul_out = irreps_out[i_out].mul
            weight_shapes.append(_infer_path_shape(mode, mul_in1, mul_in2, mul_out))

        weight_numel = sum(np.prod(shape) for shape in weight_shapes)

        key = jax.random.PRNGKey(seed)
        key_x1, key_x2, key_w, key_ref, key_cue = jax.random.split(key, 5)
        x1 = jax.random.normal(key_x1, (3, irreps_in1.dim))
        x2 = jax.random.normal(key_x2, (3, irreps_in2.dim))
        weight = jax.random.normal(key_w, (weight_numel,))

        config = CuEquivarianceConfig(enabled=True)

        def ref_forward(x1_, x2_, w_):
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
                cueq_config=config,
            )
            return tp(x1_, x2_, w_)

        ref = hk.without_apply_rng(hk.transform(ref_forward))
        params_ref = ref.init(key_ref, x1, x2, weight)
        out_ref = ref.apply(params_ref, x1, x2, weight)

        cue_tp = hk.without_apply_rng(hk.transform(cue_forward))
        params_cue = cue_tp.init(key_cue, x1, x2, weight)
        out_cue = cue_tp.apply(params_cue, x1, x2, weight)

        np.testing.assert_allclose(out_ref, out_cue, rtol=1e-5, atol=1e-5)
