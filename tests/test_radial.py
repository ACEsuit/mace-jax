import re

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from mace.modules.radial import BesselBasis as BesselBasisTorch
from mace.modules.radial import ChebychevBasis as ChebychevBasisTorch
from mace.modules.radial import RadialMLP as RadialMLPTorch

from mace_jax.modules.radial import BesselBasis as BesselBasisJAX
from mace_jax.modules.radial import ChebychevBasis as ChebychevBasisJAX
from mace_jax.modules.radial import RadialMLP as RadialMLPJax


class TestBesselBasisParity:
    @pytest.mark.parametrize('trainable', [False, True])
    @pytest.mark.parametrize('num_basis', [4, 8])
    def test_forward(self, trainable, num_basis):
        r_max = 5.0
        batch = 6

        # --------------------
        # JAX version
        # --------------------
        def forward_fn(x):
            model = BesselBasisJAX(
                r_max=r_max, num_basis=num_basis, trainable=trainable
            )
            return model(x)

        forward = hk.transform(forward_fn)

        key = jax.random.PRNGKey(0)
        x_j = jax.random.uniform(
            key, (batch, 1), minval=0.1, maxval=r_max
        )  # avoid division by 0

        params = forward.init(key, x_j)
        out_j = forward.apply(params, None, x_j)

        # --------------------
        # Torch version
        # --------------------
        model_torch = BesselBasisTorch(
            r_max=r_max, num_basis=num_basis, trainable=trainable
        )
        model_torch.eval()

        # Copy weights from JAX -> Torch if trainable
        if trainable:
            w_jax = params['bessel_basis']['bessel_weights']
            model_torch.bessel_weights.data = torch.tensor(
                np.array(w_jax), dtype=torch.float32
            )
        else:
            # For non-trainable, Torch has buffer already set, but we force sync
            w_jax = (
                np.pi / r_max * np.linspace(1.0, num_basis, num_basis, dtype=np.float32)
            )
            model_torch.bessel_weights.copy_(torch.tensor(w_jax, dtype=torch.float32))

        # Torch input
        x_t = torch.tensor(np.array(x_j), dtype=torch.float32)

        out_t = model_torch(x_t).detach().numpy()

        # --------------------
        # Compare outputs
        # --------------------
        assert out_t.shape == out_j.shape
        np.testing.assert_allclose(out_t, np.array(out_j), rtol=1e-6, atol=1e-6)


class TestChebychevBasisParity:
    @pytest.mark.parametrize('num_basis', [4, 8])
    @pytest.mark.parametrize('r_max', [3.0, 5.0])
    def test_forward(self, num_basis, r_max):
        batch = 5

        # --- Torch version ---
        model_torch = ChebychevBasisTorch(r_max=r_max, num_basis=num_basis)
        model_torch.eval()

        # Torch input
        x_torch = torch.rand(batch, 1, dtype=torch.get_default_dtype())
        out_torch = model_torch(x_torch).detach().numpy()

        # --- JAX version ---
        def forward_fn(x):
            model = ChebychevBasisJAX(r_max=r_max, num_basis=num_basis)
            return model(x)

        forward = hk.transform(forward_fn)

        key = jax.random.PRNGKey(0)
        x_jax = jnp.array(x_torch.detach().numpy())
        params = forward.init(key, x_jax)

        out_jax = np.array(forward.apply(params, None, x_jax))

        # --- Compare ---
        assert out_torch.shape == out_jax.shape
        np.testing.assert_allclose(out_jax, out_torch, rtol=1e-5, atol=1e-6)


PARAM_MAPPERS = {}


def register_mapper(torch_type):
    def decorator(fn):
        PARAM_MAPPERS[torch_type] = fn
        return fn

    return decorator


@register_mapper(torch.nn.Linear)
def map_linear(module, params):
    module.weight.data = torch.tensor(np.array(params['w'])).T
    module.bias.data = torch.tensor(np.array(params['b']))


@register_mapper(torch.nn.LayerNorm)
def map_layernorm(module, params):
    module.weight.data = torch.tensor(np.array(params['scale']))
    module.bias.data = torch.tensor(np.array(params['offset']))


class TestRadialMLP:
    """Compare RadialMLP implementations in Haiku vs PyTorch."""

    def torch_to_haiku_name(self, torch_name, scope='RadialMLP'):
        """
        Map Torch module names to Haiku param keys.

        Examples:
        ''        -> 'RadialMLP'
        'net.0'   -> 'RadialMLP/~/net_0'
        'net.1'   -> 'RadialMLP/~/net_1'
        """
        if torch_name == '':
            return scope

        parts = torch_name.split('.')
        mapped = []
        for p in parts:
            if p.isdigit():
                mapped.append(f'net_{p}')
            elif p == 'net':
                continue  # skip the container name
            else:
                mapped.append(p)
        return f'{scope}/~/' + '/'.join(mapped)

    def copy_jax_to_torch(self, torch_model, jax_params, scope='RadialMLP'):
        for name, module in torch_model.named_modules():
            if name == '':
                continue
            hk_name = self.torch_to_haiku_name(name, scope)
            print('name:', name)

            for torch_type, mapper in PARAM_MAPPERS.items():
                if isinstance(module, torch_type):
                    mapper(module, jax_params[hk_name])
                    break

    def build_jax_net(self, channels_list):
        """Wrap RadialMLPJax inside hk.transform correctly."""

        def net_fn(x):
            # Instantiated inside the transformed function
            model = RadialMLPJax(channels_list)
            return model(x)

        return hk.without_apply_rng(hk.transform(net_fn))

    @pytest.mark.parametrize(
        'channels_list',
        [
            [4, 8, 16],
            [3, 6, 6, 2],
        ],
    )
    def test_forward_match(self, channels_list):
        """Check that forward pass matches between JAX and Torch."""

        # --- Create same random input ---
        np.random.seed(0)
        x_np = np.random.randn(5, channels_list[0]).astype(np.float64)
        x_jax = jnp.array(x_np)
        x_torch = torch.tensor(x_np)

        # --- Build and init JAX net ---
        def forward_fn(x):
            # Instantiated inside the transformed function
            model = RadialMLPJax(channels_list)
            return model(x)

        transformed = hk.transform(forward_fn)
        rng = jax.random.PRNGKey(42)
        params = transformed.init(rng, x_jax)
        out_jax = transformed.apply(params, rng, x_jax)

        # --- Torch model ---
        torch_model = RadialMLPTorch(channels_list)

        # --- Copy JAX params to Torch ---
        self.copy_jax_to_torch(torch_model, params)

        # --- Run Torch version ---
        out_torch = torch_model(x_torch)

        # --- Compare outputs ---
        np.testing.assert_allclose(
            np.array(out_jax),
            out_torch.detach().numpy(),
            rtol=1e-5,
            atol=1e-6,
        )
