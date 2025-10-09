import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from mace.modules.radial import BesselBasis as BesselBasisTorch
from mace.modules.radial import ChebychevBasis as ChebychevBasisTorch
from mace.modules.radial import RadialMLP as RadialMLPTorch
from mace.modules.radial import ZBLBasis as ZBLBasisTorch

from mace_jax.modules.radial import BesselBasis as BesselBasisJAX
from mace_jax.modules.radial import ChebychevBasis as ChebychevBasisJAX
from mace_jax.modules.radial import RadialMLP as RadialMLPJax
from mace_jax.modules.radial import ZBLBasis as ZBLBasisJAX


class TestBesselBasisParity:
    @pytest.mark.parametrize('trainable', [False, True])
    @pytest.mark.parametrize('num_basis', [4, 8])
    def test_forward(self, trainable, num_basis):
        r_max = 5.0
        batch = 6

        torch_module = BesselBasisTorch(
            r_max=r_max, num_basis=num_basis, trainable=trainable
        )
        torch_module.eval()

        x_t = 0.1 + (r_max - 0.1) * torch.rand(batch, 1)
        out_t = torch_module(x_t).detach().cpu().numpy()

        model = BesselBasisJAX(
            r_max=r_max,
            num_basis=num_basis,
            trainable=trainable,
        )
        key = jax.random.PRNGKey(0)
        x_j = jnp.array(x_t.detach().cpu().numpy())
        variables = model.init(key, x_j)
        variables = BesselBasisJAX.import_from_torch(torch_module, variables)
        out_j = model.apply(variables, x_j)

        assert out_t.shape == out_j.shape
        np.testing.assert_allclose(out_t, np.array(out_j), rtol=1e-6, atol=1e-6)


class TestChebychevBasisParity:
    @pytest.mark.parametrize('num_basis', [4, 8])
    @pytest.mark.parametrize('r_max', [3.0, 5.0])
    def test_forward(self, num_basis, r_max):
        batch = 5

        model_torch = ChebychevBasisTorch(r_max=r_max, num_basis=num_basis)
        model_torch.eval()

        x_torch = torch.rand(batch, 1, dtype=torch.get_default_dtype())
        out_torch = model_torch(x_torch).detach().cpu().numpy()

        model = ChebychevBasisJAX(r_max=r_max, num_basis=num_basis)
        key = jax.random.PRNGKey(0)
        x_jax = jnp.array(x_torch.detach().cpu().numpy())
        variables = model.init(key, x_jax)
        out_jax = np.array(model.apply(variables, x_jax))

        assert out_torch.shape == out_jax.shape
        np.testing.assert_allclose(out_jax, out_torch, rtol=1e-5, atol=1e-6)


class TestRadialMLP:
    """Compare RadialMLP implementations in Flax vs PyTorch."""

    @pytest.mark.parametrize(
        'channels_list',
        [
            [4, 8, 16],
            [3, 6, 6, 2],
        ],
    )
    def test_forward_match(self, channels_list):
        np.random.seed(0)
        x_np = np.random.randn(5, channels_list[0])
        x_jax = jnp.array(x_np)
        x_torch = torch.tensor(x_np)

        torch_model = RadialMLPTorch(channels_list)
        out_torch = torch_model(x_torch)

        rng = jax.random.PRNGKey(42)
        model = RadialMLPJax(channels_list)
        variables = model.init(rng, x_jax)
        variables = RadialMLPJax.import_from_torch(torch_model, variables)
        out_jax = model.apply(variables, x_jax)

        np.testing.assert_allclose(
            np.array(out_jax),
            out_torch.detach().cpu().numpy(),
            rtol=1e-5,
            atol=1e-6,
        )


class TestZBLBasis:
    """Compare ZBLBasis in Flax vs PyTorch."""

    @pytest.mark.parametrize('trainable', [False, True])
    def test_forward_match(self, trainable):
        n_nodes = 6
        n_edges = 10
        n_species = 4

        np.random.seed(0)
        x_np = np.random.rand(n_edges, 1) + 0.5
        node_attrs_np = np.eye(n_species)[np.random.randint(0, n_species, size=n_nodes)]
        edge_index_np = np.vstack(
            [
                np.random.randint(0, n_nodes, size=n_edges),
                np.random.randint(0, n_nodes, size=n_edges),
            ]
        ).astype(np.int32)
        atomic_numbers_np = np.arange(1, n_species + 1)

        x_torch = torch.tensor(x_np)
        node_attrs_torch = torch.tensor(node_attrs_np)
        edge_index_torch = torch.tensor(edge_index_np)
        atomic_numbers_torch = torch.tensor(atomic_numbers_np)

        x_jax = jnp.array(x_np)
        node_attrs_jax = jnp.array(node_attrs_np)
        edge_index_jax = jnp.array(edge_index_np)
        atomic_numbers_jax = jnp.array(atomic_numbers_np)

        torch_model = ZBLBasisTorch(trainable=trainable)
        torch_out = (
            torch_model(
                x_torch, node_attrs_torch, edge_index_torch, atomic_numbers_torch
            )
            .detach()
            .cpu()
            .numpy()
        )

        model = ZBLBasisJAX(trainable=trainable, name='zbl_basis')
        rng = jax.random.PRNGKey(42)
        variables = model.init(
            rng, x_jax, node_attrs_jax, edge_index_jax, atomic_numbers_jax
        )
        variables = ZBLBasisJAX.import_from_torch(torch_model, variables)
        jax_out = model.apply(
            variables, x_jax, node_attrs_jax, edge_index_jax, atomic_numbers_jax
        )
        jax_out = np.array(jax_out)

        np.testing.assert_allclose(
            jax_out,
            torch_out,
            rtol=1e-5,
            atol=1e-6,
        )
