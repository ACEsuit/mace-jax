import jax.numpy as jnp
import numpy as np
import pytest
import torch
from flax import nnx

from mace_jax.nnx_utils import state_to_pure_dict

try:  # pragma: no cover - optional torch dependency for parity tests
    from mace.modules.radial import BesselBasis as BesselBasisTorch
    from mace.modules.radial import ChebychevBasis as ChebychevBasisTorch
    from mace.modules.radial import RadialMLP as RadialMLPTorch
    from mace.modules.radial import ZBLBasis as ZBLBasisTorch
except Exception as exc:  # pragma: no cover
    pytest.skip(
        f'Torch MACE radial modules unavailable: {exc}',
        allow_module_level=True,
    )

from mace_jax.modules.radial import BesselBasis as BesselBasisJAX
from mace_jax.modules.radial import ChebychevBasis as ChebychevBasisJAX
from mace_jax.modules.radial import RadialMLP as RadialMLPJax
from mace_jax.modules.radial import ZBLBasis as ZBLBasisJAX


def _split_module(module):
    graphdef, state = nnx.split(module)
    return graphdef, state_to_pure_dict(state)


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

        rngs = nnx.Rngs(0) if trainable else None
        model = BesselBasisJAX(
            r_max=r_max,
            num_basis=num_basis,
            trainable=trainable,
            rngs=rngs,
        )
        x_j = jnp.array(x_t.detach().cpu().numpy())
        graphdef, variables = _split_module(model)
        variables = BesselBasisJAX.import_from_torch(torch_module, variables)
        out_j, _ = graphdef.apply(variables)(x_j)

        assert out_t.shape == out_j.shape
        np.testing.assert_allclose(out_t, np.array(out_j), rtol=1e-6, atol=1e-6)

    def test_near_zero_parity(self):
        r_max = 5.0
        num_basis = 6
        torch_module = BesselBasisTorch(
            r_max=r_max, num_basis=num_basis, trainable=False
        )
        torch_module.eval()

        eps = torch.tensor(1e-7, dtype=torch.get_default_dtype())
        x_t = torch.full((3, 1), eps, dtype=torch.get_default_dtype())
        out_t = torch_module(x_t).detach().cpu().numpy()

        model = BesselBasisJAX(r_max=r_max, num_basis=num_basis, trainable=False)
        x_j = jnp.array(x_t.detach().cpu().numpy())
        graphdef, variables = _split_module(model)
        variables = BesselBasisJAX.import_from_torch(torch_module, variables)
        out_j, _ = graphdef.apply(variables)(x_j)
        out_j = np.array(out_j)

        np.testing.assert_allclose(out_t, out_j, rtol=1e-6, atol=1e-6)
        assert np.all(np.isfinite(out_j))

    def test_zero_distance_limit_jax(self):
        r_max = 5.0
        num_basis = 6
        model = BesselBasisJAX(r_max=r_max, num_basis=num_basis, trainable=False)
        x_j = jnp.zeros((3, 1), dtype=jnp.float64)
        graphdef, variables = _split_module(model)
        out_j, _ = graphdef.apply(variables)(x_j)
        out_j = np.array(out_j)

        prefactor = np.sqrt(2.0 / r_max)
        init_bessel = (
            np.pi
            / float(r_max)
            * np.linspace(1.0, num_basis, num_basis, dtype=out_j.dtype)
        )
        expected = prefactor * init_bessel
        expected = np.broadcast_to(expected, out_j.shape)

        np.testing.assert_allclose(out_j, expected, rtol=1e-6, atol=1e-6)
        assert np.all(np.isfinite(out_j))


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
        x_jax = jnp.array(x_torch.detach().cpu().numpy())
        graphdef, variables = _split_module(model)
        out_jax, _ = graphdef.apply(variables)(x_jax)
        out_jax = np.array(out_jax)

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

        model = RadialMLPJax(channels_list, rngs=nnx.Rngs(42))
        graphdef, variables = _split_module(model)
        variables = RadialMLPJax.import_from_torch(torch_model, variables)
        out_jax, _ = graphdef.apply(variables)(x_jax)

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

        model = ZBLBasisJAX(
            trainable=trainable, rngs=nnx.Rngs(42) if trainable else None
        )
        graphdef, variables = _split_module(model)
        variables = ZBLBasisJAX.import_from_torch(torch_model, variables)
        jax_out, _ = graphdef.apply(variables)(
            x_jax, node_attrs_jax, edge_index_jax, atomic_numbers_jax
        )
        jax_out = np.array(jax_out)

        np.testing.assert_allclose(
            jax_out,
            torch_out,
            rtol=1e-5,
            atol=1e-6,
        )
