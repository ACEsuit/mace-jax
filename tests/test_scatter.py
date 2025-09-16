import jax.numpy as jnp
import pytest
import torch
from mace.tools.scatter import _broadcast as _broadcast_torch
from mace.tools.scatter import scatter_mean as scatter_mean_torch
from mace.tools.scatter import scatter_std as scatter_std_torch
from mace.tools.scatter import scatter_sum as scatter_sum_torch

from mace_jax.tools.scatter import _broadcast as _broadcast_jax
from mace_jax.tools.scatter import scatter_mean as scatter_mean_jax
from mace_jax.tools.scatter import scatter_std as scatter_std_jax
from mace_jax.tools.scatter import scatter_sum as scatter_sum_jax


class TestBroadcastParity:
    @pytest.mark.parametrize('dim', [0, 1, -1])
    def test_same_shape(self, dim):
        x_torch = torch.ones((2, 3))
        y_torch = torch.zeros((2, 3))
        out_torch = _broadcast_torch(x_torch, y_torch, dim)

        x_jax = jnp.ones((2, 3))
        y_jax = jnp.zeros((2, 3))
        out_jax = _broadcast_jax(x_jax, y_jax, dim)

        assert out_jax.shape == tuple(out_torch.shape)
        assert jnp.allclose(out_jax, out_torch.numpy())

    def test_add_trailing_dim(self):
        x_torch = torch.arange(6).view(2, 3)  # (2, 3)
        y_torch = torch.zeros((2, 3, 4))  # (2, 3, 4)
        out_torch = _broadcast_torch(x_torch, y_torch, dim=-1)

        x_jax = jnp.arange(6).reshape(2, 3)
        y_jax = jnp.zeros((2, 3, 4))
        out_jax = _broadcast_jax(x_jax, y_jax, dim=-1)

        assert out_jax.shape == tuple(out_torch.shape)
        assert jnp.allclose(out_jax, out_torch.numpy())

    def test_add_leading_dims_for_1d(self):
        x_torch = torch.tensor([1, 2, 3])  # (3,)
        y_torch = torch.zeros((2, 3, 4))  # (2, 3, 4)
        out_torch = _broadcast_torch(x_torch, y_torch, dim=1)

        x_jax = jnp.array([1, 2, 3])
        y_jax = jnp.zeros((2, 3, 4))
        out_jax = _broadcast_jax(x_jax, y_jax, dim=1)

        assert out_jax.shape == tuple(out_torch.shape)
        assert jnp.allclose(out_jax, out_torch.numpy())

    def test_negative_dim(self):
        x_torch = torch.tensor([5, 6])  # (2,)
        y_torch = torch.zeros((3, 2, 4))  # (3, 2, 4)
        out_torch = _broadcast_torch(x_torch, y_torch, dim=-2)

        x_jax = jnp.array([5, 6])
        y_jax = jnp.zeros((3, 2, 4))
        out_jax = _broadcast_jax(x_jax, y_jax, dim=-2)

        assert out_jax.shape == tuple(out_torch.shape)
        assert jnp.allclose(out_jax, out_torch.numpy())

    def test_scalar_expand(self):
        x_torch = torch.tensor(7)  # ()
        y_torch = torch.zeros((2, 3, 4))  # (2, 3, 4)
        out_torch = _broadcast_torch(x_torch, y_torch, dim=0)

        x_jax = jnp.array(7)
        y_jax = jnp.zeros((2, 3, 4))
        out_jax = _broadcast_jax(x_jax, y_jax, dim=0)

        assert out_jax.shape == tuple(out_torch.shape)
        assert jnp.allclose(out_jax, out_torch.numpy())


class TestScatterSumParity:
    def test_basic_sum(self):
        src_t = torch.tensor([1.0, 2.0, 3.0, 4.0])
        idx_t = torch.tensor([0, 1, 0, 1])
        out_t = scatter_sum_torch(src_t, idx_t, dim=0)

        src_j = jnp.array([1.0, 2.0, 3.0, 4.0])
        idx_j = jnp.array([0, 1, 0, 1])
        out_j = scatter_sum_jax(src_j, idx_j, dim=0)

        assert out_j.shape == tuple(out_t.shape)
        assert jnp.allclose(out_j, out_t.numpy())

    def test_2d_dim0(self):
        src_t = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        idx_t = torch.tensor([0, 1, 0])
        out_t = scatter_sum_torch(src_t, idx_t, dim=0)

        src_j = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        idx_j = jnp.array([0, 1, 0])
        out_j = scatter_sum_jax(src_j, idx_j, dim=0)

        assert out_j.shape == tuple(out_t.shape)
        assert jnp.allclose(out_j, out_t.numpy())

    def test_2d_dim1(self):
        src_t = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        idx_t = torch.tensor([[0, 1, 0], [1, 0, 1]])
        out_t = scatter_sum_torch(src_t, idx_t, dim=1)

        src_j = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        idx_j = jnp.array([[0, 1, 0], [1, 0, 1]])
        out_j = scatter_sum_jax(src_j, idx_j, dim=1)

        assert out_j.shape == tuple(out_t.shape)
        assert jnp.allclose(out_j, out_t.numpy())

    def test_with_dim_size(self):
        src_t = torch.tensor([1.0, 2.0, 3.0])
        idx_t = torch.tensor([0, 1, 0])
        out_t = scatter_sum_torch(src_t, idx_t, dim=0, dim_size=3)

        src_j = jnp.array([1.0, 2.0, 3.0])
        idx_j = jnp.array([0, 1, 0])
        out_j = scatter_sum_jax(src_j, idx_j, dim=0, dim_size=3)

        assert out_j.shape == tuple(out_t.shape)
        assert jnp.allclose(out_j, out_t.numpy())


class TestScatterStdParity:
    def test_basic_1d(self):
        """1D scatter along dim 0."""
        src_t = torch.tensor([1.0, 2.0, 3.0, 4.0])
        idx_t = torch.tensor([0, 1, 0, 1])
        out_t = scatter_std_torch(src_t, idx_t, dim=0, unbiased=False)

        src_j = jnp.array([1.0, 2.0, 3.0, 4.0])
        idx_j = jnp.array([0, 1, 0, 1])
        out_j = scatter_std_jax(src_j, idx_j, dim=0, unbiased=False)

        assert out_j.shape == tuple(out_t.shape)
        assert jnp.allclose(out_j, out_t.numpy(), atol=1e-6)

    def test_2d_dim0(self):
        """2D scatter along rows (dim 0), valid shapes."""
        src_t = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        idx_t = torch.tensor([0, 1, 0])
        out_t = scatter_std_torch(src_t, idx_t, dim=0, unbiased=True)

        src_j = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        idx_j = jnp.array([0, 1, 0])
        out_j = scatter_std_jax(src_j, idx_j, dim=0, unbiased=True)

        assert out_j.shape == tuple(out_t.shape)
        assert jnp.allclose(out_j, out_t.numpy(), atol=1e-6)

    def test_2d_dim1(self):
        """2D scatter along columns (dim 1), valid shapes."""
        src_t = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        idx_t = torch.tensor([[0, 1, 0], [1, 0, 1]])
        out_t = scatter_std_torch(src_t, idx_t, dim=1, unbiased=False)

        src_j = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        idx_j = jnp.array([[0, 1, 0], [1, 0, 1]])
        out_j = scatter_std_jax(src_j, idx_j, dim=1, unbiased=False)

        assert out_j.shape == tuple(out_t.shape)
        assert jnp.allclose(out_j, out_t.numpy(), atol=1e-6)

    def test_3d_valid(self):
        """3D scatter along last dim with compatible shapes."""
        src_t = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
        idx_t = torch.tensor([[[0, 1], [1, 0]], [[0, 1], [1, 0]]])
        out_t = scatter_std_torch(src_t, idx_t, dim=2, unbiased=True)

        src_j = jnp.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
        idx_j = jnp.array([[[0, 1], [1, 0]], [[0, 1], [1, 0]]])
        out_j = scatter_std_jax(src_j, idx_j, dim=2, unbiased=True)

        assert out_j.shape == tuple(out_t.shape)
        assert jnp.allclose(out_j, out_t.numpy(), atol=1e-6)


class TestScatterMeanParity:
    def test_basic_1d(self):
        src_t = torch.tensor([1.0, 2.0, 3.0, 4.0])
        idx_t = torch.tensor([0, 1, 0, 1])
        out_t = scatter_mean_torch(src_t, idx_t, dim=0)

        src_j = jnp.array([1.0, 2.0, 3.0, 4.0])
        idx_j = jnp.array([0, 1, 0, 1])
        out_j = scatter_mean_jax(src_j, idx_j, dim=0)

        assert out_j.shape == tuple(out_t.shape)
        assert jnp.allclose(out_j, out_t.numpy(), atol=1e-6)

    def test_2d_dim0(self):
        src_t = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        idx_t = torch.tensor([0, 1, 0])
        out_t = scatter_mean_torch(src_t, idx_t, dim=0)

        src_j = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        idx_j = jnp.array([0, 1, 0])
        out_j = scatter_mean_jax(src_j, idx_j, dim=0)

        assert out_j.shape == tuple(out_t.shape)
        assert jnp.allclose(out_j, out_t.numpy(), atol=1e-6)

    def test_2d_dim1(self):
        src_t = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        idx_t = torch.tensor([[0, 1, 0], [1, 0, 1]])
        out_t = scatter_mean_torch(src_t, idx_t, dim=1)

        src_j = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        idx_j = jnp.array([[0, 1, 0], [1, 0, 1]])
        out_j = scatter_mean_jax(src_j, idx_j, dim=1)

        assert out_j.shape == tuple(out_t.shape)
        assert jnp.allclose(out_j, out_t.numpy(), atol=1e-6)

    def test_3d_valid(self):
        src_t = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
        idx_t = torch.tensor([[[0, 1], [1, 0]], [[0, 1], [1, 0]]])
        out_t = scatter_mean_torch(src_t, idx_t, dim=2)

        src_j = jnp.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
        idx_j = jnp.array([[[0, 1], [1, 0]], [[0, 1], [1, 0]]])
        out_j = scatter_mean_jax(src_j, idx_j, dim=2)

        assert out_j.shape == tuple(out_t.shape)
        assert jnp.allclose(out_j, out_t.numpy(), atol=1e-6)
