import pytest
import torch
import jax.numpy as jnp
import numpy as np

from mace.tools.cg import _wigner_nj as wigner_nj_torch
from mace.tools.cg import U_matrix_real as U_matrix_real_torch

from mace_jax.tools.cg import _wigner_nj as wigner_nj_jax
from mace_jax.tools.cg import U_matrix_real as U_matrix_real_jax


def to_numpy(x):
    """Convert torch or jax array to numpy for comparison."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    elif isinstance(x, jnp.ndarray):
        return np.array(x)
    return np.array(x)


class TestClebschGordanParity:
    @pytest.mark.parametrize("irreps_list", [["1x0e", "1x1o"], ["2x0e + 1x1o", "1x0e"]])
    def test_wigner_nj(self, irreps_list):
        out_jax = wigner_nj_jax(irreps_list)
        out_torch = wigner_nj_torch(irreps_list)

        assert len(out_jax) == len(out_torch)
        for (ir_j, tp_j, E_j), (ir_t, tp_t, E_t) in zip(out_jax, out_torch):
            # Compare irreps by their l and p
            assert ir_j.l == ir_t.l
            assert ir_j.p == ir_t.p
            # Compare tensors
            np.testing.assert_allclose(
                to_numpy(E_j), to_numpy(E_t), rtol=1e-5, atol=1e-6
            )

    @pytest.mark.parametrize(
        "irreps_in, irreps_out, corr",
        [
            ("1x0e + 1x1o", "1x0e", 1),
            ("2x0e + 1x1o", "1x1o", 2),
            ("2x0e + 1x1o", "1x0e", 2),
        ],
    )
    def test_U_matrix_real(self, irreps_in, irreps_out, corr):
        out_jax = U_matrix_real_jax(irreps_in, irreps_out, corr)
        out_torch = U_matrix_real_torch(irreps_in, irreps_out, corr)

        assert len(out_jax) == len(out_torch)
        for idx, (a, b) in enumerate(zip(out_jax, out_torch)):
            assert to_numpy(a).shape == to_numpy(b).shape, f"Mismatch at element {idx}"

            if isinstance(a, tuple) and isinstance(b, tuple):
                # tuple: (irreps, array)
                ir_a, tensor_a = a
                ir_b, tensor_b = b
                assert ir_a.l == ir_b.l and ir_a.p == ir_b.p
                np.testing.assert_allclose(
                    to_numpy(tensor_a), to_numpy(tensor_b), rtol=1e-5, atol=1e-6
                )
            elif isinstance(a, (jnp.ndarray, torch.Tensor, np.ndarray)) and isinstance(
                b, (jnp.ndarray, torch.Tensor, np.ndarray)
            ):
                # numeric array
                np.testing.assert_allclose(
                    to_numpy(a), to_numpy(b), rtol=1e-5, atol=1e-6
                )
            else:
                # string or other object, compare equality
                assert a == b
