from typing import Any, Optional, Union

import haiku as hk
import jax.numpy as jnp
import opt_einsum as oe
from e3nn_jax import Irrep, Irreps

from mace_jax.tools.cg import U_matrix_real
from mace_jax.tools.dtype import default_dtype

BATCH_EXAMPLE = 10
ALPHABET = ['w', 'x', 'v', 'n', 'z', 'r', 't', 'y', 'u', 'o', 'p', 's']


def _ensure_array_from_U(u_obj: Any) -> jnp.ndarray:
    """
    U_matrix_real may return a list of (ir, array) pairs or sometimes only an array.
    This helper returns the numeric array (the last element).
    """
    if isinstance(u_obj, jnp.ndarray):
        return u_obj
    # if it is a tuple like (ir, array)
    if isinstance(u_obj, (list, tuple)) and len(u_obj) >= 2:
        # commonly (ir, array) or list [ir_str, array]
        candidate = u_obj[-1]
        if isinstance(candidate, jnp.ndarray):
            return candidate
        # fallback: try convert
        return jnp.asarray(candidate)
    # last-resort conversion
    return jnp.asarray(u_obj)


class Contraction(hk.Module):
    """Haiku/JAX rewrite of the PyTorch Contraction.
    Instantiate *inside* hk.transform context (or via a closure).
    """

    def __init__(
        self,
        irreps_in: Irreps,
        irrep_out: Irreps,
        correlation: int,
        internal_weights: bool = True,
        use_reduced_cg: bool = False,
        num_elements: Optional[int] = None,
        weights: Optional[jnp.array] = None,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)

        self.irreps_in = Irreps(irreps_in)
        self.irrep_out = Irreps(irrep_out)
        self.coupling_irreps = Irreps([irrep.ir for irrep in irreps_in])
        self.correlation = int(correlation)
        self.internal_weights = bool(internal_weights)
        # count of scalar features (mul for (0,x) in irreps)
        self.num_features = irreps_in.count((0, 1))
        self.num_elements = int(num_elements or 1)
        # lmax (max l among output irreps)
        self.lmax = max((ir.ir.l for ir in self.irrep_out), default=0)

        # --- Precompute U matrices (numeric arrays) and path weights ---
        self.U_matrices: dict[int, jnp.ndarray] = {}
        self.zero_flags: list[bool] = []  # corresponds to path_weight (negated)

        dtype = jnp.array(0.0).dtype

        for nu in range(1, self.correlation + 1):
            # Compute U_matrix_real like PyTorch
            raw = U_matrix_real(
                irreps_in=self.coupling_irreps,
                irreps_out=self.irrep_out,
                correlation=nu,
                use_cueq_cg=use_reduced_cg,
                dtype=dtype,
            )
            # Take the last array (PyTorch uses [-1])
            last = raw[-1]
            U = _ensure_array_from_U(last)  # convert to jnp.ndarray if needed

            # Determine num_params and num_ell to match PyTorch shapes
            num_params = int(U.shape[-1])
            num_ell = int(U.shape[-2])

            # Optionally slice to match PyTorch batch-equivalent shapes if needed
            # For example, PyTorch might have slightly smaller num_ell
            # num_ell = min(num_ell, expected_from_irreps)
            # U = U[..., :num_ell, :num_params]

            # Store cleaned U_matrix
            self.U_matrices[nu] = jnp.asarray(U)

            # Compute zero flag for this path
            is_nonzero = jnp.any(self.U_matrices[nu] != 0.0)
            self.zero_flags.append(not bool(is_nonzero))

        num_equivariance = 2 * irrep_out.lmax + 1

        # --- Build main einsum expression (i == correlation) ---
        U_main = self.U_matrices[self.correlation]
        # U_main shape: (..., num_params) where last dim is parameter dim (k)
        num_params = int(U_main.shape[-1])
        num_ell = int(U_main.shape[-2])

        # prefix letters like PyTorch
        prefix = [ALPHABET[j] for j in range(self.correlation + min(self.lmax, 1) - 1)]
        main_sub = ''.join(prefix) + 'ik,ekc,bci,be->bc' + ''.join(prefix)

        # When calling contract_expression, pass shapes WITHOUT batch dims.
        # For U_main we pass exactly U_main.shape
        if num_equivariance == 1:
            shapes_main = (
                (num_ell,) * self.correlation + (num_params,),
                (self.num_elements, num_params, self.num_features),
                (BATCH_EXAMPLE, self.num_features, num_ell),
                (BATCH_EXAMPLE, self.num_elements),
            )
        else:
            shapes_main = (
                (num_equivariance,) + (num_ell,) * self.correlation + (num_params,),
                (self.num_elements, num_params, self.num_features),
                (BATCH_EXAMPLE, self.num_features, num_ell),
                (BATCH_EXAMPLE, self.num_elements),
            )
        assert shapes_main[1][2] == self.num_features, (
            "weights ekc 'c' != self.num_features"
        )
        assert shapes_main[2][1] == self.num_features, "x bci 'c' != self.num_features"

        self._main_expr = oe.contract_expression(main_sub, *shapes_main)
        self._shapes_main = shapes_main

        # --- Prepare weighting/feature expressions for i < correlation ---
        self._weight_exprs: list[Any] = []
        self._shapes_weight: list[Any] = []
        self._feature_exprs: list[Any] = []

        # iterate descending (correlation-1 .. 1) as PyTorch
        for i in range(self.correlation - 1, 0, -1):
            U_i = self.U_matrices[i]
            num_params_i = int(U_i.shape[-1])
            num_ell_i = int(U_i.shape[-2])

            prefix_i = [ALPHABET[j] for j in range(max(0, i + min(self.lmax, 1)))]
            subs_weight = ''.join(prefix_i) + 'k,ekc,be->bc' + ''.join(prefix_i)
            # feature subs: match PyTorch construction
            subs_feat = (
                'bc'
                + ''.join(prefix_i[: i - 1 + min(self.lmax, 1)])
                + 'i,bci->bc'
                + ''.join(prefix_i[: i - 1 + min(self.lmax, 1)])
            )

            # contract_expression shapes
            if num_equivariance == 1:
                shapes_weight = (
                    (num_ell_i,) * i + (num_params_i,),
                    (
                        self.num_elements,
                        num_params_i,
                        self.num_features,
                    ),  # ekc
                    (
                        BATCH_EXAMPLE,
                        self.num_elements,
                    ),
                )
            else:
                shapes_weight = (
                    (num_equivariance,) + (num_ell_i,) * i + (num_params_i,),
                    (
                        self.num_elements,
                        num_params_i,
                        self.num_features,
                    ),  # ekc
                    (
                        BATCH_EXAMPLE,
                        self.num_elements,
                    ),
                )
            expr_w = oe.contract_expression(subs_weight, *shapes_weight)
            self._weight_exprs.append(expr_w)
            self._shapes_weight.append((num_elements, num_params_i, self.num_features))

            # For feature expr: first operand is c_tensor shape WITHOUT batch
            # c_tensor shape (after weighting conv) corresponds to expr result "bc..." -> we
            # must provide the shape of that operand without batch. In PyTorch they used
            # example_inputs with shape (BATCH_EXAMPLE, self.num_features, num_equivariance, ...).
            # We construct a conservative shape consistent with "bci" second operand:
            if num_equivariance == 1:
                shapes_feat = (
                    (BATCH_EXAMPLE, self.num_features) + (num_ell_i,) * i,
                    (BATCH_EXAMPLE, self.num_features, num_ell_i),
                )
            else:
                shapes_feat = (
                    (BATCH_EXAMPLE, self.num_features, num_equivariance)
                    + (num_ell_i,) * i,
                    (BATCH_EXAMPLE, self.num_features, num_ell_i),
                )
            expr_f = oe.contract_expression(subs_feat, *shapes_feat)
            self._feature_exprs.append(expr_f)

        if not internal_weights:
            self.weights = weights[:-1]
            self.weights_max = weights[-1]

    def __call__(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        # Ensure arrays
        x = jnp.asarray(x)  # expected shape (batch, self.num_features, num_ell_main)
        y = jnp.asarray(y)  # expected shape (batch, self.num_elements)

        # expected shape: (batch, c, i) for x
        if x.ndim < 2:
            raise ValueError(
                f'x must have at least 2 dims (batch,c,...) but got shape {x.shape}'
            )
        if x.shape[1] != self.num_features:
            raise ValueError(
                f'Mismatch: x.shape[1] == {x.shape[1]} but self.num_features == {self.num_features}. '
                'Check the Irreps you passed to this Contraction and the input ordering.'
            )
        if y.ndim < 2:
            raise ValueError(f'y must have shape (batch, elements), got {y.shape}')
        if y.shape[1] != self.num_elements:
            raise ValueError(
                f'Mismatch: y.shape[1] == {y.shape[1]} but self.num_elements == {self.num_elements}.'
            )

        weights_max = hk.get_parameter(
            'weights_max',
            shape=(
                self.num_elements,
                self.U_matrices[self.correlation].shape[-1],
                self.num_features,
            ),
            init=hk.initializers.RandomNormal(
                stddev=1.0 / self.U_matrices[self.correlation].shape[-1]
            ),
            dtype=default_dtype(),
        )

        weights = []
        for i in range(1, self.correlation):
            w = hk.get_parameter(
                f'weights_{i}',
                shape=self._shapes_weight[i - 1],
                init=hk.initializers.RandomNormal(
                    stddev=1.0 / self._shapes_weight[i - 1][1]
                ),
                dtype=default_dtype(),
            )
            weights.append(w)

        # If a given path was flagged zero, use zeros instead of the parameter (like EmptyParam)
        # For weights_max it's the last path flag
        if self.zero_flags and self.zero_flags[-1]:
            w_main = jnp.zeros_like(weights_max)
        else:
            w_main = weights_max

        # Call main expr. opt_einsum supports leading batch dims and will broadcast.
        U_main = self.U_matrices[self.correlation]
        out = self._main_expr(U_main, w_main, x, y)

        # subsequent weighting + feature contractions (mirror PyTorch loop order)
        # weights list corresponds to i = correlation-1, correlation-2, ..., 1 in that order
        for idx, (w_param, expr_w, expr_f) in enumerate(
            zip(weights, self._weight_exprs, self._feature_exprs)
        ):
            # nu maps to correlation - (idx+1)
            nu = self.correlation - (idx + 1)
            nu = max(nu, 1)
            U_nu = self.U_matrices[nu]

            # zero-flag handling for this stage (indexing zero_flags[nu-1])
            zero_flag = False
            if len(self.zero_flags) >= nu:
                zero_flag = self.zero_flags[nu - 1]
            if zero_flag:
                w = jnp.zeros_like(w_param)
            else:
                w = w_param

            # compute weighting contraction: expr_w(U_nu, w, y)
            c_tensor = expr_w(U_nu, w, y)
            # add residual
            c_tensor = c_tensor + out
            # feature contraction: expr_f(c_tensor, x)
            out = expr_f(c_tensor, x)

        # reshape final result to (batch, -1) as in PyTorch
        out = jnp.reshape(out, (out.shape[0], -1))
        return out

    def U_tensors(self, nu: int) -> jnp.ndarray:
        return self.U_matrices[int(nu)]


class SymmetricContraction(hk.Module):
    def __init__(
        self,
        irreps_in: Union[str, Irreps],
        irreps_out: Union[str, Irreps],
        correlation: Union[int, dict[Irrep, int]],
        irrep_normalization: str = 'component',
        path_normalization: str = 'element',
        use_reduced_cg: bool = False,
        internal_weights: Optional[bool] = None,
        shared_weights: Optional[bool] = None,
        num_elements: Optional[int] = None,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)

        if irrep_normalization is None:
            irrep_normalization = 'component'
        if path_normalization is None:
            path_normalization = 'element'

        assert irrep_normalization in ['component', 'norm', 'none']
        assert path_normalization in ['element', 'path', 'none']

        self.irreps_in = Irreps(irreps_in)
        self.irreps_out = Irreps(irreps_out)
        self.num_elements = int(num_elements or 1)
        self.use_reduced_cg = bool(use_reduced_cg)

        # Normalize correlation into dict[Irrep, int]
        if not isinstance(correlation, dict):
            corr_val = correlation
            self.correlation = {irrep_out: corr_val for irrep_out in self.irreps_out}

        assert shared_weights or not internal_weights

        if internal_weights is None:
            internal_weights = True

        self.internal_weights = internal_weights
        self.shared_weights = shared_weights

    def __call__(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        outs = []
        for irrep_out in self.irreps_out:
            contraction = Contraction(
                irreps_in=self.irreps_in,
                irrep_out=Irreps(str(irrep_out.ir)),
                correlation=self.correlation[irrep_out],
                internal_weights=self.internal_weights,
                num_elements=self.num_elements,
                # TODO: Bug in MACE implementation, array expected but passing
                # a boolean
                weights=self.shared_weights,
                use_reduced_cg=self.use_reduced_cg,
            )
            outs.append(contraction(x, y))
        return jnp.concatenate(outs, axis=-1)
