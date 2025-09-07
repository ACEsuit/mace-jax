# mace_jax/tools/contraction.py
from typing import Dict, Optional, List, Any, Tuple, Set, Union

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import opt_einsum as oe
import e3nn_jax as e3nn
from e3nn_jax import Irreps

from mace_jax.tools.cg import U_matrix_real

BATCH_EXAMPLE = 10
ALPHABET = ["w", "x", "v", "n", "z", "r", "t", "y", "u", "o", "p", "s"]


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
        self.U_matrices: Dict[int, jnp.ndarray] = {}
        self.zero_flags: List[bool] = []  # corresponds to path_weight (negated)

        for nu in range(1, self.correlation + 1):
            # Compute U_matrix_real like PyTorch
            raw = U_matrix_real(
                irreps_in=self.coupling_irreps,
                irreps_out=self.irrep_out,
                correlation=nu,
                use_cueq_cg=use_reduced_cg,
                dtype=jnp.float32,
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
            self.U_matrices[nu] = jnp.asarray(U, dtype=jnp.float32)

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
        main_sub = "".join(prefix) + "ik,ekc,bci,be->bc" + "".join(prefix)

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

        # create hk parameters for the weights (matching PyTorch)
        self.weights_max = hk.get_parameter(
            "weights_max",
            shape=(self.num_elements, num_params, self.num_features),
            init=hk.initializers.RandomNormal(stddev=1.0 / num_params),
        )

        # --- Prepare weighting/feature expressions for i < correlation ---
        self.weights: List[jnp.ndarray] = []
        self._weight_exprs: List[Any] = []
        self._feature_exprs: List[Any] = []

        # iterate descending (correlation-1 .. 1) as PyTorch
        for i in range(self.correlation - 1, 0, -1):
            U_i = self.U_matrices[i]
            num_params_i = int(U_i.shape[-1])
            num_ell_i = int(U_i.shape[-2])

            prefix_i = [ALPHABET[j] for j in range(max(0, i + min(self.lmax, 1)))]
            subs_weight = "".join(prefix_i) + "k,ekc,be->bc" + "".join(prefix_i)
            # feature subs: match PyTorch construction
            subs_feat = (
                "bc"
                + "".join(prefix_i[: i - 1 + min(self.lmax, 1)])
                + "i,bci->bc"
                + "".join(prefix_i[: i - 1 + min(self.lmax, 1)])
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

            # create weight parameter (product-basis) like PyTorch
            w = hk.get_parameter(
                f"weights_{i}",
                shape=(self.num_elements, num_params_i, self.num_features),
                init=hk.initializers.RandomNormal(stddev=1.0 / num_params_i),
            )
            self.weights.append(w)

        # store zero flags to decide zeroing at runtime if required
        self.zero_flags = list(self.zero_flags)  # keep order for nu=1..correlation

    def __call__(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        # Ensure arrays
        x = jnp.asarray(x)  # expected shape (batch, self.num_features, num_ell_main)
        y = jnp.asarray(y)  # expected shape (batch, self.num_elements)

        # expected shape: (batch, c, i) for x
        if x.ndim < 2:
            raise ValueError(
                f"x must have at least 2 dims (batch,c,...) but got shape {x.shape}"
            )
        if x.shape[1] != self.num_features:
            raise ValueError(
                f"Mismatch: x.shape[1] == {x.shape[1]} but self.num_features == {self.num_features}. "
                "Check the Irreps you passed to this Contraction and the input ordering."
            )
        if y.ndim < 2:
            raise ValueError(f"y must have shape (batch, elements), got {y.shape}")
        if y.shape[1] != self.num_elements:
            raise ValueError(
                f"Mismatch: y.shape[1] == {y.shape[1]} but self.num_elements == {self.num_elements}."
            )

        # If a given path was flagged zero, use zeros instead of the parameter (like EmptyParam)
        # For weights_max it's the last path flag
        if self.zero_flags and self.zero_flags[-1]:
            w_main = jnp.zeros_like(self.weights_max)
        else:
            w_main = self.weights_max

        # Call main expr. opt_einsum supports leading batch dims and will broadcast.
        U_main = self.U_matrices[self.correlation]
        out = self._main_expr(U_main, w_main, x, y)

        # subsequent weighting + feature contractions (mirror PyTorch loop order)
        # weights list corresponds to i = correlation-1, correlation-2, ..., 1 in that order
        for idx, (w_param, expr_w, expr_f) in enumerate(
            zip(self.weights, self._weight_exprs, self._feature_exprs)
        ):
            # nu maps to correlation - (idx+1)
            nu = self.correlation - (idx + 1)
            if nu < 1:
                nu = 1
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
        correlation: int,
        keep_irrep_out: Set[e3nn.Irrep],
        num_species: int,
        gradient_normalization: Union[str, float] = None,
        symmetric_tensor_product_basis: bool = True,
        off_diagonal: bool = False,
    ):
        super().__init__()
        self.correlation = correlation

        if gradient_normalization is None:
            gradient_normalization = e3nn.config("gradient_normalization")
        if isinstance(gradient_normalization, str):
            gradient_normalization = {"element": 0.0, "path": 1.0}[
                gradient_normalization
            ]
        self.gradient_normalization = gradient_normalization

        if isinstance(keep_irrep_out, str):
            keep_irrep_out = e3nn.Irreps(keep_irrep_out)
            assert all(mul == 1 for mul, _ in keep_irrep_out)

        self.keep_irrep_out = {e3nn.Irrep(ir) for ir in keep_irrep_out}
        self.num_species = num_species
        self.symmetric_tensor_product_basis = symmetric_tensor_product_basis
        self.off_diagonal = off_diagonal

    def __call__(self, input: e3nn.IrrepsArray, index: jnp.ndarray) -> e3nn.IrrepsArray:
        def fn(input: e3nn.IrrepsArray, index: jnp.ndarray):
            # - This operation is parallel on the feature dimension (but each feature has its own parameters)
            # This operation is an efficient implementation of
            # vmap(lambda w, x: FunctionalLinear(irreps_out)(w, concatenate([x, tensor_product(x, x), tensor_product(x, x, x), ...])))(w, x)
            # up to x power self.correlation
            assert input.ndim == 2  # [num_features, irreps_x.dim]
            assert index.ndim == 0  # int

            out = dict()

            for order in range(self.correlation, 0, -1):  # correlation, ..., 1
                if self.off_diagonal:
                    x_ = jnp.roll(input.array, A025582[order - 1])
                else:
                    x_ = input.array

                if self.symmetric_tensor_product_basis:
                    U = e3nn.reduced_symmetric_tensor_product_basis(
                        input.irreps, order, keep_ir=self.keep_irrep_out
                    )
                else:
                    U = e3nn.reduced_tensor_product_basis(
                        [input.irreps] * order, keep_ir=self.keep_irrep_out
                    )
                # U = U / order  # normalization TODO(mario): put back after testing
                # NOTE(mario): The normalization constants (/order and /mul**0.5)
                # has been numerically checked to be correct.

                # TODO(mario) implement norm_p

                # ((w3 x + w2) x + w1) x
                #  \-----------/
                #       out

                for (mul, ir_out), u in zip(U.irreps, U.list):
                    u = u.astype(x_.dtype)
                    # u: ndarray [(irreps_x.dim)^order, multiplicity, ir_out.dim]

                    w = hk.get_parameter(
                        f"w{order}_{ir_out}",
                        (self.num_species, mul, input.shape[0]),
                        dtype=jnp.float32,
                        init=hk.initializers.RandomNormal(
                            stddev=(mul**-0.5) ** (1.0 - self.gradient_normalization)
                        ),
                    )[index]  # [multiplicity, num_features]
                    w = (
                        w * (mul**-0.5) ** self.gradient_normalization
                    )  # normalize weights

                    if ir_out not in out:
                        out[ir_out] = (
                            "special",
                            jnp.einsum("...jki,kc,cj->c...i", u, w, x_),
                        )  # [num_features, (irreps_x.dim)^(oder-1), ir_out.dim]
                    else:
                        out[ir_out] += jnp.einsum(
                            "...ki,kc->c...i", u, w
                        )  # [num_features, (irreps_x.dim)^order, ir_out.dim]

                # ((w3 x + w2) x + w1) x
                #  \----------------/
                #         out (in the normal case)

                for ir_out in out:
                    if isinstance(out[ir_out], tuple):
                        out[ir_out] = out[ir_out][1]
                        continue  # already done (special case optimization above)

                    out[ir_out] = jnp.einsum(
                        "c...ji,cj->c...i", out[ir_out], x_
                    )  # [num_features, (irreps_x.dim)^(oder-1), ir_out.dim]

                # ((w3 x + w2) x + w1) x
                #  \-------------------/
                #           out

            # out[irrep_out] : [num_features, ir_out.dim]
            irreps_out = e3nn.Irreps(sorted(out.keys()))
            return e3nn.IrrepsArray.from_list(
                irreps_out,
                [out[ir][:, None, :] for (_, ir) in irreps_out],
                (input.shape[0],),
            )

        # Treat batch indices using vmap
        shape = jnp.broadcast_shapes(input.shape[:-2], index.shape)
        input = input.broadcast_to(shape + input.shape[-2:])
        index = jnp.broadcast_to(index, shape)

        fn_mapped = fn
        for _ in range(input.ndim - 2):
            fn_mapped = hk.vmap(fn_mapped, split_rng=False)

        return fn_mapped(input, index)
