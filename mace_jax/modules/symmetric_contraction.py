from typing import Set, Optional, Union

import e3nn_jax as e3nn
import haiku as hk
import jax.numpy as jnp


class SymmetricContraction(hk.Module):
    def __init__(
        self,
        correlation: int,
        keep_irrep_out: Set[e3nn.Irrep],
        num_species: int,
        max_poly_order: Optional[int] = None,
        input_poly_order: int = 0,
        gradient_normalization: Union[str, float] = None,
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
        self.max_poly_order = max_poly_order
        self.input_poly_order = input_poly_order

    def __call__(self, x: e3nn.IrrepsArray, y: jnp.ndarray) -> e3nn.IrrepsArray:
        def fn(x: e3nn.IrrepsArray, y: jnp.ndarray):
            # - This operation is parallel on the feature dimension (but each feature has its own parameters)
            # - y is trivially contracted with the parameters
            # This operation is an efficient implementation of
            # vmap(lambda w, x: FunctionalLinear(irreps_out)(w, concatenate([x, tensor_product(x, x), tensor_product(x, x, x), ...])))(w, x)
            # up to x power self.correlation
            assert x.ndim == 2  # [num_features, irreps_x.dim]
            assert y.ndim == 0  # int

            out = dict()

            for order in range(self.correlation, 0, -1):  # correlation, ..., 1
                U = e3nn.reduced_symmetric_tensor_product_basis(
                    x.irreps,
                    order,
                    keep_ir=self.keep_irrep_out,
                    max_order=self.max_poly_order - order * self.input_poly_order
                    if self.max_poly_order is not None
                    else None,
                )
                U = U / order  # normalization
                # NOTE(mario): The normalization constants (/order and /mul**0.5)
                # has been numerically checked to be correct.

                # TODO(mario) implement norm_p

                # ((w3 x + w2) x + w1) x
                #  \-----------/
                #       out

                for (mul, ir_out), u in zip(U.irreps, U.list):
                    # u: ndarray [(irreps_x.dim)^order, multiplicity, ir_out.dim]

                    w = hk.get_parameter(
                        f"w{order}_{ir_out}",
                        (self.num_species, mul, x.shape[0]),
                        dtype=jnp.float32,
                        init=hk.initializers.RandomNormal(
                            stddev=(mul**-0.5) ** (1.0 - self.gradient_normalization)
                        ),
                    )[
                        y
                    ]  # [multiplicity, num_features]
                    w = (
                        w * (mul**-0.5) ** self.gradient_normalization
                    )  # normalize weights

                    if ir_out not in out:
                        out[ir_out] = (
                            "special",
                            jnp.einsum("...jki,kc,cj->c...i", u, w, x.array),
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
                        "c...ji,cj->c...i", out[ir_out], x.array
                    )  # [num_features, (irreps_x.dim)^(oder-1), ir_out.dim]

                # ((w3 x + w2) x + w1) x
                #  \-------------------/
                #           out

            # out[irrep_out] : [num_features, ir_out.dim]
            irreps_out = e3nn.Irreps(sorted(out.keys()))
            return e3nn.IrrepsArray.from_list(
                irreps_out,
                [out[ir][:, None, :] for (_, ir) in irreps_out],
                (x.shape[0],),
            )

        # Treat batch indices using vmap
        shape = jnp.broadcast_shapes(x.shape[:-2], y.shape)
        x = x.broadcast_to(shape + x.shape[-2:])
        y = jnp.broadcast_to(y, shape)

        fn_mapped = fn
        for _ in range(x.ndim - 2):
            fn_mapped = hk.vmap(fn_mapped, split_rng=False)

        return fn_mapped(x, y)
