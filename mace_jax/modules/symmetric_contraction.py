from typing import Set, Optional

import e3nn_jax as e3nn
import haiku as hk
import jax.numpy as jnp


class SymmetricContraction(hk.Module):
    def __init__(
        self,
        correlation: int,
        keep_irrep_out: Set[e3nn.Irrep],
        max_poly_order: Optional[int] = None,
    ):
        super().__init__()
        self.correlation = correlation

        if isinstance(keep_irrep_out, str):
            keep_irrep_out = e3nn.Irreps(keep_irrep_out)
            assert all(mul == 1 for mul, _ in keep_irrep_out)

        self.keep_irrep_out = {e3nn.Irrep(ir) for ir in keep_irrep_out}
        self.max_poly_order = max_poly_order

    def __call__(self, x: e3nn.IrrepsArray, y: jnp.ndarray) -> e3nn.IrrepsArray:
        def fn(x: e3nn.IrrepsArray, y: jnp.ndarray):
            # - This operation is parallel on the feature dimension (but each feature has its own parameters)
            # - y is trivially contracted with the parameters
            # This operation is an efficient implementation of
            # vmap(lambda w, x: FunctionalLinear(irreps_out)(w, concatenate([x, tensor_product(x, x), tensor_product(x, x, x), ...])))(w, x)
            # up to x power self.correlation
            assert x.ndim == 2  # [num_features, irreps_x.dim]
            assert y.ndim == 1  # [num_elements]

            out = dict()

            for order in range(self.correlation, 0, -1):  # correlation, ..., 1
                U = e3nn.reduced_symmetric_tensor_product_basis(
                    x.irreps,
                    order,
                    keep_ir=self.keep_irrep_out,
                    max_order=self.max_poly_order,
                )

                # ((w3 x + w2) x + w1) x
                #  \-----------/
                #       out

                for (mul, ir_out), u in zip(U.irreps, U.list):
                    # u: ndarray [(irreps_x.dim)^order, multiplicity, ir_out.dim]

                    w = hk.get_parameter(
                        f"w{order}_{ir_out}",
                        (mul, y.shape[0], x.shape[0]),
                        dtype=jnp.float32,
                        init=hk.initializers.RandomNormal(),
                    )  # [multiplicity, num_elements, num_features]
                    w = w / w.shape[0]  # normalize
                    if ir_out not in out:
                        out[ir_out] = (
                            "special",
                            jnp.einsum("...jki,kec,e,cj->c...i", u, w, y, x.array),
                        )  # [num_features, (irreps_x.dim)^(oder-1), ir_out.dim]
                    else:
                        out[ir_out] += jnp.einsum(
                            "...ki,kec,e->c...i", u, w, y
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
        shape = jnp.broadcast_shapes(x.shape[:-2], y.shape[:-1])
        x = x.broadcast_to(shape + x.shape[-2:])
        y = jnp.broadcast_to(y, shape + y.shape[-1:])

        fn_mapped = fn
        for _ in range(x.ndim - 2):
            fn_mapped = hk.vmap(fn_mapped, split_rng=False)

        return fn_mapped(x, y)
