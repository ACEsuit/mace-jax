from typing import Optional
import e3nn_jax as e3nn
import haiku as hk
import jax.numpy as jnp

from .symmetric_contraction import SymmetricContraction


class EquivariantProductBasisBlock(hk.Module):
    def __init__(
        self,
        target_irreps: e3nn.Irreps,
        correlation: int,
    ) -> None:
        super().__init__()

        target_irreps = e3nn.Irreps(target_irreps)

        self.symmetric_contractions = SymmetricContraction(
            keep_irrep_out=target_irreps, correlation=correlation
        )

        self.linear = e3nn.Linear(target_irreps)

    def __call__(
        self,
        node_feats: e3nn.IrrepsArray,
        node_attrs: jnp.ndarray,
        sc: Optional[e3nn.IrrepsArray] = None,
    ) -> e3nn.IrrepsArray:
        node_feats = self.symmetric_contractions(node_feats, node_attrs)
        node_feats = self.linear(node_feats)

        return node_feats + sc if sc is not None else node_feats
