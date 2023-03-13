import functools
import math
from typing import Callable, Optional, Union

import e3nn_jax as e3nn
import haiku as hk
import jax
import jax.numpy as jnp

from ..tools import safe_norm
from .blocks import (
    EquivariantProductBasisBlock,
    InteractionBlock,
    LinearNodeEmbeddingBlock,
    LinearReadoutBlock,
    NonLinearReadoutBlock,
    RadialEmbeddingBlock,
)

try:
    from profile_nn_jax import profile
except ImportError:

    def profile(_, x, __=None):
        return x


class MACE(hk.Module):
    def __init__(
        self,
        *,
        output_irreps: e3nn.Irreps,  # Irreps of the output, default 1x0e
        r_max: float,
        num_interactions: int,  # Number of interactions (layers), default 2
        hidden_irreps: e3nn.Irreps,  # 256x0e or 128x0e + 128x1o
        readout_mlp_irreps: e3nn.Irreps,  # Hidden irreps of the MLP in last readout, default 16x0e
        avg_num_neighbors: float,
        num_species: int,
        num_features: int = None,  # Number of features per node, default gcd of hidden_irreps multiplicities
        avg_r_min: float = None,
        radial_basis: Callable[[jnp.ndarray], jnp.ndarray],
        radial_envelope: Callable[[jnp.ndarray], jnp.ndarray],
        # Number of zero derivatives at small and large distances, default 4 and 2
        # If both are None, it uses a smooth C^inf envelope function
        max_ell: int = 3,  # Max spherical harmonic degree, default 3
        epsilon: Optional[float] = None,
        correlation: int = 3,  # Correlation order at each layer (~ node_features^correlation), default 3
        gate: Callable = jax.nn.silu,  # activation function
        symmetric_tensor_product_basis: bool = True,
        off_diagonal: bool = False,
        interaction_irreps: Union[str, e3nn.Irreps] = "o3_restricted",  # or o3_full
        node_embedding: hk.Module = LinearNodeEmbeddingBlock,
    ):
        super().__init__()

        output_irreps = e3nn.Irreps(output_irreps)
        hidden_irreps = e3nn.Irreps(hidden_irreps)
        readout_mlp_irreps = e3nn.Irreps(readout_mlp_irreps)

        if num_features is None:
            self.num_features = functools.reduce(
                math.gcd, (mul for mul, _ in hidden_irreps)
            )
            self.hidden_irreps = e3nn.Irreps(
                [(mul // self.num_features, ir) for mul, ir in hidden_irreps]
            )
        else:
            self.num_features = num_features
            self.hidden_irreps = hidden_irreps

        if interaction_irreps == "o3_restricted":
            self.interaction_irreps = e3nn.Irreps.spherical_harmonics(max_ell)
        elif interaction_irreps == "o3_full":
            self.interaction_irreps = e3nn.Irreps(e3nn.Irrep.iterator(max_ell))
        else:
            self.interaction_irreps = e3nn.Irreps(interaction_irreps)

        self.r_max = r_max
        self.correlation = correlation
        self.avg_num_neighbors = avg_num_neighbors
        self.epsilon = epsilon
        self.readout_mlp_irreps = readout_mlp_irreps
        self.activation = gate
        self.num_interactions = num_interactions
        self.output_irreps = output_irreps
        self.num_species = num_species
        self.symmetric_tensor_product_basis = symmetric_tensor_product_basis
        self.off_diagonal = off_diagonal
        self.max_ell = max_ell

        # Embeddings
        self.node_embedding = node_embedding(
            self.num_species, self.num_features * self.hidden_irreps
        )
        self.radial_embedding = RadialEmbeddingBlock(
            r_max=r_max,
            avg_r_min=avg_r_min,
            basis_functions=radial_basis,
            envelope_function=radial_envelope,
        )

    def __call__(
        self,
        vectors: jnp.ndarray,  # [n_edges, 3]
        node_specie: jnp.ndarray,  # [n_nodes] int between 0 and num_species-1
        senders: jnp.ndarray,  # [n_edges]
        receivers: jnp.ndarray,  # [n_edges]
        node_mask: Optional[jnp.ndarray] = None,  # [n_nodes] only used for profiling
    ) -> e3nn.IrrepsArray:
        assert vectors.ndim == 2 and vectors.shape[1] == 3
        assert node_specie.ndim == 1
        assert senders.ndim == 1 and receivers.ndim == 1
        assert vectors.shape[0] == senders.shape[0] == receivers.shape[0]

        if node_mask is None:
            node_mask = jnp.ones(node_specie.shape[0], dtype=jnp.bool_)

        # Embeddings
        node_feats = self.node_embedding(node_specie).astype(
            vectors.dtype
        )  # [n_nodes, feature * irreps]
        node_feats = profile("embedding: node_feats", node_feats, node_mask[:, None])

        radial_embedding = self.radial_embedding(safe_norm(vectors, axis=-1))
        vectors = e3nn.IrrepsArray("1o", vectors)

        # Interactions
        outputs = []
        for i in range(self.num_interactions):
            first = i == 0
            last = i == self.num_interactions - 1

            hidden_irreps = (
                self.hidden_irreps
                if not last
                else self.hidden_irreps.filter(self.output_irreps)
            )

            node_outputs, node_feats = MACELayer(
                first=first,
                last=last,
                num_features=self.num_features,
                interaction_irreps=self.interaction_irreps,
                hidden_irreps=hidden_irreps,
                max_ell=self.max_ell,
                avg_num_neighbors=self.avg_num_neighbors,
                activation=self.activation,
                num_species=self.num_species,
                epsilon=self.epsilon,
                correlation=self.correlation,
                output_irreps=self.output_irreps,
                readout_mlp_irreps=self.readout_mlp_irreps,
                symmetric_tensor_product_basis=self.symmetric_tensor_product_basis,
                off_diagonal=self.off_diagonal,
                name=f"layer_{i}",
            )(
                vectors,
                node_feats,
                node_specie,
                radial_embedding,
                senders,
                receivers,
                node_mask,
            )
            outputs += [node_outputs]  # list of [n_nodes, output_irreps]

        return e3nn.stack(outputs, axis=1)  # [n_nodes, num_interactions, output_irreps]


class MACELayer(hk.Module):
    def __init__(
        self,
        *,
        first: bool,
        last: bool,
        num_features: int,
        interaction_irreps: e3nn.Irreps,
        hidden_irreps: e3nn.Irreps,
        activation: Callable,
        num_species: int,
        epsilon: Optional[float],
        name: Optional[str],
        # InteractionBlock:
        max_ell: int,
        avg_num_neighbors: float,
        # EquivariantProductBasisBlock:
        correlation: int,
        symmetric_tensor_product_basis: bool,
        off_diagonal: bool,
        # ReadoutBlock:
        output_irreps: e3nn.Irreps,
        readout_mlp_irreps: e3nn.Irreps,
    ) -> None:
        super().__init__(name=name)

        self.first = first
        self.last = last
        self.num_features = num_features
        self.interaction_irreps = interaction_irreps
        self.hidden_irreps = hidden_irreps
        self.max_ell = max_ell
        self.avg_num_neighbors = avg_num_neighbors
        self.activation = activation
        self.num_species = num_species
        self.epsilon = epsilon
        self.correlation = correlation
        self.output_irreps = output_irreps
        self.readout_mlp_irreps = readout_mlp_irreps
        self.symmetric_tensor_product_basis = symmetric_tensor_product_basis
        self.off_diagonal = off_diagonal

    def __call__(
        self,
        vectors: e3nn.IrrepsArray,  # [n_edges, 3]
        node_feats: e3nn.IrrepsArray,  # [n_nodes, irreps]
        node_specie: jnp.ndarray,  # [n_nodes] int between 0 and num_species-1
        radial_embedding: jnp.ndarray,  # [n_edges, radial_embedding_dim]
        senders: jnp.ndarray,  # [n_edges]
        receivers: jnp.ndarray,  # [n_edges]
        node_mask: Optional[jnp.ndarray] = None,  # [n_nodes] only used for profiling
    ):
        if node_mask is None:
            node_mask = jnp.ones(node_specie.shape[0], dtype=jnp.bool_)

        node_feats = profile(f"{self.name}: input", node_feats, node_mask[:, None])

        sc = None
        if not self.first:
            sc = e3nn.haiku.Linear(
                self.num_features * self.hidden_irreps,
                num_indexed_weights=self.num_species,
                name="skip_tp",
            )(
                node_specie, node_feats
            )  # [n_nodes, feature * hidden_irreps]
            sc = profile(f"{self.name}: self-connexion", sc, node_mask[:, None])

        node_feats = InteractionBlock(
            target_irreps=self.num_features * self.interaction_irreps,
            avg_num_neighbors=self.avg_num_neighbors,
            max_ell=self.max_ell,
            activation=self.activation,
        )(
            vectors=vectors,
            node_feats=node_feats,
            radial_embedding=radial_embedding,
            receivers=receivers,
            senders=senders,
        )

        if self.epsilon is not None:
            node_feats *= self.epsilon
        else:
            node_feats /= jnp.sqrt(self.avg_num_neighbors)

        node_feats = profile(
            f"{self.name}: interaction", node_feats, node_mask[:, None]
        )

        if self.first:
            # Selector TensorProduct
            node_feats = e3nn.haiku.Linear(
                self.num_features * self.interaction_irreps,
                num_indexed_weights=self.num_species,
                name="skip_tp_first",
            )(node_specie, node_feats)
            node_feats = profile(
                f"{self.name}: skip_tp_first", node_feats, node_mask[:, None]
            )
            sc = None

        node_feats = EquivariantProductBasisBlock(
            target_irreps=self.num_features * self.hidden_irreps,
            correlation=self.correlation,
            num_species=self.num_species,
            symmetric_tensor_product_basis=self.symmetric_tensor_product_basis,
            off_diagonal=self.off_diagonal,
        )(node_feats=node_feats, node_specie=node_specie)

        node_feats = profile(
            f"{self.name}: tensor power", node_feats, node_mask[:, None]
        )

        if sc is not None:
            node_feats = node_feats + sc  # [n_nodes, feature * hidden_irreps]

        if not self.last:
            node_outputs = LinearReadoutBlock(self.output_irreps)(
                node_feats
            )  # [n_nodes, output_irreps]
        else:  # Non linear readout for last layer
            node_outputs = NonLinearReadoutBlock(
                self.readout_mlp_irreps,
                self.output_irreps,
                activation=self.activation,
            )(
                node_feats
            )  # [n_nodes, output_irreps]

        node_outputs = profile(f"{self.name}: output", node_outputs, node_mask[:, None])
        return node_outputs, node_feats
