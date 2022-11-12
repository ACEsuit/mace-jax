from typing import Callable, Dict, Optional

import e3nn_jax as e3nn
import haiku as hk
import jax
import jax.numpy as jnp
import jraph
import numpy as np

from ..tools import get_edge_relative_vectors
from .blocks import (
    AgnosticResidualInteractionBlock,
    EquivariantProductBasisBlock,
    LinearNodeEmbeddingBlock,
    LinearReadoutBlock,
    NonLinearReadoutBlock,
    RadialEmbeddingBlock,
    ScaleShiftBlock,
)
from .utils import safe_norm, sum_nodes_of_the_same_graph


try:
    from profile_nn_jax import profile
except ImportError:
    profile = lambda _, x: x


class GeneralMACE(hk.Module):
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
        avg_r_min: float = 0.0,
        num_bessel: int = 8,  # Number of Bessel functions, default 8
        num_deriv_in_zero: Optional[int] = None,
        num_deriv_in_one: Optional[int] = None,
        # Number of zero derivatives at small and large distances, default 4 and 2
        # If both are None, it uses a smooth C^inf envelope function
        max_ell: int = 3,  # Max spherical harmonic degree, default 3
        epsilon: Optional[float] = 0.5,
        correlation: int = 3,  # Correlation order at each layer (~ node_features^correlation), default 3
        max_poly_order: Optional[int] = None,  # TODO (mario): implement it back?
        gate: Callable = jax.nn.silu,  # activation function
    ):
        super().__init__()

        output_irreps = e3nn.Irreps(output_irreps)
        hidden_irreps = e3nn.Irreps(hidden_irreps)
        readout_mlp_irreps = e3nn.Irreps(readout_mlp_irreps)

        self.num_features = hidden_irreps.count(e3nn.Irrep("0e"))
        if not all(mul == self.num_features for mul, _ in hidden_irreps):
            raise ValueError(
                f"All hidden irrep must have the same multiplicity, got {hidden_irreps}"
            )

        self.sh_irreps = e3nn.Irreps.spherical_harmonics(max_ell)
        self.hidden_irreps = e3nn.Irreps([ir for _, ir in hidden_irreps])
        self.interaction_irreps = e3nn.Irreps(e3nn.Irrep.iterator(max_ell))

        self.r_max = r_max
        self.correlation = correlation
        self.max_poly_order = max_poly_order
        self.avg_num_neighbors = avg_num_neighbors
        self.epsilon = epsilon
        self.readout_mlp_irreps = readout_mlp_irreps
        self.activation = gate
        self.num_interactions = num_interactions
        self.output_irreps = output_irreps
        self.num_species = num_species

        # Embeddings
        self.node_embedding = LinearNodeEmbeddingBlock(
            self.num_species, self.num_features, self.hidden_irreps
        )
        self.radial_embedding = RadialEmbeddingBlock(
            r_max=r_max,
            num_bessel=num_bessel,
            num_deriv_in_zero=num_deriv_in_zero,
            num_deriv_in_one=num_deriv_in_one,
            avg_r_min=avg_r_min,
        )

    def __call__(
        self,
        vectors: jnp.ndarray,  # [n_edges, 3]
        node_specie: jnp.ndarray,  # [n_nodes] int between 0 and num_species-1
        senders: jnp.ndarray,  # [n_edges]
        receivers: jnp.ndarray,  # [n_edges]
    ) -> e3nn.IrrepsArray:
        assert vectors.ndim == 2 and vectors.shape[1] == 3
        assert node_specie.ndim == 1
        assert senders.ndim == 1 and receivers.ndim == 1
        assert vectors.shape[0] == senders.shape[0] == receivers.shape[0]

        # Embeddings
        node_feats = self.node_embedding(node_specie)  # [n_nodes, feature, irreps]
        # poly_order = 0  # polynomial order in atom positions of the node features
        # NOTE: we assume hidden_irreps to be scalar only
        node_feats = profile("embedding: node_feats", node_feats)

        # TODO (mario): use jax_md formalism to compute the relative vectors and lengths

        lengths = safe_norm(vectors, axis=-1, keepdims=True)
        edge_attrs = e3nn.spherical_harmonics(
            self.sh_irreps,
            vectors / lengths,
            normalize=False,
            normalization="component",
        )
        edge_feats = self.radial_embedding(lengths)
        edge_feats = e3nn.IrrepsArray(
            f"{edge_feats.shape[-1]}x0e", edge_feats
        )  # [n_edges, irreps]

        edge_attrs = profile("embedding: edge_attrs", edge_attrs)
        edge_feats = profile("embedding: edge_feats", edge_feats)

        # Interactions
        outputs = []
        for i in range(self.num_interactions):
            node_outputs, node_feats = MACELayer(
                first=i == 0,
                last=i == self.num_interactions - 1,
                num_features=self.num_features,
                interaction_irreps=self.interaction_irreps,
                hidden_irreps=self.hidden_irreps,
                avg_num_neighbors=self.avg_num_neighbors,
                activation=self.activation,
                num_species=self.num_species,
                epsilon=self.epsilon,
                correlation=self.correlation,
                output_irreps=self.output_irreps,
                readout_mlp_irreps=self.readout_mlp_irreps,
                name=f"layer_{i}",
            )(
                node_feats,
                node_specie,
                edge_attrs,
                edge_feats,
                senders,
                receivers,
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
        avg_num_neighbors: float,
        activation: Callable,
        num_species: int,
        epsilon: Optional[float],
        correlation: int,
        output_irreps: e3nn.Irreps,
        readout_mlp_irreps: e3nn.Irreps,
        name: Optional[str],
    ) -> None:
        super().__init__(name=name)

        self.first = first
        self.last = last
        self.num_features = num_features
        self.interaction_irreps = interaction_irreps
        self.hidden_irreps = hidden_irreps
        self.avg_num_neighbors = avg_num_neighbors
        self.activation = activation
        self.num_species = num_species
        self.epsilon = epsilon
        self.correlation = correlation
        self.output_irreps = output_irreps
        self.readout_mlp_irreps = readout_mlp_irreps

    def __call__(
        self,
        node_feats: e3nn.IrrepsArray,  # [n_nodes, feature, irreps]
        node_specie: jnp.ndarray,  # [n_nodes] int between 0 and num_species-1
        edge_attrs: e3nn.IrrepsArray,  # [n_edges, irreps]
        edge_feats: e3nn.IrrepsArray,  # [n_edges, irreps]
        senders: jnp.ndarray,  # [n_edges]
        receivers: jnp.ndarray,  # [n_edges]
    ):
        node_feats = profile(f"{self.name}: node_feats", node_feats)

        node_feats, sc = AgnosticResidualInteractionBlock(
            num_features=self.num_features,
            target_irreps=self.interaction_irreps,
            hidden_irreps=self.hidden_irreps,
            avg_num_neighbors=self.avg_num_neighbors,
            activation=self.activation,
            num_species=self.num_species,
        )(
            node_specie=node_specie,
            node_feats=node_feats,
            edge_attrs=edge_attrs,
            edge_feats=edge_feats,
            receivers=receivers,
            senders=senders,
        )

        if self.first:
            # Selector TensorProduct
            node_feats = e3nn.Linear(
                self.interaction_irreps, self.num_features, num_weights=self.num_species
            )(node_specie, node_feats)
            sc = None

        node_feats = profile(f"{self.name}: node_feats after interaction", node_feats)
        if sc is not None:
            sc = profile(f"{self.name}: self-connexion", sc)

        if self.epsilon is not None:
            node_feats *= self.epsilon
        else:
            node_feats /= jnp.sqrt(self.avg_num_neighbors)

        # if self.max_poly_order is None:
        #     new_poly_order = self.correlation * (poly_order + self.sh_irreps.lmax)
        # else:
        #     new_poly_order = self.correlation * poly_order + self.max_poly_order

        node_feats = EquivariantProductBasisBlock(
            num_features=self.num_features,
            target_irreps=self.hidden_irreps,
            correlation=self.correlation,
            # max_poly_order=new_poly_order,
            # input_poly_order=poly_order,
            num_species=self.num_species,
        )(node_feats=node_feats, node_specie=node_specie)
        node_feats = profile(f"{self.name}: node_feats after tensor power", node_feats)

        # poly_order = new_poly_order

        if sc is not None:
            node_feats = node_feats + sc  # [n_nodes, feature, hidden_irreps]

        if not self.last:
            node_outputs = LinearReadoutBlock(self.output_irreps)(
                node_feats.axis_to_mul()
            )  # [n_nodes, output_irreps]
        else:  # Non linear readout for last layer
            node_outputs = NonLinearReadoutBlock(
                self.readout_mlp_irreps,
                self.output_irreps,
                activation=self.activation,
            )(
                node_feats.axis_to_mul()
            )  # [n_nodes, output_irreps]

            # polynomial order of node outputs is infinite in the last layer because of the non-polynomial activation function

        node_outputs = profile(f"{self.name}: node_outputs", node_outputs)
        return node_outputs, node_feats


class MACE(hk.Module):
    def __init__(
        self,
        *,
        atomic_inter_scale: float = 1.0,
        atomic_inter_shift: float = 0.0,
        atomic_energies: np.ndarray,
        **kwargs,
    ):
        super().__init__()
        self.scale_shift = ScaleShiftBlock(
            scale=atomic_inter_scale, shift=atomic_inter_shift
        )
        self.energy_model = GeneralMACE(output_irreps="0e", **kwargs)
        self.atomic_energies = jnp.asarray(atomic_energies)

    def __call__(self, graph: jraph.GraphsTuple) -> Dict[str, jnp.ndarray]:
        def energy_fn(positions):
            vectors = get_edge_relative_vectors(
                positions=positions,
                senders=graph.senders,
                receivers=graph.receivers,
                shifts=graph.edges.shifts,
                cell=graph.globals.cell,
                n_edge=graph.n_edge,
            )

            node_e0 = self.atomic_energies[graph.nodes.specie]  # [n_nodes, ]
            contributions = self.energy_model(
                vectors, graph.nodes.specie, graph.senders, graph.receivers
            )  # [n_nodes, num_interactions, 0e]
            contributions = contributions.array[:, :, 0]  # [n_nodes, num_interactions]

            node_energies = node_e0 + self.scale_shift(
                jnp.sum(contributions, axis=1)
            )  # [n_nodes, ]
            return jnp.sum(node_energies), node_energies

        minus_forces, node_energies = jax.grad(energy_fn, has_aux=True)(
            graph.nodes.positions
        )

        graph_energies = sum_nodes_of_the_same_graph(
            graph, node_energies
        )  # [ n_graphs,]

        return {
            "energy": graph_energies,  # [n_graphs,]
            "forces": -minus_forces,  # [n_nodes, 3]
        }
