from typing import Callable, Dict, Optional, Type

import e3nn_jax as e3nn
import haiku as hk
import jax
import jax.numpy as jnp
import jraph
import numpy as np

from ..tools import get_edge_relative_vectors
from .blocks import (
    AgnosticInteractionBlock,
    AgnosticResidualInteractionBlock,
    AtomicEnergiesBlock,
    EquivariantProductBasisBlock,
    InteractionBlock,
    LinearNodeEmbeddingBlock,
    LinearReadoutBlock,
    NonLinearReadoutBlock,
    RadialEmbeddingBlock,
    ScaleShiftBlock,
)
from .utils import safe_norm, sum_nodes_of_the_same_graph


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
        num_bessel: int = 8,  # Number of Bessel functions, default 8
        num_deriv_in_zero: Optional[int] = None,
        num_deriv_in_one: Optional[int] = None,
        # Number of zero derivatives at small and large distances, default 4 and 2
        # If both are None, it uses a smooth C^inf envelope function
        max_ell: int = 3,  # Max spherical harmonic degree, default 3
        interaction_cls: Type[InteractionBlock] = AgnosticResidualInteractionBlock,
        interaction_cls_first: Type[InteractionBlock] = AgnosticInteractionBlock,
        epsilon: Optional[float] = 0.5,
        correlation: int = 3,  # Correlation order at each layer (~ node_features^correlation), default 3
        gate: Optional[
            Callable
        ] = jax.nn.silu,  # Gate function for the MLP in last readout
    ):
        super().__init__()

        output_irreps = e3nn.Irreps(output_irreps)
        hidden_irreps = e3nn.Irreps(hidden_irreps)
        readout_mlp_irreps = e3nn.Irreps(readout_mlp_irreps)

        self.r_max = r_max
        self.correlation = correlation
        self.hidden_irreps = hidden_irreps
        self.avg_num_neighbors = avg_num_neighbors
        self.epsilon = epsilon
        self.readout_mlp_irreps = readout_mlp_irreps
        self.gate = gate
        self.interaction_cls_first = interaction_cls_first
        self.interaction_cls = interaction_cls
        self.num_interactions = num_interactions
        self.output_irreps = output_irreps

        # Embeddings
        self.node_embedding = LinearNodeEmbeddingBlock(self.hidden_irreps)
        self.radial_embedding = RadialEmbeddingBlock(
            r_max=r_max,
            num_bessel=num_bessel,
            num_deriv_in_zero=num_deriv_in_zero,
            num_deriv_in_one=num_deriv_in_one,
        )

        self.sh_irreps = e3nn.Irreps.spherical_harmonics(max_ell)
        num_features = hidden_irreps.count(e3nn.Irrep("0e"))
        self.interaction_irreps = e3nn.Irreps(
            [(num_features, ir) for ir in e3nn.Irrep.iterator(max_ell)]
        )

    def __call__(
        self,
        vectors: jnp.ndarray,  # [n_edges, 3]
        node_attrs: jnp.ndarray,  # [n_nodes, #scalar_features]
        senders: jnp.ndarray,  # [n_edges]
        receivers: jnp.ndarray,  # [n_edges]
    ) -> e3nn.IrrepsArray:
        # Embeddings
        node_attrs = e3nn.IrrepsArray(f"{node_attrs.shape[-1]}x0e", node_attrs)
        node_feats = self.node_embedding(node_attrs)

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

        # Interactions
        outputs = []
        for i in range(self.num_interactions):

            if i == 0:  # No residual connection for first layer
                inter = self.interaction_cls_first
            else:
                inter = self.interaction_cls

            node_feats: e3nn.IrrepsArray
            sc: Optional[e3nn.IrrepsArray]
            node_feats, sc = inter(
                target_irreps=self.interaction_irreps,
                hidden_irreps=self.hidden_irreps,
                avg_num_neighbors=self.avg_num_neighbors,
            )(
                node_attrs=node_attrs,
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                receivers=receivers,
                senders=senders,
            )

            if self.epsilon is not None:
                node_feats *= self.epsilon
            else:
                node_feats /= jnp.sqrt(self.avg_num_neighbors)

            node_feats = EquivariantProductBasisBlock(
                target_irreps=self.hidden_irreps, correlation=self.correlation
            )(node_feats=node_feats, node_attrs=node_attrs)

            if sc is not None:
                node_feats = node_feats + sc

            if i == self.num_interactions - 1:  # Non linear readout for last layer
                node_outputs = NonLinearReadoutBlock(
                    self.readout_mlp_irreps, self.output_irreps, activation=self.gate
                )(
                    node_feats
                )  # [n_nodes, output_irreps]
            else:
                node_outputs = LinearReadoutBlock(self.output_irreps)(
                    node_feats
                )  # [n_nodes, output_irreps]

            outputs += [node_outputs]  # list of [n_nodes, output_irreps]

        return e3nn.stack(outputs, axis=1)  # [n_nodes, num_interactions, output_irreps]


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
        self.atomic_energies_fn = AtomicEnergiesBlock(atomic_energies)

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

            node_e0 = self.atomic_energies_fn(graph.nodes.attrs)  # [n_nodes, ]
            contributions = self.energy_model(
                vectors, graph.nodes.attrs, graph.senders, graph.receivers
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
