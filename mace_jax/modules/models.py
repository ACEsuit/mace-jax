from typing import Callable, Dict, List, Optional, Type

import e3nn_jax as e3nn
import haiku as hk
import jax
import jax.numpy as jnp
import jraph
import numpy as np

from .blocks import (
    AtomicEnergiesBlock,
    EquivariantProductBasisBlock,
    InteractionBlock,
    LinearNodeEmbeddingBlock,
    LinearReadoutBlock,
    NonLinearReadoutBlock,
    RadialEmbeddingBlock,
    ScaleShiftBlock,
)
from .utils import get_edge_vectors_and_lengths, safe_norm, sum_nodes_of_the_same_graph


class GeneralMACE(hk.Module):
    def __init__(
        self,
        *,
        r_max: float,
        num_bessel: int,  # Number of Bessel functions, default 8
        num_deriv_in_zero: int,  # Number of zero derivatives at small distances, default 4
        num_deriv_in_one: int,  # Number of zero derivatives at large distances, default 2
        max_ell: int,  # Max spherical harmonic degree, default 3
        interaction_cls: Type[InteractionBlock],
        interaction_cls_first: Type[InteractionBlock],
        num_interactions: int,  # Number of interactions (layers), default 2
        hidden_irreps: e3nn.Irreps,  # 256x0e or 128x0e + 128x1o
        MLP_irreps: e3nn.Irreps,  # Hidden irreps of the MLP in last readout, default 16x0e
        avg_num_neighbors: float,
        correlation: int,  # Correlation order at each layer (~ node_features^correlation), default 3
        gate: Optional[Callable],  # Gate function for the MLP in last readout
        output_irreps: e3nn.Irreps,  # Irreps of the output, default 1x0e
    ):
        super().__init__()
        self.r_max = r_max
        self.correlation = correlation
        self.hidden_irreps = hidden_irreps
        self.avg_num_neighbors = avg_num_neighbors
        self.MLP_irreps = MLP_irreps
        self.gate = gate
        self.interaction_cls_first = interaction_cls_first
        self.interaction_cls = interaction_cls
        self.num_interactions = num_interactions
        self.output_irreps = output_irreps

        # Embeddings
        self.node_embedding = LinearNodeEmbeddingBlock(
            self.hidden_irreps.filter(["0e"])
        )
        self.radial_embedding = RadialEmbeddingBlock(
            r_max=r_max,
            num_bessel=num_bessel,
            num_deriv_in_zero=num_deriv_in_zero,
            num_deriv_in_one=num_deriv_in_one,
        )

        self.sh_irreps = e3nn.Irreps.spherical_harmonics(max_ell)
        num_features = hidden_irreps.count(e3nn.Irrep("0e"))
        self.interaction_irreps = e3nn.Irreps(
            [(num_features, ir) for _, ir in self.sh_irreps]
        )

    def __call__(self, graph: jraph.GraphsTuple) -> e3nn.IrrepsArray:
        # Embeddings
        node_attrs = e3nn.IrrepsArray(
            f"{graph.nodes.attrs.shape[-1]}x0e", graph.nodes.attrs
        )
        node_feats = self.node_embedding(node_attrs)

        # TODO (mario): use jax_md formalism to compute the relative vectors and lengths
        (vectors, lengths,) = get_edge_vectors_and_lengths(
            positions=graph.nodes.positions,
            senders=graph.senders,
            receivers=graph.receivers,
            shifts=graph.edges.shifts,
            cell=graph.globals.cell,
            n_edge=graph.n_edge,
        )
        edge_attrs = e3nn.spherical_harmonics(
            self.sh_irreps,
            vectors / safe_norm(vectors, axis=-1, keepdims=True),
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
            if i == self.num_interactions - 1:  # Select only scalars for last layer
                hidden_irreps_out = self.hidden_irreps.filter(["0e"])
            else:
                hidden_irreps_out = self.hidden_irreps

            if i == 0:  # No residual connection for first layer
                inter = self.interaction_cls_first
            else:
                inter = self.interaction_cls

            node_feats, sc = inter(
                target_irreps=self.interaction_irreps,
                hidden_irreps=hidden_irreps_out,
                avg_num_neighbors=self.avg_num_neighbors,
            )(
                node_attrs=node_attrs,
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                receivers=graph.receivers,
                senders=graph.senders,
            )
            node_feats = EquivariantProductBasisBlock(
                target_irreps=hidden_irreps_out, correlation=self.correlation
            )(node_feats=node_feats, node_attrs=node_attrs)

            if sc is not None:
                node_feats = node_feats + sc

            if i == self.num_interactions - 1:  # Non linear readout for last layer
                node_outputs = NonLinearReadoutBlock(
                    self.MLP_irreps, self.output_irreps, self.gate
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
        r_max: float,
        num_bessel: int,
        num_deriv_in_zero: int,
        num_deriv_in_one: int,
        max_ell: int,
        interaction_cls: Type[InteractionBlock],
        interaction_cls_first: Type[InteractionBlock],
        num_interactions: int,
        hidden_irreps: e3nn.Irreps,
        MLP_irreps: e3nn.Irreps,
        avg_num_neighbors: float,
        atomic_numbers: List[int],  # TODO (mario): Remove this?
        correlation: int,
        gate: Optional[Callable],
        atomic_energies: np.ndarray,
    ):
        super().__init__()
        self.energy_model = GeneralMACE(
            r_max=r_max,
            num_bessel=num_bessel,
            num_deriv_in_zero=num_deriv_in_zero,
            num_deriv_in_one=num_deriv_in_one,
            max_ell=max_ell,
            interaction_cls=interaction_cls,
            interaction_cls_first=interaction_cls_first,
            num_interactions=num_interactions,
            hidden_irreps=hidden_irreps,
            MLP_irreps=MLP_irreps,
            avg_num_neighbors=avg_num_neighbors,
            correlation=correlation,
            gate=gate,
            output_irreps="0e",
        )
        self.atomic_energies_fn = AtomicEnergiesBlock(atomic_energies)

    def __call__(self, graph: jraph.GraphsTuple) -> Dict[str, jnp.ndarray]:
        def energy_fn(positions):
            node_e0 = self.atomic_energies_fn(graph.nodes.attrs)  # [n_nodes, ]
            contributions = self.energy_model(
                graph._replace(nodes=graph.nodes._replace(positions=positions))
            )  # [n_nodes, num_interactions, 0e]
            contributions = contributions.array[:, :, 0]  # [n_nodes, num_interactions]

            node_energies = node_e0 + jnp.sum(contributions, axis=1)  # [n_nodes, ]
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


class ScaleShiftMACE(hk.Module):
    def __init__(
        self,
        atomic_inter_scale: float,
        atomic_inter_shift: float,
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
            node_e0 = self.atomic_energies_fn(graph.nodes.attrs)
            contributions = self.energy_model(
                graph._replace(nodes=graph.nodes._replace(positions=positions))
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

        output = {
            "energy": graph_energies,  # [n_graphs,]
            "forces": -minus_forces,  # [n_nodes, 3]
        }

        return output
