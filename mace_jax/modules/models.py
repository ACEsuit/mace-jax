from typing import Any, Callable, Dict, List, Optional, Type

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
from .utils import get_edge_vectors_and_lengths, safe_norm


class EnergyMACE(hk.Module):
    def __init__(
        self,
        r_max: float,
        num_bessel: int,
        num_polynomial_cutoff: int,
        max_ell: int,
        interaction_cls: Type[InteractionBlock],
        interaction_cls_first: Type[InteractionBlock],
        num_interactions: int,
        hidden_irreps: e3nn.Irreps,
        MLP_irreps: e3nn.Irreps,
        atomic_energies: np.ndarray,  # TODO (mario): remove this ?
        avg_num_neighbors: float,
        atomic_numbers: List[int],
        correlation: int,
        gate: Optional[Callable],
    ):
        super().__init__()
        self.r_max = r_max
        self.atomic_numbers = atomic_numbers
        self.correlation = correlation
        self.hidden_irreps = hidden_irreps
        self.avg_num_neighbors = avg_num_neighbors
        self.MLP_irreps = MLP_irreps
        self.gate = gate
        self.interaction_cls_first = interaction_cls_first
        self.interaction_cls = interaction_cls
        self.num_interactions = num_interactions
        # Embedding
        node_feats_irreps = e3nn.Irreps([(hidden_irreps.count(e3nn.Irrep("0e")), "0e")])
        self.node_embedding = LinearNodeEmbeddingBlock(irreps_out=node_feats_irreps)
        self.radial_embedding = RadialEmbeddingBlock(
            r_max=r_max,
            num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
        )

        self.sh_irreps = e3nn.Irreps.spherical_harmonics(max_ell)
        num_features = hidden_irreps.count(e3nn.Irrep("0e"))
        self.interaction_irreps = (self.sh_irreps * num_features).sort()[0].simplify()

    def __call__(self, graph: jraph.GraphsTuple) -> Dict[str, Any]:
        # Embeddings
        node_attrs = e3nn.IrrepsArray(
            f"{graph.nodes.attrs.shape[-1]}x0e", graph.nodes.attrs
        )
        node_feats = self.node_embedding(node_attrs)
        vectors, lengths = get_edge_vectors_and_lengths(
            positions=graph.nodes.positions,
            senders=graph.senders,
            receivers=graph.receivers,
            shifts=graph.edges.shifts,
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
            if i == self.num_interactions - 1:
                hidden_irreps_out = str(
                    self.hidden_irreps[0]
                )  # Select only scalars for last layer
            else:
                hidden_irreps_out = self.hidden_irreps
            if i == 0:
                inter = self.interaction_cls_first(
                    target_irreps=self.interaction_irreps,
                    hidden_irreps=hidden_irreps_out,
                    avg_num_neighbors=self.avg_num_neighbors,
                )
            else:
                inter = self.interaction_cls(
                    target_irreps=self.interaction_irreps,
                    hidden_irreps=hidden_irreps_out,
                    avg_num_neighbors=self.avg_num_neighbors,
                )
            node_feats, sc = inter(
                node_attrs=node_attrs,
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                receivers=graph.receivers,
                senders=graph.senders,
            )
            prod = EquivariantProductBasisBlock(
                target_irreps=hidden_irreps_out, correlation=self.correlation
            )
            node_feats = prod(node_feats=node_feats, sc=sc, node_attrs=node_attrs)
            if i == self.num_interactions - 1:
                readout = NonLinearReadoutBlock(self.MLP_irreps, self.gate)
                node_energies = readout(node_feats).array.squeeze(-1)  # [n_nodes, ]
            else:
                readout = LinearReadoutBlock()
                node_energies = readout(node_feats).array.squeeze(-1)  # [n_nodes, ]
            outputs += [node_energies]

        return jnp.stack(outputs, axis=1)  # [n_nodes, num_interactions]


class MACE(hk.Module):
    def __init__(
        self,
        r_max: float,
        num_bessel: int,
        num_polynomial_cutoff: int,
        max_ell: int,
        interaction_cls: Type[InteractionBlock],
        interaction_cls_first: Type[InteractionBlock],
        num_interactions: int,
        hidden_irreps: e3nn.Irreps,
        MLP_irreps: e3nn.Irreps,
        atomic_energies: np.ndarray,
        avg_num_neighbors: float,
        atomic_numbers: List[int],
        correlation: int,
        gate: Optional[Callable],
    ):
        super().__init__()
        self.energy_model = EnergyMACE(
            r_max=r_max,
            num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
            max_ell=max_ell,
            interaction_cls=interaction_cls,
            interaction_cls_first=interaction_cls_first,
            num_interactions=num_interactions,
            hidden_irreps=hidden_irreps,
            MLP_irreps=MLP_irreps,
            atomic_energies=atomic_energies,
            avg_num_neighbors=avg_num_neighbors,
            atomic_numbers=atomic_numbers,
            correlation=correlation,
            gate=gate,
        )
        self.atomic_energies_fn = AtomicEnergiesBlock(atomic_energies)

    def __call__(self, graph: jraph.GraphsTuple) -> Dict[str, Any]:
        def energy_fn(positions):
            node_e0 = self.atomic_energies_fn(graph.nodes.attrs)  # [n_nodes, ]
            contributions = self.energy_model(
                graph._replace(nodes=graph.nodes._replace(positions=positions))
            )
            node_energies = jnp.sum(contributions, axis=1)  # [n_nodes, ]
            node_energies = node_energies + node_e0  # [n_nodes, ]
            return jnp.sum(node_energies), node_energies

        minus_forces, node_energies = jax.grad(energy_fn, has_aux=True)(
            graph.nodes.positions
        )

        num_graphs = graph.n_node.shape[0]
        num_nodes = graph.nodes.positions.shape[0]
        graph_index = jnp.repeat(
            jnp.arange(num_graphs), graph.n_node, total_repeat_length=num_nodes
        )  # (node,)
        graph_energies = e3nn.index_add(
            indices=graph_index, input=node_energies, out_dim=num_graphs
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
        **kwargs,
    ):
        super().__init__()
        self.scale_shift = ScaleShiftBlock(
            scale=atomic_inter_scale, shift=atomic_inter_shift
        )
        self.energy_model = EnergyMACE(**kwargs)
        self.atomic_energies_fn = AtomicEnergiesBlock(kwargs["atomic_energies"])

    def __call__(self, graph: jraph.GraphsTuple) -> Dict[str, Any]:
        def energy_fn(positions):
            node_e0 = self.atomic_energies_fn(
                graph.nodes.attrs
            )  # TODO (mario): check if we need this variable
            contributions = self.energy_model(
                graph._replace(nodes=graph.nodes._replace(positions=positions))
            )  # [n_nodes, num_interactions]
            node_energies = jnp.sum(contributions, axis=1)  # [n_nodes, ]
            node_inter_es = self.scale_shift(node_energies)  # [n_nodes, ]

            node_energies = node_energies + node_inter_es  # [n_nodes, ]
            return jnp.sum(node_energies), node_energies

        minus_forces, node_energies = jax.grad(energy_fn, has_aux=True)(
            graph.nodes.positions
        )

        num_graphs = graph.n_node.shape[0]
        num_nodes = graph.nodes.positions.shape[0]
        graph_index = jnp.repeat(
            jnp.arange(num_graphs), graph.n_node, total_repeat_length=num_nodes
        )  # (node,)
        graph_energies = e3nn.index_add(
            indices=graph_index, input=node_energies, out_dim=num_graphs
        )  # [ n_graphs,]

        output = {
            "energy": graph_energies,  # [n_graphs,]
            "forces": -minus_forces,  # [n_nodes, 3]
        }

        return output
