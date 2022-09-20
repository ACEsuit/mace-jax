from typing import Any, Callable, Dict, List, Optional, Type

import numpy as np
import jax
import jax.numpy as jnp
import e3nn_jax as e3nn
import haiku as hk
import jraph

from mace_jax.data import AtomicData
from mace_jax.tools.scatter import scatter_sum

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
from .utils import compute_forces, get_edge_vectors_and_lengths


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

        # Interactions and readout
        self.atomic_energies_fn = AtomicEnergiesBlock(atomic_energies)

    def __call__(self, graph: jraph.GraphTuple) -> Dict[str, Any]:
        def energy_fn(positions):
            # Setup
            num_graphs = graph.n_node.shape[0]
            num_nodes = positions.shape[0]
            num_edges = len(graph.senders)
            graph_index = jnp.repeat(
                jnp.arange(num_graphs), graph.n_node, total_repeat_length=num_nodes
            )  # (node,)

            # Atomic energies
            node_e0 = self.atomic_energies_fn(graph.nodes.attrs)
            e0 = e3nn.index_add(
                indices=graph_index, input=node_e0, out_dim=num_graphs
            )  # [n_graphs,]
            # pyg: batch=[0,0,0,0,1,1,1] #[n_nodes]
            # jraph: n_node=[4,3] #[n_graphs]

            # Embeddings
            node_attrs = e3nn.IrrepsArray(
                f"{graph.nodes.attrs.shape[-1]}x0e", graph.nodes.attrs
            )
            node_feats = self.node_embedding(node_attrs)
            vectors, lengths = get_edge_vectors_and_lengths(
                positions=positions,
                senders=graph.senders,
                receivers=graph.receivers,
                shifts=graph.edges.shifts,
            )
            edge_attrs = e3nn.spherical_harmonics(
                self.sh_irreps, vectors, normalize=True, normalization="component"
            )
            edge_feats = self.radial_embedding(lengths)
            edge_feats = e3nn.IrrepsArray(
                f"{edge_feats.shape[-1]}x0e", edge_feats
            )  # [n_edges, irreps]

            # Interactions
            energies = [e0]
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
                    target_irreps=hidden_irreps_out, correlation=self.correlation,
                )
                node_feats = prod(node_feats=node_feats, sc=sc, node_attrs=node_attrs)
                if i == self.num_interactions - 1:
                    readout = NonLinearReadoutBlock(self.MLP_irreps, self.gate)
                    node_energies = readout(node_feats).array.squeeze(-1)  # [n_nodes, ]
                else:
                    readout = LinearReadoutBlock(self.hidden_irreps)
                    node_energies = readout(node_feats).array.squeeze(-1)  # [n_nodes, ]
                energy = e3nn.index_add(
                    indices=graph_index, input=node_energies, out_dim=num_graphs
                )  # [n_graphs,]
                energies.append(energy)

            # Sum over energy contributions
            contributions = jnp.stack(energies, axis=0)  # [num_layers, n_graphs]
            total_energy = jnp.sum(contributions, axis=0)  # [n_graphs, ]
            return jnp.sum(total_energy), total_energy, contributions

        (_, (total_energy, contributions)), minus_forces = jax.value_and_grad(
            energy_fn, has_aux=True
        )(graph.nodes.positions)

        return {
            "energy": total_energy,
            "contributions": contributions,
            "forces": -minus_forces,
        }


class ScaleShiftMACE(MACE):
    def __init__(
        self, atomic_inter_scale: float, atomic_inter_shift: float, **kwargs,
    ):
        super().__init__(**kwargs)
        self.scale_shift = ScaleShiftBlock(
            scale=atomic_inter_scale, shift=atomic_inter_shift
        )

    def forward(self, data: AtomicData, training=False) -> Dict[str, Any]:
        # Setup
        data.positions.requires_grad = True

        # Atomic energies
        node_e0 = self.atomic_energies_fn(data.node_attrs)
        e0 = scatter_sum(
            src=node_e0, index=data.batch, dim=-1, dim_size=data.num_graphs
        )  # [n_graphs,]

        # Embeddings
        node_feats = self.node_embedding(data.node_attrs)
        vectors, lengths = get_edge_vectors_and_lengths(
            positions=data.positions, edge_index=data.edge_index, shifts=data.shifts
        )
        edge_attrs = self.spherical_harmonics(vectors)
        edge_feats = self.radial_embedding(lengths)

        # Interactions
        node_es_list = []
        for interaction, product, readout in zip(
            self.interactions, self.products, self.readouts
        ):
            node_feats, sc = interaction(
                node_attrs=data.node_attrs,
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=data.edge_index,
            )
            node_feats = product(
                node_feats=node_feats, sc=sc, node_attrs=data.node_attrs
            )
            node_es_list.append(readout(node_feats).squeeze(-1))  # {[n_nodes, ], }

        # Sum over interactions
        node_inter_es = torch.sum(
            torch.stack(node_es_list, dim=0), dim=0
        )  # [n_nodes, ]
        node_inter_es = self.scale_shift(node_inter_es)

        # Sum over nodes in graph
        inter_e = scatter_sum(
            src=node_inter_es, index=data.batch, dim=-1, dim_size=data.num_graphs
        )  # [n_graphs,]

        # Add E_0 and (scaled) interaction energy
        total_e = e0 + inter_e

        output = {
            "energy": total_e,
            "forces": compute_forces(
                energy=inter_e, positions=data.positions, training=training
            ),
        }

        return output
