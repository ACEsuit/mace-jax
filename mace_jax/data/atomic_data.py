from typing import Optional, Sequence

import torch
import torch.utils.data

from mace_jax.tools import torch_geometric

from .neighborhood import get_neighborhood
from .utils import Configuration


class AtomicData(torch_geometric.data.Data):
    num_graphs: torch.Tensor
    num_nodes: torch.Tensor
    n_edge: torch.Tensor
    batch: torch.Tensor
    edge_index: torch.Tensor
    node_attrs: torch.Tensor
    edge_vectors: torch.Tensor
    edge_lengths: torch.Tensor
    positions: torch.Tensor
    shifts: torch.Tensor
    cell: torch.Tensor
    forces: torch.Tensor
    energy: torch.Tensor
    weight: torch.Tensor

    def __init__(
        self,
        edge_index: torch.Tensor,  # [2, n_edges]
        node_attrs: torch.Tensor,  # [n_nodes, n_node_feats]
        positions: torch.Tensor,  # [n_nodes, 3]
        shifts: torch.Tensor,  # [n_edges, 3],
        cell: Optional[torch.Tensor],  # [3,3]
        weight: Optional[torch.Tensor],  # [,]
        forces: Optional[torch.Tensor],  # [n_nodes, 3]
        energy: Optional[torch.Tensor],  # [, ]
    ):
        # Check shapes
        num_nodes = node_attrs.shape[0]
        n_edge = edge_index.shape[1]

        assert edge_index.shape[0] == 2 and len(edge_index.shape) == 2
        assert positions.shape == (num_nodes, 3)
        assert shifts.shape == (n_edge, 3)
        assert len(node_attrs.shape) in [1, 2]  # one-hot or not
        assert weight is None or len(weight.shape) == 0
        assert cell is None or cell.shape == (3, 3)
        assert forces is None or forces.shape == (num_nodes, 3)
        assert energy is None or len(energy.shape) == 0
        # Aggregate data
        data = {
            "num_nodes": num_nodes,
            "n_edge": n_edge,
            "edge_index": edge_index,
            "positions": positions,
            "shifts": shifts,
            "cell": cell[None, :, :] if cell is not None else None,
            "node_attrs": node_attrs,
            "weight": weight,
            "forces": forces,
            "energy": energy,
        }
        super().__init__(**data)

    @classmethod
    def from_config(
        cls,
        config: Configuration,
        cutoff: float,
    ) -> "AtomicData":
        edge_index, shifts, cell = get_neighborhood(
            positions=config.positions, cutoff=cutoff, pbc=config.pbc, cell=config.cell
        )

        cell = (
            torch.tensor(cell, dtype=torch.get_default_dtype())
            if config.cell is not None
            else None
        )

        weight = (
            torch.tensor(config.weight, dtype=torch.get_default_dtype())
            if config.weight is not None
            else 1
        )

        forces = (
            torch.tensor(config.forces, dtype=torch.get_default_dtype())
            if config.forces is not None
            else None
        )
        energy = (
            torch.tensor(config.energy, dtype=torch.get_default_dtype())
            if config.energy is not None
            else None
        )

        return cls(
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            positions=torch.tensor(config.positions, dtype=torch.get_default_dtype()),
            shifts=torch.tensor(shifts, dtype=torch.long),
            cell=cell,
            node_attrs=torch.tensor(config.atomic_numbers, dtype=torch.long),
            weight=weight,
            forces=forces,
            energy=energy,
        )


def get_data_loader(
    dataset: Sequence[AtomicData],
    batch_size: int,
    shuffle=True,
    drop_last=False,
) -> torch.utils.data.DataLoader:
    return torch_geometric.dataloader.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
    )
