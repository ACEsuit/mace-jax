from typing import Callable, Dict, Optional

import jax

from .blocks import (
    AtomicEnergiesBlock,
    EquivariantProductBasisBlock,
    LinearNodeEmbeddingBlock,
    LinearReadoutBlock,
    NonLinearReadoutBlock,
    RadialEmbeddingBlock,
    AgnosticResidualInteractionBlock,
    ScaleShiftBlock,
)

from .loss import WeightedEnergyForcesLoss
from .models import GeneralMACE, MACE
from .symmetric_contraction import SymmetricContraction

from .utils import (
    compute_avg_num_neighbors,
    compute_avg_min_neighbor_distance,
    compute_mean_rms_energy_forces,
    compute_mean_std_atomic_inter_energy,
    sum_nodes_of_the_same_graph,
)

from .message_passing import MessagePassingConvolution


scaling_classes: Dict[str, Callable] = {
    "std_scaling": compute_mean_std_atomic_inter_energy,
    "rms_forces_scaling": compute_mean_rms_energy_forces,
}

gate_dict: Dict[str, Optional[Callable]] = {
    "abs": jax.numpy.abs,
    "tanh": jax.numpy.tanh,
    "silu": jax.nn.silu,
    "None": None,
}

__all__ = [
    "AtomicEnergiesBlock",
    "EquivariantProductBasisBlock",
    "LinearNodeEmbeddingBlock",
    "LinearReadoutBlock",
    "NonLinearReadoutBlock",
    "RadialEmbeddingBlock",
    "AgnosticResidualInteractionBlock",
    "ScaleShiftBlock",
    "WeightedEnergyForcesLoss",
    "GeneralMACE",
    "MACE",
    "SymmetricContraction",
    "compute_avg_num_neighbors",
    "compute_avg_min_neighbor_distance",
    "compute_mean_rms_energy_forces",
    "compute_mean_std_atomic_inter_energy",
    "sum_nodes_of_the_same_graph",
    "MessagePassingConvolution",
    "interaction_classes",
    "scaling_classes",
    "gate_dict",
]
