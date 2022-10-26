from typing import Callable, Dict, Optional, Type

import jax

from .blocks import (
    AtomicEnergiesBlock,
    EquivariantProductBasisBlock,
    InteractionBlock,
    LinearNodeEmbeddingBlock,
    LinearReadoutBlock,
    NonLinearReadoutBlock,
    RadialEmbeddingBlock,
    AgnosticInteractionBlock,
    AgnosticResidualInteractionBlock,
    ScaleShiftBlock,
)

from .loss import WeightedEnergyForcesLoss
from .models import GeneralMACE, MACE
from .symmetric_contraction import SymmetricContraction

from .utils import (
    compute_avg_num_neighbors,
    compute_mean_rms_energy_forces,
    compute_mean_std_atomic_inter_energy,
)

interaction_classes: Dict[str, Type[InteractionBlock]] = {
    "AgnosticResidualInteractionBlock": AgnosticResidualInteractionBlock,
    "AgnosticInteractionBlock": AgnosticInteractionBlock,
}

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
    "InteractionBlock",
    "LinearNodeEmbeddingBlock",
    "LinearReadoutBlock",
    "NonLinearReadoutBlock",
    "RadialEmbeddingBlock",
    "AgnosticInteractionBlock",
    "AgnosticResidualInteractionBlock",
    "ScaleShiftBlock",
    "WeightedEnergyForcesLoss",
    "GeneralMACE",
    "MACE",
    "SymmetricContraction",
    "compute_avg_num_neighbors",
    "compute_mean_rms_energy_forces",
    "compute_mean_std_atomic_inter_energy",
    "interaction_classes",
    "scaling_classes",
    "gate_dict",
]
