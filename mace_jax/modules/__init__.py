from typing import Callable, Optional

import jax

from .blocks import (
    AtomicEnergiesBlock,
    EquivariantProductBasisBlock,
    InteractionBlock,
    LinearDipolePolarReadoutBlock,
    LinearDipoleReadoutBlock,
    LinearNodeEmbeddingBlock,
    LinearReadoutBlock,
    NonLinearBiasReadoutBlock,
    NonLinearDipolePolarReadoutBlock,
    NonLinearDipoleReadoutBlock,
    NonLinearReadoutBlock,
    RadialEmbeddingBlock,
    RealAgnosticAttResidualInteractionBlock,
    RealAgnosticDensityInteractionBlock,
    RealAgnosticDensityResidualInteractionBlock,
    RealAgnosticInteractionBlock,
    RealAgnosticResidualInteractionBlock,
    RealAgnosticResidualNonLinearInteractionBlock,
    ScaleShiftBlock,
)
from .loss import WeightedEnergyForcesStressLoss, uber_loss
from .models import MACE, ScaleShiftMACE

__all__ = [
    'AtomicEnergiesBlock',
    'InteractionBlock',
    'EquivariantProductBasisBlock',
    'LinearNodeEmbeddingBlock',
    'LinearDipolePolarReadoutBlock',
    'LinearDipoleReadoutBlock',
    'LinearReadoutBlock',
    'NonLinearBiasReadoutBlock',
    'NonLinearDipolePolarReadoutBlock',
    'NonLinearDipoleReadoutBlock',
    'NonLinearReadoutBlock',
    'RadialEmbeddingBlock',
    'ScaleShiftBlock',
    'WeightedEnergyForcesStressLoss',
    'uber_loss',
    'MACE',
    'ScaleShiftMACE',
]

interaction_classes: dict[str, type[InteractionBlock]] = {
    'RealAgnosticResidualInteractionBlock': RealAgnosticResidualInteractionBlock,
    'RealAgnosticAttResidualInteractionBlock': RealAgnosticAttResidualInteractionBlock,
    'RealAgnosticInteractionBlock': RealAgnosticInteractionBlock,
    'RealAgnosticDensityInteractionBlock': RealAgnosticDensityInteractionBlock,
    'RealAgnosticDensityResidualInteractionBlock': RealAgnosticDensityResidualInteractionBlock,
    'RealAgnosticResidualNonLinearInteractionBlock': RealAgnosticResidualNonLinearInteractionBlock,
}

readout_classes: dict[str, type[LinearReadoutBlock]] = {
    'LinearReadoutBlock': LinearReadoutBlock,
    'LinearDipoleReadoutBlock': LinearDipoleReadoutBlock,
    'NonLinearDipoleReadoutBlock': NonLinearDipoleReadoutBlock,
    'NonLinearReadoutBlock': NonLinearReadoutBlock,
    'NonLinearBiasReadoutBlock': NonLinearBiasReadoutBlock,
}

gate_dict: dict[str, Optional[Callable]] = {
    'abs': jax.numpy.abs,
    'tanh': jax.nn.tanh,
    'silu': jax.nn.silu,
    'None': None,
}
