from collections.abc import Callable

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
from .loss import (
    DipolePolarLoss,
    DipoleSingleLoss,
    UniversalLoss,
    WeightedEnergyForcesDipoleLoss,
    WeightedEnergyForcesL1L2Loss,
    WeightedEnergyForcesLoss,
    WeightedEnergyForcesStressLoss,
    WeightedEnergyForcesVirialsLoss,
    WeightedForcesLoss,
    WeightedHuberEnergyForcesStressLoss,
    uber_loss,
)
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
    'WeightedEnergyForcesLoss',
    'WeightedEnergyForcesStressLoss',
    'WeightedEnergyForcesVirialsLoss',
    'WeightedForcesLoss',
    'WeightedHuberEnergyForcesStressLoss',
    'WeightedEnergyForcesDipoleLoss',
    'WeightedEnergyForcesL1L2Loss',
    'UniversalLoss',
    'DipoleSingleLoss',
    'DipolePolarLoss',
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

gate_dict: dict[str, Callable | None] = {
    'abs': jax.numpy.abs,
    'tanh': jax.nn.tanh,
    'silu': jax.nn.silu,
    'None': None,
}
