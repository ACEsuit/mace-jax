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
from .les import MACELES
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
from .magnetic import MagneticScaleShiftMACE, MagneticSCFMACE
from .magnetic_blocks import (
    MagneticRealAgnosticResidueSpinOrbitCoupledDensityInteractionBlock,
    MagneticRealAgnosticSpinOrbitCoupledDensityInteractionBlock,
)
from .models import (
    MACE,
    AtomicDielectricMACE,
    AtomicDipolesMACE,
    EnergyDipolesMACE,
    ScaleShiftMACE,
)
from .polar import PolarMACE

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
    'MACELES',
    'PolarMACE',
    'MagneticScaleShiftMACE',
    'MagneticSCFMACE',
    'MagneticRealAgnosticResidueSpinOrbitCoupledDensityInteractionBlock',
    'MagneticRealAgnosticSpinOrbitCoupledDensityInteractionBlock',
    'MACE',
    'ScaleShiftMACE',
    'AtomicDipolesMACE',
    'EnergyDipolesMACE',
    'AtomicDielectricMACE',
]

# Optional: dict-layout experimental model. Import lazily to avoid hard failures
# when cuequivariance_jax backends are not available.
try:  # pragma: no cover - optional dependency path
    from .ir_dict_mace import MACEIrDict  # noqa: F401

    __all__.append('MACEIrDict')
except Exception:
    pass

interaction_classes: dict[str, type[InteractionBlock]] = {
    'MagneticRealAgnosticSpinOrbitCoupledDensityInteractionBlock': MagneticRealAgnosticSpinOrbitCoupledDensityInteractionBlock,
    'MagneticRealAgnosticResidueSpinOrbitCoupledDensityInteractionBlock': MagneticRealAgnosticResidueSpinOrbitCoupledDensityInteractionBlock,
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
    'LinearDipolePolarReadoutBlock': LinearDipolePolarReadoutBlock,
    'NonLinearDipolePolarReadoutBlock': NonLinearDipolePolarReadoutBlock,
}

gate_dict: dict[str, Callable | None] = {
    'abs': jax.numpy.abs,
    'tanh': jax.nn.tanh,
    'silu': jax.nn.silu,
    'None': None,
}
