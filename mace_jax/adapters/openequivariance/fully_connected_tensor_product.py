"""Unsupported OpenEquivariance FCTP used only by failure diagnostics.

Do not expose this module through the public adapter package. OEQ 0.6.8 can
silently corrupt its input gradients when multiple FCTP problems execute in a
single process. The implementation remains isolated here so the upstream
regression reproducer can exercise the affected backend.
"""

from __future__ import annotations

from e3nn_jax import Irreps
from flax import nnx

from mace_jax.adapters.nnx.torch import nxx_auto_import_from_torch

from .tensor_product import TensorProduct


@nxx_auto_import_from_torch(allow_missing_mapper=True)
class FullyConnectedTensorProduct(TensorProduct):
    """Diagnostic-only weighted ``uvw`` tensor product."""

    def __init__(
        self,
        irreps_in1: Irreps,
        irreps_in2: Irreps,
        irreps_out: Irreps,
        shared_weights: bool = True,
        internal_weights: bool = True,
        *,
        layout: str = 'mul_ir',
        group: str = 'O3_e3nn',
        rngs: nnx.Rngs | None = None,
    ) -> None:
        irreps_in1 = Irreps(irreps_in1)
        irreps_in2 = Irreps(irreps_in2)
        irreps_out = Irreps(irreps_out)
        instructions = [
            (i1, i2, iout, 'uvw', True, 1.0)
            for i1, (_, ir1) in enumerate(irreps_in1)
            for i2, (_, ir2) in enumerate(irreps_in2)
            for iout, (_, irout) in enumerate(irreps_out)
            if irout in ir1 * ir2
        ]
        super().__init__(
            irreps_in1,
            irreps_in2,
            irreps_out,
            instructions=instructions,
            shared_weights=shared_weights,
            internal_weights=internal_weights,
            conv_fusion=False,
            layout=layout,
            group=group,
            rngs=rngs,
            _unsafe_allow_standalone_for_diagnostics=True,
        )
