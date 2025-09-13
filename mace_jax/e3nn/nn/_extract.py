from typing import Tuple, List, Optional
import haiku as hk
import jax.numpy as jnp
from e3nn_jax import Irreps, IrrepsArray


class Extract(hk.Module):
    """Extract sub-sets of irreps from an IrrepsArray."""

    def __init__(
        self,
        irreps_in: Irreps,
        irreps_outs: List[Irreps],
        instructions: List[Tuple[int, ...]],
        squeeze_out: bool = False,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self.irreps_in = Irreps(irreps_in)
        self.irreps_outs = [Irreps(ir) for ir in irreps_outs]
        self.instructions = instructions
        self.squeeze_out = squeeze_out

        assert len(self.irreps_outs) == len(self.instructions)
        for ir_out, ins in zip(self.irreps_outs, self.instructions):
            assert len(ir_out) == len(ins)

        # Precompute slices for each output irreps
        self._slices_out = []
        for ir_out, ins in zip(self.irreps_outs, self.instructions):
            slices = []
            for idx in ins:
                start = sum(mul_ir.dim for mul_ir in self.irreps_in[:idx])
                end = start + self.irreps_in[idx].dim
                slices.append(slice(start, end))
            self._slices_out.append(slices)

    def __call__(self, x: IrrepsArray) -> Tuple[IrrepsArray, ...]:
        outputs = []
        for ir_out, slices in zip(self.irreps_outs, self._slices_out):
            # concatenate the slices along the last axis
            arr = jnp.concatenate([x.array[..., s] for s in slices], axis=-1)
            outputs.append(IrrepsArray(ir_out, arr))

        if self.squeeze_out and len(outputs) == 1:
            return outputs[0]
        return tuple(outputs)


class ExtractIr(Extract):
    """Extract a single Irrep from an IrrepsArray."""

    def __init__(self, irreps_in: Irreps, ir) -> None:
        ir = Irreps(ir) if isinstance(ir, str) else ir  # ensure Irrep type
        irreps_in = Irreps(irreps_in)
        irreps_out = Irreps([mul_ir for mul_ir in irreps_in if mul_ir.ir == ir])
        instructions = [
            tuple(i for i, mul_ir in enumerate(irreps_in) if mul_ir.ir == ir)
        ]
        super().__init__(irreps_in, [irreps_out], instructions, squeeze_out=True)
