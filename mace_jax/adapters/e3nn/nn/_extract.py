from collections.abc import Sequence
from typing import Optional, Union

import haiku as hk
import jax.numpy as jnp
from e3nn_jax import Irreps


class Extract(hk.Module):
    """Extract sub-sets of irreps from a raw array."""

    def __init__(
        self,
        irreps_in: Irreps,
        irreps_outs: Sequence[Irreps],
        instructions: Sequence[tuple[int, ...]],
        squeeze_out: bool = False,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self.irreps_in = Irreps(irreps_in)
        self.irreps_outs = [Irreps(ir) for ir in irreps_outs]
        self.instructions = list(instructions)
        self.squeeze_out = squeeze_out

        assert len(self.irreps_outs) == len(self.instructions)
        for ir_out, ins in zip(self.irreps_outs, self.instructions):
            assert len(ir_out) == len(ins)

        # Precompute slices for each output irreps
        self._slices_out: list[list[slice]] = []
        for ir_out, ins in zip(self.irreps_outs, self.instructions):
            slices = []
            for idx in ins:
                start = sum(mul_ir.dim for mul_ir in self.irreps_in[:idx])
                end = start + self.irreps_in[idx].dim
                slices.append(slice(start, end))
            self._slices_out.append(slices)

    def __call__(self, x: jnp.ndarray) -> Union[jnp.ndarray, tuple[jnp.ndarray, ...]]:
        """
        x: jnp.ndarray [..., irreps_in.dim]

        Returns: either a tuple of arrays (each [..., irreps_out.dim]),
        or a single array if squeeze_out=True and only one output.
        """
        if x.shape[-1] != self.irreps_in.dim:
            raise ValueError(
                f'Invalid input shape: expected last dim {self.irreps_in.dim}, '
                f'got {x.shape[-1]}'
            )

        outputs: list[jnp.ndarray] = []
        for ir_out, slices in zip(self.irreps_outs, self._slices_out):
            if not slices:  # empty irreps_out
                arr = jnp.zeros(x.shape[:-1] + (0,), dtype=x.dtype)
            else:
                parts = [x[..., s] for s in slices]
                arr = parts[0] if len(parts) == 1 else jnp.concatenate(parts, axis=-1)
            outputs.append(arr)

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
