"""Utilities for slicing arrays according to ``e3nn`` irreps layouts."""

from collections.abc import Sequence
from typing import Optional, Union

import jax.numpy as jnp
from e3nn_jax import Irreps, IrrepsArray


class Extract:
    """Extract contiguous irreps slices following ``e3nn`` indexing rules."""

    def __init__(
        self,
        irreps_in: Irreps,
        irreps_outs: Sequence[Irreps],
        instructions: Sequence[tuple[int, ...]],
        squeeze_out: bool = False,
        name: Optional[str] = None,
    ):
        del name  # kept for backward-compatible signature
        self.irreps_in = Irreps(irreps_in)
        self.irreps_outs = [Irreps(ir) for ir in irreps_outs]
        self.instructions = [tuple(ins) for ins in instructions]
        self.squeeze_out = squeeze_out

        if len(self.irreps_outs) != len(self.instructions):
            raise ValueError(
                'Number of output irreps must match number of instruction sets.'
            )
        for ir_out, ins in zip(self.irreps_outs, self.instructions):
            if len(ir_out) != len(ins):
                raise ValueError('Instruction length must match irreps length.')

        dims = [mul_ir.dim for mul_ir in self.irreps_in]
        offsets = [0]
        for dim in dims:
            offsets.append(offsets[-1] + dim)

        self._slices_out: list[tuple[slice, ...]] = []
        for ins in self.instructions:
            slices: list[slice] = []
            for idx in ins:
                start = offsets[idx]
                end = offsets[idx + 1]
                slices.append(slice(start, end))
            self._slices_out.append(tuple(slices))

    def __call__(
        self, x: Union[jnp.ndarray, IrrepsArray]
    ) -> Union[jnp.ndarray, tuple[jnp.ndarray, ...]]:
        """Return the slices specified by the instruction set."""
        array = x.array if isinstance(x, IrrepsArray) else x

        if array.shape[-1] != self.irreps_in.dim:
            raise ValueError(
                f'Invalid input shape: expected last dim {self.irreps_in.dim}, '
                f'got {array.shape[-1]}'
            )

        outputs: list[jnp.ndarray] = []
        for slices in self._slices_out:
            if not slices:  # empty irreps_out
                arr = jnp.zeros(array.shape[:-1] + (0,), dtype=array.dtype)
            else:
                parts = [array[..., s] for s in slices]
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
