from typing import Callable, Optional

import haiku as hk
import jax.numpy as jnp
from e3nn_jax import Irreps, IrrepsArray

from .._tensor_product._sub import ElementwiseTensorProduct
from ._activation import Activation
from ._extract import Extract


class _Sortcut(hk.Module):
    """Helper to extract subsets of irreps and manage instructions."""

    def __init__(self, *irreps_outs: Irreps, name: Optional[str] = None):
        super().__init__(name=name)
        self.irreps_outs = tuple(Irreps(ir).simplify() for ir in irreps_outs)
        irreps_in = sum(self.irreps_outs, Irreps([]))

        i = 0
        instructions = []
        for irreps_out in self.irreps_outs:
            instructions.append(tuple(range(i, i + len(irreps_out))))
            i += len(irreps_out)
        assert len(irreps_in) == i, (len(irreps_in), i)

        # Sort input irreps and update instructions
        irreps_in, p, _ = irreps_in.sort()
        instructions = [tuple(p[i] for i in x) for x in instructions]

        self.cut = Extract(irreps_in, self.irreps_outs, instructions)
        self.irreps_in = irreps_in.simplify()

    def __call__(self, x: IrrepsArray) -> tuple[IrrepsArray, ...]:
        return self.cut(x)  # returns tuple of extracted IrrepsArrays


class Gate(hk.Module):
    """
    Gate activation function: scalars pass through act_scalars,
    gated irreps are multiplied by gates passed through act_gates.
    """

    def __init__(
        self,
        irreps_scalars: Irreps,
        act_scalars: list[Optional[Callable]],
        irreps_gates: Irreps,
        act_gates: list[Optional[Callable]],
        irreps_gated: Irreps,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)

        irreps_scalars = Irreps(irreps_scalars)
        irreps_gates = Irreps(irreps_gates)
        irreps_gated = Irreps(irreps_gated)

        if len(irreps_gates) > 0 and irreps_gates.lmax > 0:
            raise ValueError(f'Gate scalars must be scalars, got {irreps_gates}')
        if len(irreps_scalars) > 0 and irreps_scalars.lmax > 0:
            raise ValueError(f'Scalars must be scalars, got {irreps_scalars}')
        if irreps_gates.num_irreps != irreps_gated.num_irreps:
            raise ValueError(
                f'Mismatch: {irreps_gated.num_irreps} irreps in gated, '
                f'{irreps_gates.num_irreps} in gates'
            )

        # Extract scalars, gates, and gated irreps
        self.sc = _Sortcut(irreps_scalars, irreps_gates, irreps_gated)
        self.irreps_scalars, self.irreps_gates, self.irreps_gated = self.sc.irreps_outs
        self._irreps_in = self.sc.irreps_in

        # Activation functions
        self.act_scalars = Activation(irreps_scalars, act_scalars)
        self.act_gates = Activation(irreps_gates, act_gates)

        # Elementwise multiplication
        self.mul = ElementwiseTensorProduct(irreps_gated, self.act_gates.irreps_out)
        self._irreps_out = self.act_scalars.irreps_out + self.mul.irreps_out

    def __call__(self, features: IrrepsArray) -> IrrepsArray:
        scalars, gates, gated = self.sc(features)

        scalars = self.act_scalars(scalars)
        if gates.shape[-1] > 0:
            gates = self.act_gates(gates)
            gated = self.mul(gated, gates)
            out = IrrepsArray.from_array(
                self._irreps_out, jnp.concatenate([scalars.array, gated.array], axis=-1)
            )
        else:
            out = scalars
        return out

    @property
    def irreps_in(self) -> Irreps:
        return self._irreps_in

    @property
    def irreps_out(self) -> Irreps:
        return self._irreps_out
