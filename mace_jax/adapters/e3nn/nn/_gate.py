"""Flax port of the ``e3nn`` gated non-linearity module."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import e3nn_jax as e3nn
import jax.numpy as jnp
from e3nn_jax import Irreps, IrrepsArray
from flax import linen as fnn

from mace_jax.adapters.flax.torch import auto_import_from_torch_flax

from ._activation import Activation
from ._extract import Extract


def _sort_irreps_like_torch(irreps: Irreps) -> tuple[Irreps, tuple[int, ...]]:
    """Match the legacy Torch ``e3nn`` irreps ordering used by the gated block.

    ``e3nn`` in PyTorch sorts representations by increasing ``l`` but places odd
    parity irreps before even ones when ``l`` matches. ``e3nn_jax`` keeps the even
    parity first, so a direct call to :meth:`Irreps.sort` would generate a
    permutation that disagrees with the Torch module.
    """

    irreps = Irreps(irreps)
    indexed_irreps = list(enumerate(irreps))

    ordered = sorted(
        indexed_irreps,
        key=lambda item: (item[1].ir.l, item[1].ir.p == 1, item[0]),
    )

    sorted_irreps = Irreps([mul_ir for _, mul_ir in ordered])
    permutation = [0] * len(ordered)
    for new_idx, (orig_idx, _) in enumerate(ordered):
        permutation[orig_idx] = new_idx

    return sorted_irreps, tuple(permutation)


class _Sortcut:
    """Prepare sorted irreps layout and associated extraction instructions."""

    def __init__(self, *irreps_outs: Irreps) -> None:
        self.irreps_outs = tuple(Irreps(ir).simplify() for ir in irreps_outs)
        irreps_in = sum(self.irreps_outs, Irreps([]))

        index = 0
        instructions = []
        for irreps_out in self.irreps_outs:
            instructions.append(tuple(range(index, index + len(irreps_out))))
            index += len(irreps_out)
        if len(irreps_in) != index:
            raise ValueError(
                f'Instruction mismatch: expected {len(irreps_in)} entries, found {index}.'
            )

        irreps_in_sorted, permutation = _sort_irreps_like_torch(irreps_in)
        instructions = [tuple(permutation[i] for i in ins) for ins in instructions]

        self.extract = Extract(irreps_in_sorted, self.irreps_outs, instructions)
        self.irreps_in = irreps_in_sorted.simplify()

    def __call__(self, x: jnp.ndarray | IrrepsArray) -> tuple[jnp.ndarray, ...]:
        """Return the scalar, gate, and gated views of the input array."""
        return self.extract(x)


def _as_irreps(value) -> Irreps:
    return value if isinstance(value, Irreps) else Irreps(value)


@auto_import_from_torch_flax(allow_missing_mapper=True)
class Gate(fnn.Module):
    """Combine scalar activations with gated higher-order features.

    The gate expects its input features to be organised as the concatenation of
    ``(scalars, gates, gated)`` irreps expressed in cue-style ``ir_mul`` layout.
    It normalises the scalar and gate channels separately, applies the provided
    non-linearities, and finally modulates each gated irrep by the
    corresponding gate via an element-wise tensor product.  The output packs the
    activated scalars followed by the gated channels so it can flow back into
    other ``e3nn_jax`` blocks.

    Args:
        irreps_scalars: Irreps describing the scalar channels that receive
            direct activations and are forwarded without gating.
        act_scalars: Sequence of scalar activation functions (or ``None``) that
            matches ``irreps_scalars`` one-to-one.
        irreps_gates: Scalar irreps used as multiplicative gates; each entry
            must be an ``l=0`` irrep.
        act_gates: Activations applied to the gate channels prior to modulation.
        irreps_gated: Irreps carrying the features that will be modulated by
            the gates.  Must contain the same number of entries as
            ``irreps_gates``.
        normalize_act: Whether to match the activation variance with the Torch
            reference via ``normalize2mom``.
    """

    irreps_scalars: Irreps
    act_scalars: Sequence[Callable | None]
    irreps_gates: Irreps
    act_gates: Sequence[Callable | None]
    irreps_gated: Irreps
    normalize_act: bool = True

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(self, 'act_scalars', tuple(self.act_scalars))
        object.__setattr__(self, 'act_gates', tuple(self.act_gates))

    def setup(self) -> None:
        irreps_scalars = Irreps(self.irreps_scalars).simplify()
        irreps_gates = Irreps(self.irreps_gates).simplify()
        irreps_gated = Irreps(self.irreps_gated).simplify()

        max_gate_l = max((mul_ir.ir.l for mul_ir in irreps_gates), default=0)
        max_scalar_l = max((mul_ir.ir.l for mul_ir in irreps_scalars), default=0)

        if len(irreps_gates) > 0 and max_gate_l > 0:
            raise ValueError(f'Gate scalars must be scalars, got {irreps_gates}')
        if len(irreps_scalars) > 0 and max_scalar_l > 0:
            raise ValueError(f'Scalars must be scalars, got {irreps_scalars}')
        if len(irreps_gates) != len(irreps_gated):
            raise ValueError(
                f'Mismatch: {len(irreps_gated)} irreps in gated, '
                f'{len(irreps_gates)} in gates'
            )

        self._irreps_scalars = irreps_scalars
        self._irreps_gates = irreps_gates
        self._irreps_gated = irreps_gated

        self._scalar_activation = Activation(
            self._irreps_scalars,
            self.act_scalars,
            normalize_act=self.normalize_act,
        )
        self._gate_activation = Activation(
            self._irreps_gates,
            self.act_gates,
            normalize_act=self.normalize_act,
        )

        self._sortcut = _Sortcut(
            self._irreps_scalars,
            self._irreps_gates,
            self._irreps_gated,
        )

        self._irreps_in = self._sortcut.irreps_in
        self._irreps_scalars_out = self._scalar_activation.irreps_out
        self._irreps_gates_out = self._gate_activation.irreps_out

        irreps_gated_dim = _as_irreps(self._irreps_gated).dim
        irreps_gates_out_dim = _as_irreps(self._irreps_gates_out).dim

        if irreps_gated_dim > 0 and irreps_gates_out_dim > 0:
            sample_gated = e3nn.zeros(_as_irreps(self._irreps_gated), ())
            sample_gates = e3nn.zeros(_as_irreps(self._irreps_gates_out), ())
            self._mul_irreps_out = e3nn.elementwise_tensor_product(
                sample_gated,
                sample_gates,
            ).irreps
        else:
            self._mul_irreps_out = Irreps([])

        self._irreps_out = _as_irreps(self._irreps_scalars_out) + _as_irreps(
            self._mul_irreps_out
        )

    def __call__(self, features: IrrepsArray | jnp.ndarray) -> IrrepsArray:
        """Apply scalar activations and gated tensor products to ``features``.

        Args:
            features: Either an ``IrrepsArray`` with irreps ``irreps_in`` or a
                raw array whose last dimension equals ``irreps_in.dim``.

        Returns:
            An ``IrrepsArray`` whose irreps are ``irreps_out`` containing the
            activated scalars followed by the gated feature blocks.
        """
        if isinstance(features, IrrepsArray):
            array = features.array
        else:
            array = features
            if array.shape[-1] != _as_irreps(self._irreps_in).dim:
                raise ValueError(
                    f'Invalid input shape: expected last dim {_as_irreps(self._irreps_in).dim}, '
                    f'got {array.shape[-1]}'
                )

        scalars, gates, gated = self._sortcut(array)

        scalars_act = self._scalar_activation(scalars)
        gates_act = gates
        if gates.shape[-1] > 0:
            gates_act = self._gate_activation(gates)

        outputs: list[jnp.ndarray] = []
        if scalars_act.shape[-1] > 0:
            outputs.append(scalars_act)

        if gates_act.shape[-1] > 0 and gated.shape[-1] > 0:
            gated_ir = e3nn.IrrepsArray(_as_irreps(self._irreps_gated), gated)
            gates_ir = e3nn.IrrepsArray(_as_irreps(self._irreps_gates_out), gates_act)
            gated_prod = e3nn.elementwise_tensor_product(gated_ir, gates_ir).array
            if gated_prod.shape[-1] > 0:
                outputs.append(gated_prod)

        if not outputs:
            empty = jnp.zeros(array.shape[:-1] + (0,), dtype=array.dtype)
            return IrrepsArray(_as_irreps(self._irreps_out), empty)

        concatenated = (
            outputs[0] if len(outputs) == 1 else jnp.concatenate(outputs, axis=-1)
        )
        return IrrepsArray(_as_irreps(self._irreps_out), concatenated)

    @property
    def irreps_in(self) -> Irreps:
        return _as_irreps(self._irreps_in)

    @property
    def irreps_out(self) -> Irreps:
        return _as_irreps(self._irreps_out)
