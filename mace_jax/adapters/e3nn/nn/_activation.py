"""Flax-friendly scalar activation logic mirroring ``e3nn.nn._activation``."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Callable

import e3nn_jax as e3nn
import jax.numpy as jnp
from e3nn_jax import Irreps, IrrepsArray


class Activation:
    """Apply scalar activations chunk-wise according to an ``Irreps`` layout.

    This class mirrors the Torch ``e3nn.nn.Activation`` helper but emits plain
    JAX arrays by default, allowing it to sit inside larger Flax modules.
    """

    def __init__(
        self,
        irreps_in: Irreps,
        acts: Sequence[Callable | None],
        *,
        normalize_act: bool = True,
        name: str | None = None,
    ) -> None:
        del name  # preserved for backward compatibility

        self.irreps_in = Irreps(irreps_in)
        self._acts = tuple(acts)
        self._normalize_act = normalize_act

        if len(self.irreps_in) != len(self._acts):
            raise ValueError(
                'Irreps in and number of activation functions does not match: '
                f'{len(self._acts)}, ({self.irreps_in}, {self._acts})'
            )

        sample = e3nn.zeros(self.irreps_in, ())
        sample_out = e3nn.scalar_activation(
            sample,
            acts=self._acts,
            normalize_act=self._normalize_act,
        )
        self.irreps_out = sample_out.irreps

    def __repr__(self) -> str:
        acts_str = ''.join('x' if act is not None else ' ' for act in self._acts)
        return f'{self.__class__.__name__} [{acts_str}] ({self.irreps_in} -> {self.irreps_out})'

    def __call__(
        self,
        features: jnp.ndarray | IrrepsArray,
        axis: int = -1,
    ) -> jnp.ndarray:
        """Apply the configured scalar activations to ``features``."""
        if isinstance(features, IrrepsArray):
            if features.irreps != self.irreps_in:
                raise ValueError(
                    f'Activation expects irreps {self.irreps_in}, got {features.irreps}'
                )
            irreps_array = features
            array = features.array
            moved = False
        else:
            array = features
            if axis != -1:
                array = jnp.moveaxis(array, axis, -1)
                moved = True
            else:
                moved = False

            if array.shape[-1] != self.irreps_in.dim:
                raise ValueError(
                    f'Invalid input shape: expected last dimension {self.irreps_in.dim}, '
                    f'got {array.shape[-1]}'
                )

            irreps_array = e3nn.IrrepsArray(self.irreps_in, array)

        activated = e3nn.scalar_activation(
            irreps_array,
            acts=self._acts,
            normalize_act=self._normalize_act,
        ).array

        if isinstance(features, IrrepsArray):
            return activated

        if moved:
            activated = jnp.moveaxis(activated, -1, axis)
        return activated
