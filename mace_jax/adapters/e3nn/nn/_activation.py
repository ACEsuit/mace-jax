"""Flax-friendly scalar activation logic mirroring ``e3nn.nn._activation``."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import e3nn_jax as e3nn
import jax
import jax.numpy as jnp
from e3nn_jax import Irreps, IrrepsArray
from e3nn_jax._src.activation import parity_function, scalar_activation

from mace_jax.adapters.e3nn.math import normalize2mom, register_normalize2mom_const


class Activation:
    """Apply scalar activations chunk-wise according to an ``Irreps`` layout.

    This class mirrors the Torch ``e3nn.nn.Activation`` helper but emits plain
    JAX arrays by default, allowing it to sit inside larger Flax modules. We
    also take care to preserve Torch-side normalize2mom constants so imported
    models use identical activation scaling without re-estimation.
    """

    def __init__(
        self,
        irreps_in: Irreps,
        acts: Sequence[Callable | None],
        *,
        normalize_act: bool = True,
    ) -> None:
        self.irreps_in = Irreps(irreps_in)

        # Map common Torch activations to their JAX equivalents so we can apply
        # JAX-side normalisation while still honouring Torch-stored constants.
        def _to_jax_act(act: Callable | None) -> Callable | None:
            if act is None:
                return None
            if hasattr(act, '__name__'):
                name = act.__name__.lower()
                if name in ('silu', 'swish'):
                    return jax.nn.silu
            cls_name = getattr(getattr(act, '__class__', None), '__name__', '').lower()
            if cls_name in ('silu', 'swish'):
                return jax.nn.silu
            # Torch normalize2mom wrapper stores the underlying torch function as ``f``.
            wrapped = getattr(act, 'f', None)
            if wrapped is not None and wrapped is not act:
                mapped = _to_jax_act(wrapped)
                if mapped is not None:
                    return mapped
            return act

        # Extract any Torch normalize2mom constant and original function.
        # Torch normalize2mom exposes constants as ``cst`` and the wrapped
        # callable as ``f`` rather than the JAX-side private attributes.
        def _maybe_get_const(act):
            if act is None:
                return None, None
            const = getattr(act, '_normalize2mom_const', None)
            if const is None:
                const = getattr(act, 'cst', None)  # Torch normalize2mom module
            orig = getattr(act, '_normalize2mom_original', None)
            if orig is None:
                orig = getattr(act, 'f', None)  # Torch normalize2mom module
            return const, orig

        processed_acts: list[Callable | None] = []
        for act in acts:
            jax_act = _to_jax_act(act)
            const, orig = _maybe_get_const(act)
            if const is not None:
                # Allow normalize2mom to reuse the Torch constant.
                register_normalize2mom_const(orig or jax_act or act, const)
            use_norm = normalize_act or const is not None
            if use_norm:
                processed_acts.append(
                    normalize2mom(jax_act) if jax_act is not None else None
                )
            else:
                processed_acts.append(jax_act)

        acts = processed_acts

        self._acts = tuple(acts)
        self._normalize_act = False

        if len(self.irreps_in) != len(self._acts):
            raise ValueError(
                'Irreps in and number of activation functions does not match: '
                f'{len(self._acts)}, ({self.irreps_in}, {self._acts})'
            )

        self.irreps_out = self._infer_irreps_out()

    def _infer_irreps_out(self) -> Irreps:
        """Determine the output irreps without tracing through e3nn.

        The original implementation staged a scalar activation call that
        executed eagerly at construction time. When ``Activation`` is created
        while JIT tracing (e.g. inside ``flax.linen.Module.setup`` during
        ``jax.jit``) that call attempts to convert a traced scalar to bool,
        causing a ``TracerBoolConversionError``. We instead mirror the logic
        from :func:`e3nn_jax.scalar_activation` to compute the outgoing irrep
        metadata purely symbolically.

        Returns
        -------
        Irreps
            Output irreps after applying the configured scalar activations.
        """
        result: list[tuple[int, tuple[int, int]]] = []
        for (mul, ir), act in zip(self.irreps_in, self._acts):
            l_in, p_in = ir.l, ir.p
            if act is None:
                result.append((mul, (l_in, p_in)))
                continue

            if l_in != 0:
                raise ValueError(
                    'Activation: cannot apply an activation function to a non-scalar input. '
                    f'{self.irreps_in} {self._acts}'
                )

            if p_in == -1:
                p_out = parity_function(act)
                if p_out == 0:
                    raise ValueError(
                        'Activation: the parity is violated! The input scalar is odd but the '
                        'activation is neither even nor odd.'
                    )
            else:
                p_out = p_in

            result.append((mul, (0, p_out)))
        return Irreps(result)

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

        activated = scalar_activation(
            irreps_array,
            acts=self._acts,
            normalize_act=self._normalize_act,
        ).array

        if isinstance(features, IrrepsArray):
            return IrrepsArray(self.irreps_out, activated)

        if moved:
            activated = jnp.moveaxis(activated, -1, axis)
        return activated
