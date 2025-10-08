"""Fully connected network layers compatible with ``e3nn`` Torch modules."""

from collections.abc import Sequence
from typing import Callable, Optional

import jax
import jax.numpy as jnp
from flax import linen as fnn
from flax.core import freeze, unfreeze

from mace_jax.adapters.flax.torch import (
    _resolve_scope,
    auto_import_from_torch_flax,
    register_import_mapper,
)

from ..math import normalize2mom


@auto_import_from_torch_flax(allow_missing_mapper=True)
class Layer(fnn.Module):
    """Flax version of e3nn.nn._fc._Layer."""

    h_in: int
    h_out: int
    act: Optional[Callable]
    var_in: float
    var_out: float

    def __repr__(self) -> str:
        act = self.act
        act_name = (
            act.__name__ if callable(act) and hasattr(act, '__name__') else str(act)
        )
        return f'{self.__class__.__name__}({self.h_in}->{self.h_out}, act={act_name})'

    @fnn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        def init(rng):
            return jax.random.normal(rng, (self.h_in, self.h_out))

        weight = self.param('weight', init)

        if self.act is not None:
            scaled = weight / jnp.sqrt(self.h_in * self.var_in)
            y = x @ scaled
            y = self.act(y)
            y = y * jnp.sqrt(self.var_out)
            return y

        scaled = weight / jnp.sqrt(self.h_in * self.var_in / self.var_out)
        return x @ scaled


@auto_import_from_torch_flax(allow_missing_mapper=True)
class FullyConnectedNet(fnn.Module):
    """Stack of variance-aware dense layers compatible with ``e3nn``.

    The module mirrors the Torch ``e3nn.nn.FullyConnectedNet`` constructor so
    trained weights can be imported verbatim.  Each hidden layer optionally
    applies a normalised activation and tracks the variance flowing through the
    network so that the final layer can rescale its weights to match the desired
    output variance.

    Args:
        hs: Sequence of layer widths ``[in_dim, hidden..., out_dim]``.
        act: Optional scalar activation applied after each hidden layer.  When
            provided it is normalised with :func:`normalize2mom` to match the
            Torch implementation.
        variance_in: Expected variance of the input features; used to scale the
            first weight matrix.
        variance_out: Target variance for the final layer when ``out_act`` is
            ``False``.
        out_act: If ``True`` apply ``act`` after the final layer as well.
    """

    hs: Sequence[int]
    act: Optional[Callable] = None
    variance_in: float = 1.0
    variance_out: float = 1.0
    out_act: bool = False

    def setup(self) -> None:
        if len(self.hs) < 2:
            raise ValueError('hs must contain at least input and output dimensions.')

        act = self.act
        if act is not None:
            act = normalize2mom(act)

        var_in = self.variance_in
        layers = []
        num_layers = len(self.hs) - 1

        for idx, (h_in, h_out) in enumerate(zip(self.hs, self.hs[1:])):
            is_last = idx == num_layers - 1
            var_out = self.variance_out if is_last else 1.0
            if is_last:
                activation = act if self.out_act else None
            else:
                activation = act
            layer = Layer(
                h_in=h_in,
                h_out=h_out,
                act=activation,
                var_in=var_in,
                var_out=var_out,
                name=f'layer{idx}',
            )
            layers.append(layer)
            var_in = var_out

        self.layers = layers

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply each dense layer in sequence to the input array.

        Args:
            x: Array whose last dimension equals ``hs[0]``.

        Returns:
            Array with the same leading shape as ``x`` and last dimension
            ``hs[-1]``.
        """
        for layer in self.layers:
            x = layer(x)
        return x

    def __repr__(self) -> str:
        hs_str = ', '.join(str(h) for h in self.hs)
        return f'{self.__class__.__name__}([{hs_str}])'


@register_import_mapper('e3nn.nn._fc._Layer')
def _import_e3nn_fc_layer(module, variables, scope):
    target = _resolve_scope(variables, scope)
    weight = jnp.asarray(module.weight.detach().cpu().numpy())
    if 'weight' in target:
        target['weight'] = weight.astype(target['weight'].dtype, copy=False)
    elif 'kernel' in target:
        reshaped = weight.reshape(target['kernel'].shape)
        target['kernel'] = reshaped.astype(target['kernel'].dtype, copy=False)
    if hasattr(module, 'bias') and module.bias is not None:
        bias = jnp.asarray(module.bias.detach().cpu().numpy())
        if 'bias' in target:
            target['bias'] = bias.astype(target['bias'].dtype, copy=False)


@register_import_mapper('e3nn.nn._fc.FullyConnectedNet')
def _import_e3nn_fc(module, variables, scope):
    node = variables.setdefault('params', {})
    for key in scope:
        node = node.setdefault(key, {})
    wrapped = freeze({'params': node})
    updated = FullyConnectedNet.import_from_torch(module, wrapped)
    updated_params = unfreeze(updated).get('params', {})
    node.clear()
    node.update(updated_params)
