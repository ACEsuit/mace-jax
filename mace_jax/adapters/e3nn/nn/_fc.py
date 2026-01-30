"""Fully connected network layers compatible with ``e3nn`` Torch modules."""

from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp
from flax import nnx

from mace_jax.adapters.nnx.torch import (
    _resolve_scope,
    nxx_auto_import_from_torch,
    nxx_register_import_mapper,
)
from mace_jax.tools.dtype import default_dtype

from ..math import normalize2mom


@nxx_auto_import_from_torch(allow_missing_mapper=True)
class Layer(nnx.Module):
    """Flax version of e3nn.nn._fc._Layer."""

    h_in: int
    h_out: int
    act: Callable | None
    var_in: float
    var_out: float

    def __repr__(self) -> str:
        act = self.act
        act_name = (
            act.__name__ if callable(act) and hasattr(act, '__name__') else str(act)
        )
        return f'{self.__class__.__name__}({self.h_in}->{self.h_out}, act={act_name})'

    def __init__(
        self,
        h_in: int,
        h_out: int,
        act: Callable | None,
        var_in: float,
        var_out: float,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.h_in = h_in
        self.h_out = h_out
        self.act = act
        self.var_in = var_in
        self.var_out = var_out
        self._use_activation = self.act is not None
        if self._use_activation:
            normalized = normalize2mom(self.act)  # compute reference constant
            const = getattr(normalized, '_normalize2mom_const', 1.0)
            self._act_fn = self.act
            self._act_scale_init = jnp.asarray(const)
        else:
            self._act_fn = None
            self._act_scale_init = 1.0
        self.weight = nnx.Param(
            jax.random.normal(
                rngs(),
                (self.h_in, self.h_out),
                dtype=default_dtype(),
            )
        )
        if self._use_activation:
            self.act_scale = nnx.Param(
                jnp.asarray(self._act_scale_init, dtype=default_dtype()),
                is_mutable=False,
            )
        else:
            self.act_scale = None

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        weight = self.weight

        if self._use_activation:
            scaled = weight / jnp.sqrt(self.h_in * self.var_in)
            y = x @ scaled
            act_scale = (
                jax.lax.stop_gradient(jnp.asarray(self.act_scale, dtype=y.dtype))
                if self.act_scale is not None
                else jnp.asarray(self._act_scale_init, dtype=y.dtype)
            )
            y = self._act_fn(y) * act_scale
            y = y * jnp.sqrt(self.var_out)
            return y

        scaled = weight / jnp.sqrt(self.h_in * self.var_in / self.var_out)
        return x @ scaled


@nxx_auto_import_from_torch(allow_missing_mapper=True)
class FullyConnectedNet(nnx.Module):
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
    act: Callable | None = None
    variance_in: float = 1.0
    variance_out: float = 1.0
    out_act: bool = False

    def __init__(
        self,
        hs: Sequence[int],
        act: Callable | None = None,
        variance_in: float = 1.0,
        variance_out: float = 1.0,
        out_act: bool = False,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.hs = tuple(hs)
        self.act = act
        self.variance_in = variance_in
        self.variance_out = variance_out
        self.out_act = out_act
        if len(self.hs) < 2:
            raise ValueError('hs must contain at least input and output dimensions.')

        var_in = self.variance_in
        layers = []
        num_layers = len(self.hs) - 1

        for idx, (h_in, h_out) in enumerate(zip(self.hs, self.hs[1:])):
            is_last = idx == num_layers - 1
            var_out = self.variance_out if is_last else 1.0
            if is_last:
                activation = self.act if self.out_act else None
            else:
                activation = self.act
            layer = Layer(
                h_in=h_in,
                h_out=h_out,
                act=activation,
                var_in=var_in,
                var_out=var_out,
                rngs=rngs,
            )
            layers.append(layer)
            var_in = var_out

        self.layers = nnx.List(layers)

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


@nxx_register_import_mapper('e3nn.nn._fc._Layer')
def _import_e3nn_fc_layer(module, variables, scope):
    target = _resolve_scope(variables, scope)
    weight = jnp.asarray(module.weight.detach().cpu().numpy())
    if 'weight' in target:
        target['weight'] = weight.astype(target['weight'].dtype, copy=False)
    elif 'kernel' in target:
        reshaped = weight.reshape(target['kernel'].shape)
        target['kernel'] = reshaped.astype(target['kernel'].dtype, copy=False)
    if getattr(module, 'act', None) is not None and 'act_scale' in target:
        const = getattr(module.act, 'cst', None)
        if const is not None:
            target['act_scale'] = jnp.asarray(const, dtype=target['act_scale'].dtype)
    if hasattr(module, 'bias') and module.bias is not None:
        bias = jnp.asarray(module.bias.detach().cpu().numpy())
        if 'bias' in target:
            target['bias'] = bias.astype(target['bias'].dtype, copy=False)


@nxx_register_import_mapper('e3nn.nn._fc.FullyConnectedNet')
def _import_e3nn_fc(module, variables, scope):
    target = _resolve_scope(variables, scope)
    layers = target.get('layers')
    if isinstance(layers, dict):
        for name, child in module.named_children():
            if not name.startswith('layer'):
                continue
            suffix = name[len('layer') :]
            if not suffix.isdigit():
                continue
            idx = int(suffix)
            if idx not in layers:
                continue
            _import_e3nn_fc_layer(child, layers, [idx])
        return

    if hasattr(FullyConnectedNet, '_import_from_torch_impl'):
        updated = FullyConnectedNet._import_from_torch_impl(
            module,
            target,
            skip_root=True,
        )
    else:
        updated = FullyConnectedNet.import_from_torch(module, target)
    if updated is not None and updated is not target:
        target.clear()
        target.update(updated)
