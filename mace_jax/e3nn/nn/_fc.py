from typing import Callable, Optional

import haiku as hk
import jax.numpy as jnp

from mace_jax.e3nn.math import normalize2mom
from mace_jax.haiku.torch import copy_torch_to_jax, register_import


@register_import('e3nn.nn._fc._Layer')
class Layer(hk.Module):
    """JAX/Haiku version of _Layer.

    Parameters
    ----------
    h_in : int
        Input dimensionality.
    h_out : int
        Output dimensionality.
    act : Callable or None
        Activation function (e.g. jax.nn.relu). If None, no activation is applied.
    var_in : float
        Input variance.
    var_out : float
        Output variance.
    """

    def __init__(
        self,
        h_in: int,
        h_out: int,
        act: Optional[Callable],
        var_in: float,
        var_out: float,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self.h_in = h_in
        self.h_out = h_out
        self.act = act
        self.var_in = var_in
        self.var_out = var_out

        # For repr/profiling
        act_name = act.__name__ if hasattr(act, '__name__') else str(act)
        self._profiling_str = f'Layer({h_in}->{h_out}, act={act_name})'

    def __repr__(self) -> str:
        return self._profiling_str

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # Initialize weight
        w = hk.get_parameter(
            'weight',
            shape=(self.h_in, self.h_out),
            init=hk.initializers.RandomNormal(),
        )

        if self.act is not None:
            w = w / (self.h_in * self.var_in) ** 0.5
            x = x @ w
            x = self.act(x)
            x = x * self.var_out**0.5
        else:
            w = w / (self.h_in * self.var_in / self.var_out) ** 0.5
            x = x @ w

        return x

    @classmethod
    def import_from_torch(cls, torch_module, hk_params, scope):
        """
        Copy Torch Layer weights directly into Haiku params dict.
        """
        hk_params = hk.data_structures.to_mutable_dict(hk_params)

        hk_params[scope]['weight'] = jnp.array(torch_module.weight.detach().numpy())

        return hk.data_structures.to_immutable_dict(hk_params)


@register_import('e3nn.nn._fc.FullyConnectedNet')
class FullyConnectedNet(hk.Module):
    """Fully-connected Neural Network (Haiku version).

    Parameters
    ----------
    hs : list of int
        Input, hidden, and output dimensions.
    act : callable, optional
        Activation function φ. Will be normalized by normalize2mom.
    variance_in : float
        Input variance.
    variance_out : float
        Output variance.
    out_act : bool
        Whether to apply the activation function on the output layer.
    """

    hs: list[int]

    def __init__(
        self,
        hs: list[int],
        act: Optional[Callable] = None,
        variance_in: float = 1.0,
        variance_out: float = 1.0,
        out_act: bool = False,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self.hs = list(hs)

        if act is not None:
            act = normalize2mom(act)

        var_in = variance_in
        self.layers = []

        for i, (h1, h2) in enumerate(zip(self.hs, self.hs[1:])):
            if i == len(self.hs) - 2:  # last layer
                var_out = variance_out
                a = act if out_act else None
            else:
                var_out = 1.0
                a = act

            layer = Layer(h1, h2, a, var_in, var_out, name=f'layer{i}')
            self.layers.append(layer)

            var_in = var_out

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for layer in self.layers:
            x = layer(x)
        return x

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}{self.hs}'

    @classmethod
    def import_from_torch(cls, torch_model, hk_params, scope):
        hk_params = hk.data_structures.to_mutable_dict(hk_params)

        for name, module in torch_model.named_modules():
            # Skip container itself
            if name == '':
                continue  # Build Haiku key
            hk_key = f'{scope}/~/{name}'
            # Delegate to Layer.import_from_torch if leaf
            hk_params = copy_torch_to_jax(module, hk_params, scope=hk_key)

        return hk.data_structures.to_immutable_dict(hk_params)
