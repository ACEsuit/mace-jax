import haiku as hk
import jax.numpy as jnp
import torch

from .utility import camel_to_snake

_IMPORT_MAPPERS = {}


def register_import_mapper(torch_type):
    """Decorator to register a function that maps Torch params -> JAX params."""

    def decorator(fn):
        _IMPORT_MAPPERS[torch_type] = fn
        return fn

    return decorator


@register_import_mapper(torch.nn.Linear)
def _import_linear(module, params):
    # Note: Torch stores weight as [out, in], Haiku wants [in, out]
    params['w'] = jnp.array(module.weight.detach().numpy().T)
    params['b'] = jnp.array(module.bias.detach().numpy())


@register_import_mapper(torch.nn.LayerNorm)
def _import_layernorm(module, params):
    params['scale'] = jnp.array(module.weight.detach().numpy())
    params['offset'] = jnp.array(module.bias.detach().numpy())


def copy_torch_to_jax(torch_module, jax_params, scope=None, hk_name=None):
    """
    Copy parameters from Torch -> JAX Haiku params dict.
    Returns a new params dict (immutability preserved).
    """
    name = torch_module.__class__.__name__

    if hk_name is None:
        hk_name = camel_to_snake(name)

    key = f'{scope}/~/{hk_name}'

    # Convert to mutable dict so we can update
    jax_params = hk.data_structures.to_mutable_dict(jax_params)

    for torch_type, mapper in _IMPORT_MAPPERS.items():
        if isinstance(torch_module, torch_type):
            mapper(torch_module, jax_params[key])
            break
    else:
        # Only complain if module actually has parameters
        if any(p.requires_grad for p in torch_module.parameters(recurse=False)):
            raise NotImplementedError(
                f"No import mapper registered for Torch module '{name}' "
                f"of type {type(torch_module)} (Haiku key '{hk_name}')"
            )

    return hk.data_structures.to_immutable_dict(jax_params)
