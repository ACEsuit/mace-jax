import haiku as hk
import jax.numpy as jnp

from .utility import camel_to_snake

_IMPORT_MAPPERS = {}


def register_import(torch_type_str: str):
    """
    Register a Haiku module for Torch->JAX import.
    torch_type_str is a string like "torch.nn.Linear"
    """

    def decorator(cls):
        if not hasattr(cls, 'import_from_torch'):
            raise ValueError(f"{cls.__name__} must define 'import_from_torch'")
        _IMPORT_MAPPERS[torch_type_str] = cls.import_from_torch
        return cls

    return decorator


def register_import_mapper(torch_type):
    """Decorator to register a function that maps Torch params -> JAX params."""

    def decorator(fn):
        _IMPORT_MAPPERS[torch_type] = fn
        return fn

    return decorator


@register_import_mapper('e3nn.math._normalize_activation.normalize2mom')
def _import_normalize2mom(module, params, scope):
    return params


@register_import_mapper('torch.nn.modules.linear.Linear')
def _import_linear(module, params, scope):
    # Note: Torch stores weight as [out, in], Haiku wants [in, out]
    params[scope]['w'] = jnp.array(module.weight.detach().numpy().T)
    params[scope]['b'] = jnp.array(module.bias.detach().numpy())
    return params


@register_import_mapper('torch.nn.modules.normalization.LayerNorm')
def _import_layernorm(module, params, scope):
    params[scope]['scale'] = jnp.array(module.weight.detach().numpy())
    params[scope]['offset'] = jnp.array(module.bias.detach().numpy())
    return params


@register_import_mapper('torch.nn.modules.container.Sequential')
def _import_sequential(module, params, scope):
    for idx, child in enumerate(module):
        params = copy_torch_to_jax(child, params, f'{scope}_{idx}')
    return params


def copy_torch_to_jax(torch_module, jax_params, scope=None):
    """
    Copy parameters from Torch -> JAX Haiku params dict.
    Returns a new params dict (immutability preserved).
    """

    if scope is None:
        scope = f'{camel_to_snake(torch_module.__class__.__name__)}'

    # Convert to mutable dict so we can update
    jax_params = hk.data_structures.to_mutable_dict(jax_params)

    torch_module_str = (
        f'{torch_module.__class__.__module__}.{torch_module.__class__.__name__}'
    )

    if (mapper := _IMPORT_MAPPERS.get(torch_module_str, None)) is not None:
        jax_params = mapper(torch_module, jax_params, scope=scope)
    else:
        raise NotImplementedError(
            f"No import mapper registered for Torch module '{torch_module_str}' "
            f'of type {type(torch_module)})'
        )

    return hk.data_structures.to_immutable_dict(jax_params)


def auto_import_from_torch(separator: str = '~'):
    """
    Decorator that adds a generic `import_from_torch` classmethod
    to a Haiku module. It automatically:
    - copies torch Parameters (weight, bias, etc.)
    - imports submodules recursively
    """

    def decorator(cls):
        @classmethod
        def import_from_torch(cls, torch_module, hk_params, scope):
            hk_params = hk.data_structures.to_mutable_dict(hk_params)

            # --- Copy direct parameters ---
            for name, param in torch_module.named_parameters(recurse=False):
                if param is not None:
                    hk_params[scope][name] = jnp.array(param.detach().cpu().numpy())

            # --- Copy submodules ---
            for name, submodule in torch_module.named_children():
                if separator == '' or separator is None:
                    subscope = f'{scope}/{name}'
                else:
                    subscope = f'{scope}/{separator}/{name}'

                hk_params = copy_torch_to_jax(
                    submodule,
                    hk_params,
                    scope=subscope,
                )

            return hk.data_structures.to_immutable_dict(hk_params)

        cls.import_from_torch = import_from_torch
        return cls

    return decorator
