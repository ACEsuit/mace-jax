"""Utilities for importing PyTorch module parameters into Flax adapters."""

from __future__ import annotations

import re
from collections.abc import MutableMapping, Sequence
from typing import Callable

import jax.numpy as jnp

from flax.core import freeze, unfreeze

_IMPORT_MAPPERS: dict[str, Callable] = {}


_FIRST_CAP_RE = re.compile('(.)([A-Z][a-z]+)')
_ALL_CAP_RE = re.compile('([a-z0-9])([A-Z])')


def camel_to_snake(value: str) -> str:
    """Return a snake_case version of ``value``.

    Torch modules expose CamelCase class names, whereas the Flax parameter trees
    in this project follow the snake_case convention.  When the caller does not
    provide an explicit scope for ``copy_torch_to_flax`` we convert the class
    name automatically.

    Args:
        value: Input string in CamelCase/PascalCase form.

    Returns:
        The snake_case representation of ``value``.
    """
    step1 = _FIRST_CAP_RE.sub(r'\1_\2', value)
    step2 = _ALL_CAP_RE.sub(r'\1_\2', step1)
    return step2.lower()


def register_import_mapper(torch_type: str) -> Callable[[Callable], Callable]:
    """Register a function that copies parameters from a Torch module.

    The decorator records the supplied callable inside ``_IMPORT_MAPPERS`` keyed
    by the fully qualified Torch class name.  When the generic importer walks a
    module tree it consults this registry to decide whether a custom mapper is
    available for a given submodule.

    Args:
        torch_type: Fully qualified name of the Torch class to handle.

    Returns:
        The decorated callable, unchanged.
    """

    def decorator(fn: Callable) -> Callable:
        _IMPORT_MAPPERS[torch_type] = fn
        return fn

    return decorator


def _copy_direct_parameters(torch_module, variables: MutableMapping) -> None:
    """Best-effort copy that matches direct Torch parameters to Flax leaves.

    The fallback iterates over the non-recursive parameters of ``torch_module``
    and replaces any entries with the same name under ``variables['params']``.
    It performs updates in place so callers can reuse the partially mutated
    variable tree when combining multiple transfers.
    """

    params = variables.setdefault('params', {})
    if not params:
        return

    for name, param in torch_module.named_parameters(recurse=False):
        if name in params:
            params[name] = jnp.asarray(
                param.detach().cpu().numpy(), dtype=params[name].dtype
            )


def _iter_module_tree(module, path: tuple[str, ...] = ()):
    """Yield ``(path, module)`` pairs while traversing ``module`` depth-first."""

    yield path, module
    for name, child in module.named_children():
        yield from _iter_module_tree(child, path + (name,))


def _apply_module_import(
    module,
    variables: MutableMapping,
    scope: tuple[str, ...],
    *,
    fallback: Callable | None,
) -> None:
    """Apply the registered mapper (or fallback) for a single Torch submodule."""

    module_name = _torch_module_name(module)
    mapper = _IMPORT_MAPPERS.get(module_name)
    has_parameters = any(True for _ in module.parameters(recurse=False))

    if not scope:
        if mapper is not None:
            # Root-level import requires an explicit scope; defer to fallback.
            if fallback is not None:
                fallback(module, variables)
            elif has_parameters:
                raise NotImplementedError(
                    f'No scope available for Torch module {module_name!r}; '
                    'register a mapper or provide a fallback copier.'
                )
        elif has_parameters:
            if fallback is not None:
                fallback(module, variables)
            else:
                raise NotImplementedError(
                    f'No Flax import mapper registered for Torch module {module_name!r}'
                )
        return

    if mapper is not None:
        mapper(module, variables, list(scope))
        return

    if has_parameters:
        if fallback is not None:
            fallback(module, variables)
            return
        raise NotImplementedError(
            f'No Flax import mapper registered for Torch module {module_name!r}'
        )


def auto_import_from_torch_flax(
    *,
    allow_missing_mapper: bool = False,
    fallback: Callable | None = None,
):
    """Decorator that adds a generic ``import_from_torch`` method to a Flax module.

    The generated classmethod performs a depth-first traversal over the Torch
    module hierarchy.  For each module encountered it decides whether a mapper
    has been registered (via ``register_import_mapper`` or
    ``register_flax_module``).  If so, the mapper is invoked with the current
    variables tree and a scope path that mirrors the traversal.  If no mapper is
    available but ``allow_missing_mapper`` was requested, the fallback copier is
    used instead; otherwise an informative ``NotImplementedError`` is raised.

    Args:
        allow_missing_mapper: When ``True`` the decorator will attempt to copy
            parameters directly if no specialised mapper exists for a module.
        fallback: Optional callable that receives the Torch module and the
            mutable variables tree.  The callable should perform any desired
            updates in place.  When ``None``, a sensible default is used if
            ``allow_missing_mapper`` is enabled.

    Returns:
        A decorator that attaches the generated ``import_from_torch`` classmethod
        to the annotated Flax module class.
    """

    def decorator(cls):
        @classmethod
        def import_from_torch(cls, torch_module, flax_variables):
            variables_mut = unfreeze(flax_variables)
            fallback_fn = fallback or (
                _copy_direct_parameters if allow_missing_mapper else None
            )

            for scope, module in _iter_module_tree(torch_module):
                try:
                    _apply_module_import(
                        module,
                        variables_mut,
                        scope,
                        fallback=fallback_fn,
                    )
                except (KeyError, ValueError):
                    if fallback_fn is not None:
                        fallback_fn(module, variables_mut)
                    else:
                        raise

            return freeze(variables_mut)

        cls.import_from_torch = import_from_torch
        return cls

    return decorator


def _torch_module_name(module) -> str:
    """Return the fully qualified class name for a Torch module."""
    return f'{module.__class__.__module__}.{module.__class__.__name__}'


def _resolve_scope(variables: MutableMapping, scope: Sequence[str]):
    """Traverse the mutable variables tree and return the mapping at ``scope``.

    Args:
        variables: Mutable view of the Flax variables tree.
        scope: Sequence of path components describing the desired sub-dictionary.

    Returns:
        The mutable mapping residing at the requested scope.

    Raises:
        KeyError: If the scope does not exist (it is expected to be created by
        the template module during ``init``).
    """
    node = variables.setdefault('params', {})
    for key in scope:
        node = node[key]
    return node


def register_flax_module(torch_type: str):
    """Register a Flax module class to handle Torch weight imports.

    This decorator is similar in spirit to ``register_import_mapper`` but allows
    the mapper to be expressed as a Flax module class.  The resulting mapper
    simply delegates to ``cls.import_from_torch`` after isolating the parameter
    subtree for the requested scope.

    Args:
        torch_type: Fully qualified name of the Torch module to map.

    Returns:
        A decorator that registers the module class and returns it unchanged.
    """

    def decorator(cls):
        def mapper(module, variables, scope: Sequence[str]):
            target = _resolve_scope(variables, scope)
            wrapped = freeze({'params': target})
            updated = cls.import_from_torch(module, wrapped)
            updated_params = unfreeze(updated).get('params', {})
            target.clear()
            target.update(updated_params)

        _IMPORT_MAPPERS[torch_type] = mapper
        return cls

    return decorator


def copy_torch_to_flax(
    torch_module, flax_variables, scope: str | Sequence[str] | None = None
):
    """Copy parameters from a Torch module into a Flax variables FrozenDict.

    Args:
        torch_module: Source Torch module whose parameters will be copied.
        flax_variables: FrozenDict produced by the destination Flax module.
        scope: Optional explicit scope identifying the parameter subtree
            receiving the copied weights.  When omitted a snake_case version of
            the Torch class name is used.

    Returns:
        A FrozenDict mirroring ``flax_variables`` but with the selected scope
        overwritten by parameters taken from ``torch_module``.

    Raises:
        NotImplementedError: If no mapper has been registered for the Torch
        module's type.
    """

    module_name = _torch_module_name(torch_module)
    mapper = _IMPORT_MAPPERS.get(module_name)
    if mapper is None:
        raise NotImplementedError(
            f"No Flax import mapper registered for Torch module '{module_name}'."
        )

    if scope is None:
        scope = [camel_to_snake(torch_module.__class__.__name__)]
    elif isinstance(scope, str):
        scope = [key for key in scope.split('/') if key]

    variables_mut = unfreeze(flax_variables)
    mapper(torch_module, variables_mut, scope)
    return freeze(variables_mut)


def init_from_torch(
    module,
    torch_module,
    rngs,
    *init_args,
    **init_kwargs,
):
    """Initialise ``module`` and load matching parameters from ``torch_module``.

    The helper mirrors the pattern used throughout the tests:

    1. Call ``module.init`` with the provided RNGs and sample inputs to produce
       the Flax variables tree.
    2. Delegate to ``module.__class__.import_from_torch`` so registered mappers
       can copy parameters from the Torch reference implementation.

    Args:
        module: Instantiated Flax module.
        torch_module: Torch module whose learned parameters should be copied.
        rngs: RNG key or dict passed to ``module.init``.
        *init_args: Positional arguments forwarded to ``module.init``.
        **init_kwargs: Keyword arguments forwarded to ``module.init``.

    Returns:
        A tuple ``(module, variables)`` where ``variables`` contains the copied
        parameters ready for ``module.apply``.
    """

    variables = module.init(rngs, *init_args, **init_kwargs)
    variables = module.__class__.import_from_torch(torch_module, variables)
    return module, variables


@register_import_mapper('torch.nn.modules.linear.Linear')
def _import_linear(module, variables, scope: Sequence[str]) -> None:
    """Import mapper for ``torch.nn.Linear`` → Flax dense kernel/bias."""
    target = _resolve_scope(variables, scope)
    target['kernel'] = jnp.asarray(
        module.weight.detach().cpu().numpy().T, dtype=target['kernel'].dtype
    )
    if module.bias is not None and 'bias' in target:
        target['bias'] = jnp.asarray(
            module.bias.detach().cpu().numpy(), dtype=target['bias'].dtype
        )


@register_import_mapper('torch.nn.modules.normalization.LayerNorm')
def _import_layernorm(module, variables, scope: Sequence[str]) -> None:
    """Import mapper for ``torch.nn.LayerNorm`` parameters."""
    target = _resolve_scope(variables, scope)
    target['scale'] = jnp.asarray(
        module.weight.detach().cpu().numpy(), dtype=target['scale'].dtype
    )
    target['bias'] = jnp.asarray(
        module.bias.detach().cpu().numpy(), dtype=target['bias'].dtype
    )


@register_import_mapper('torch.nn.modules.sparse.Embedding')
def _import_embedding(module, variables, scope: Sequence[str]) -> None:
    """Import mapper for ``torch.nn.Embedding`` tables."""
    target = _resolve_scope(variables, scope)
    target['embedding'] = jnp.asarray(
        module.weight.detach().cpu().numpy(), dtype=target['embedding'].dtype
    )


@register_import_mapper('e3nn.nn._activation.Activation')
def _import_e3nn_activation(module, variables, scope: Sequence[str]) -> None:
    """No-op mapper for activation wrappers with no parameters."""
    return variables


@register_import_mapper('e3nn.math._normalize_activation.normalize2mom')
def _import_e3nn_normalize2mom(module, variables, scope: Sequence[str]) -> None:
    """No-op mapper for normalisation helpers with no parameters."""
    return variables


@register_import_mapper('e3nn.o3._tensor_product._sub.ElementwiseTensorProduct')
def _import_e3nn_elementwise_tp(module, variables, scope: Sequence[str]) -> None:
    """No-op mapper for elementwise tensor products (parameter-free)."""
    return variables


@register_import_mapper('mace.modules.irreps_tools.reshape_irreps')
def _import_mace_reshape_irreps(module, variables, scope: Sequence[str]) -> None:
    """No-op mapper for reshape helpers (parameter-free)."""
    return variables


@register_import_mapper('e3nn.o3._linear.Linear')
def _import_e3nn_linear(module, variables, scope: Sequence[str]) -> None:
    """Import mapper for ``e3nn.o3.Linear`` weight/bias tensors."""
    target = _resolve_scope(variables, scope)
    weight = module.weight.detach().cpu().numpy()
    if 'weight' in target:
        target['weight'] = jnp.asarray(
            weight.reshape(target['weight'].shape), dtype=target['weight'].dtype
        )
    if hasattr(module, 'bias') and module.bias is not None and 'bias' in target:
        target['bias'] = jnp.asarray(
            module.bias.detach().cpu().numpy(),
            dtype=target['bias'].dtype,
        )
