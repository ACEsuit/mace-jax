import copy
import math

import jax
import jax.numpy as jnp
from flax import nnx

from mace_jax.adapters.e3nn.math import register_normalize2mom_const
from mace_jax.nnx_config import ConfigDict, ConfigVar
from mace_jax.tools.dtype import default_dtype


def extract_config_mace_model(torch_model):
    """Extract Torch config, recovering supported legacy architecture metadata.

    Some Torch MACE versions access ``bessel_weights`` even for Gaussian and
    Chebyshev bases. Supply that size on shallow module copies so extraction
    works without changing the source model or copying its parameter tensors.
    """
    from mace.tools.scripts_utils import (  # noqa: PLC0415
        extract_config_mace_model as torch_extract_config,
    )

    radial = getattr(torch_model, 'radial_embedding', None)
    basis = getattr(radial, 'bessel_fn', None)
    extraction_model = torch_model
    if basis is not None and not hasattr(basis, 'bessel_weights'):
        extraction_model = copy.copy(torch_model)
        extraction_model._modules = dict(torch_model._modules)
        radial_copy = copy.copy(radial)
        radial_copy._modules = dict(radial._modules)
        basis_copy = copy.copy(basis)
        # Only len() is used by the extractor; no additional Torch state is needed.
        basis_copy.bessel_weights = range(radial.out_dim)
        radial_copy.bessel_fn = basis_copy
        extraction_model.radial_embedding = radial_copy

    config = torch_extract_config(extraction_model)
    if (
        'error' not in config
        and not hasattr(torch_model, 'use_edge_irreps_first')
        and getattr(torch_model, 'edge_irreps', None) is not None
    ):
        config['use_edge_irreps_first'] = _infer_use_edge_irreps_first(torch_model)
    return config


def _infer_use_edge_irreps_first(torch_model) -> bool:
    """Recover a missing flag from a supported first block, without mutation.

    Older MH1 checkpoints omit the model flag although their first interaction
    stores the projected edge representation. Default-compatible projections
    retain False; only the known nonlinear block can infer True. Explicit flags
    and models without edge irreps never enter this fallback.
    """
    from e3nn import o3  # noqa: PLC0415

    message = (
        'Cannot infer missing use_edge_irreps_first from the checkpoint: '
        'the first interaction layout is unsupported or inconsistent. '
        'Re-export the Torch checkpoint with explicit use_edge_irreps_first metadata.'
    )
    try:
        first = torch_model.interactions[0]
        nonlinear = (
            type(first).__name__ == 'RealAgnosticResidualNonLinearInteractionBlock'
        )
        node_irreps = o3.Irreps(str(first.node_feats_irreps)).simplify()
        edge_channels = o3.Irreps(str(torch_model.edge_irreps)).count(o3.Irrep('0e'))
        edge_irreps = o3.Irreps(f'{edge_channels}x0e')
        projected_irreps = o3.Irreps(str(first.linear_up.irreps_out)).simplify()
        if (
            not node_irreps
            or node_irreps != o3.Irreps(f'{node_irreps.count(o3.Irrep("0e"))}x0e')
            or o3.Irreps(str(first.linear_up.irreps_in)).simplify() != node_irreps
            or o3.Irreps(str(first.edge_irreps)).simplify() != projected_irreps
            or (
                nonlinear
                and o3.Irreps(str(first.linear_res.irreps_in)).simplify()
                != projected_irreps
            )
        ):
            raise ValueError(message)
    except (AttributeError, IndexError, TypeError, ValueError) as exc:
        raise ValueError(message) from exc

    if projected_irreps == node_irreps:
        # Equal node/edge widths yield the same architecture for either flag.
        # Preserve the legacy default in that case as well.
        return False
    if nonlinear and edge_channels and projected_irreps == edge_irreps:
        return True
    raise ValueError(message)


def _format_tree_path(path) -> str:
    return '/'.join(str(key) for key in path) if path else '<root>'


def _activation_name(activation) -> str | None:
    """Return the stable JAX registry key for a Torch activation."""
    if activation is None:
        return None
    name = getattr(activation, '__name__', None)
    if name is None:
        name = getattr(getattr(activation, '__class__', None), '__name__', None)
    if not name:
        return None
    name = str(name).lower()
    return 'silu' if name in {'silu', 'swish'} else name


def _extract_norm_consts(torch_model=None) -> dict[str, float]:
    """Fetch exact Torch normalize2mom constants for all model gates.

    Parity relies on reusing the exact normalize2mom constants that Torch
    precomputed for its activation wrappers. If they cannot be obtained we raise
    instead of silently recomputing a different value.
    """
    try:
        import torch  # noqa: PLC0415
        from e3nn.math._normalize_activation import (  # noqa: PLC0415
            normalize2mom as torch_norm,
        )
    except Exception as exc:
        raise ImportError(
            'Torch e3nn (and torch) are required to import activation '
            'normalization constants; parity cannot be guaranteed without them.'
        ) from exc

    try:
        silu = float(torch_norm(torch.nn.functional.silu).cst)
        sigmoid = float(torch_norm(torch.sigmoid).cst)
    except Exception as exc:
        raise RuntimeError(
            'Unable to compute normalize2mom constants for silu/sigmoid '
            'during import; parity cannot be guaranteed.'
        ) from exc

    constants = {'silu': silu, 'swish': silu, 'sigmoid': sigmoid}
    model_constants: dict[str, float] = {}

    # Preserve both legacy e3nn wrappers and newer MACE blocks, which store
    # their normalization data in plain fields rather than child modules.
    if torch_model is not None:
        modules = (
            torch_model.modules()
            if callable(getattr(torch_model, 'modules', None))
            else ()
        )
        for module in modules:
            attributes = [
                ('_act_scalar', '_scalar_cst'),
                ('_act_gate', '_gate_cst'),
            ]
            if isinstance(module, torch_norm):
                attributes.append(('f', 'cst'))
            for activation_attr, constant_attr in attributes:
                activation = getattr(module, activation_attr, None)
                constant = getattr(module, constant_attr, None)
                key = _activation_name(activation)
                if key is None or constant is None:
                    continue
                value = float(constant)
                previous = model_constants.get(key)
                if previous is not None and not math.isclose(
                    previous, value, rel_tol=1e-12, abs_tol=1e-12
                ):
                    raise ValueError(
                        f'Conflicting normalize2mom constants for {key}: '
                        f'{previous} and {value}.'
                    )
                model_constants[key] = value
                constants[key] = value
                if key == 'silu':
                    constants['swish'] = value

    return constants


def import_from_torch(jax_model, torch_model, variables):
    torch_use_reduced_cg = getattr(torch_model, 'use_reduced_cg', None)
    jax_use_reduced_cg = getattr(jax_model, 'use_reduced_cg', None)
    if (
        torch_use_reduced_cg is not None
        and jax_use_reduced_cg is not None
        and bool(torch_use_reduced_cg) != bool(jax_use_reduced_cg)
    ):
        raise ValueError(
            'Torch model was built with use_reduced_cg='
            f'{torch_use_reduced_cg!r} but the target MACE-JAX module '
            f'uses {jax_use_reduced_cg!r}. Please construct the JAX model '
            'with matching settings before importing parameters.'
        )
    if isinstance(variables, nnx.State):

        def _extract_with_nan(value):
            if isinstance(value, ConfigVar):
                config_val = value.get_value()
                if isinstance(config_val, dict) and not isinstance(
                    config_val, ConfigDict
                ):
                    return ConfigDict(config_val)
                return config_val
            if isinstance(value, nnx.Param):
                arr = value.get_value()
                if isinstance(arr, jnp.ndarray) and jnp.issubdtype(
                    arr.dtype, jnp.floating
                ):
                    return jnp.full_like(arr, jnp.nan)
                return arr
            if isinstance(value, nnx.Variable):
                return value.get_value()
            return value

        variables_pure = nnx.to_pure_dict(variables, extract_fn=_extract_with_nan)
    else:
        variables_pure = variables
        variables_pure = jax.tree_util.tree_map(
            lambda x: (
                jnp.full_like(x, jnp.nan)
                if isinstance(x, jnp.ndarray) and jnp.issubdtype(x.dtype, jnp.floating)
                else x
            ),
            variables_pure,
        )
    if getattr(jax_model.import_from_torch, '_supports_target_module', False):
        variables_pure = jax_model.import_from_torch(
            torch_model, variables_pure, target_module=jax_model
        )
    else:
        variables_pure = jax_model.import_from_torch(torch_model, variables_pure)

    # Persist normalize2mom constants in non-trainable collections.
    norm_consts = _extract_norm_consts(torch_model)
    if norm_consts:
        for key, value in norm_consts.items():
            register_normalize2mom_const(key, float(value))
        target = variables_pure.get('_normalize2mom_consts_var')
        if isinstance(target, ConfigDict):
            updated = {k: v for k, v in target.items()}
            updated_type = ConfigDict
        elif isinstance(target, dict):
            updated = dict(target)
            updated_type = dict
        else:
            updated = None
            updated_type = None
        if updated is not None:
            for key, value in norm_consts.items():
                dtype = None
                if key in updated and isinstance(updated[key], jnp.ndarray):
                    dtype = updated[key].dtype
                updated[key] = jax.lax.stop_gradient(
                    jnp.asarray(value, dtype=dtype or default_dtype())
                )
            variables_pure['_normalize2mom_consts_var'] = (
                updated_type(updated) if updated_type is not None else updated
            )

    nan_leaves: list[str] = []
    for path, leaf in jax.tree_util.tree_flatten_with_path(variables_pure)[0]:
        if isinstance(leaf, jnp.ndarray) and jnp.issubdtype(leaf.dtype, jnp.floating):
            if jnp.isnan(leaf).any():
                nan_leaves.append(_format_tree_path(path))
    if nan_leaves:
        joined = '\n  - '.join(nan_leaves)
        raise ValueError(
            'Torch parameter import failed for the following leaves (still NaN):\n'
            f'  - {joined}'
        )

    if isinstance(variables, nnx.State):
        nnx.replace_by_pure_dict(variables, variables_pure)
        if norm_consts:
            cfg = variables.get('_normalize2mom_consts_var', None)
            if isinstance(cfg, ConfigVar):
                current = cfg.get_value()
                if isinstance(current, dict):
                    updated = dict(current)
                    for key, value in norm_consts.items():
                        dtype = None
                        if key in updated and isinstance(updated[key], jnp.ndarray):
                            dtype = updated[key].dtype
                        updated[key] = jax.lax.stop_gradient(
                            jnp.asarray(value, dtype=dtype or default_dtype())
                        )
                    cfg.set_value(updated)
        # Keep the module in sync so downstream nnx.split() sees imported weights.
        # Rebuild a new module from the updated state to handle immutable params.
        try:
            graphdef = nnx.graphdef(jax_model)
            merged = nnx.merge(graphdef, variables)
            jax_model.__dict__.clear()
            jax_model.__dict__.update(merged.__dict__)
        except Exception:
            # Fallback: update only mutable variables if merge fails.
            try:
                mutable_state = variables.filter(
                    lambda _path, value: (
                        isinstance(value, nnx.Variable)
                        and getattr(value, 'is_mutable', False)
                    )
                )
                nnx.update(jax_model, mutable_state)
            except Exception:
                # If update fails (e.g., incompatible graph/state), keep the state-only
                # update to preserve existing behavior.
                pass
        return variables
    return variables_pure
