import jax
import jax.numpy as jnp
from flax import nnx

from mace_jax.adapters.e3nn.math import register_normalize2mom_const
from mace_jax.nnx_config import ConfigDict, ConfigVar
from mace_jax.tools.dtype import default_dtype


def _format_tree_path(path) -> str:
    return '/'.join(str(key) for key in path) if path else '<root>'


def _extract_norm_consts() -> dict[str, float]:
    """Fetch Torch normalize2mom constants for common gates (fail fast).

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
        const = float(torch_norm(torch.nn.functional.silu).cst)
    except Exception as exc:
        raise RuntimeError(
            'Unable to compute normalize2mom constant for torch.nn.functional.silu '
            'during import; parity cannot be guaranteed.'
        ) from exc

    return {'silu': const, 'swish': const}


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
    variables_pure = jax_model.import_from_torch(torch_model, variables_pure)

    # Persist normalize2mom constants in non-trainable collections.
    norm_consts = _extract_norm_consts()
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
                    lambda _path, value: isinstance(value, nnx.Variable)
                    and getattr(value, 'is_mutable', False)
                )
                nnx.update(jax_model, mutable_state)
            except Exception:
                # If update fails (e.g., incompatible graph/state), keep the state-only
                # update to preserve existing behavior.
                pass
        return variables
    return variables_pure
