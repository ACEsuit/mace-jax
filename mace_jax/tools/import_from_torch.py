import jax
import jax.numpy as jnp


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
    variables = jax.tree_util.tree_map(
        lambda x: (
            jnp.full_like(x, jnp.nan)
            if isinstance(x, jnp.ndarray) and jnp.issubdtype(x.dtype, jnp.floating)
            else x
        ),
        variables,
    )
    variables = jax_model.import_from_torch(torch_model, variables)

    # Persist normalize2mom constants in a non-trainable collection.
    norm_consts = _extract_norm_consts()
    if norm_consts:
        vars_mut = (
            variables.unfreeze() if hasattr(variables, 'unfreeze') else dict(variables)
        )
        consts = vars_mut.get('constants', {})
        consts = consts.copy() if hasattr(consts, 'copy') else dict(consts)
        consts['normalize2mom_consts'] = {
            k: jax.lax.stop_gradient(jnp.asarray(v, dtype=jnp.float32))
            for k, v in norm_consts.items()
        }
        vars_mut['constants'] = consts
        try:
            from flax.core import freeze  # noqa: PLC0415

            variables = freeze(vars_mut)
        except Exception:
            variables = vars_mut

    nan_leaves: list[str] = []
    for path, leaf in jax.tree_util.tree_flatten_with_path(variables)[0]:
        if isinstance(leaf, jnp.ndarray) and jnp.issubdtype(leaf.dtype, jnp.floating):
            if jnp.isnan(leaf).any():
                nan_leaves.append(_format_tree_path(path))
    if nan_leaves:
        joined = '\n  - '.join(nan_leaves)
        raise ValueError(
            'Torch parameter import failed for the following leaves (still NaN):\n'
            f'  - {joined}'
        )

    return variables
