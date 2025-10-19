import jax
import jax.numpy as jnp


def _format_tree_path(path) -> str:
    return '/'.join(str(key) for key in path) if path else '<root>'


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
