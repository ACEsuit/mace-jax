"""Shared OpenEquivariance tensor-product problem construction."""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

import jax.numpy as jnp
import numpy as np
from e3nn_jax import Irreps

from mace_jax.tools.dtype import default_dtype


Instruction = tuple[int, int, int, str, bool, float]


def load_openeq():
    """Import the optional JAX backend without importing its Torch bindings."""
    os.environ.setdefault('OEQ_NOTORCH', '1')
    try:
        import openequivariance as oeq
        import openequivariance.jax as oeq_jax
    except (ImportError, OSError, AttributeError) as exc:
        raise RuntimeError(
            'OpenEquivariance was selected but its JAX packages are unavailable. '
            'Install openequivariance[jax]==0.6.8, then install '
            'openequivariance_extjax==0.6.8 with --no-build-isolation.'
        ) from exc
    return oeq, oeq_jax


def normalize_instructions(
    instructions: Sequence[Sequence[Any]] | None,
    *,
    allowed_modes: frozenset[str] | None = None,
) -> tuple[Instruction, ...]:
    """Validate and normalize e3nn tensor-product instructions."""
    if instructions is None:
        raise ValueError('OpenEquivariance requires explicit instructions.')
    result = []
    for instruction in instructions:
        if len(instruction) == 5:
            i1, i2, iout, mode, weighted = instruction
            path_weight = 1.0
        elif len(instruction) == 6:
            i1, i2, iout, mode, weighted, path_weight = instruction
        else:
            raise ValueError(
                'OpenEquivariance instructions must have length 5 or 6; '
                f'got {len(instruction)}.'
            )
        normalized = (
            int(i1),
            int(i2),
            int(iout),
            str(mode),
            bool(weighted),
            float(path_weight),
        )
        if not normalized[4]:
            raise ValueError('OpenEquivariance adapters require weighted instructions.')
        if allowed_modes is not None and normalized[3] not in allowed_modes:
            modes = ', '.join(sorted(allowed_modes))
            raise ValueError(
                f'OpenEquivariance instruction mode must be one of {modes}; '
                f'got {normalized[3]!r}.'
            )
        result.append(normalized)
    return tuple(result)


def build_tp_problem(
    irreps_in1: Irreps,
    irreps_in2: Irreps,
    irreps_out: Irreps,
    instructions: Sequence[Sequence[Any]],
    *,
    shared_weights: bool,
    layout: str = 'mul_ir',
    group: str = 'O3_e3nn',
    allowed_modes: frozenset[str] | None = None,
):
    """Build a strictly validated OpenEquivariance ``TPProblem``."""
    if layout not in ('mul_ir', 'ir_mul'):
        raise ValueError(f'OpenEquivariance layout must be mul_ir or ir_mul, got {layout!r}.')
    if group != 'O3_e3nn':
        raise ValueError(
            "OpenEquivariance requires group='O3_e3nn'; " f'got {group!r}.'
        )
    dtype = jnp.dtype(default_dtype())
    if dtype not in (jnp.dtype(jnp.float32), jnp.dtype(jnp.float64)):
        raise TypeError(
            'OpenEquivariance supports only float32 and float64; '
            f'received {dtype}.'
        )
    np_dtype = np.float64 if dtype == jnp.dtype(jnp.float64) else np.float32
    normalized = normalize_instructions(instructions, allowed_modes=allowed_modes)
    oeq, oeq_jax = load_openeq()
    try:
        problem = oeq.TPProblem(
            oeq.Irreps(str(Irreps(irreps_in1))),
            oeq.Irreps(str(Irreps(irreps_in2))),
            oeq.Irreps(str(Irreps(irreps_out))),
            normalized,
            shared_weights=bool(shared_weights),
            internal_weights=False,
            irrep_normalization='component',
            path_normalization='element',
            irrep_dtype=np_dtype,
            weight_dtype=np_dtype,
            layout=layout,
        )
    except Exception as exc:
        raise RuntimeError(
            'OpenEquivariance could not construct the requested CUDA tensor product.'
        ) from exc
    return problem, oeq_jax, dtype, normalized


def weight_permutation(operator, weight_numel: int, *, shared_weights: bool) -> tuple[int, ...]:
    """Return indices mapping canonical e3nn weights to backend weight order."""
    canonical = np.arange(weight_numel, dtype=np.int32)
    candidate = canonical if shared_weights else canonical[None, :]
    reordered = operator.reorder_weights_from_e3nn(
        candidate, has_batch_dim=not shared_weights
    )
    permutation = tuple(int(x) for x in np.asarray(reordered).reshape(-1))
    if sorted(permutation) != list(range(weight_numel)):
        raise RuntimeError('OpenEquivariance returned an invalid weight permutation.')
    return permutation
