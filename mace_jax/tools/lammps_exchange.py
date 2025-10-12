from __future__ import annotations

from typing import Any

import numpy as np

import jax
import jax.numpy as jnp


def _forward_exchange_numpy(arr: np.ndarray, lammps_class: Any) -> np.ndarray:
    if not hasattr(lammps_class, 'forward_exchange'):
        raise AttributeError(
            'LAMMPS class does not implement forward_exchange; '
            'ensure the simulation provides the expected interface.'
        )
    out = np.empty_like(arr)
    lammps_class.forward_exchange(arr, out, arr.shape[-1])
    return out


def forward_exchange(
    node_feats: jnp.ndarray,
    lammps_class: Any,
) -> jnp.ndarray:
    """Exchange features across domains via the LAMMPS communicator."""

    if not hasattr(lammps_class, 'forward_exchange'):
        raise AttributeError(
            'LAMMPS class does not implement forward_exchange; '
            'ensure the simulation provides the expected interface.'
        )

    result_spec = jax.ShapeDtypeStruct(node_feats.shape, node_feats.dtype)

    def _callback(x: np.ndarray) -> np.ndarray:
        return _forward_exchange_numpy(x, lammps_class)

    return jax.pure_callback(_callback, result_spec, node_feats)
