from __future__ import annotations

from typing import Any

import numpy as np

import jax
import jax.numpy as jnp

try:  # pragma: no cover - optional runtime dependency
    from lammps.mliap import jax as _lammps_mliap_jax
except ImportError:  # pragma: no cover - optional runtime dependency
    _HAS_LAMMPS_JAX = False
    _ffi_forward_exchange = None
else:
    _HAS_LAMMPS_JAX = True
    _ffi_forward_exchange = getattr(_lammps_mliap_jax, 'forward_exchange', None)


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

    pair_handle = getattr(lammps_class, '_mace_jax_pair_handle', None)
    if _HAS_LAMMPS_JAX and _ffi_forward_exchange is not None and pair_handle is not None:
        return _ffi_forward_exchange(node_feats, pair_handle)

    if not hasattr(lammps_class, 'forward_exchange'):
        raise AttributeError(
            'LAMMPS class does not implement forward_exchange; '
            'ensure the simulation provides the expected interface.'
        )

    result_spec = jax.ShapeDtypeStruct(node_feats.shape, node_feats.dtype)

    def _callback(x: np.ndarray) -> np.ndarray:
        return _forward_exchange_numpy(x, lammps_class)

    return jax.pure_callback(_callback, result_spec, node_feats)
