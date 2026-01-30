"""NNX helper utilities for MACE-JAX."""

from __future__ import annotations

from typing import Any

from flax import nnx

from mace_jax.nnx_config import ConfigDict, ConfigVar


def state_to_pure_dict(state: nnx.State) -> dict[str, Any]:
    """Convert an NNX State into a pure dict of arrays."""

    def _extract(value):
        if isinstance(value, ConfigVar):
            config_val = value.get_value()
            if isinstance(config_val, dict) and not isinstance(config_val, ConfigDict):
                return ConfigDict(config_val)
            return config_val
        if isinstance(value, nnx.Variable):
            return value.get_value()
        return value

    return nnx.to_pure_dict(state, extract_fn=_extract)


def state_to_serializable_dict(state: nnx.State) -> dict[str, Any]:
    """Convert an NNX State into a pure dict safe for serialization."""

    def _convert(obj):
        if isinstance(obj, ConfigDict):
            return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return type(obj)(_convert(v) for v in obj)
        return obj

    return _convert(state_to_pure_dict(state))


def replace_state_from_pure_dict(state: nnx.State, values: dict[str, Any]) -> nnx.State:
    """Replace values in an NNX State from a pure dict."""
    nnx.replace_by_pure_dict(state, values)
    return state
