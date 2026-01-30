"""NNX config variables and pytree registrations."""

from __future__ import annotations

from typing import Any

import jax
from flax import nnx


class ConfigVar(nnx.Variable):
    """Non-trainable configuration/state variable."""


class ConfigDict:
    """Lightweight config container that is not treated as a Mapping."""

    def __init__(self, data: dict[str, Any]):
        self._data = dict(data)

    def get(self, key: str, default: Any | None = None):
        return self._data.get(key, default)

    def items(self):
        return self._data.items()

    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()

    def __bool__(self):
        return bool(self._data)

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f'ConfigDict({self._data!r})'


def _configdict_flatten(value: ConfigDict):
    keys = tuple(value.keys())
    children = tuple(value._data[k] for k in keys)
    return children, keys


def _configdict_unflatten(keys, children):
    return ConfigDict(dict(zip(keys, children)))


try:  # pragma: no cover - registration is idempotent per process
    jax.tree_util.register_pytree_node(
        ConfigDict, _configdict_flatten, _configdict_unflatten
    )
except ValueError:
    pass
