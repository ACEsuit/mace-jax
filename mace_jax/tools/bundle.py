"""Helpers for loading serialized MACE-JAX bundles."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import jax
from flax import core as flax_core
from flax import serialization

from mace_jax.cli import mace_torch2jax

DEFAULT_CONFIG_NAME = 'config.json'
DEFAULT_PARAMS_NAME = 'params.msgpack'


@dataclass(frozen=True)
class ModelBundle:
    config: dict
    params: dict
    module: object


def resolve_model_paths(model_arg: str) -> tuple[Path, Path]:
    path = Path(model_arg).expanduser().resolve()
    if path.is_dir():
        config_path = path / DEFAULT_CONFIG_NAME
        params_path = path / DEFAULT_PARAMS_NAME
    elif path.suffix == '.json':
        config_path = path
        params_path = path.with_suffix('.msgpack')
    else:
        params_path = path
        config_path = path.with_suffix('.json')
    if not config_path.exists():
        raise FileNotFoundError(
            f'Unable to locate JAX model configuration at {config_path}'
        )
    if not params_path.exists():
        raise FileNotFoundError(
            f'Unable to locate serialized JAX parameters at {params_path}'
        )
    return config_path, params_path


def load_model_bundle(
    model_arg: str,
    dtype: str,
    *,
    wrapper: str | None = None,
) -> ModelBundle:
    config_path, params_path = resolve_model_paths(model_arg)
    config = json.loads(config_path.read_text())

    if dtype and dtype.lower() == 'float64':
        jax.config.update('jax_enable_x64', True)
    elif dtype:
        jax.config.update('jax_enable_x64', False)

    if wrapper not in (None, '', 'mace'):
        raise ValueError('mace-jax only supports the built-in MACE wrapper.')

    module = mace_torch2jax._build_jax_model(config)
    template = mace_torch2jax._prepare_template_data(config)
    variables = module.init(jax.random.PRNGKey(0), template)
    variables = serialization.from_bytes(variables, params_path.read_bytes())
    variables = flax_core.freeze(variables)
    return ModelBundle(config=config, params=variables, module=module)


__all__ = ['ModelBundle', 'load_model_bundle', 'resolve_model_paths']
