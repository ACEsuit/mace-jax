#!/usr/bin/env python
"""Convert a pre-trained Torch MACE foundation model to MACE-JAX parameters."""

from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import replace
from pathlib import Path
from typing import Any

import jax
import numpy as np

from mace_jax.modules.wrapper_ops import CuEquivarianceConfig

warnings.filterwarnings(
    'ignore',
    message='Environment variable TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD detected.*',
    category=UserWarning,
)

import torch
from flax import core as flax_core
from flax import serialization
from mace.calculators import foundations_models
from mace.tools.scripts_utils import extract_config_mace_model

from mace_jax.data import utils as data_utils
from mace_jax.data.utils import graph_from_configuration
from mace_jax.tools.gin_model import _graph_to_data  # type: ignore[attr-defined]
from mace_jax.tools.import_from_torch import import_from_torch
from mace_jax.tools.model_builder import (
    _as_irreps,
    _build_jax_model,
    _prepare_template_data,
)


def _load_torch_model_from_foundations(
    source: str, model: str | None
) -> torch.nn.Module:
    source = source.lower()
    if source not in {'mp', 'off', 'anicc', 'omol'}:
        raise ValueError(
            "Unknown foundation source. Supported values are 'mp', 'off', 'anicc', 'omol'."
        )

    loader_kwargs: dict[str, Any] = {'device': 'cpu'}
    loader = None
    if source in {'mp', 'off', 'omol'}:
        loader = getattr(
            foundations_models, f'mace_{"mp" if source == "mp" else source}'
        )
        if model is not None:
            loader_kwargs['model'] = model
    else:  # anicc
        loader = foundations_models.mace_anicc
        if model is not None:
            loader_kwargs['model_path'] = model

    try:
        return loader(return_raw_model=True, **loader_kwargs)
    except Exception:
        calc = loader(return_raw_model=False, **loader_kwargs)
        torch_model = getattr(calc, 'model', None)
        if torch_model is None:
            models_attr = getattr(calc, 'models', None)
            if models_attr:
                torch_model = models_attr[0]
        if torch_model is None:
            raise
        return torch_model


def _maybe_update_hidden_irreps_from_torch(
    torch_model: torch.nn.Module, config: dict[str, Any]
) -> None:
    try:
        num_interactions = int(config.get('num_interactions', 0))
    except Exception:
        return
    if num_interactions != 1:
        return

    if 'hidden_irreps' not in config:
        return

    torch_hidden = None
    try:
        products = getattr(torch_model, 'products', None)
        if products:
            linear = getattr(products[0], 'linear', None)
            if linear is not None:
                torch_hidden = getattr(linear, 'irreps_out', None)
        if torch_hidden is None:
            torch_hidden = getattr(torch_model, 'hidden_irreps', None)
    except Exception:
        torch_hidden = None

    if torch_hidden is None:
        return

    try:
        torch_irreps = _as_irreps(torch_hidden)
        config_irreps = _as_irreps(config['hidden_irreps'])
    except Exception:
        return

    # Torch started collapsing single-interaction hidden irreps in
    # mace commit f599b0e ("fix the 1 layer model cueq"); switch to legacy
    # mode when the Torch model still carries multiple irreps.
    collapse_hidden = len(torch_irreps) <= 1
    config['collapse_hidden_irreps'] = collapse_hidden

    if torch_irreps != config_irreps:
        config['hidden_irreps'] = str(torch_irreps)


def convert_model(
    torch_model,
    config: dict[str, Any],
    *,
    cueq_config: CuEquivarianceConfig | None = None,
):
    def _ensure_nontrainable_collections(
        vars_imported: flax_core.FrozenDict, template_vars: flax_core.FrozenDict
    ) -> flax_core.FrozenDict:
        # import_from_torch only populates parameter leaves; unless we copy the
        # template’s auxiliary collections (config/constants) back in, the
        # exported bundle won’t carry normalize2mom/layout metadata, breaking
        # parity when reloading. This keeps the non-trainable collections from
        # the template alongside the imported params.
        merged = flax_core.unfreeze(vars_imported)
        template_unfrozen = flax_core.unfreeze(template_vars)
        for collection in ('config', 'constants', 'meta'):
            if collection not in merged and collection in template_unfrozen:
                merged[collection] = template_unfrozen[collection]
        return flax_core.freeze(merged)

    _maybe_update_hidden_irreps_from_torch(torch_model, config)

    try:
        jax_model = _build_jax_model(
            config,
            cueq_config=cueq_config,
            init_normalize2mom_consts=False,
        )
    except TypeError as exc:
        if 'cueq_config' in str(exc):
            jax_model = _build_jax_model(
                config,
                init_normalize2mom_consts=False,
            )
        else:
            raise
    template_data = _prepare_template_data(config)
    template_vars = jax_model.init(jax.random.PRNGKey(0), template_data)

    variables = import_from_torch(jax_model, torch_model, template_vars)
    variables = _ensure_nontrainable_collections(variables, template_vars)
    consts_loaded = None
    if isinstance(variables, dict) or hasattr(variables, 'get'):
        config_coll = variables.get('config', {}) or variables.get('constants', {})
        consts_loaded = config_coll.get('normalize2mom_consts', None)
    if consts_loaded:
        config['normalize2mom_consts'] = {
            key: float(np.asarray(val)) for key, val in consts_loaded.items()
        }
    return jax_model, variables, template_data


def main():
    parser = argparse.ArgumentParser(
        description='Convert Torch MACE model to JAX parameters'
    )
    parser.add_argument(
        '--torch-model',
        help='Optional path to a Torch checkpoint. If omitted, a foundation model is downloaded.',
    )
    parser.add_argument(
        '--foundation',
        default='mp',
        choices=['mp', 'off', 'anicc', 'omol'],
        help='Foundation family to download when --torch-model is not provided.',
    )
    parser.add_argument(
        '--model-name',
        help='Specific foundation variant (e.g., "medium-mpa-0"). See foundations_models for options.',
    )
    parser.add_argument(
        '--output',
        help=(
            "Output file for serialized JAX parameters. Defaults to '<checkpoint>-jax.npz' "
            "(or '<source>-<model>-jax.npz' for foundation downloads)."
        ),
    )
    parser.add_argument(
        '--predict',
        help='Optional path to an XYZ file for prediction. Results are written to stdout.',
    )
    args = parser.parse_args()

    if args.torch_model:
        bundle = torch.load(args.torch_model, map_location='cpu')
        torch_model = (
            bundle['model']
            if isinstance(bundle, dict) and 'model' in bundle
            else bundle
        )
        default_output = Path(args.torch_model).with_name(
            Path(args.torch_model).stem + '-jax.npz'
        )
    else:
        torch_model = _load_torch_model_from_foundations(
            args.foundation, args.model_name
        )
    torch_model.eval()

    if args.output is None:
        if args.torch_model:
            output_path = default_output
        else:
            model_tag = args.model_name or args.foundation
            output_path = Path(f'{args.foundation}-{model_tag}-jax.npz')
    else:
        output_path = Path(args.output)

    config = extract_config_mace_model(torch_model)
    if 'error' in config:
        raise RuntimeError(config['error'])
    config['torch_model_class'] = torch_model.__class__.__name__

    jax_model, variables, template_data = convert_model(torch_model, config)

    params_bytes = serialization.to_bytes(variables)
    output_path.write_bytes(params_bytes)
    print(f'Serialized JAX parameters written to {output_path}')
    # Persist config alongside parameters.
    config_path = output_path.with_suffix('.json')
    config_path.write_text(json.dumps(config, indent=2))
    print(f'Config written to {config_path}')

    if args.predict:
        _, configurations = data_utils.load_from_xyz(args.predict)
        if not configurations:
            raise ValueError(f'No configurations found in {args.predict}')

        species_table = data_utils.AtomicNumberTable(
            [int(z) for z in config['atomic_numbers']]
        )

        for idx, configuration in enumerate(configurations):
            species_indices = data_utils.atomic_numbers_to_indices(
                configuration.atomic_numbers, species_table
            )
            indexed_config = replace(configuration, atomic_numbers=species_indices)

            graph = graph_from_configuration(
                indexed_config,
                cutoff=float(config['r_max']),
            )
            data_dict = _graph_to_data(graph, num_species=len(species_table))
            outputs = jax_model.apply(
                variables,
                data_dict,
                compute_force=True,
                compute_stress=False,
            )
            forces = np.asarray(outputs['forces'])
            energy = float(np.asarray(outputs['energy']).sum())

            print(f'Configuration {idx}:')
            print(f'  Energy: {energy}')
            print(f'  Forces (shape {forces.shape}):')
            print(forces)


if __name__ == '__main__':
    main()
