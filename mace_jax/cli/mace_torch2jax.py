#!/usr/bin/env python
"""Convert a pre-trained Torch MACE foundation model to MACE-JAX parameters."""

from __future__ import annotations

import argparse
from dataclasses import replace
import warnings
from pathlib import Path
from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
warnings.filterwarnings(
    'ignore',
    message='Environment variable TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD detected.*',
    category=UserWarning,
)

import torch
from e3nn_jax import Irreps
from flax import serialization
from mace.calculators import foundations_models
from mace.tools.scripts_utils import extract_config_mace_model

from mace_jax.data import utils as data_utils
from mace_jax.data.utils import Configuration, graph_from_configuration
from mace_jax.modules import interaction_classes, readout_classes
from mace_jax.modules.models import MACE, ScaleShiftMACE
from mace_jax.tools.gin_model import _graph_to_data  # type: ignore[attr-defined]


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

def _as_irreps(value: Any) -> Irreps:
    if isinstance(value, Irreps):
        return value
    if isinstance(value, str):
        return Irreps(value)
    if isinstance(value, int):
        return Irreps(f'{value}x0e')
    return Irreps(str(value))


def _interaction(name_or_cls: Any):
    name = name_or_cls if isinstance(name_or_cls, str) else name_or_cls.__name__
    if name not in interaction_classes:
        raise ValueError(f'Unsupported interaction class {name!r} in Torch model')
    return interaction_classes[name]


def _readout(name_or_cls: Any):
    if name_or_cls is None:
        return readout_classes['NonLinearReadoutBlock']
    name = name_or_cls if isinstance(name_or_cls, str) else name_or_cls.__name__
    return readout_classes.get(name, readout_classes['NonLinearReadoutBlock'])


def _build_configuration(
    atomic_numbers: tuple[int, ...], r_max: float
) -> Configuration:
    num_atoms = len(atomic_numbers)
    spacing = max(r_max / max(num_atoms, 1), 0.5)
    positions = np.zeros((num_atoms, 3), dtype=float)
    for i in range(num_atoms):
        positions[i, 0] = spacing * i
        positions[i, 1] = spacing * (i % 2)
        positions[i, 2] = 0.0
    return Configuration(
        atomic_numbers=np.array(atomic_numbers, dtype=int),
        positions=positions,
        energy=np.array(0.0),
        forces=np.zeros_like(positions),
        stress=np.zeros((3, 3)),
        cell=np.eye(3) * (spacing * max(num_atoms, 1) * 2),
        pbc=(False, False, False),
    )


def _prepare_template_data(config: dict[str, Any]) -> dict[str, jnp.ndarray]:
    atomic_numbers = tuple(int(z) for z in config['atomic_numbers'])
    configuration = _build_configuration(atomic_numbers, config['r_max'])
    graph = graph_from_configuration(configuration, cutoff=config['r_max'])
    data = _graph_to_data(graph, num_species=len(atomic_numbers))
    return data


def _build_jax_model(config: dict[str, Any]):
    common_kwargs = dict(
        r_max=config['r_max'],
        num_bessel=config['num_bessel'],
        num_polynomial_cutoff=config['num_polynomial_cutoff'],
        max_ell=config['max_ell'],
        interaction_cls=_interaction(config['interaction_cls']),
        interaction_cls_first=_interaction(config['interaction_cls_first']),
        num_interactions=config['num_interactions'],
        num_elements=len(config['atomic_numbers']),
        hidden_irreps=_as_irreps(config['hidden_irreps']),
        MLP_irreps=_as_irreps(config['MLP_irreps']),
        atomic_numbers=tuple(int(z) for z in config['atomic_numbers']),
        atomic_energies=np.asarray(config['atomic_energies'], dtype=np.float32),
        avg_num_neighbors=float(config['avg_num_neighbors']),
        correlation=config['correlation'],
        radial_type=config.get('radial_type', 'bessel'),
        pair_repulsion=config.get('pair_repulsion', False),
        distance_transform=config.get('distance_transform', None),
        embedding_specs=config.get('embedding_specs'),
        use_so3=config.get('use_so3', False),
        use_reduced_cg=config.get('use_reduced_cg', True),
        use_agnostic_product=config.get('use_agnostic_product', False),
        use_last_readout_only=config.get('use_last_readout_only', False),
        use_embedding_readout=config.get('use_embedding_readout', False),
        readout_cls=_readout(config.get('readout_cls', None)),
    )

    if config.get('radial_MLP') is not None:
        common_kwargs['radial_MLP'] = tuple(int(x) for x in config['radial_MLP'])

    if config.get('edge_irreps') is not None:
        common_kwargs['edge_irreps'] = _as_irreps(config['edge_irreps'])

    if config.get('apply_cutoff') is not None:
        common_kwargs['apply_cutoff'] = bool(config['apply_cutoff'])

    torch_class = config.get('torch_model_class', 'MACE')
    if torch_class == 'ScaleShiftMACE' or 'atomic_inter_scale' in config:
        return ScaleShiftMACE(
            atomic_inter_scale=np.asarray(config.get('atomic_inter_scale', 1.0)),
            atomic_inter_shift=np.asarray(config.get('atomic_inter_shift', 0.0)),
            **common_kwargs,
        )
    return MACE(**common_kwargs)


def convert_model(torch_model, config: dict[str, Any]):
    jax_model = _build_jax_model(config)
    template_data = _prepare_template_data(config)
    variables = jax_model.init(jax.random.PRNGKey(0), template_data)
    variables = jax_model.import_from_torch(torch_model, variables)
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
        '--output', required=True, help='Output file for serialized JAX parameters'
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
    else:
        torch_model = _load_torch_model_from_foundations(
            args.foundation, args.model_name
        )
    torch_model.eval()

    config = extract_config_mace_model(torch_model)
    if 'error' in config:
        raise RuntimeError(config['error'])
    config['torch_model_class'] = torch_model.__class__.__name__

    jax_model, variables, template_data = convert_model(torch_model, config)

    params_bytes = serialization.to_bytes(variables)
    Path(args.output).write_bytes(params_bytes)
    print(f'Serialized JAX parameters written to {args.output}')

    if args.predict:
        _, configurations = data_utils.load_from_xyz(args.predict)
        if not configurations:
            raise ValueError(f'No configurations found in {args.predict}')

        species_table = data_utils.AtomicNumberTable([
            int(z) for z in config['atomic_numbers']
        ])

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
