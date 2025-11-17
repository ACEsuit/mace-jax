#!/usr/bin/env python
"""Convert a pre-trained Torch MACE foundation model to MACE-JAX parameters."""

from __future__ import annotations

import argparse
import warnings
from dataclasses import replace
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from mace_jax.modules.wrapper_ops import CuEquivarianceConfig

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

from mace_jax.adapters.flax.torch import resolve_gate_callable
from mace_jax.data import utils as data_utils
from mace_jax.data.utils import Configuration, graph_from_configuration
from mace_jax.modules import interaction_classes, readout_classes
from mace_jax.modules.models import MACE, ScaleShiftMACE
from mace_jax.tools.gin_model import _graph_to_data  # type: ignore[attr-defined]
from mace_jax.tools.import_from_torch import import_from_torch


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


def _parse_parity(parity: Any) -> int:
    if parity is None:
        return 1
    if isinstance(parity, str):
        p = parity.strip().lower()
        if p in {'e', 'even'}:
            return 1
        if p in {'o', 'odd'}:
            return -1
    try:
        parity_int = int(parity)
    except (TypeError, ValueError):
        return 1
    return 1 if parity_int >= 0 else -1


def _as_irrep_entry(entry: Any):
    if isinstance(entry, dict):
        mul = entry.get('mul') or entry.get('multiplicity') or entry.get('n')
        rep = entry.get('irrep') or entry.get('rep') or entry.get('l')
        parity = entry.get('p') or entry.get('parity')
        if isinstance(rep, dict):
            l_val = rep.get('l')
            parity = parity or rep.get('p') or rep.get('parity')
        else:
            l_val = rep
        if mul is None or l_val is None:
            return None
        return int(mul), (int(l_val), _parse_parity(parity))

    if isinstance(entry, (list, tuple)):
        if len(entry) == 2 and isinstance(entry[0], (int, np.integer)):
            mul = int(entry[0])
            rep = entry[1]
            if isinstance(rep, (list, tuple)):
                if not rep:
                    return None
                l_val = int(rep[0])
                parity = _parse_parity(rep[1] if len(rep) > 1 else None)
                return mul, (l_val, parity)
            if isinstance(rep, dict):
                l_val = rep.get('l')
                parity = rep.get('p') or rep.get('parity')
                if l_val is None:
                    return None
                return mul, (int(l_val), _parse_parity(parity))
            if isinstance(rep, (int, np.integer)):
                return mul, (int(rep), 1)
    return None


def _normalize_irreps(value: Any):
    if isinstance(value, dict):
        value = [value]

    if isinstance(value, (list, tuple)):
        if value and _as_irrep_entry(value) is not None:
            entries = [value]
        else:
            entries = value

        parsed = []
        for item in entries:
            entry = _as_irrep_entry(item)
            if entry is None:
                return None
            parsed.append(entry)

        return parsed

    return None


def _as_irreps(value: Any) -> Irreps:
    if isinstance(value, Irreps):
        return value
    if isinstance(value, str):
        return Irreps(value)
    if isinstance(value, int):
        return Irreps(f'{value}x0e')
    normalized = _normalize_irreps(value)
    if normalized is not None:
        return Irreps(normalized)
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


def _build_jax_model(
    config: dict[str, Any],
    *,
    cueq_config: CuEquivarianceConfig | None = None,
):
    cue_config_obj: CuEquivarianceConfig | None = None
    if cueq_config is not None:
        cue_config_obj = cueq_config
    elif config.get('cue_conv_fusion'):
        # Mirror the torch wrapper behaviour: allow conv_fusion without enabling
        # the full cue acceleration stack so symmetric contraction stays on the
        # pure-JAX implementation. The tensor product layer will only switch to
        # cue when conv_fusion=True but other ops keep the default backend.
        cue_config_obj = CuEquivarianceConfig(
            enabled=False,
            optimize_channelwise=True,
            conv_fusion=bool(config['cue_conv_fusion']),
            layout='mul_ir',
        )
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
        gate=resolve_gate_callable(config.get('gate', None)),
        cueq_config=cue_config_obj,
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


def convert_model(
    torch_model,
    config: dict[str, Any],
    *,
    cueq_config: CuEquivarianceConfig | None = None,
):
    try:
        jax_model = _build_jax_model(config, cueq_config=cueq_config)
    except TypeError as exc:
        if 'cueq_config' in str(exc):
            jax_model = _build_jax_model(config)
        else:
            raise
    template_data = _prepare_template_data(config)
    variables = jax_model.init(jax.random.PRNGKey(0), template_data)

    variables = import_from_torch(jax_model, torch_model, variables)
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
