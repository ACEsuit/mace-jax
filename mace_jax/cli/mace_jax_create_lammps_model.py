from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any

import jax
import torch
from flax import serialization
from mace.tools.scripts_utils import extract_config_mace_model

from mace_jax.calculators.lammps_mliap_mace import create_lammps_mliap_calculator

from .mace_torch2jax import convert_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Create a JAX MLIAP LAMMPS model from a Torch checkpoint.',
    )
    parser.add_argument('model_path', help='Path to the Torch checkpoint (.pt).')
    parser.add_argument(
        '--head',
        help='Optional head name to select when the Torch model has multiple heads.',
    )
    parser.add_argument(
        '--dtype',
        default='float64',
        choices=['float64', 'float32'],
        help='Convert parameters to the requested dtype before saving.',
    )
    parser.add_argument(
        '--output',
        help='Destination file for the exported model. Defaults to <model_path>-jax-lammps.pkl',
    )
    return parser.parse_args()


def load_torch_model(path: Path) -> Any:
    bundle = torch.load(path, map_location='cpu')
    if isinstance(bundle, dict) and 'model' in bundle:
        model = bundle['model']
    else:
        model = bundle
    model.eval()
    return model


def main() -> None:
    args = parse_args()
    torch_model = load_torch_model(Path(args.model_path))

    config = extract_config_mace_model(torch_model)
    if 'error' in config:
        raise RuntimeError(config['error'])
    config['torch_model_class'] = torch_model.__class__.__name__

    jax_model, variables, _ = convert_model(torch_model, config)

    if args.dtype == 'float32':
        variables = jax.tree_util.tree_map(lambda x: x.astype('float32'), variables)

    lammps_wrapper = create_lammps_mliap_calculator(
        jax_model,
        variables,
        head=args.head,
    )

    artifact = {
        'config': config,
        'params': serialization.to_bytes(variables),
        'head': args.head,
    }

    output_path = (
        Path(args.output)
        if args.output is not None
        else Path(args.model_path)
        .with_suffix('')
        .with_name(Path(args.model_path).name + '-jax-lammps.pkl')
    )

    with output_path.open('wb') as fh:
        pickle.dump(artifact, fh)

    print(f'Wrote JAX LAMMPS artifact to {output_path}')
    print('To instantiate inside Python:')
    print(
        '  from mace_jax.calculators.lammps_mliap_mace import create_lammps_mliap_calculator'
    )
    print('  artifact = pickle.load(open(path, "rb"))')
    print('  jax_model, variables, _ = convert_model(torch_model, artifact["config"])')
    print(
        '  wrapper = create_lammps_mliap_calculator(jax_model, variables, head=artifact["head"])'
    )
    _ = lammps_wrapper  # silence unused warning


if __name__ == '__main__':
    main()
