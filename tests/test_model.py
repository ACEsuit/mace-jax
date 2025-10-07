import json
from pathlib import Path

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from ase import Atoms
from ase.build import bulk
from mace.data.atomic_data import AtomicData
from mace.data.utils import config_from_atoms
from mace.tools import torch_geometric
from mace.tools.model_script_utils import configure_model as configure_model_torch
from mace.tools.multihead_tools import AtomicNumberTable, prepare_default_head
from mace.tools.torch_geometric.batch import Batch

from mace_jax.haiku.torch import copy_torch_to_jax

jax.config.update('jax_enable_x64', True)


def _load_statistics(path: Path) -> dict:
    data = json.loads(path.read_text())
    stats = dict(data)
    stats['atomic_numbers'] = AtomicNumberTable(stats['atomic_numbers'])
    stats['atomic_energies'] = [
        stats['atomic_energies'][str(z)] for z in stats['atomic_numbers'].zs
    ]
    return stats


def configure_model_jax(
    args,
    atomic_energies,
    z_table=None,
    model_foundation=None,
    head_configs=None,
):
    import ast  # noqa: PLC0415

    from e3nn_jax import Irreps  # noqa: PLC0415

    from mace_jax import modules  # noqa: PLC0415

    model_config = dict(
        r_max=args.r_max,
        num_bessel=args.num_radial_basis,
        num_polynomial_cutoff=args.num_cutoff_basis,
        max_ell=args.max_ell,
        interaction_cls=modules.interaction_classes[args.interaction],
        num_interactions=args.num_interactions,
        num_elements=len(z_table),
        hidden_irreps=Irreps(args.hidden_irreps),
        edge_irreps=Irreps(args.edge_irreps) if args.edge_irreps else None,
        atomic_energies=atomic_energies,
        apply_cutoff=args.apply_cutoff,
        avg_num_neighbors=args.avg_num_neighbors,
        atomic_numbers=z_table.zs,
        use_reduced_cg=args.use_reduced_cg,
        use_so3=args.use_so3,
        cueq_config=None,
    )
    return modules.ScaleShiftMACE(
        **model_config,
        pair_repulsion=args.pair_repulsion,
        distance_transform=args.distance_transform,
        correlation=args.correlation,
        gate=modules.gate_dict[args.gate],
        interaction_cls_first=modules.interaction_classes[args.interaction_first],
        MLP_irreps=Irreps(args.MLP_irreps),
        atomic_inter_scale=args.std,
        atomic_inter_shift=args.mean,
        radial_MLP=ast.literal_eval(args.radial_MLP),
        radial_type=args.radial_type,
        heads=args.heads,
        embedding_specs=args.embedding_specs,
        use_embedding_readout=args.use_embedding_readout,
        use_last_readout_only=args.use_last_readout_only,
        use_agnostic_product=args.use_agnostic_product,
    )


def _batch_to_jax(batch: Batch) -> dict:
    converted = {}
    for key in batch.keys:
        value = batch[key]
        if isinstance(value, torch.Tensor):
            converted[key] = jnp.asarray(value.detach().cpu().numpy())
        else:
            converted[key] = value
    return converted


class TestModelEquivalence:
    stats_path = Path(__file__).parent / 'test_model_statistics.json'
    statistics = _load_statistics(stats_path)

    @classmethod
    def setup_class(cls):
        cls.structures = cls._build_structures()
        atomic_data_list = []
        for atoms in cls.structures:
            config = config_from_atoms(atoms)
            config.pbc = [bool(x) for x in config.pbc]
            atomic_data_list.append(
                AtomicData.from_config(
                    config,
                    z_table=cls.statistics['atomic_numbers'],
                    cutoff=2.0,
                )
            )

        cls.batch = torch_geometric.batch.Batch.from_data_list(atomic_data_list)
        cls.batch_jax = _batch_to_jax(cls.batch)

        cls.args = cls._default_args()
        cls._prepare_args(cls.args)

        cls.torch_model, _ = configure_model_torch(
            cls.args,
            train_loader=[],
            atomic_energies=cls.statistics['atomic_energies'],
            heads=cls.args.heads,
            z_table=cls.statistics['atomic_numbers'],
        )
        cls.torch_model.eval()

        torch_output = cls.torch_model(cls.batch, compute_stress=True)
        cls.torch_energy = torch_output['energy'].detach().cpu().numpy()
        cls.torch_forces = torch_output['forces'].detach().cpu().numpy()
        cls.torch_stress = torch_output['stress'].detach().cpu().numpy()

        def forward_fn(batch):
            model = configure_model_jax(
                cls.args,
                atomic_energies=cls.statistics['atomic_energies'],
                z_table=cls.statistics['atomic_numbers'],
            )
            return model(batch, compute_stress=True)

        cls.jax_transform = hk.transform_with_state(forward_fn)
        init_rng = jax.random.PRNGKey(0)
        apply_rng = jax.random.PRNGKey(1)
        cls.jax_params, cls.jax_state = cls.jax_transform.init(init_rng, cls.batch_jax)
        cls.jax_params = copy_torch_to_jax(cls.torch_model, cls.jax_params)
        jax_output, cls.jax_state = cls.jax_transform.apply(
            cls.jax_params,
            cls.jax_state,
            apply_rng,
            cls.batch_jax,
        )

        cls.jax_energy = np.asarray(jax_output['energy'])
        cls.jax_forces = np.asarray(jax_output['forces'])
        cls.jax_stress = np.asarray(jax_output['stress'])

    @classmethod
    def _build_structures(cls):
        structures: list[Atoms] = []
        repeats = [(2, 2, 1), (2, 2, 2)]
        displacement_scales = [0.08, 0.12]
        strain_matrices = [
            np.zeros((3, 3)),
            np.array(
                [
                    [0.00, 0.02, 0.00],
                    [0.02, 0.00, 0.00],
                    [0.00, 0.00, -0.015],
                ]
            ),
        ]

        cation_species = ['Na', 'K']
        anion_species = ['Cl', 'Br']

        for idx, repeat in enumerate(repeats):
            atoms = bulk('NaCl', 'rocksalt', a=5.64).repeat(repeat)

            cation_idx = idx
            anion_idx = idx
            for atom in atoms:
                if atom.symbol == 'Na':
                    atom.symbol = cation_species[cation_idx % len(cation_species)]
                    cation_idx += 1
                else:
                    atom.symbol = anion_species[anion_idx % len(anion_species)]
                    anion_idx += 1

            rng = np.random.default_rng(seed=42 + idx)
            atoms.positions += displacement_scales[idx] * rng.normal(
                size=atoms.positions.shape
            )

            strain = strain_matrices[idx]
            if np.any(strain):
                deformation = np.identity(3) + strain
                atoms.set_cell(atoms.cell @ deformation, scale_atoms=True)

            atoms.wrap()
            structures.append(atoms)

        return structures

    @classmethod
    def _default_args(cls):
        from mace.tools import build_default_arg_parser, check_args  # noqa: PLC0415

        arguments = [
            '--name',
            'MACE_large_density',
            '--interaction_first',
            'RealAgnosticDensityInteractionBlock',
            '--interaction',
            'RealAgnosticDensityResidualInteractionBlock',
            '--num_channels',
            '128',
            '--max_L',
            '2',
            '--max_ell',
            '3',
            '--num_interactions',
            '3',
            '--correlation',
            '3',
            '--num_radial_basis',
            '8',
            '--MLP_irreps',
            '16x0e',
            '--distance_transform',
            'Agnesi',
            '--pair_repulsion',
            '--only_cueq',
            'True',
        ]

        args = build_default_arg_parser().parse_args(arguments)
        args, _ = check_args(args)
        return args

    @classmethod
    def _prepare_args(cls, args):
        args.mean = cls.statistics['mean']
        args.std = cls.statistics['std']
        args.compute_energy = True
        args.compute_dipole = False
        args.key_specification = None
        args.heads = prepare_default_head(args)
        args.avg_num_neighbors = cls.statistics['avg_num_neighbors']
        args.r_max = cls.statistics['r_max']
        args.scaling = 'no_scaling'

    def test_model_outputs_match(self):
        cls = self.__class__
        np.testing.assert_allclose(
            cls.jax_energy,
            cls.torch_energy,
            rtol=1e-2,
            atol=1e-2,
        )
        np.testing.assert_allclose(
            cls.jax_forces,
            cls.torch_forces,
            rtol=3e-2,
            atol=3e-2,
        )
        np.testing.assert_allclose(
            cls.jax_stress,
            cls.torch_stress,
            rtol=1e-2,
            atol=1e-2,
        )
