import json
from pathlib import Path

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from ase import Atoms
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

    atoms = Atoms(
        symbols=['H', 'H', 'Ne', 'O'],
        positions=np.array(
            [
                [0.0, 0.0, 0.0],
                [0.5, 0.0, 0.0],
                [0.0, 0.4, 0.0],
                [0.0, 0.3, 0.3],
            ]
        ),
        cell=np.identity(3),
        pbc=[True, True, False],
    )

    config = config_from_atoms(atoms)
    config.pbc = [bool(x) for x in config.pbc]
    atomic_data = AtomicData.from_config(
        config,
        z_table=statistics['atomic_numbers'],
        cutoff=2.0,
    )
    batch = torch_geometric.batch.Batch.from_data_list([atomic_data])

    @pytest.fixture
    def torch_model(self):
        args = self._default_args()
        self._prepare_args(args)
        model, _ = configure_model_torch(
            args,
            train_loader=[],
            atomic_energies=self.statistics['atomic_energies'],
            heads=args.heads,
            z_table=self.statistics['atomic_numbers'],
        )
        model.eval()
        model._configure_args = args
        return model

    @pytest.fixture
    def jax_model(self, torch_model):
        args = torch_model._configure_args

        def forward_fn(batch):
            model = configure_model_jax(
                args,
                atomic_energies=self.statistics['atomic_energies'],
                z_table=self.statistics['atomic_numbers'],
            )
            return model(batch, compute_stress=True)

        return hk.transform_with_state(forward_fn)

    def _default_args(self):
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

    def _prepare_args(self, args):
        args.mean = self.statistics['mean']
        args.std = self.statistics['std']
        args.compute_energy = True
        args.compute_dipole = False
        args.key_specification = None
        args.heads = prepare_default_head(args)
        args.avg_num_neighbors = self.statistics['avg_num_neighbors']
        args.r_max = self.statistics['r_max']
        args.scaling = 'no_scaling'

    def test_model_outputs_match(self, torch_model, jax_model):
        batch_torch = self.batch
        torch_output = torch_model(batch_torch, compute_stress=True)

        batch_jax = _batch_to_jax(batch_torch)
        rng = jax.random.PRNGKey(0)
        params, state = jax_model.init(rng, batch_jax)
        params = copy_torch_to_jax(torch_model, params)

        jax_output, _ = jax_model.apply(params, state, rng, batch_jax)

        np.testing.assert_allclose(
            np.asarray(jax_output['energy']),
            torch_output['energy'].detach().cpu().numpy(),
            rtol=1e-2,
            atol=1e-2,
        )
        np.testing.assert_allclose(
            np.asarray(jax_output['forces']),
            torch_output['forces'].detach().cpu().numpy(),
            rtol=3e-2,
            atol=3e-2,
        )
        np.testing.assert_allclose(
            np.asarray(jax_output['stress']),
            torch_output['stress'].detach().cpu().numpy(),
            rtol=1e-2,
            atol=1e-2,
        )
