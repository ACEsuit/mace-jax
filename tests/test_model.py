import itertools
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from ase import Atoms
from ase.build import bulk
from e3nn_jax import Irreps
from flax import nnx
from mace.data.atomic_data import AtomicData
from mace.data.utils import config_from_atoms
from mace.tools import torch_geometric

_TORCH_MODEL_IMPORT_ERROR = None
try:  # pragma: no cover - may fail when Torch cue ops unavailable
    from mace.tools.model_script_utils import configure_model as configure_model_torch
    from mace.tools.multihead_tools import AtomicNumberTable, prepare_default_head
    from mace.tools.torch_geometric.batch import Batch
except Exception as exc:  # pragma: no cover
    configure_model_torch = None
    AtomicNumberTable = None
    prepare_default_head = None
    Batch = None
    _TORCH_MODEL_IMPORT_ERROR = exc

from mace_jax import modules
from mace_jax.modules.wrapper_ops import CuEquivarianceConfig
from mace_jax.tools.import_from_torch import import_from_torch


def _torch_cuda_available() -> bool:
    return torch.cuda.is_available() and torch.cuda.device_count() > 0


_TORCH_CUDA_AVAILABLE = _torch_cuda_available()

pytestmark = [
    pytest.mark.skipif(
        _TORCH_MODEL_IMPORT_ERROR is not None,
        reason=(
            'Unable to import Torch cuequivariance components: '
            f'{_TORCH_MODEL_IMPORT_ERROR}'
        ),
    ),
    pytest.mark.skipif(
        not _TORCH_CUDA_AVAILABLE,
        reason='CUDA not available for torch cuequivariance tests.',
    ),
]


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
    rngs: nnx.Rngs | None = None,
):
    import ast  # noqa: PLC0415

    if rngs is None:
        rngs = nnx.Rngs(0)

    cueq_config = None
    if getattr(args, 'only_cueq', False):
        cueq_config = CuEquivarianceConfig(
            enabled=True,
            layout='ir_mul',
            group='O3_e3nn',
            optimize_all=True,
            conv_fusion=(getattr(args, 'device', None) == 'cuda'),
        )
    elif getattr(args, 'enable_cueq', False):
        cueq_config = CuEquivarianceConfig(
            enabled=True,
            layout='mul_ir',
            group='O3',
            optimize_all=True,
            conv_fusion=(getattr(args, 'device', None) == 'cuda'),
        )

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
        atomic_numbers=tuple(int(z) for z in z_table.zs),
        use_reduced_cg=args.use_reduced_cg,
        use_so3=args.use_so3,
        cueq_config=cueq_config,
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
        rngs=rngs,
        radial_MLP=ast.literal_eval(args.radial_MLP),
        radial_type=args.radial_type,
        heads=tuple(args.heads) if args.heads is not None else None,
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


class ModelEquivalenceTestBase:
    stats_path = Path(__file__).parent / 'test_model_statistics.json'
    statistics = None
    structure_repeats: list[tuple[int, int, int]] = [(2, 2, 1), (2, 2, 2)]
    displacement_scales: list[float] = [0.08, 0.12]
    strain_matrices: list[np.ndarray] = [
        np.zeros((3, 3)),
        np.array(
            [
                [0.00, 0.02, 0.00],
                [0.02, 0.00, 0.00],
                [0.00, 0.00, -0.015],
            ]
        ),
    ]
    arguments: list[str] = [
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

    @classmethod
    def setup_class(cls):
        if _TORCH_MODEL_IMPORT_ERROR is not None:
            pytest.skip(f'Torch model helpers unavailable: {_TORCH_MODEL_IMPORT_ERROR}')
        if cls.statistics is None:
            cls.statistics = _load_statistics(cls.stats_path)
        cls.structures = cls._build_structures()
        atomic_data_list = []
        for atoms in cls.structures:
            config = config_from_atoms(atoms)
            config.pbc = [bool(x) for x in config.pbc]
            atomic_data_list.append(
                AtomicData.from_config(
                    config,
                    z_table=cls.statistics['atomic_numbers'],
                    cutoff=float(cls.statistics['r_max']),
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
        cls.torch_node_energy = (
            torch_output['node_energy'].detach().cpu().numpy()
            if 'node_energy' in torch_output
            else None
        )
        cls.torch_interaction_energy = (
            torch_output['interaction_energy'].detach().cpu().numpy()
            if 'interaction_energy' in torch_output
            else None
        )
        cls.torch_node_feats = (
            torch_output['node_feats'].detach().cpu().numpy()
            if 'node_feats' in torch_output
            else None
        )

        cls.jax_model = configure_model_jax(
            cls.args,
            atomic_energies=cls.statistics['atomic_energies'],
            z_table=cls.statistics['atomic_numbers'],
            rngs=nnx.Rngs(0),
        )
        cls.jax_params = nnx.state(cls.jax_model)
        cls.jax_params = import_from_torch(
            cls.jax_model,
            cls.torch_model,
            cls.jax_params,
        )
        cls.jax_graphdef, cls.jax_params = nnx.split(cls.jax_model)
        jax_output, _ = cls.jax_graphdef.apply(cls.jax_params)(
            cls.batch_jax,
            compute_stress=True,
        )

        cls.jax_energy = np.asarray(jax_output['energy'])
        cls.jax_forces = np.asarray(jax_output['forces'])
        cls.jax_stress = np.asarray(jax_output['stress'])
        cls.jax_node_energy = np.asarray(jax_output['node_energy'])
        cls.jax_interaction_energy = np.asarray(jax_output['interaction_energy'])
        cls.jax_node_feats = np.asarray(jax_output['node_feats'])

        cls._collect_block_diagnostics()

        if cls.torch_node_energy is not None:
            cls.max_node_energy_diff = float(
                np.max(np.abs(cls.torch_node_energy - cls.jax_node_energy))
            )
        else:
            cls.max_node_energy_diff = float('nan')

        if cls.torch_interaction_energy is not None:
            cls.max_interaction_energy_diff = float(
                np.max(
                    np.abs(cls.torch_interaction_energy - cls.jax_interaction_energy)
                )
            )
        else:
            cls.max_interaction_energy_diff = float('nan')

    @classmethod
    def _build_structures(cls):
        structures: list[Atoms] = []
        cation_species = ['Na', 'K']
        anion_species = ['Cl', 'Br']

        for idx, repeat in enumerate(cls.structure_repeats):
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
            scale_idx = min(idx, len(cls.displacement_scales) - 1)
            atoms.positions += cls.displacement_scales[scale_idx] * rng.normal(
                size=atoms.positions.shape
            )

            strain_idx = min(idx, len(cls.strain_matrices) - 1)
            strain = cls.strain_matrices[strain_idx]
            if np.any(strain):
                deformation = np.identity(3) + strain
                atoms.set_cell(atoms.cell @ deformation, scale_atoms=True)

            atoms.wrap()
            structures.append(atoms)

        return structures

    @classmethod
    def _default_args(cls):
        from mace.tools import build_default_arg_parser, check_args  # noqa: PLC0415

        args = build_default_arg_parser().parse_args(cls.arguments)
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
        cls._customise_args(args)

    @classmethod
    def _customise_args(cls, args):
        """Hook for subclasses to adjust parsed arguments."""

    @classmethod
    def _block_output_dims(cls) -> list[int]:
        hidden_irreps = Irreps(cls.args.hidden_irreps)
        dims = [hidden_irreps.dim]
        if cls.args.num_interactions <= 1:
            return dims

        base_dim = hidden_irreps.dim
        last_dim = Irreps(str(hidden_irreps[0])).dim
        for idx in range(cls.args.num_interactions - 1):
            if idx == cls.args.num_interactions - 2:
                dims.append(last_dim)
            else:
                dims.append(base_dim)
        return dims

    @classmethod
    def _collect_block_diagnostics(cls) -> None:
        from mace.modules.utils import (  # noqa: PLC0415
            prepare_graph as prepare_graph_torch,
        )

        if cls.torch_node_feats is None or cls.jax_node_feats is None:
            cls.block_feature_dims = tuple()
            cls.blockwise_max_diff = tuple()
            cls.blockwise_mean_diff = tuple()
            cls.interaction_blockwise_max_diff = tuple()
            cls.interaction_blockwise_mean_diff = tuple()
            return

        # --- Torch side unroll ---
        batch = cls.batch
        torch_data: dict[str, torch.Tensor] = {}
        key_list = [
            'positions',
            'shifts',
            'unit_shifts',
            'edge_index',
            'batch',
            'ptr',
            'cell',
            'node_attrs',
            'head',
        ]
        if hasattr(batch, 'pbc'):
            key_list.append('pbc')
        for key in key_list:
            value = getattr(batch, key)
            if isinstance(value, torch.Tensor):
                torch_data[key] = value.detach().clone()

        float_keys = ['positions', 'shifts', 'unit_shifts', 'cell', 'node_attrs']
        for key in float_keys:
            torch_data[key] = torch_data[key].to(torch.get_default_dtype())
        torch_data['edge_index'] = torch_data['edge_index'].long()
        torch_data['batch'] = torch_data['batch'].long()
        torch_data['ptr'] = torch_data['ptr'].long()
        torch_data['head'] = torch_data['head'].long()

        with torch.no_grad():
            ctx_torch = prepare_graph_torch(
                torch_data,
                compute_virials=False,
                compute_stress=True,
                compute_displacement=False,
                lammps_mliap=False,
            )
            node_feats_torch = cls.torch_model.node_embedding(torch_data['node_attrs'])
            edge_attrs_torch = cls.torch_model.spherical_harmonics(ctx_torch.vectors)
            edge_feats_torch, cutoff_torch = cls.torch_model.radial_embedding(
                ctx_torch.lengths,
                torch_data['node_attrs'],
                torch_data['edge_index'],
                cls.torch_model.atomic_numbers,
            )

            torch_interactions: list[np.ndarray] = []
            torch_products: list[np.ndarray] = []
            current = node_feats_torch
            for idx, (interaction, product) in enumerate(
                zip(cls.torch_model.interactions, cls.torch_model.products)
            ):
                node_attrs_slice = torch_data['node_attrs']
                current, sc = interaction(
                    node_attrs=node_attrs_slice,
                    node_feats=current,
                    edge_attrs=edge_attrs_torch,
                    edge_feats=edge_feats_torch,
                    edge_index=torch_data['edge_index'],
                    cutoff=cutoff_torch,
                    first_layer=(idx == 0),
                )
                torch_interactions.append(current.detach().cpu().numpy())
                current = product(
                    node_feats=current,
                    sc=sc,
                    node_attrs=node_attrs_slice,
                )
                torch_products.append(current.detach().cpu().numpy())

        # --- JAX side capture ---
        data_jax = cls.batch_jax

        from mace_jax.modules.utils import (  # noqa: PLC0415
            prepare_graph as prepare_graph_jax,
        )

        def _to_numpy(value):
            if isinstance(value, tuple):
                value = value[0]
            if hasattr(value, 'array'):
                return np.asarray(value.array)
            return np.asarray(value)

        model = nnx.merge(cls.jax_graphdef, cls.jax_params)
        ctx_jax = prepare_graph_jax(data_jax)

        node_attrs = data_jax['node_attrs']
        need_node_attrs_index = model.pair_repulsion or model.distance_transform in {
            'Agnesi',
            'Soft',
        }
        if model.cueq_config is not None and getattr(
            model.cueq_config, 'enabled', False
        ):
            need_node_attrs_index = need_node_attrs_index or bool(
                getattr(model.cueq_config, 'optimize_all', False)
                or getattr(model.cueq_config, 'optimize_symmetric', False)
            )
        node_attrs_index = data_jax.get('node_attrs_index')
        if node_attrs_index is None:
            node_attrs_index = data_jax.get('node_type')
        if node_attrs_index is None:
            node_attrs_index = data_jax.get('species')
        if node_attrs_index is not None and getattr(node_attrs_index, 'ndim', 1) != 1:
            node_attrs_index = None
        if node_attrs_index is None and need_node_attrs_index:
            node_attrs_index = jnp.argmax(node_attrs, axis=1)
        if node_attrs_index is not None:
            node_attrs_index = jnp.asarray(node_attrs_index, dtype=jnp.int32)

        node_feats = model.node_embedding(node_attrs)
        edge_attrs = model.spherical_harmonics(ctx_jax.vectors)
        edge_feats, cutoff = model.radial_embedding(
            ctx_jax.lengths,
            node_attrs,
            data_jax['edge_index'],
            model._atomic_numbers,
            node_attrs_index=node_attrs_index,
        )

        if model._embedding_specs:
            embedding_features = {
                name: data_jax[name] for name in model._embedding_names
            }
            node_feats = node_feats + model.joint_embedding(
                data_jax['batch'], embedding_features
            )

        jax_interactions: list[np.ndarray] = []
        jax_products: list[np.ndarray] = []
        for idx, (interaction, product) in enumerate(
            zip(model.interactions, model.products)
        ):
            node_feats, sc = interaction(
                node_attrs=node_attrs,
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=data_jax['edge_index'],
                cutoff=cutoff,
                n_real=None,
                first_layer=(idx == 0),
            )
            jax_interactions.append(_to_numpy(node_feats))
            node_feats = product(
                node_feats=node_feats,
                sc=sc,
                node_attrs=node_attrs,
                node_attrs_index=node_attrs_index,
            )
            jax_products.append(_to_numpy(node_feats))

        def _align_to_reference(
            j_block: np.ndarray, ref_block: np.ndarray
        ) -> np.ndarray:
            if j_block.shape == ref_block.shape:
                return j_block
            if j_block.size != ref_block.size or j_block.ndim != ref_block.ndim:
                raise AssertionError(
                    'Unable to align arrays with shapes '
                    f'{j_block.shape} and {ref_block.shape}.'
                )
            for perm in itertools.permutations(range(j_block.ndim)):
                permuted_shape = tuple(j_block.shape[i] for i in perm)
                if permuted_shape == ref_block.shape:
                    return np.transpose(j_block, perm)
            raise AssertionError(
                'Unable to find permutation aligning shapes '
                f'{j_block.shape} and {ref_block.shape}.'
            )

        jax_products_raw = list(jax_products)
        jax_interactions_raw = list(jax_interactions)
        jax_products = [
            _align_to_reference(j_block, t_block)
            for j_block, t_block in zip(jax_products_raw, torch_products, strict=False)
        ]
        jax_interactions = [
            _align_to_reference(j_block, t_block)
            for j_block, t_block in zip(
                jax_interactions_raw, torch_interactions, strict=False
            )
        ]

        cls.block_feature_dims = tuple(out.shape[-1] for out in torch_products)
        expected_dims = cls._block_output_dims()
        if len(expected_dims) != len(cls.block_feature_dims):
            raise AssertionError(
                f'Expected {len(expected_dims)} blocks but observed '
                f'{len(cls.block_feature_dims)} blocks from product unroll.'
            )
        if sum(cls.block_feature_dims) != cls.torch_node_feats.shape[-1]:
            raise AssertionError(
                'Concatenated block dimensions do not match Torch node features.'
            )

        torch_concat = np.concatenate(torch_products, axis=-1)
        jax_concat_original = np.concatenate(jax_products_raw, axis=-1)
        if not np.allclose(torch_concat, cls.torch_node_feats):
            raise AssertionError(
                'Torch product outputs do not recombine to stored node features.'
            )
        if not np.allclose(jax_concat_original, cls.jax_node_feats):
            raise AssertionError(
                'JAX product outputs do not recombine to stored node features.'
            )

        cls.blockwise_max_diff = tuple(
            float(np.max(np.abs(t_block - j_block)))
            for t_block, j_block in zip(torch_products, jax_products, strict=False)
        )
        cls.blockwise_mean_diff = tuple(
            float(np.mean(np.abs(t_block - j_block)))
            for t_block, j_block in zip(torch_products, jax_products, strict=False)
        )
        cls.interaction_blockwise_max_diff = tuple(
            float(np.max(np.abs(t_block - j_block)))
            for t_block, j_block in zip(
                torch_interactions, jax_interactions, strict=False
            )
        )
        cls.interaction_blockwise_mean_diff = tuple(
            float(np.mean(np.abs(t_block - j_block)))
            for t_block, j_block in zip(
                torch_interactions, jax_interactions, strict=False
            )
        )

    def test_model_outputs_match(self):
        cls = self.__class__
        np.testing.assert_allclose(
            cls.jax_energy,
            cls.torch_energy,
            rtol=5e-5,
            atol=5e-5,
        )
        np.testing.assert_allclose(
            cls.jax_forces,
            cls.torch_forces,
            rtol=2e-5,
            atol=2e-5,
        )
        np.testing.assert_allclose(
            cls.jax_stress,
            cls.torch_stress,
            rtol=5e-6,
            atol=5e-6,
        )

    def test_blockwise_node_features_within_threshold(self):
        cls = self.__class__
        if not cls.blockwise_max_diff:
            pytest.skip('Block-wise diagnostics unavailable.')

        thresholds = []
        last_idx = len(cls.blockwise_max_diff) - 1
        for idx in range(len(cls.blockwise_max_diff)):
            if idx == 0:
                thresholds.append(0.05)
            elif idx == last_idx:
                thresholds.append(0.04)
            else:
                thresholds.append(0.02)
        for idx, (max_diff, limit) in enumerate(
            zip(cls.blockwise_max_diff, thresholds, strict=False)
        ):
            assert max_diff < limit, (
                f'Interaction block {idx} exceeds tolerance: '
                f'max |Δ|={max_diff:.3f}, limit={limit:.2f}'
            )

    def test_interaction_block_outputs_within_threshold(self):
        cls = self.__class__
        if not cls.interaction_blockwise_max_diff:
            pytest.skip('Interaction diagnostics unavailable.')
        thresholds = []
        last_idx = len(cls.interaction_blockwise_max_diff) - 1
        for idx in range(len(cls.interaction_blockwise_max_diff)):
            if idx == 0 or idx == last_idx:
                thresholds.append(0.03)
            else:
                thresholds.append(0.015)
        for idx, (max_diff, limit) in enumerate(
            zip(cls.interaction_blockwise_max_diff, thresholds, strict=False)
        ):
            assert max_diff < limit, (
                f'Interaction block {idx} pre-product output exceeds tolerance: '
                f'max |Δ|={max_diff:.3f}, limit={limit:.2f}'
            )

    def test_interaction_energy_within_threshold(self):
        cls = self.__class__
        if np.isnan(cls.max_interaction_energy_diff):
            pytest.skip('Interaction energy diagnostics unavailable.')
        assert cls.max_interaction_energy_diff < 5e-4, (
            'Interaction energy deviation too large: '
            f'max |Δ|={cls.max_interaction_energy_diff:.3f}'
        )


class TestModelEquivalenceSmall(ModelEquivalenceTestBase):
    structure_repeats = [(1, 1, 1)]
    displacement_scales = [0.05]
    strain_matrices = [np.zeros((3, 3))]
    arguments = [
        '--name',
        'MACE_small_density',
        '--interaction_first',
        'RealAgnosticInteractionBlock',
        '--interaction',
        'RealAgnosticInteractionBlock',
        '--num_channels',
        '32',
        '--max_L',
        '1',
        '--max_ell',
        '2',
        '--num_interactions',
        '1',
        '--correlation',
        '1',
        '--num_radial_basis',
        '4',
        '--MLP_irreps',
        '8x0e',
        '--distance_transform',
        'Agnesi',
        '--pair_repulsion',
    ]

    @classmethod
    def _customise_args(cls, args):
        args.only_cueq = True
        args.enable_cueq = True
        args.hidden_irreps = '32x0e+32x1o'
        args.radial_MLP = '[32]'
        args.use_agnostic_product = False
        args.gate = 'silu'


class TestModelEquivalenceSmallFullCG(TestModelEquivalenceSmall):
    """Smoke-test full Clebsch–Gordan parameters on a tiny configuration."""

    @classmethod
    def _customise_args(cls, args):
        super()._customise_args(args)
        args.use_reduced_cg = False


class TestModelEquivalenceSmallJitted(TestModelEquivalenceSmall):
    """Exercise the small configuration under jitted apply to reproduce the regression."""

    @classmethod
    def _customise_args(cls, args):
        # Keep the test lightweight while ensuring a non-linear readout is used.
        super()._customise_args(args)
        args.num_interactions = 2
        args.num_channels = 16
        args.hidden_irreps = '16x0e+16x1o'
        args.MLP_irreps = '8x0e'
        args.radial_MLP = '[16]'
        args.correlation = 2

    def test_jitted_apply_matches_eager(self):
        """Ensure the jitted apply produces identical outputs to eager mode."""

        def apply_fn(params, batch, compute_stress=False):
            outputs, _ = self.jax_graphdef.apply(params)(
                batch, compute_stress=compute_stress
            )
            return outputs

        jitted = jax.jit(apply_fn)
        traced = jitted(self.jax_params, self.batch_jax, compute_stress=False)
        eager = apply_fn(self.jax_params, self.batch_jax, compute_stress=False)
        for key, eager_value in eager.items():
            traced_value = traced[key]
            if eager_value is None or traced_value is None:
                # Optional outputs (e.g. stress without compute_stress) return
                # zero arrays rather than ``None`` after JIT, so verify the data
                # is numerically zero in either representation.
                array = traced_value if eager_value is None else eager_value
                np.testing.assert_allclose(
                    np.asarray(array),
                    0.0,
                    rtol=1e-6,
                    atol=1e-6,
                    err_msg=f'Expected zero-valued array for optional output {key!r}',
                )
                continue
            np.testing.assert_allclose(
                np.asarray(traced_value),
                np.asarray(eager_value),
                rtol=1e-6,
                atol=1e-6,
                err_msg=f'JIT output mismatch for key {key!r}',
            )


@pytest.mark.slow
class TestModelEquivalenceLarge(ModelEquivalenceTestBase):
    """Full-size model equivalence aligned with the Torch integration test."""
