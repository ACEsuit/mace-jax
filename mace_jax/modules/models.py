from typing import Any, Callable, Optional, Union

import haiku as hk
import jax.numpy as jnp
import numpy as np
from e3nn_jax import Irrep, Irreps

from mace_jax.e3nn.o3 import SphericalHarmonics
from mace_jax.haiku.torch import (
    auto_import_from_torch,
    register_import,
)
from mace_jax.modules.embeddings import GenericJointEmbedding
from mace_jax.modules.radial import ZBLBasis
from mace_jax.tools.scatter import scatter_sum

from .blocks import (
    AtomicEnergiesBlock,
    EquivariantProductBasisBlock,
    InteractionBlock,
    LinearNodeEmbeddingBlock,
    LinearReadoutBlock,
    NonLinearReadoutBlock,
    RadialEmbeddingBlock,
    ScaleShiftBlock,
)
from .utils import (
    prepare_graph,
)


@register_import('mace.modules.models.Mace')
@auto_import_from_torch(separator='~')
class MACE(hk.Module):
    def __init__(
        self,
        r_max: float,
        num_bessel: int,
        num_polynomial_cutoff: int,
        max_ell: int,
        interaction_cls: type[InteractionBlock],
        interaction_cls_first: type[InteractionBlock],
        num_interactions: int,
        num_elements: int,
        hidden_irreps: Irreps,
        MLP_irreps: Irreps,
        atomic_energies: np.ndarray,
        avg_num_neighbors: float,
        atomic_numbers: list[int],
        correlation: Union[int, list[int]],
        gate: Optional[Callable],
        pair_repulsion: bool = False,
        apply_cutoff: bool = True,
        use_reduced_cg: bool = True,
        use_so3: bool = False,
        use_agnostic_product: bool = False,
        use_last_readout_only: bool = False,
        use_embedding_readout: bool = False,
        distance_transform: str = 'None',
        edge_irreps: Optional[Irreps] = None,
        radial_MLP: Optional[list[int]] = None,
        radial_type: Optional[str] = 'bessel',
        heads: Optional[list[str]] = None,
        cueq_config: Optional[dict[str, Any]] = None,
        embedding_specs: Optional[dict[str, Any]] = None,
        oeq_config: Optional[dict[str, Any]] = None,
        readout_cls: Optional[type[NonLinearReadoutBlock]] = NonLinearReadoutBlock,
    ):
        super().__init__()
        self.atomic_numbers = jnp.array(atomic_numbers, dtype=jnp.int64)
        self.r_max = float(r_max)
        self.num_interactions = int(num_interactions)
        if heads is None:
            heads = ['Default']
        self.heads = heads
        if isinstance(correlation, int):
            correlation = [correlation] * num_interactions
        self.apply_cutoff = apply_cutoff
        self.edge_irreps = edge_irreps
        self.use_reduced_cg = use_reduced_cg
        self.use_agnostic_product = use_agnostic_product
        self.use_so3 = use_so3
        self.use_last_readout_only = use_last_readout_only

        # Embedding
        node_attr_irreps = Irreps([(num_elements, (0, 1))])
        node_feats_irreps = Irreps([(hidden_irreps.count(Irrep(0, 1)), (0, 1))])
        self.node_embedding = LinearNodeEmbeddingBlock(
            irreps_in=node_attr_irreps,
            irreps_out=node_feats_irreps,
            cueq_config=cueq_config,
            name='node_embedding',
        )
        embedding_size = node_feats_irreps.count(Irrep(0, 1))
        if embedding_specs is not None:
            self.embedding_specs = embedding_specs
            self.joint_embedding = GenericJointEmbedding(
                base_dim=embedding_size,
                embedding_specs=embedding_specs,
                out_dim=embedding_size,
                name='joint_embedding',
            )
            if use_embedding_readout:
                self.embedding_readout = LinearReadoutBlock(
                    node_feats_irreps,
                    Irreps(f'{len(heads)}x0e'),
                    cueq_config,
                    oeq_config,
                    name='embedding_readout',
                )

        self.radial_embedding = RadialEmbeddingBlock(
            r_max=r_max,
            num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
            radial_type=radial_type,
            distance_transform=distance_transform,
            apply_cutoff=apply_cutoff,
            name='radial_embedding',
        )
        edge_feats_irreps = Irreps(f'{self.radial_embedding.out_dim}x0e')
        if pair_repulsion:
            self.pair_repulsion_fn = ZBLBasis(
                p=num_polynomial_cutoff, name='pair_repulsion_fn'
            )
            self.pair_repulsion = True

        if not use_so3:
            sh_irreps = Irreps.spherical_harmonics(max_ell)
        else:
            sh_irreps = Irreps.spherical_harmonics(max_ell, p=1)
        num_features = hidden_irreps.count(Irrep(0, 1))

        # interaction_irreps = (sh_irreps * num_features).sort()[0].simplify()
        def generate_irreps(l):
            str_irrep = '+'.join([f'1x{i}e+1x{i}o' for i in range(l + 1)])
            return Irreps(str_irrep)

        sh_irreps_inter = sh_irreps
        if hidden_irreps.count(Irrep(0, -1)) > 0:
            sh_irreps_inter = generate_irreps(max_ell)
        interaction_irreps = (sh_irreps_inter * num_features).sort()[0].simplify()
        interaction_irreps_first = (sh_irreps * num_features).sort()[0].simplify()

        # TODO
        self.spherical_harmonics = SphericalHarmonics(
            sh_irreps, normalize=True, normalization='component'
        )
        if radial_MLP is None:
            radial_MLP = [64, 64, 64]
        # Interactions and readout
        self.atomic_energies_fn = AtomicEnergiesBlock(atomic_energies)

        inter = interaction_cls_first(
            node_attrs_irreps=node_attr_irreps,
            node_feats_irreps=node_feats_irreps,
            edge_attrs_irreps=sh_irreps,
            edge_feats_irreps=edge_feats_irreps,
            target_irreps=interaction_irreps_first,
            hidden_irreps=hidden_irreps,
            avg_num_neighbors=avg_num_neighbors,
            radial_MLP=radial_MLP,
            cueq_config=cueq_config,
            oeq_config=oeq_config,
        )
        # TODO: Check torch.ModuleList -> List
        self.interactions = [inter]

        # Use the appropriate self connection at the first layer for proper E0
        use_sc_first = False
        if 'Residual' in str(interaction_cls_first):
            use_sc_first = True

        node_feats_irreps_out = inter.target_irreps
        prod = EquivariantProductBasisBlock(
            node_feats_irreps=node_feats_irreps_out,
            target_irreps=hidden_irreps,
            correlation=correlation[0],
            num_elements=num_elements,
            use_sc=use_sc_first,
            cueq_config=cueq_config,
            oeq_config=oeq_config,
            use_reduced_cg=use_reduced_cg,
            use_agnostic_product=use_agnostic_product,
        )
        # TODO: Check torch.ModuleList -> List
        self.products = [prod]

        # TODO: Check torch.ModuleList -> List
        self.readouts = []
        if not use_last_readout_only:
            self.readouts.append(
                LinearReadoutBlock(
                    hidden_irreps,
                    Irreps(f'{len(heads)}x0e'),
                    cueq_config,
                    oeq_config,
                )
            )

        for i in range(num_interactions - 1):
            if i == num_interactions - 2:
                hidden_irreps_out = str(
                    hidden_irreps[0]
                )  # Select only scalars for last layer
            else:
                hidden_irreps_out = hidden_irreps
            inter = interaction_cls(
                node_attrs_irreps=node_attr_irreps,
                node_feats_irreps=hidden_irreps,
                edge_attrs_irreps=sh_irreps,
                edge_feats_irreps=edge_feats_irreps,
                target_irreps=interaction_irreps,
                hidden_irreps=hidden_irreps_out,
                avg_num_neighbors=avg_num_neighbors,
                edge_irreps=edge_irreps,
                radial_MLP=radial_MLP,
                cueq_config=cueq_config,
                oeq_config=oeq_config,
            )
            self.interactions.append(inter)
            prod = EquivariantProductBasisBlock(
                node_feats_irreps=interaction_irreps,
                target_irreps=hidden_irreps_out,
                correlation=correlation[i + 1],
                num_elements=num_elements,
                use_sc=True,
                cueq_config=cueq_config,
                oeq_config=oeq_config,
                use_reduced_cg=use_reduced_cg,
                use_agnostic_product=use_agnostic_product,
            )
            self.products.append(prod)
            if i == num_interactions - 2:
                self.readouts.append(
                    readout_cls(
                        hidden_irreps_out,
                        (len(heads) * MLP_irreps).simplify(),
                        gate,
                        Irreps(f'{len(heads)}x0e'),
                        len(heads),
                        cueq_config,
                        oeq_config,
                    )
                )
            elif not use_last_readout_only:
                self.readouts.append(
                    LinearReadoutBlock(
                        hidden_irreps,
                        Irreps(f'{len(heads)}x0e'),
                        cueq_config,
                        oeq_config,
                    )
                )

    def __call__(
        self,
        data: dict[str, jnp.ndarray],
        training: bool = False,
        compute_force: bool = True,
        compute_virials: bool = False,
        compute_stress: bool = False,
        compute_displacement: bool = False,
        compute_hessian: bool = False,
        compute_edge_forces: bool = False,
        compute_atomic_stresses: bool = False,
    ) -> dict[str, Optional[jnp.ndarray]]:
        # Setup
        # TODO
        ctx = prepare_graph(
            data,
            compute_virials=compute_virials,
            compute_stress=compute_stress,
            compute_displacement=compute_displacement,
        )
        num_atoms_arange = jnp.asarray(ctx.num_atoms_arange, dtype=jnp.int64)
        num_graphs = ctx.num_graphs
        displacement = ctx.displacement
        positions = ctx.positions
        vectors = ctx.vectors
        lengths = ctx.lengths
        cell = ctx.cell
        node_heads = jnp.asarray(ctx.node_heads, dtype=jnp.int64)

        # Atomic energies
        node_e0 = self.atomic_energies_fn(data['node_attrs'])[
            num_atoms_arange, node_heads
        ]
        e0 = scatter_sum(
            src=node_e0, index=data['batch'], dim=0, dim_size=num_graphs
        ).astype(vectors.dtype)  # [n_graphs, n_heads]
        # Embeddings
        node_feats = self.node_embedding(data['node_attrs'])
        edge_attrs = self.spherical_harmonics(vectors)
        edge_feats, cutoff = self.radial_embedding(
            lengths, data['node_attrs'], data['edge_index'], self.atomic_numbers
        )
        if hasattr(self, 'pair_repulsion'):
            pair_node_energy = self.pair_repulsion_fn(
                lengths, data['node_attrs'], data['edge_index'], self.atomic_numbers
            )
            pair_energy = scatter_sum(
                src=pair_node_energy, index=data['batch'], dim=-1, dim_size=num_graphs
            )  # [n_graphs,]
        else:
            pair_node_energy = jnp.zeros_like(node_e0)
            pair_energy = jnp.zeros_like(e0)

        if hasattr(self, 'joint_embedding'):
            embedding_features: dict[str, jnp.ndarray] = {}
            for name, _ in self.embedding_specs.items():
                embedding_features[name] = data[name]
            node_feats += self.joint_embedding(
                data['batch'],
                embedding_features,
            )
            if hasattr(self, 'embedding_readout'):
                embedding_node_energy = self.embedding_readout(
                    node_feats, node_heads
                ).squeeze(-1)
                embedding_energy = scatter_sum(
                    src=embedding_node_energy,
                    index=data['batch'],
                    dim=0,
                    dim_size=num_graphs,
                )
                e0 += embedding_energy

        # Interactions
        energies = [e0, pair_energy]
        node_energies_list = [node_e0, pair_node_energy]
        node_feats_concat: list[jnp.ndarray] = []

        for i, (interaction, product) in enumerate(
            zip(self.interactions, self.products)
        ):
            node_attrs_slice = data['node_attrs']
            node_feats, sc = interaction(
                node_attrs=node_attrs_slice,
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=data['edge_index'],
                cutoff=cutoff,
                first_layer=(i == 0),
            )
            node_feats = product(
                node_feats=node_feats, sc=sc, node_attrs=node_attrs_slice
            )
            node_feats_concat.append(node_feats)

        for i, readout in enumerate(self.readouts):
            feat_idx = -1 if len(self.readouts) == 1 else i
            node_es = readout(node_feats_concat[feat_idx], node_heads)[
                num_atoms_arange, node_heads
            ]
            energy = scatter_sum(node_es, data['batch'], dim=0, dim_size=num_graphs)
            energies.append(energy)
            node_energies_list.append(node_es)

        contributions = jnp.stack(energies, axis=-1)
        total_energy = jnp.sum(contributions, axis=-1)
        node_energy = jnp.sum(jnp.stack(node_energies_list, axis=-1), axis=-1)
        node_feats_out = jnp.concatenate(node_feats_concat, axis=-1)

        # TODO: Compute full output set

        return {
            'energy': total_energy,
            'node_energy': node_energy,
            'contributions': contributions,
            'displacement': displacement,
            'node_feats': node_feats_out,
        }


@register_import('mace.modules.models.Mace')
@auto_import_from_torch(separator='~')
class ScaleShiftMACE(MACE):
    def __init__(
        self,
        atomic_inter_scale: float,
        atomic_inter_shift: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.scale_shift = ScaleShiftBlock(
            scale=atomic_inter_scale, shift=atomic_inter_shift
        )

    def __call__(
        self,
        data: dict[str, jnp.ndarray],
        training: bool = False,
        compute_force: bool = True,
        compute_virials: bool = False,
        compute_stress: bool = False,
        compute_displacement: bool = False,
        compute_hessian: bool = False,
        compute_edge_forces: bool = False,
        compute_atomic_stresses: bool = False,
    ) -> dict[str, Optional[jnp.ndarray]]:
        # Setup
        # TODO
        ctx = prepare_graph(
            data,
            compute_virials=compute_virials,
            compute_stress=compute_stress,
            compute_displacement=compute_displacement,
        )

        num_atoms_arange = ctx.num_atoms_arange.astype(jnp.int64)
        num_graphs = ctx.num_graphs
        displacement = ctx.displacement
        positions = ctx.positions
        vectors = ctx.vectors
        lengths = ctx.lengths
        cell = ctx.cell
        node_heads = ctx.node_heads.astype(jnp.int64)

        # Atomic energies
        node_e0 = self.atomic_energies_fn(data['node_attrs'])[
            num_atoms_arange, node_heads
        ]
        e0 = scatter_sum(
            src=node_e0, index=data['batch'], dim=0, dim_size=num_graphs
        ).astype(vectors.dtype)  # [n_graphs, num_heads]

        # Embeddings
        node_feats = self.node_embedding(data['node_attrs'])
        edge_attrs = self.spherical_harmonics(vectors)
        edge_feats, cutoff = self.radial_embedding(
            lengths, data['node_attrs'], data['edge_index'], self.atomic_numbers
        )

        if hasattr(self, 'pair_repulsion'):
            pair_node_energy = self.pair_repulsion_fn(
                lengths, data['node_attrs'], data['edge_index'], self.atomic_numbers
            )
        else:
            pair_node_energy = jnp.zeros_like(node_e0)

        # Embeddings of additional features
        if hasattr(self, 'joint_embedding'):
            embedding_features: dict[str, jnp.ndarray] = {}
            for name, _ in self.embedding_specs.items():
                embedding_features[name] = data[name]
            node_feats += self.joint_embedding(
                data['batch'],
                embedding_features,
            )
            if hasattr(self, 'embedding_readout'):
                embedding_node_energy = self.embedding_readout(
                    node_feats, node_heads
                ).squeeze(-1)
                embedding_energy = scatter_sum(
                    src=embedding_node_energy,
                    index=data['batch'],
                    dim=0,
                    dim_size=num_graphs,
                )
                e0 += embedding_energy

        # Interactions
        node_es_list = [pair_node_energy]
        node_feats_list: list[jnp.ndarray] = []

        for i, (interaction, product) in enumerate(
            zip(self.interactions, self.products)
        ):
            node_attrs_slice = data['node_attrs']
            node_feats, sc = interaction(
                node_attrs=node_attrs_slice,
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=data['edge_index'],
                cutoff=cutoff,
                first_layer=(i == 0),
            )
            node_feats = product(
                node_feats=node_feats, sc=sc, node_attrs=node_attrs_slice
            )
            node_feats_list.append(node_feats)

        for i, readout in enumerate(self.readouts):
            feat_idx = -1 if len(self.readouts) == 1 else i
            node_es_list.append(
                readout(node_feats_list[feat_idx], node_heads)[
                    num_atoms_arange, node_heads
                ]
            )

        node_feats_out = jnp.concatenate(node_feats_list, axis=-1)
        node_inter_es = jnp.sum(jnp.stack(node_es_list, axis=0), axis=0)
        node_inter_es = self.scale_shift(node_inter_es, node_heads)
        inter_e = scatter_sum(node_inter_es, data['batch'], dim=-1, dim_size=num_graphs)

        total_energy = e0 + inter_e
        node_energy = node_e0.clone().double() + node_inter_es.clone().double()

        # TODO: Compute full output set

        return {
            'energy': total_energy,
            'node_energy': node_energy,
            'interaction_energy': inter_e,
            'displacement': displacement,
            'node_feats': node_feats_out,
        }
