from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Callable, Optional, Tuple

import jax.numpy as jnp
import numpy as np
from e3nn_jax import Irrep, Irreps
from flax import linen as fnn
from flax import struct

from mace_jax.adapters.e3nn.o3 import SphericalHarmonics
from mace_jax.adapters.flax.torch import auto_import_from_torch_flax
from mace_jax.modules.embeddings import GenericJointEmbedding
from mace_jax.modules.radial import ZBLBasis
from mace_jax.tools.dtype import default_dtype
from mace_jax.tools.lammps_exchange import forward_exchange as lammps_forward_exchange
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
from .utils import add_output_interface, prepare_graph


def _apply_lammps_exchange(
    node_feats: jnp.ndarray,
    lammps_class: Optional[Any],
    lammps_natoms: Tuple[int, int],
) -> jnp.ndarray:
    """Host exchange helper mirroring the Torch LAMMPS MP behaviour."""

    if lammps_class is None:
        return node_feats

    n_real = int(lammps_natoms[0])
    n_ghosts = int(lammps_natoms[1])
    if n_ghosts <= 0:
        return node_feats

    pad = jnp.zeros((n_ghosts, node_feats.shape[1]), dtype=node_feats.dtype)
    padded = jnp.concatenate((node_feats, pad), axis=0)
    exchanged = lammps_forward_exchange(padded, lammps_class)
    return exchanged[:n_real]


def _as_tuple(value: Sequence[int] | int, repeats: int) -> tuple[int, ...]:
    if isinstance(value, int):
        return tuple([value] * repeats)
    return tuple(value)


@auto_import_from_torch_flax(allow_missing_mapper=True)
@add_output_interface
class MACE(fnn.Module):
    r_max: float
    num_bessel: int
    num_polynomial_cutoff: int
    max_ell: int
    interaction_cls: type[InteractionBlock] = struct.field(pytree_node=False)
    interaction_cls_first: type[InteractionBlock] = struct.field(pytree_node=False)
    atomic_energies: np.ndarray = struct.field(pytree_node=False)
    atomic_numbers: tuple[int, ...] = struct.field(pytree_node=False)
    num_interactions: int = 3
    num_elements: int = 1
    hidden_irreps: Irreps = Irreps('1x0e')
    MLP_irreps: Irreps = Irreps('1x0e')
    avg_num_neighbors: float = 1.0
    correlation: int | Sequence[int] = 1
    gate: Callable | None = struct.field(pytree_node=False, default=None)
    pair_repulsion: bool = False
    apply_cutoff: bool = True
    use_reduced_cg: bool = True
    use_so3: bool = False
    use_agnostic_product: bool = False
    use_last_readout_only: bool = False
    use_embedding_readout: bool = False
    distance_transform: str = 'None'
    edge_irreps: Irreps | None = None
    radial_MLP: Sequence[int] | None = None
    radial_type: str = 'bessel'
    heads: Sequence[str] | None = struct.field(pytree_node=False, default=None)
    cueq_config: dict[str, Any] | None = struct.field(pytree_node=False, default=None)
    embedding_specs: dict[str, Any] | None = struct.field(
        pytree_node=False, default=None
    )
    readout_cls: type[NonLinearReadoutBlock] = struct.field(
        pytree_node=False, default=NonLinearReadoutBlock
    )

    def setup(self) -> None:
        self._heads = tuple(self.heads) if self.heads is not None else ('Default',)
        correlation = _as_tuple(self.correlation, self.num_interactions)
        if len(correlation) != self.num_interactions:
            raise ValueError(
                'Length of correlation list must match num_interactions '
                f'(expected {self.num_interactions}, got {len(correlation)})'
            )
        self._correlation = correlation

        self._atomic_numbers = jnp.asarray(self.atomic_numbers, dtype=jnp.int32)
        self._atomic_energies = jnp.asarray(self.atomic_energies, dtype=default_dtype())

        hidden_irreps = (
            self.hidden_irreps
            if isinstance(self.hidden_irreps, Irreps)
            else Irreps(self.hidden_irreps)
        )
        mlp_irreps = (
            self.MLP_irreps
            if isinstance(self.MLP_irreps, Irreps)
            else Irreps(self.MLP_irreps)
        )
        self._hidden_irreps = hidden_irreps
        self._mlp_irreps = mlp_irreps

        node_attr_irreps = Irreps([(self.num_elements, (0, 1))])
        scalar_mul = hidden_irreps.count(Irrep(0, 1))
        node_feats_irreps = Irreps([(scalar_mul, (0, 1))])

        self.node_embedding = LinearNodeEmbeddingBlock(
            irreps_in=node_attr_irreps,
            irreps_out=node_feats_irreps,
            cueq_config=self.cueq_config,
            name='node_embedding',
        )

        self._embedding_specs = self.embedding_specs or {}
        if self._embedding_specs:
            self._embedding_names = tuple(self._embedding_specs.keys())
            self.joint_embedding = GenericJointEmbedding(
                base_dim=node_feats_irreps.count(Irrep(0, 1)),
                embedding_specs=self._embedding_specs,
                out_dim=node_feats_irreps.count(Irrep(0, 1)),
                name='joint_embedding',
            )
            if self.use_embedding_readout:
                self.embedding_readout = LinearReadoutBlock(
                    node_feats_irreps,
                    Irreps(f'{len(self._heads)}x0e'),
                    self.cueq_config,
                    name='embedding_readout',
                )

        self.radial_embedding = RadialEmbeddingBlock(
            r_max=self.r_max,
            num_bessel=self.num_bessel,
            num_polynomial_cutoff=self.num_polynomial_cutoff,
            radial_type=self.radial_type,
            distance_transform=self.distance_transform,
            apply_cutoff=self.apply_cutoff,
            name='radial_embedding',
        )
        edge_feats_irreps = Irreps(f'{self.radial_embedding.out_dim}x0e')

        if self.pair_repulsion:
            self.pair_repulsion_fn = ZBLBasis(
                p=self.num_polynomial_cutoff,
                name='pair_repulsion_fn',
            )

        if not self.use_so3:
            sh_irreps = Irreps.spherical_harmonics(self.max_ell)
        else:
            sh_irreps = Irreps.spherical_harmonics(self.max_ell, p=1)

        num_features = hidden_irreps.count(Irrep(0, 1))

        def _generate_irreps(l_val: int) -> Irreps:
            repr_str = '+'.join([f'1x{i}e+1x{i}o' for i in range(l_val + 1)])
            return Irreps(repr_str)

        sh_irreps_inter = sh_irreps
        if self.hidden_irreps.count(Irrep(0, -1)) > 0:
            sh_irreps_inter = _generate_irreps(self.max_ell)

        interaction_irreps = (sh_irreps_inter * num_features).sort()[0].simplify()
        interaction_irreps_first = (sh_irreps * num_features).sort()[0].simplify()

        self.spherical_harmonics = SphericalHarmonics(
            sh_irreps,
            normalize=True,
            normalization='component',
            name='spherical_harmonics',
        )

        radial_mlp = (
            list(self.radial_MLP) if self.radial_MLP is not None else [64, 64, 64]
        )
        self.atomic_energies_fn = AtomicEnergiesBlock(self._atomic_energies)

        interactions = []
        products = []
        readouts = []

        interaction_first = self.interaction_cls_first(
            node_attrs_irreps=node_attr_irreps,
            node_feats_irreps=node_feats_irreps,
            edge_attrs_irreps=sh_irreps,
            edge_feats_irreps=edge_feats_irreps,
            target_irreps=interaction_irreps_first,
            hidden_irreps=hidden_irreps,
            avg_num_neighbors=self.avg_num_neighbors,
            radial_MLP=radial_mlp,
            cueq_config=self.cueq_config,
            name='interactions_0',
        )
        interactions.append(interaction_first)

        use_sc_first = 'Residual' in self.interaction_cls_first.__name__
        product_first = EquivariantProductBasisBlock(
            node_feats_irreps=interaction_first.target_irreps,
            target_irreps=self.hidden_irreps,
            correlation=self._correlation[0],
            num_elements=self.num_elements,
            use_sc=use_sc_first,
            cueq_config=self.cueq_config,
            use_reduced_cg=self.use_reduced_cg,
            use_agnostic_product=self.use_agnostic_product,
            name='products_0',
        )
        products.append(product_first)

        if not self.use_last_readout_only:
            readouts.append(
                LinearReadoutBlock(
                    hidden_irreps,
                    Irreps(f'{len(self._heads)}x0e'),
                    self.cueq_config,
                    name=f'readouts_{len(readouts)}',
                )
            )

        for idx in range(self.num_interactions - 1):
            if idx == self.num_interactions - 2:
                hidden_irreps_out = Irreps(str(hidden_irreps[0]))
            else:
                hidden_irreps_out = hidden_irreps

            interaction = self.interaction_cls(
                node_attrs_irreps=node_attr_irreps,
                node_feats_irreps=hidden_irreps,
                edge_attrs_irreps=sh_irreps,
                edge_feats_irreps=edge_feats_irreps,
                target_irreps=interaction_irreps,
                hidden_irreps=hidden_irreps_out,
                avg_num_neighbors=self.avg_num_neighbors,
                edge_irreps=self.edge_irreps,
                radial_MLP=radial_mlp,
                cueq_config=self.cueq_config,
                name=f'interactions_{len(interactions)}',
            )
            interactions.append(interaction)

            product = EquivariantProductBasisBlock(
                node_feats_irreps=interaction_irreps,
                target_irreps=hidden_irreps_out,
                correlation=self._correlation[idx + 1],
                num_elements=self.num_elements,
                use_sc=True,
                cueq_config=self.cueq_config,
                use_reduced_cg=self.use_reduced_cg,
                use_agnostic_product=self.use_agnostic_product,
                name=f'products_{len(products)}',
            )
            products.append(product)

            if idx == self.num_interactions - 2:
                readouts.append(
                    self.readout_cls(
                        hidden_irreps_out,
                        (len(self._heads) * mlp_irreps).simplify(),
                        self.gate,
                        Irreps(f'{len(self._heads)}x0e'),
                        len(self._heads),
                        self.cueq_config,
                        name=f'readouts_{len(readouts)}',
                    )
                )
            elif not self.use_last_readout_only:
                readouts.append(
                    LinearReadoutBlock(
                        hidden_irreps,
                        Irreps(f'{len(self._heads)}x0e'),
                        self.cueq_config,
                        name=f'readouts_{len(readouts)}',
                    )
                )

        self.interactions = tuple(interactions)
        self.products = tuple(products)
        self.readouts = tuple(readouts)

    def __call__(
        self,
        data: dict[str, jnp.ndarray],
        *,
        lammps_mliap: bool = False,
        lammps_class: Any | None = None,
    ) -> dict[str, jnp.ndarray]:
        ctx = prepare_graph(
            data,
            lammps_mliap=lammps_mliap,
            lammps_class=data.get('lammps_class', lammps_class),
        )
        num_atoms_arange = ctx.num_atoms_arange
        node_heads = ctx.node_heads
        interaction_kwargs = ctx.interaction_kwargs
        lammps_class = interaction_kwargs.lammps_class
        lammps_natoms = interaction_kwargs.lammps_natoms
        n_real = int(num_atoms_arange.shape[0])
        if lammps_class is not None:
            n_real = int(lammps_natoms[0])
        node_e0 = self.atomic_energies_fn(data['node_attrs'])[
            num_atoms_arange, node_heads
        ]
        e0 = scatter_sum(
            src=node_e0,
            index=data['batch'],
            dim=0,
            dim_size=ctx.num_graphs,
        ).astype(ctx.vectors.dtype)

        node_feats = self.node_embedding(data['node_attrs'])
        edge_attrs = self.spherical_harmonics(ctx.vectors)
        edge_feats, cutoff = self.radial_embedding(
            ctx.lengths,
            data['node_attrs'],
            data['edge_index'],
            self._atomic_numbers,
        )

        if self.pair_repulsion:
            pair_node_energy = self.pair_repulsion_fn(
                ctx.lengths,
                data['node_attrs'],
                data['edge_index'],
                self._atomic_numbers,
            )
            if lammps_class is not None:
                pair_node_energy = pair_node_energy[:n_real]
            pair_energy = scatter_sum(
                src=pair_node_energy,
                index=data['batch'],
                dim=-1,
                dim_size=ctx.num_graphs,
            )
        else:
            pair_node_energy = jnp.zeros_like(node_e0)
            pair_energy = jnp.zeros_like(e0)

        if self._embedding_specs:
            embedding_features = {name: data[name] for name in self._embedding_names}
            node_feats += self.joint_embedding(data['batch'], embedding_features)
            if self.use_embedding_readout:
                embedding_node_energy = self.embedding_readout(
                    node_feats, node_heads
                ).squeeze(-1)
                embedding_energy = scatter_sum(
                    src=embedding_node_energy,
                    index=data['batch'],
                    dim=0,
                    dim_size=ctx.num_graphs,
                )
                e0 += embedding_energy

        energies = [e0, pair_energy]
        node_energies_list = [node_e0, pair_node_energy]
        node_feats_concat: list[jnp.ndarray] = []

        node_attrs_full = data['node_attrs']

        for idx, (interaction, product) in enumerate(
            zip(self.interactions, self.products)
        ):
            if lammps_class is not None and idx > 0:
                node_feats = _apply_lammps_exchange(
                    node_feats, lammps_class, lammps_natoms
                )

            node_attrs_slice = node_attrs_full
            if lammps_class is not None and idx > 0:
                node_attrs_slice = node_attrs_slice[:n_real]

            node_feats, sc = interaction(
                node_attrs=node_attrs_slice,
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=data['edge_index'],
                cutoff=cutoff,
                first_layer=(idx == 0),
            )
            node_feats = product(
                node_feats=node_feats,
                sc=sc,
                node_attrs=node_attrs_slice,
            )
            if lammps_class is not None:
                node_feats = node_feats[:n_real]

            node_feats_concat.append(node_feats)

        for idx, readout in enumerate(self.readouts):
            feat_idx = -1 if len(self.readouts) == 1 else idx
            node_es = readout(node_feats_concat[feat_idx], node_heads)[
                num_atoms_arange, node_heads
            ]
            energy = scatter_sum(
                src=node_es,
                index=data['batch'],
                dim=0,
                dim_size=ctx.num_graphs,
            )
            energies.append(energy)
            node_energies_list.append(node_es)

        contributions = jnp.stack(energies, axis=-1)
        total_energy = jnp.sum(contributions, axis=-1)
        node_energy = jnp.sum(jnp.stack(node_energies_list, axis=-1), axis=-1)
        node_feats_out = (
            jnp.concatenate(node_feats_concat, axis=-1)
            if node_feats_concat
            else node_feats
        )
        return {
            'energy': total_energy,
            'node_energy': node_energy,
            'contributions': contributions,
            'node_feats': node_feats_out,
            'interaction_energy': total_energy - e0,
            'displacement': ctx.displacement,
            'lammps_natoms': ctx.interaction_kwargs.lammps_natoms,
        }


@auto_import_from_torch_flax(allow_missing_mapper=True)
@add_output_interface
class ScaleShiftMACE(MACE):
    atomic_inter_scale: float = 1.0
    atomic_inter_shift: float = 0.0

    def setup(self) -> None:
        super().setup()
        self.scale_shift = ScaleShiftBlock(
            scale=self.atomic_inter_scale,
            shift=self.atomic_inter_shift,
        )

    def __call__(
        self,
        data: dict[str, jnp.ndarray],
        *,
        lammps_mliap: bool = False,
        lammps_class: Optional[Any] = None,
    ) -> dict[str, jnp.ndarray]:
        ctx = prepare_graph(
            data,
            lammps_mliap=lammps_mliap,
            lammps_class=data.get('lammps_class', lammps_class),
        )
        num_atoms_arange = ctx.num_atoms_arange
        node_heads = ctx.node_heads
        interaction_kwargs = ctx.interaction_kwargs
        lammps_class = interaction_kwargs.lammps_class
        lammps_natoms = interaction_kwargs.lammps_natoms
        n_real = int(num_atoms_arange.shape[0])
        if lammps_class is not None:
            n_real = int(lammps_natoms[0])
        node_e0 = self.atomic_energies_fn(data['node_attrs'])[
            num_atoms_arange, node_heads
        ]
        e0 = scatter_sum(
            src=node_e0,
            index=data['batch'],
            dim=0,
            dim_size=ctx.num_graphs,
        ).astype(ctx.vectors.dtype)

        node_feats = self.node_embedding(data['node_attrs'])
        edge_attrs = self.spherical_harmonics(ctx.vectors)
        edge_feats, cutoff = self.radial_embedding(
            ctx.lengths,
            data['node_attrs'],
            data['edge_index'],
            self._atomic_numbers,
        )

        if self.pair_repulsion:
            pair_node_energy = self.pair_repulsion_fn(
                ctx.lengths,
                data['node_attrs'],
                data['edge_index'],
                self._atomic_numbers,
            )
            if lammps_class is not None:
                pair_node_energy = pair_node_energy[:n_real]
        else:
            pair_node_energy = jnp.zeros_like(node_e0)

        if self._embedding_specs:
            embedding_features = {name: data[name] for name in self._embedding_names}
            node_feats += self.joint_embedding(data['batch'], embedding_features)
            if self.use_embedding_readout:
                embedding_node_energy = self.embedding_readout(
                    node_feats, node_heads
                ).squeeze(-1)
                e0 += scatter_sum(
                    src=embedding_node_energy,
                    index=data['batch'],
                    dim=0,
                    dim_size=ctx.num_graphs,
                )

        node_energies_list = [pair_node_energy]
        node_feats_list: list[jnp.ndarray] = []

        node_attrs_full = data['node_attrs']

        for idx, (interaction, product) in enumerate(
            zip(self.interactions, self.products)
        ):
            if lammps_class is not None and idx > 0:
                node_feats = _apply_lammps_exchange(
                    node_feats, lammps_class, lammps_natoms
                )

            node_attrs_slice = node_attrs_full
            if lammps_class is not None and idx > 0:
                node_attrs_slice = node_attrs_slice[:n_real]

            node_feats, sc = interaction(
                node_attrs=node_attrs_slice,
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=data['edge_index'],
                cutoff=cutoff,
                first_layer=(idx == 0),
            )
            node_feats = product(
                node_feats=node_feats,
                sc=sc,
                node_attrs=node_attrs_slice,
            )
            if lammps_class is not None:
                node_feats = node_feats[:n_real]

            node_feats_list.append(node_feats)

        for idx, readout in enumerate(self.readouts):
            feat_idx = -1 if len(self.readouts) == 1 else idx
            node_energies_list.append(
                readout(node_feats_list[feat_idx], node_heads)[
                    num_atoms_arange, node_heads
                ]
            )

        node_feats_out = (
            jnp.concatenate(node_feats_list, axis=-1) if node_feats_list else node_feats
        )
        node_inter_es = jnp.sum(jnp.stack(node_energies_list, axis=0), axis=0)
        node_inter_es = self.scale_shift(node_inter_es, node_heads)
        inter_e = scatter_sum(
            node_inter_es,
            index=data['batch'],
            dim=-1,
            dim_size=ctx.num_graphs,
        )

        total_energy = e0 + inter_e
        node_energy = node_e0 + node_inter_es
        contributions = jnp.stack((e0, inter_e), axis=-1)
        return {
            'energy': total_energy,
            'node_energy': node_energy,
            'contributions': contributions,
            'node_feats': node_feats_out,
            'interaction_energy': inter_e,
            'displacement': ctx.displacement,
            'lammps_natoms': ctx.interaction_kwargs.lammps_natoms,
        }
