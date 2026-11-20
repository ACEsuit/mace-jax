"""PolarMACE with charge/spin constraints and native electrostatic response."""

import jax
import jax.numpy as jnp
import numpy as np
from e3nn_jax import Irreps
from flax import nnx

from mace_jax.adapters.nnx.torch import nxx_auto_import_from_torch
from mace_jax.modules import gto
from mace_jax.modules.blocks import NonLinearBiasReadoutBlock
from mace_jax.modules.field_blocks import (
    AgnosticEmbeddedOneBodyVariableUpdate,
    EnvironmentDependentSpinSourceBlock,
    MultiLayerFeatureMixer,
    OneBodyMLPFieldReadout,
)
from mace_jax.modules.models import ScaleShiftMACE
from mace_jax.modules.utils import add_output_interface
from mace_jax.nnx_config import ConfigVar
from mace_jax.tools.dtype import default_dtype
from mace_jax.tools.scatter import scatter_sum

POLAR_CONFIG_KEYS = (
    'kspace_cutoff_factor',
    'atomic_multipoles_max_l',
    'atomic_multipoles_smearing_width',
    'field_feature_max_l',
    'field_feature_widths',
    'num_recursion_steps',
    'field_si',
    'include_electrostatic_self_interaction',
    'add_local_electron_energy',
    'quadrupole_feature_corrections',
    'return_electrostatic_potentials',
    'field_feature_norms',
    'field_norm_factor',
    'fixedpoint_update_config',
    'field_readout_config',
    'pbc_handling',
    'reciprocal_max_index',
)


@nxx_auto_import_from_torch(allow_missing_mapper=True)
@add_output_interface
class PolarMACE(ScaleShiftMACE):
    cell_dependent = True
    field_dependent = True

    def __init__(
        self,
        *,
        kspace_cutoff_factor=1.5,
        atomic_multipoles_max_l=0,
        atomic_multipoles_smearing_width=1.0,
        field_feature_max_l=0,
        field_feature_widths=(1.0,),
        num_recursion_steps=1,
        field_si=False,
        include_electrostatic_self_interaction=False,
        add_local_electron_energy=False,
        quadrupole_feature_corrections=False,
        return_electrostatic_potentials=False,
        field_feature_norms=None,
        field_norm_factor=0.02,
        fixedpoint_update_config=None,
        field_readout_config=None,
        pbc_handling='auto',
        reciprocal_max_index=8,
        rngs,
        **kwargs,
    ):
        kwargs['keep_last_layer_irreps'] = True
        kwargs.setdefault('use_agnostic_product', True)
        if (
            kwargs.get('use_last_readout_only')
            or kwargs.get('embedding_specs')
            or kwargs.get('pair_repulsion')
        ):
            raise ValueError(
                'PolarMACE currently requires ordinary layer readouts, no embeddings, and no pair repulsion.'
            )
        super().__init__(rngs=rngs, **kwargs)
        if getattr(self.equivariance_config, 'layout', 'mul_ir') != 'mul_ir':
            raise ValueError('PolarMACE currently requires mul_ir layout.')
        if atomic_multipoles_max_l not in (0, 1) or field_feature_max_l not in (0, 1):
            raise ValueError(
                'The Polar electrostatic backend supports angular orders 0 and 1.'
            )
        if num_recursion_steps < 0:
            raise ValueError('num_recursion_steps must be nonnegative.')
        widths = tuple(float(x) for x in field_feature_widths)
        if not widths or any(
            not np.isfinite(x) or x <= 0
            for x in (*widths, atomic_multipoles_smearing_width, kspace_cutoff_factor)
        ):
            raise ValueError(
                'Gaussian widths and kspace_cutoff_factor must be finite and positive.'
            )
        if pbc_handling not in (
            'auto',
            'realspace',
            'pbc',
            'slab',
            'molecule_in_box',
            'mixed_periodic',
        ):
            raise ValueError(
                f'Unsupported electrostatic boundary mode {pbc_handling!r}.'
            )
        self.kspace_cutoff_factor = kspace_cutoff_factor
        self.atomic_multipoles_max_l = atomic_multipoles_max_l
        self.atomic_multipoles_smearing_width = atomic_multipoles_smearing_width
        self.field_feature_max_l = field_feature_max_l
        self.field_feature_widths = widths
        self.num_recursion_steps = num_recursion_steps
        self.field_si = field_si
        self.include_electrostatic_self_interaction = (
            include_electrostatic_self_interaction
        )
        self.add_local_electron_energy = add_local_electron_energy
        self.quadrupole_feature_corrections = quadrupole_feature_corrections
        self.return_electrostatic_potentials = return_electrostatic_potentials
        self.field_norm_factor = field_norm_factor
        self.pbc_handling = pbc_handling
        self.reciprocal_max_index = reciprocal_max_index
        cutoff = (
            kspace_cutoff_factor
            * 2.25
            / min(atomic_multipoles_smearing_width, *widths)
            * (max(atomic_multipoles_max_l, field_feature_max_l) + 1) ** 0.3
        )
        self.kspace_cutoff = ConfigVar(jnp.asarray(cutoff, dtype=default_dtype()))
        norms = (
            tuple(field_feature_norms)
            if field_feature_norms is not None
            else (1.0,) * (len(widths) * (field_feature_max_l + 1))
        )
        if len(norms) != len(widths) * (field_feature_max_l + 1) or any(
            not np.isfinite(x) or x == 0 for x in norms
        ):
            raise ValueError(
                'field_feature_norms must supply a finite nonzero normalization for each degree and width.'
            )
        self._field_feature_norms = norms
        expanded = [
            norms[degree * len(widths) + i]
            for degree in range(field_feature_max_l + 1)
            for i in range(len(widths))
            for _ in range(2 * degree + 1)
        ]
        self.field_feature_norms = ConfigVar(
            jnp.asarray(expanded, dtype=default_dtype())
        )
        features = self._hidden_irreps
        sh = Irreps.spherical_harmonics(atomic_multipoles_max_l)
        self.charges_irreps = sh + sh
        self.field_irreps = (
            (Irreps.spherical_harmonics(field_feature_max_l) * len(widths))
            .sort()
            .irreps.simplify()
        )
        self.potential_irreps = self.field_irreps + self.field_irreps
        self.lr_source_maps = nnx.List(
            [
                EnvironmentDependentSpinSourceBlock(
                    features, atomic_multipoles_max_l, rngs=rngs
                )
                for _ in range(self.num_interactions)
            ]
        )
        self.fukui_source_map = NonLinearBiasReadoutBlock(
            features, self._mlp_irreps.simplify(), self.gate, Irreps('2x0e'), rngs=rngs
        )
        self.layer_feature_mixer = MultiLayerFeatureMixer(
            features, self.num_interactions, rngs=rngs
        )
        self._fixedpoint_update_config = dict(fixedpoint_update_config or {})
        self._field_readout_config = dict(field_readout_config or {})
        for config in (self._fixedpoint_update_config, self._field_readout_config):
            for key in ('type', 'potential_embedding_cls', 'nonlinearity_cls'):
                if callable(config.get(key)):
                    config[key] = config[key].__name__
        if (
            self._fixedpoint_update_config.get(
                'type', 'AgnosticEmbeddedOneBodyVariableUpdate'
            )
            != 'AgnosticEmbeddedOneBodyVariableUpdate'
        ):
            raise ValueError('Unsupported Polar field update class.')
        if (
            self._field_readout_config.get('type', 'OneBodyMLPFieldReadout')
            != 'OneBodyMLPFieldReadout'
        ):
            raise ValueError('Unsupported Polar field readout class.')
        if (
            self._fixedpoint_update_config.get(
                'potential_embedding_cls',
                'AgnosticChargeBiasedLinearPotentialEmbedding',
            )
            != 'AgnosticChargeBiasedLinearPotentialEmbedding'
        ):
            raise ValueError('Unsupported Polar potential embedding class.')
        self.field_dependent_charges_maps = nnx.List(
            [
                AgnosticEmbeddedOneBodyVariableUpdate(
                    Irreps(f'{self.num_elements}x0e'),
                    features,
                    self.potential_irreps,
                    self.charges_irreps,
                    field_norm_factor or 1.0,
                    rngs=rngs,
                )
                for _ in range(num_recursion_steps)
            ]
        )
        self.local_electron_energy = OneBodyMLPFieldReadout(
            features, self.potential_irreps, self.charges_irreps, rngs=rngs
        )

    def _edge_vectors_for_harmonics(self, vectors):
        return vectors[:, [1, 2, 0]]

    def __call__(self, data, *, compute_node_feats=True, **kwargs):
        if (
            kwargs.get('lammps_mliap')
            or kwargs.get('lammps_class') is not None
            or data.get('lammps_class') is not None
            or kwargs.get('comm') is not None
        ):
            raise ValueError('PolarMACE requires complete graphs for electrostatics.')
        output = ScaleShiftMACE._energy_fn(
            self, data, compute_node_feats=True, **kwargs
        )
        features = jnp.split(output['node_feats'], self.num_interactions, axis=-1)
        batch, n_graphs = data['batch'], len(data['ptr']) - 1
        mask = jnp.asarray(
            data.get('node_mask', jnp.ones(len(batch))), dtype=data['positions'].dtype
        )
        pool = lambda x: scatter_sum(x, batch, dim=0, dim_size=n_graphs)
        coefficients = (
            sum(block(x) for block, x in zip(self.lr_source_maps, features))
            * mask[:, None]
        )
        coefficients = coefficients.reshape(len(batch), 2, -1)
        charge = jnp.asarray(data.get('total_charge', jnp.zeros(n_graphs))).reshape(-1)
        spin = jnp.asarray(data.get('total_spin', jnp.ones(n_graphs))).reshape(-1) - 1
        targets = jnp.stack(((charge + spin) / 2, (charge - spin) / 2), axis=-1)

        def constrain(coeffs, fukui):
            fukui = getattr(fukui, 'array', fukui)
            fukui = fukui * mask[:, None]
            norm = pool(fukui)[batch]
            # Isolated atoms can have zero local features and hence zero Fukui
            # weights. A uniform fallback still enforces charge and spin totals.
            uniform = mask[:, None] / jnp.maximum(pool(mask)[batch, None], 1)
            fukui = jnp.where(norm != 0, fukui / jnp.where(norm == 0, 1, norm), uniform)
            delta = targets - pool(coeffs[:, :, 0])
            coeffs = coeffs.at[:, :, 0].add(fukui * delta[batch])
            return coeffs, fukui

        coefficients, fukui = constrain(
            coefficients, self.fukui_source_map(features[-1])
        )
        initial = coefficients
        mixed = self.layer_feature_mixer(features)
        pbc = data.get('pbc', jnp.zeros((n_graphs, 3), dtype=bool)).reshape(n_graphs, 3)
        graph_valid = pool(mask) > 0
        use_real = self.pbc_handling == 'realspace'
        if self.pbc_handling == 'auto':
            use_real = ~jnp.any(pbc & graph_valid[:, None])
        geometry = gto.reciprocal_geometry(
            data,
            self.kspace_cutoff[...],
            self.reciprocal_max_index,
            active=~jnp.asarray(use_real),
        )
        sigma, widths = self.atomic_multipoles_smearing_width, self.field_feature_widths
        external = jnp.asarray(data.get('external_field', jnp.zeros((n_graphs, 3))))
        if external.shape != (n_graphs, 3):
            raise ValueError('external_field must have shape [number of graphs, 3].')
        barycenter = pool(data['positions'] * mask[:, None]) / jnp.maximum(
            pool(mask)[:, None], 1
        )
        external_potential = jnp.sum(
            (data['positions'] - barycenter[batch]) * external[batch], axis=-1
        )
        external_features = 0.5 * gto.field_features(
            external_potential, external[batch], widths, self.field_feature_max_l
        )

        def potential(coeffs):
            def open_fn():
                return gto.real_features(
                    data, coeffs, sigma, widths, self.field_feature_max_l, self.field_si
                )

            def periodic_fn():
                value = gto.periodic_features(
                    coeffs,
                    sigma,
                    widths,
                    self.field_feature_max_l,
                    geometry,
                    self.field_si,
                )
                _, pot, field = gto.boundary_corrections(
                    dict(data, pbc=pbc), coeffs, geometry[-1], self.pbc_handling
                )
                return value + gto.field_features(
                    pot, field, widths, self.field_feature_max_l
                )

            value = jax.lax.cond(use_real, open_fn, periodic_fn)
            return (value + external_features) / self.field_feature_norms[...]

        potentials = jnp.zeros(
            (len(batch), self.potential_irreps.dim), dtype=coefficients.dtype
        )
        for block in self.field_dependent_charges_maps:
            potentials = (
                jnp.concatenate(
                    (potential(coefficients[:, 0]), potential(coefficients[:, 1])),
                    axis=-1,
                )
                * mask[:, None]
            )
            update = (
                block(
                    data['node_attrs'],
                    mixed,
                    potentials,
                    coefficients.reshape(len(batch), -1),
                )
                * mask[:, None]
            )
            coefficients += update[:, :-2].reshape(coefficients.shape)
            coefficients, fukui = constrain(coefficients, update[:, -2:])
        density = coefficients.sum(axis=1)
        spin_density = coefficients[:, 0] - coefficients[:, 1]
        electron_energy = jnp.zeros(n_graphs, dtype=density.dtype)
        if self.add_local_electron_energy:
            electron_energy = pool(
                self.local_electron_energy(
                    features[-1],
                    potentials,
                    initial.reshape(len(batch), -1),
                    coefficients.reshape(len(batch), -1),
                )
                * mask
            )

        def periodic_energy():
            value = gto.periodic_energy(
                data,
                density,
                sigma,
                geometry,
                self.include_electrostatic_self_interaction,
            )
            correction, _, _ = gto.boundary_corrections(
                dict(data, pbc=pbc), density, geometry[-1], self.pbc_handling
            )
            return value + correction

        electrostatic = jax.lax.cond(
            use_real,
            lambda: gto.real_energy(
                data, density, sigma, self.include_electrostatic_self_interaction
            ),
            periodic_energy,
        )
        dipole = pool(data['positions'] * density[:, :1])
        if self.atomic_multipoles_max_l:
            dipole += pool(density[:, [3, 1, 2]])
        output.update(
            energy=output['energy']
            + electron_energy
            + electrostatic
            + jnp.sum(external * dipole, axis=-1),
            density_coefficients=density,
            spin_density=spin_density,
            spin_charge_density=coefficients,
            charges=density[:, 0],
            spins=spin_density[:, 0],
            dipole=dipole,
            total_charge=pool(density[:, 0]),
            electron_energy=electron_energy,
            electrostatic_energy=electrostatic,
            fukui_functions=fukui,
            charges_history=jax.lax.stop_gradient(coefficients[..., None]),
            fermi_level=jnp.zeros(n_graphs),
            external_field=external,
            polarizability=None,
            dipole_from_energy=None,
            electrostatic_potentials=None,
        )
        if not compute_node_feats:
            output['node_feats'] = None
        return output
