from collections.abc import Callable

import jax
import jraph
import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.stress import full_3x3_to_voigt_6_stress
from flax import traverse_util
from jax import config as jax_config

from mace_jax import data
from mace_jax.data.features import extract_features
from mace_jax.data.utils import (
    AtomicNumberTable,
    atomic_numbers_to_indices,
    graph_from_configuration,
)


def stop_grad(variables):
    flat_vars = traverse_util.flatten_dict(variables)
    new_vars = {k: jax.lax.stop_gradient(v) for k, v in flat_vars.items()}
    return traverse_util.unflatten_dict(new_vars)


class MACEJAXCalculator(Calculator):
    """MACE ASE Calculator"""

    implemented_properties = ['energy', 'free_energy', 'forces', 'stress']

    def __init__(
        self,
        model: Callable,
        params: dict,
        r_max: float,
        energy_units_to_eV: float = 1.0,
        length_units_to_A: float = 1.0,
        default_dtype='float64',
        atomic_numbers: list[int] | None = None,
        embedding_specs: dict | None = None,
        **kwargs,
    ):
        Calculator.__init__(self, **kwargs)
        self.results = {}
        self.model = model
        self.embedding_specs = (
            embedding_specs
            if embedding_specs is not None
            else getattr(model, 'model_config', {}).get('embedding_specs')
        )
        self._embedding_feature_values = None
        self._charge_inputs = None
        self.model_class = getattr(model, 'model_config', {}).get(
            'torch_model_class', 'MACE'
        )
        self.has_energy = self.model_class not in (
            'AtomicDipolesMACE',
            'AtomicDielectricMACE',
        )
        self.implemented_properties = (
            list(type(self).implemented_properties) if self.has_energy else []
        )
        if self.has_energy:
            self.implemented_properties += ['stresses', 'hessian', 'atomic_virials']
        if self.model_class in (
            'AtomicDipolesMACE',
            'EnergyDipolesMACE',
            'AtomicDielectricMACE',
        ):
            self.implemented_properties += ['dipole', 'atomic_dipoles', 'dmu_dr']
        if self.model_class == 'AtomicDielectricMACE':
            self.implemented_properties += [
                'charges',
                'polarizability',
                'polarizability_sh',
                'dalpha_dr',
            ]
        self.params = stop_grad(params)

        def predict(w, g, atomic=False, hessian=False, dielectric=False):
            flags = {}
            if atomic:
                flags['compute_atomic_stresses'] = True
            if hessian:
                flags['compute_hessian'] = True
            if dielectric:
                flags['compute_dielectric_derivatives'] = True
            return self.model(
                w,
                g,
                compute_force=self.has_energy,
                compute_stress=self.has_energy,
                **flags,
            )

        self.predictor = jax.jit(
            predict, static_argnames=('atomic', 'hessian', 'dielectric')
        )

        self.r_max = r_max
        self.energy_units_to_eV = energy_units_to_eV
        self.length_units_to_A = length_units_to_A
        self.z_table = None
        self.min_n_edge = 0

        if atomic_numbers is not None:
            self.z_table = lambda x: atomic_numbers_to_indices(
                x, AtomicNumberTable([int(z) for z in atomic_numbers])
            )
        if default_dtype == 'float64':
            jax_config.update('jax_enable_x64', True)

    def check_state(self, atoms, tol=1e-15):
        changes = super().check_state(atoms, tol=tol)
        if self.atoms is not None and self._embedding_feature_values is not None:
            previous = self._embedding_feature_values
            current = extract_features(atoms, self.embedding_specs)
            if any(
                not np.array_equal(old[name], new[name])
                for old, new in zip(previous, current)
                for name in (old or {})
            ):
                changes.append('embedding_features')
        if self._charge_inputs is not None:
            current = self._get_charge_inputs(atoms)
            if any(
                not np.array_equal(old, new)
                for old, new in zip(self._charge_inputs, current)
            ):
                changes.append('charge_inputs')
        return changes

    @staticmethod
    def _get_charge_inputs(atoms):
        charges = np.array(
            atoms.arrays.get(
                'charges', atoms.info.get('charges', atoms.get_initial_charges())
            ),
            copy=True,
        )
        return charges, np.array(
            atoms.info.get('total_charge', charges.sum()), copy=True
        )

    def get_hessian(self, atoms=None):
        return self.get_property('hessian', atoms)

    def get_atomic_virials(self, atoms=None):
        return self.get_property('atomic_virials', atoms)

    # pylint: disable=dangerous-default-value
    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        """
        Calculate properties.
        :param atoms: ase.Atoms object
        :param properties: [str], properties to be computed, used by ASE internally
        :param system_changes: [str], system changes since last calculation, used by ASE internally
        :return:
        """
        # call to base-class to set atoms attribute
        Calculator.calculate(self, atoms)

        # prepare data
        # ASE's copy has no calculator attached. Reading labels from the input
        # atoms would otherwise call back into this calculator recursively.
        self._charge_inputs = self._get_charge_inputs(self.atoms)
        config = data.config_from_atoms(
            self.atoms.copy(), embedding_specs=self.embedding_specs
        )
        # ASE copies info dictionaries shallowly; cache independent feature arrays
        # so in-place edits to graph inputs also invalidate the calculated results.
        self._embedding_feature_values = tuple(
            {
                name: np.array(value, copy=True)
                for name, value in (features or {}).items()
            }
            for features in (config.node_features, config.graph_features)
        )
        if self.z_table is not None:
            config.atomic_numbers = self.z_table(config.atomic_numbers)

        graph_config = graph_from_configuration(config, cutoff=self.r_max)
        n_edge = int(np.asarray(graph_config.n_edge).item())
        if self.min_n_edge == 0 or self.min_n_edge <= n_edge:
            self.min_n_edge = n_edge + max(int(n_edge // 10), 10)
        # pad graph with dummy atoms
        n_node = int(np.asarray(graph_config.n_node).item()) + 1
        graph = jraph.pad_with_graphs(
            graph_config,
            n_node=n_node,
            n_edge=self.min_n_edge,
            n_graph=2,
        )
        properties = properties or self.implemented_properties[:1]
        out = self.predictor(
            self.params,
            graph,
            atomic=any(key in properties for key in ('stresses', 'atomic_virials')),
            hessian='hessian' in properties,
            dielectric=any(key in properties for key in ('dmu_dr', 'dalpha_dr')),
        )
        out = jax.device_get(out)
        n_atoms = len(self.atoms)
        self.results = {}
        if out.get('energy') is not None:
            self.results['energy'] = float(out['energy'][0]) * self.energy_units_to_eV
            self.results['free_energy'] = self.results['energy']
        if out.get('forces') is not None:
            self.results['forces'] = (
                np.asarray(out['forces'])[:n_atoms]
                * self.energy_units_to_eV
                / self.length_units_to_A
            )
        if out.get('stress') is not None:
            stress = (
                np.asarray(out['stress'])[0]
                * self.energy_units_to_eV
                / self.length_units_to_A**3
            )
            self.results['stress'] = full_3x3_to_voigt_6_stress(stress)
        if out.get('atomic_stresses') is not None:
            stress = (
                np.asarray(out['atomic_stresses'])[:n_atoms]
                * self.energy_units_to_eV
                / self.length_units_to_A**3
            )
            self.results['stresses'] = full_3x3_to_voigt_6_stress(stress)
        if out.get('atomic_virials') is not None:
            self.results['atomic_virials'] = (
                np.asarray(out['atomic_virials'])[:n_atoms] * self.energy_units_to_eV
            )
        if out.get('hessian') is not None:
            self.results['hessian'] = (
                np.asarray(out['hessian'])[: 3 * n_atoms, :n_atoms]
                * self.energy_units_to_eV
                / self.length_units_to_A**2
            )
        # ASE dipoles use e Angstrom, including for legacy Debye readouts.
        dipole_factor = self.length_units_to_A
        if self.model_class in ('AtomicDipolesMACE', 'EnergyDipolesMACE'):
            dipole_factor /= 299792458.0 * 1.602176634e-19 / 1e-11
        for key in ('dipole', 'polarizability', 'polarizability_sh'):
            if out.get(key) is not None:
                factor = dipole_factor if key == 'dipole' else self.length_units_to_A**3
                self.results[key] = np.asarray(out[key])[0] * factor
        for key in ('charges', 'atomic_dipoles'):
            if out.get(key) is not None:
                self.results[key] = np.asarray(out[key])[:n_atoms] * (
                    dipole_factor if key == 'atomic_dipoles' else 1.0
                )
        for key in ('dmu_dr', 'dalpha_dr'):
            if out.get(key) is not None:
                factor = (
                    dipole_factor / self.length_units_to_A
                    if key == 'dmu_dr'
                    else self.length_units_to_A**2
                )
                self.results[key] = np.asarray(out[key])[:, :n_atoms] * factor
