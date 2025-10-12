from typing import Callable, List, Optional

import jax
import jraph
import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.stress import full_3x3_to_voigt_6_stress
from flax import traverse_util
from jax import config as jax_config

from mace_jax import data
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
        atomic_numbers: Optional[List[int]] = None,
        **kwargs,
    ):
        Calculator.__init__(self, **kwargs)
        self.results = {}
        self.model = model
        self.params = stop_grad(params)
        self.predictor = jax.jit(
            lambda w, g: self.model(w, g, compute_force=True, compute_stress=True)
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
        config = data.config_from_atoms(atoms)
        if self.z_table is not None:
            config.atomic_numbers = self.z_table(config.atomic_numbers)

        graph_config = graph_from_configuration(config, cutoff=self.r_max)
        if self.min_n_edge == 0 or self.min_n_edge <= graph_config.n_edge:
            self.min_n_edge = graph_config.n_edge + max(
                int(graph_config.n_edge // 10), 10
            )
        # pad graph with dummy atoms
        graph = jraph.pad_with_graphs(
            graph_config,
            n_node=graph_config.n_node + 1,
            n_edge=self.min_n_edge,
            n_graph=2,
        )
        # predict + extract data
        out = self.predictor(self.params, graph)
        energy = np.array(jax.lax.stop_gradient(out['energy']))[0]
        forces = np.array(jax.lax.stop_gradient(out['forces']))[:-1, :]

        # store results
        E = energy * self.energy_units_to_eV
        self.results = {
            'energy': E,
            'free_energy': E,
            # force has units eng / len:
            'forces': forces * (self.energy_units_to_eV / self.length_units_to_A),
        }

        # even though compute_stress is True, stress can be none if pbc is False
        # not sure if correct ASE thing is to have no dict key, or dict key with value None
        if out['stress'] is not None:
            stress = np.array(jax.lax.stop_gradient(out['stress']))[:3, :3]
            # stress has units eng / len^3:
            self.results['stress'] = (
                stress * (self.energy_units_to_eV / self.length_units_to_A**3)
            )[0]
            self.results['stress'] = full_3x3_to_voigt_6_stress(self.results['stress'])
