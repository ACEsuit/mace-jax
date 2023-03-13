from typing import Callable
import jax
from ase.calculators.calculator import Calculator, all_changes
from ase.stress import full_3x3_to_voigt_6_stress
from jax.config import config
import numpy as np

from mace_jax import data, tools
from mace_jax.data.utils import graph_from_configuration


class MACEJAXCalculator(Calculator):
    """MACE ASE Calculator"""

    implemented_properties = ["energy", "free_energy", "forces", "stress"]

    def __init__(
        self,
        model: Callable,
        params: dict,
        r_max: float,
        energy_units_to_eV: float = 1.0,
        length_units_to_A: float = 1.0,
        default_dtype="float64",
        **kwargs
    ):
        Calculator.__init__(self, **kwargs)
        self.results = {}
        self.model = model
        self.params = params
        self.predictor = jax.jit(
            lambda w, g: tools.predict_energy_forces_stress(
                lambda *x: self.model(w, *x), g
            )
        )

        self.r_max = r_max
        self.energy_units_to_eV = energy_units_to_eV
        self.length_units_to_A = length_units_to_A
        if default_dtype == "float64":
            config.update("jax_enable_x64", True)

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
        graph = graph_from_configuration(config, cutoff=self.r_max)

        # predict + extract data
        out = self.predictor(self.params, graph)
        energy = np.array(out["energy"])
        forces = np.array(out["forces"])

        # store results
        E = energy * self.energy_units_to_eV
        self.results = {
            "energy": E,
            "free_energy": E,
            # force has units eng / len:
            "forces": forces * (self.energy_units_to_eV / self.length_units_to_A),
        }

        # even though compute_stress is True, stress can be none if pbc is False
        # not sure if correct ASE thing is to have no dict key, or dict key with value None
        if out["stress"] is not None:
            stress = np.array(out["stress"])
            # stress has units eng / len^3:
            self.results["stress"] = (
                stress * (self.energy_units_to_eV / self.length_units_to_A ** 3)
            )[0]
            self.results["stress"] = full_3x3_to_voigt_6_stress(self.results["stress"])
