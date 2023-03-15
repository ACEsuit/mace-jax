from typing import Callable, List, Optional
import jax
from ase.calculators.calculator import Calculator, all_changes
from ase.stress import full_3x3_to_voigt_6_stress
from jax.config import config
import numpy as np
import jraph
from jax import jit, vmap

from mace_jax import data, tools
from mace_jax.data.utils import (
    AtomicNumberTable,
    atomic_numbers_to_indices,
)
import time
from jax_md import partition, space, simulate
from jax.lax import fori_loop
from jax_md import quantity
import jax.numpy as jnp
from functools import partial
from flax import traverse_util


K_B = 8.617e-5


def stop_grad(variables):
    flat_vars = traverse_util.flatten_dict(variables)
    new_vars = {k: jax.lax.stop_gradient(v) for k, v in flat_vars.items()}
    return traverse_util.unflatten_dict(new_vars)


class MACEJAXmd:
    def __init__(
        self,
        model: Callable,
        params: dict,
        r_max: float,
        energy_units_to_eV: float = 1.0,
        length_units_to_A: float = 1.0,
        default_dtype="float64",
        atomic_numbers: Optional[List[int]] = None,
        **kwargs
    ):
        self.results = {}
        self.model = model
        self.params = stop_grad(params)
        self.predictor = jax.jit(lambda w, *x: self.model(w, *x))

        self.r_max = r_max
        self.energy_units_to_eV = energy_units_to_eV
        self.length_units_to_A = length_units_to_A
        self.z_table = None

        if atomic_numbers is not None:
            self.z_table = lambda x: atomic_numbers_to_indices(
                x, AtomicNumberTable([int(z) for z in atomic_numbers])
            )
        if default_dtype == "float64":
            config.update("jax_enable_x64", True)

    def setup_NPT(
        self, atoms, inner_steps=100, P_start=0.0, kT=1.0, dt=0.001, mass=1.0, skin=1.0,
    ):
        config = data.config_from_atoms(atoms)
        cell = config.cell
        position_initial = atoms.get_scaled_positions(False)
        species = config.atomic_numbers
        displacement, shift = space.periodic_general(cell, fractional_coordinates=True)
        self.neighbor_fn = partition.neighbor_list(
            displacement,
            cell,
            self.r_max,
            dr_threshold=skin,
            format=partition.Sparse,
            fractional_coordinates=True,
        )
        self.shift = shift
        self.displacement = displacement
        neighbor = self.neighbor_fn.allocate(position_initial)
        species = jax.numpy.array(self.z_table(species))

        @jit
        def energy_fn(position, neighbor, **kwargs):
            senders, receivers = neighbor.idx
            d = vmap(partial(self.displacement))
            vectors = d(position[senders], position[receivers])

            mask = partition.neighbor_list_mask(neighbor)
            vectors = jnp.where(mask[:, None], vectors, 0)
            t0 = time.process_time()
            node_energies = self.predictor(
                self.params, vectors, species, senders, receivers
            )  # [n_nodes, ]
            t = time.process_time() - t0
            return jnp.sum(node_energies)

        init_fn, step_fn = simulate.npt_nose_hoover(
            jax.jit(energy_fn), self.shift, dt, P_start, kT
        )
        step_fn = jax.jit(step_fn)

        @jit
        def take_steps(state, nbrs, pressure):
            def sim_fn(i, state_nbrs):
                state, nbrs = state_nbrs
                state = step_fn(state, pressure=pressure, neighbor=nbrs)
                nbrs = nbrs.update(state.position, box=state.box)
                return state, nbrs

            return fori_loop(0, inner_steps, jax.jit(sim_fn), (state, nbrs))

        @jit
        def compute_diagnostics(state, nbrs):
            temperature = quantity.temperature(momentum=state.momentum, mass=mass) / K_B
            kinetic_energy = quantity.kinetic_energy(momentum=state.momentum, mass=mass)
            pressure = quantity.pressure(
                jax.jit(energy_fn),
                state.position,
                state.box,
                kinetic_energy,
                neighbor=nbrs,
            )
            position = space.transform(state.box, state.position)
            return temperature, pressure, position

        return init_fn, take_steps, compute_diagnostics, neighbor, step_fn, energy_fn
