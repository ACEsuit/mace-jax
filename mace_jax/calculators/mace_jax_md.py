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
from jax_md import partition, space, simulate
from jax.lax import fori_loop
from jax_md import quantity
import jax.numpy as jnp
from functools import partial


K_B = 8.617e-5


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
        self.params = params
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
        self,
        position_initial,
        species,
        cell,
        inner_steps=1000,
        P_start=1.0,
        kT=1.0,
        dt=0.001,
        mass=1.0,
    ):
        neighbor = self.neighbor_fn.allocate(position_initial)
        species = self.z_table(species)
        displacement, shift = space.periodic_general(cell)
        self.shift = shift
        self.displacement = displacement
        self.neighbor_fn = partition.neighbor_list(
            displacement, cell, self.r_max, format=partition.Sparse
        )

        @jit
        def energy_fn(position, neighbor, **kwargs):
            senders, receivers = neighbor.update(position).idx

            d = vmap(partial(self.displacement))
            vectors = d(position[senders], position[receivers.idx[1, :]])

            mask = partition.neighbor_list_mask(neighbor)
            vectors = jnp.where(mask[:, None], vectors, 0)
            node_energies = self.predictor(
                vectors, species, senders, receivers
            )  # [n_nodes, ]
            assert node_energies.shape == (
                len(position),
            ), "model output needs to be an array of shape (n_nodes, )"
            return node_energies

        init_fn, step_fn = simulate.npt_nose_hoover(
            energy_fn, self.shift, dt, P_start, kT
        )

        @jit
        def take_steps(state, nbrs, pressure):
            def sim_fn(i, state_nbrs):
                state, nbrs = state_nbrs
                state = step_fn(state, pressure=pressure, neighbor=nbrs)
                nbrs = nbrs.update(state.position, box=state.box)
                return state, nbrs

            return fori_loop(0, inner_steps, sim_fn, (state, nbrs))

        @jit
        def compute_diagnostics(state, nbrs):
            temperature = quantity.temperature(momentum=state.momentum, mass=mass) / K_B
            kinetic_energy = quantity.kinetic_energy(momentum=state.momentum, mass=mass)
            pressure = quantity.pressure(
                energy_fn, state.position, state.box, kinetic_energy, neighbor=nbrs
            )
            position = space.transform(state.box, state.position)
            return temperature, pressure, position

        return init_fn, take_steps, compute_diagnostics, neighbor
