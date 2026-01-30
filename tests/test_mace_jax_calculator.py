import jax.numpy as jnp
import numpy as np
import pytest
from ase import Atoms

from mace_jax.calculators import MACEJAXCalculator


def _toy_model(params, graph, *, compute_force=True, compute_stress=True):
    positions = graph.nodes.positions
    energy = jnp.asarray([jnp.sum(positions) * params['scale']])
    forces = None
    if compute_force:
        forces = jnp.ones_like(positions) * params['force']
    stress = None
    if compute_stress:
        stress = jnp.zeros((1, 3, 3), dtype=positions.dtype)
    return {'energy': energy, 'forces': forces, 'stress': stress}


def test_mace_jax_calculator_runs():
    atoms = Atoms(
        numbers=[1, 6],
        positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        cell=[3.0, 3.0, 3.0],
        pbc=[True, True, True],
    )

    params = {'scale': jnp.asarray(1.0), 'force': jnp.asarray(0.0)}
    calc = MACEJAXCalculator(
        model=_toy_model,
        params=params,
        r_max=5.0,
        atomic_numbers=[1, 6],
        default_dtype='float32',
    )

    calc.calculate(atoms)

    assert 'energy' in calc.results
    assert 'free_energy' in calc.results
    assert 'forces' in calc.results
    assert 'stress' in calc.results

    assert np.asarray(calc.results['forces']).shape == (2, 3)
    assert np.asarray(calc.results['stress']).shape == (6,)
    assert pytest.approx(calc.results['energy']) == calc.results['free_energy']
