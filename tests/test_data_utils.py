import ast
import io
import json

import ase
import ase.io
import h5py
import numpy as np
import pytest
from ase.calculators.singlepoint import SinglePointCalculator

from mace_jax import data, tools
from mace_jax.cli import mace_jax_preprocess


@pytest.mark.parametrize('energy_key', ['REF_energy', 'energy'])
def test_isolated_atom_energies(energy_key):
    xyz = io.StringIO(
        f'1\nconfig_type=IsolatedAtom {energy_key}=-150.0\nC 0 0 0\n'
        f'2\n{energy_key}=-290.0\nC 0 0 0\nC 1 0 0\n'
    )
    energies, configs = data.load_from_xyz(
        xyz, energy_key=energy_key, extract_atomic_energies=True
    )
    assert energies == {6: -150.0}
    assert len(configs) == 1
    assert len(configs[0].atomic_numbers) == 2


def test_isolated_atom_without_energy_is_not_assigned_a_zero_reference(caplog):
    energies, configs = data.load_from_xyz(
        io.StringIO('1\nconfig_type=IsolatedAtom\nC 0 0 0\n'),
        extract_atomic_energies=True,
    )
    assert energies == {}
    assert configs == []
    assert 'does not contain an energy' in caplog.text


def test_preprocess_uses_isolated_energies_for_statistics(tmp_path):
    train = tmp_path / 'train.xyz'
    train.write_text(
        '1\nconfig_type=IsolatedAtom REF_energy=-150.0\nC 0 0 0\n'
        + '2\nREF_energy=-290.0\nC 0 0 0\nC 1 0 0\n' * 4
    )
    output = tmp_path / 'processed'
    args = tools.build_preprocess_arg_parser().parse_args(
        [
            '--train_file',
            str(train),
            '--energy_key',
            'REF_energy',
            '--h5_prefix',
            str(output),
            '--compute_statistics',
            '--num_process',
            '1',
        ]
    )
    mace_jax_preprocess.run(args)
    statistics = json.loads((output / 'statistics.json').read_text())
    assert ast.literal_eval(statistics['atomic_energies']) == {6: -150.0}
    assert statistics['mean'] == pytest.approx(5.0)
    for path in output.glob('*/*.h5'):
        with h5py.File(path) as handle:
            for config in handle['config_batch_0'].values():
                assert len(config['atomic_numbers']) == 2


def test_ase_calculator_labels_are_preserved():
    atoms = ase.Atoms('H2', positions=[[0, 0, 0], [1, 0, 0]])
    atoms.calc = SinglePointCalculator(
        atoms, energy=-1.0, forces=np.ones((2, 3)), stress=np.arange(6.0)
    )
    xyz = io.StringIO()
    ase.io.write(xyz, atoms, format='extxyz')
    xyz.seek(0)
    _, configs = data.load_from_xyz(xyz)
    config = configs[0]
    assert config.energy == -1.0
    np.testing.assert_array_equal(config.forces, np.ones((2, 3)))
    assert config.stress.shape == (3, 3)
    assert config.energy_weight == config.forces_weight == config.stress_weight == 1.0


def test_config_weights_match_mace_metadata():
    atoms = ase.Atoms(
        'H',
        info={
            'config_type': 'test',
            'config_weight': 0.25,
            'energy': -1.0,
            'config_energy_weight': 2.0,
            'config_forces_weight': 5.0,
        },
    )
    config = data.config_from_atoms(atoms, config_type_weights={'test': 3.0})
    assert config.weight == 0.75
    assert config.energy_weight == 2.0
    assert config.forces_weight == 0.0  # No force labels, despite the metadata.


def test_legacy_property_weights_remain_supported():
    atoms = ase.Atoms('H', info={'energy': -1.0, 'energy_weight': 0.5})
    assert data.config_from_atoms(atoms).energy_weight == 0.5


def test_graph_accepts_python_scalar_energy_from_custom_xyz_key():
    _, configs = data.load_from_xyz(
        io.StringIO('2\nREF_energy=-1.5\nH 0 0 0\nH 1 0 0\n'),
        energy_key='REF_energy',
    )
    graph = data.graph_from_configuration(configs[0], cutoff=2.0)
    np.testing.assert_array_equal(graph.globals.energy, [-1.5])
