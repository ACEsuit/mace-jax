import jax
import jax.numpy as jnp
import numpy as np
import pytest
from e3nn_jax import Irreps

from mace_jax import modules
from mace_jax.calculators.lammps_mliap_mace import (
    LAMMPS_MLIAP_MACE,
    create_lammps_mliap_calculator,
)


class DummyLAMMPSData:
    def __init__(
        self,
        elems,
        rij,
        pair_i,
        pair_j,
        *,
        nlocal=None,
        ntotal=None,
        positions=None,
        unit_shifts=None,
        shifts=None,
        cell=None,
    ):
        self.nlocal = len(elems) if nlocal is None else int(nlocal)
        self.ntotal = len(elems) if ntotal is None else int(ntotal)
        self.npairs = len(pair_i)
        self.elems = np.asarray(elems, dtype=np.int64)
        self.rij = np.asarray(rij, dtype=float)
        self.pair_i = np.asarray(pair_i, dtype=np.int32)
        self.pair_j = np.asarray(pair_j, dtype=np.int32)
        self.eatoms = np.zeros(self.ntotal, dtype=float)
        self.energy = 0.0
        self.updated_pair_forces = None
        self.exchange_calls = 0
        if positions is not None:
            self.positions = np.asarray(positions, dtype=float)
        if unit_shifts is not None:
            self.unit_shifts = np.asarray(unit_shifts, dtype=float)
        if shifts is not None:
            self.shifts = np.asarray(shifts, dtype=float)
        if cell is not None:
            self.cell = np.asarray(cell, dtype=float)

    def update_pair_forces_gpu(self, values):
        self.updated_pair_forces = np.asarray(values, dtype=float)

    def forward_exchange(self, src, dst, vec_len):
        self.exchange_calls += 1
        np.copyto(dst, src)


def _build_test_model(num_interactions: int = 1):
    return modules.ScaleShiftMACE(
        r_max=5.0,
        num_bessel=2,
        num_polynomial_cutoff=2,
        max_ell=1,
        interaction_cls=modules.interaction_classes[
            'RealAgnosticResidualInteractionBlock'
        ],
        interaction_cls_first=modules.interaction_classes[
            'RealAgnosticResidualInteractionBlock'
        ],
        num_interactions=num_interactions,
        num_elements=1,
        hidden_irreps=Irreps('1x0e'),
        MLP_irreps=Irreps('1x0e'),
        atomic_energies=np.zeros((1,), dtype=np.float64),
        avg_num_neighbors=1.0,
        atomic_numbers=(1,),
        correlation=1,
        gate=None,
        pair_repulsion=False,
        distance_transform='None',
        atomic_inter_scale=np.asarray(1.0),
        atomic_inter_shift=np.asarray(0.0),
    )


def _build_lammps_batch(
    vectors,
    pair_i,
    pair_j,
    natoms,
    *,
    n_ghosts: int = 0,
    positions=None,
    unit_shifts=None,
    shifts=None,
    cell=None,
):
    dtype = vectors.dtype
    if positions is None:
        positions_arr = jnp.zeros((natoms, 3), dtype=dtype)
    else:
        positions_arr = jnp.asarray(positions, dtype=dtype)
    if unit_shifts is None:
        unit_shifts_arr = jnp.zeros_like(vectors)
    else:
        unit_shifts_arr = jnp.asarray(unit_shifts, dtype=dtype)
    if shifts is not None:
        shifts_arr = jnp.asarray(shifts, dtype=dtype)
    elif unit_shifts is not None and cell is not None:
        cell_tensor = jnp.asarray(cell, dtype=dtype)
        if cell_tensor.ndim == 2:
            shifts_arr = unit_shifts_arr @ cell_tensor
        else:
            shifts_arr = unit_shifts_arr @ cell_tensor[0]
    else:
        shifts_arr = jnp.zeros_like(vectors)
    if cell is None:
        cell_arr = jnp.zeros((2, 3, 3), dtype=dtype)
    else:
        cell_tensor = jnp.asarray(cell, dtype=dtype)
        if cell_tensor.ndim == 2:
            cell_arr = jnp.stack((cell_tensor, cell_tensor), axis=0)
        else:
            if cell_tensor.shape[0] == 1:
                cell_arr = jnp.repeat(cell_tensor, 2, axis=0)
            elif cell_tensor.shape[0] >= 2:
                cell_arr = cell_tensor[:2]
            else:
                cell_arr = jnp.zeros((2, 3, 3), dtype=dtype)
    return {
        'vectors': vectors,
        'node_attrs': jax.nn.one_hot(
            jnp.zeros(natoms, dtype=jnp.int32), num_classes=1, dtype=dtype
        ),
        'edge_index': jnp.stack((pair_j, pair_i), axis=0),
        'batch': jnp.zeros(natoms, dtype=jnp.int32),
        'natoms': (natoms, n_ghosts),
        'ptr': jnp.asarray([0, natoms, natoms + n_ghosts], dtype=jnp.int32),
        'positions': positions_arr,
        'unit_shifts': unit_shifts_arr,
        'shifts': shifts_arr,
        'cell': cell_arr,
        'lammps_class': None,
    }


def test_lammps_mliap_wrapper_matches_direct_model():
    model = _build_test_model()

    pair_i = jnp.asarray([0, 1], dtype=jnp.int32)
    pair_j = jnp.asarray([1, 0], dtype=jnp.int32)
    vectors = jnp.asarray([[0.8, 0.0, 0.0], [-0.8, 0.0, 0.0]], dtype=jnp.float64)

    lammps_batch = _build_lammps_batch(vectors, pair_i, pair_j, natoms=2)

    variables = model.init(
        jax.random.PRNGKey(0),
        lammps_batch,
        lammps_mliap=True,
    )

    direct_out = model.apply(
        variables,
        lammps_batch,
        lammps_mliap=True,
    )

    def energy_with_vectors(edge_vectors):
        batch = dict(lammps_batch)
        batch['vectors'] = edge_vectors
        out = model.apply(
            variables,
            batch,
            lammps_mliap=True,
        )
        return jnp.sum(out['energy'])

    grad_vectors = jax.grad(energy_with_vectors)(lammps_batch['vectors'])
    expected_pair_forces = -np.asarray(grad_vectors)

    calculator = LAMMPS_MLIAP_MACE(model, variables)

    dummy_data = DummyLAMMPSData(
        elems=[0, 0],
        rij=vectors,
        pair_i=pair_i,
        pair_j=pair_j,
    )

    calculator.compute_forces(dummy_data)

    node_energy = np.asarray(direct_out['node_energy'])
    total_energy = float(np.asarray(direct_out['energy']).sum())

    np.testing.assert_allclose(
        dummy_data.eatoms[: dummy_data.nlocal],
        node_energy[: dummy_data.nlocal],
        rtol=1e-9,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        dummy_data.updated_pair_forces,
        expected_pair_forces,
        rtol=1e-9,
        atol=1e-9,
    )
    assert dummy_data.energy == pytest.approx(total_energy)


def test_create_lammps_factory_returns_working_wrapper():
    model = _build_test_model()

    pair_i = jnp.asarray([0, 1], dtype=jnp.int32)
    pair_j = jnp.asarray([1, 0], dtype=jnp.int32)
    vectors = jnp.asarray([[0.8, 0.0, 0.0], [-0.8, 0.0, 0.0]], dtype=jnp.float64)

    lammps_batch = _build_lammps_batch(vectors, pair_i, pair_j, natoms=2)

    variables = model.init(
        jax.random.PRNGKey(0),
        lammps_batch,
        lammps_mliap=True,
    )

    direct_out = model.apply(
        variables,
        lammps_batch,
        lammps_mliap=True,
    )

    wrapper = create_lammps_mliap_calculator(model, variables)

    dummy_data = DummyLAMMPSData(
        elems=[0, 0],
        rij=vectors,
        pair_i=pair_i,
        pair_j=pair_j,
    )

    wrapper.compute_forces(dummy_data)

    node_energy = np.asarray(direct_out['node_energy'])
    total_energy = float(np.asarray(direct_out['energy']).sum())

    np.testing.assert_allclose(
        dummy_data.eatoms[: dummy_data.nlocal],
        node_energy[: dummy_data.nlocal],
        rtol=1e-9,
        atol=1e-9,
    )
    assert dummy_data.energy == pytest.approx(total_energy)


def test_lammps_mliap_wrapper_handles_ghost_atoms():
    model = _build_test_model()

    pair_i = jnp.asarray([0, 1], dtype=jnp.int32)
    pair_j = jnp.asarray([1, 0], dtype=jnp.int32)
    vectors = jnp.asarray([[0.8, 0.0, 0.0], [-0.8, 0.0, 0.0]], dtype=jnp.float64)

    n_real = 2
    n_ghosts = 1
    lammps_batch = _build_lammps_batch(
        vectors, pair_i, pair_j, natoms=n_real, n_ghosts=n_ghosts
    )

    variables = model.init(
        jax.random.PRNGKey(0),
        lammps_batch,
        lammps_mliap=True,
    )

    class _ExchangeStub:
        def forward_exchange(self, src, dst, vec_len):
            np.copyto(dst, src)

    exchange_stub = _ExchangeStub()

    direct_out = model.apply(
        variables,
        lammps_batch,
        lammps_mliap=True,
        lammps_class=exchange_stub,
    )

    def energy_with_vectors(edge_vectors):
        batch = dict(lammps_batch)
        batch['vectors'] = edge_vectors
        out = model.apply(
            variables,
            batch,
            lammps_mliap=True,
            lammps_class=exchange_stub,
        )
        return jnp.sum(out['energy'])

    grad_vectors = jax.grad(energy_with_vectors)(lammps_batch['vectors'])
    expected_pair_forces = -np.asarray(grad_vectors)

    calculator = LAMMPS_MLIAP_MACE(model, variables)

    dummy_data = DummyLAMMPSData(
        elems=[0, 0],
        rij=vectors,
        pair_i=pair_i,
        pair_j=pair_j,
        nlocal=n_real,
        ntotal=n_real + n_ghosts,
    )

    calculator.compute_forces(dummy_data)

    node_energy = np.asarray(direct_out['node_energy'])
    total_energy = float(np.asarray(direct_out['energy']).sum())

    np.testing.assert_allclose(
        dummy_data.eatoms[: dummy_data.nlocal],
        node_energy[: dummy_data.nlocal],
        rtol=1e-9,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        dummy_data.updated_pair_forces,
        expected_pair_forces,
        rtol=1e-9,
        atol=1e-9,
    )
    assert dummy_data.energy == pytest.approx(total_energy)
    assert dummy_data.ntotal - dummy_data.nlocal == n_ghosts


def test_prepare_batch_uses_lammps_geometry_metadata():
    model = _build_test_model()

    pair_i = jnp.asarray([0, 1], dtype=jnp.int32)
    pair_j = jnp.asarray([1, 0], dtype=jnp.int32)
    vectors = jnp.asarray([[0.8, 0.0, 0.0], [-0.8, 0.0, 0.0]], dtype=jnp.float64)

    variables = model.init(
        jax.random.PRNGKey(0),
        _build_lammps_batch(vectors, pair_i, pair_j, natoms=2),
        lammps_mliap=True,
    )
    calculator = LAMMPS_MLIAP_MACE(model, variables)

    cell = np.array(
        [
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 2.0],
        ],
        dtype=float,
    )
    positions = np.array(
        [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9],
        ],
        dtype=float,
    )
    unit_shifts = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    shifts = unit_shifts @ cell

    dummy_data = DummyLAMMPSData(
        elems=[0, 0, 0],
        rij=vectors,
        pair_i=pair_i,
        pair_j=pair_j,
        nlocal=2,
        ntotal=3,
        positions=positions,
        unit_shifts=unit_shifts,
        shifts=shifts,
        cell=cell,
    )

    batch = calculator._prepare_batch(
        dummy_data,
        dummy_data.nlocal,
        dummy_data.ntotal - dummy_data.nlocal,
        jnp.asarray(dummy_data.elems, dtype=jnp.int64),
    )

    np.testing.assert_allclose(
        np.asarray(batch['positions']),
        positions[: dummy_data.nlocal],
    )
    np.testing.assert_allclose(
        np.asarray(batch['unit_shifts']),
        unit_shifts,
    )
    np.testing.assert_allclose(
        np.asarray(batch['shifts']),
        shifts,
    )
    expected_cell = np.stack((cell, cell), axis=0)
    np.testing.assert_allclose(
        np.asarray(batch['cell']),
        expected_cell,
    )
    assert batch['vectors'].device.platform == calculator.device.platform


def test_lammps_mliap_wrapper_periodic_image_example():
    model = _build_test_model()

    pair_i = jnp.asarray([0, 1], dtype=jnp.int32)
    pair_j = jnp.asarray([1, 0], dtype=jnp.int32)

    cell = np.diag([2.0, 2.0, 2.0])
    positions = np.array(
        [
            [0.1, 0.2, 0.3],
            [1.9, 0.2, 0.3],
        ],
        dtype=float,
    )
    unit_shifts = np.array(
        [
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )
    shifts = unit_shifts @ cell
    vectors = jnp.asarray(
        positions[pair_j] - positions[pair_i] + shifts,
        dtype=jnp.float64,
    )

    lammps_batch = _build_lammps_batch(
        vectors,
        pair_i,
        pair_j,
        natoms=2,
        positions=positions,
        unit_shifts=unit_shifts,
        shifts=shifts,
        cell=cell,
    )

    variables = model.init(
        jax.random.PRNGKey(0),
        lammps_batch,
        lammps_mliap=True,
    )

    direct_out = model.apply(
        variables,
        lammps_batch,
        lammps_mliap=True,
    )

    def energy_with_vectors(edge_vectors):
        batch = dict(lammps_batch)
        batch['vectors'] = edge_vectors
        out = model.apply(
            variables,
            batch,
            lammps_mliap=True,
        )
        return jnp.sum(out['energy'])

    grad_vectors = jax.grad(energy_with_vectors)(lammps_batch['vectors'])
    expected_pair_forces = -np.asarray(grad_vectors)

    calculator = LAMMPS_MLIAP_MACE(model, variables)

    dummy_data = DummyLAMMPSData(
        elems=[0, 0],
        rij=vectors,
        pair_i=pair_i,
        pair_j=pair_j,
        positions=positions,
        unit_shifts=unit_shifts,
        shifts=shifts,
        cell=cell,
    )

    calculator.compute_forces(dummy_data)

    node_energy = np.asarray(direct_out['node_energy'])
    total_energy = float(np.asarray(direct_out['energy']).sum())

    np.testing.assert_allclose(
        dummy_data.eatoms[: dummy_data.nlocal],
        node_energy[: dummy_data.nlocal],
        rtol=1e-9,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        dummy_data.updated_pair_forces,
        expected_pair_forces,
        rtol=1e-9,
        atol=1e-9,
    )
    assert dummy_data.energy == pytest.approx(total_energy)


def test_lammps_calculator_respects_force_cpu(monkeypatch):
    monkeypatch.setenv('MACE_FORCE_CPU', 'true')
    model = _build_test_model()

    pair_i = jnp.asarray([0], dtype=jnp.int32)
    pair_j = jnp.asarray([0], dtype=jnp.int32)
    vectors = jnp.asarray([[0.5, 0.0, 0.0]], dtype=jnp.float64)

    lammps_batch = _build_lammps_batch(vectors, pair_i, pair_j, natoms=1)

    variables = model.init(
        jax.random.PRNGKey(0),
        lammps_batch,
        lammps_mliap=True,
    )

    calculator = LAMMPS_MLIAP_MACE(model, variables)
    assert calculator.device.platform == 'cpu'
