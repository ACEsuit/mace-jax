"""Geometry regressions against an independent periodic-image enumeration."""

import itertools

import numpy as np
import pytest

from mace_jax.data.neighborhood import get_neighborhood


def _reference_edges(positions, cell, pbc, cutoff):
    periodic_axes = np.flatnonzero(pbc)
    limits = np.zeros(3, dtype=int)
    if periodic_axes.size:
        reciprocal = np.linalg.pinv(cell[periodic_axes])
        # |n + (r_j-r_i) @ reciprocal| <= cutoff * |reciprocal|.
        limits[periodic_axes] = np.ceil(
            np.ptp(positions @ reciprocal, axis=0)
            + cutoff * np.linalg.norm(reciprocal, axis=0)
        ).astype(int)
    edges = set()
    for shift in itertools.product(*(range(-n, n + 1) for n in limits)):
        for sender, receiver in itertools.product(range(len(positions)), repeat=2):
            if sender == receiver and not any(shift):
                continue
            vector = positions[receiver] - positions[sender] + np.array(shift) @ cell
            if np.linalg.norm(vector) < cutoff:
                edges.add((sender, receiver, *shift))
    return edges


@pytest.mark.parametrize('pbc', list(itertools.product([False, True], repeat=3)))
@pytest.mark.parametrize('translate', [False, True])
def test_neighbors_match_periodic_images_for_tilted_cells(pbc, translate):
    cell = np.array([[2.5, 0.3, 1.2], [0.7, 3.0, -0.8], [-1.0, 0.6, 3.5]])
    positions = np.random.default_rng(4).uniform(-0.7, 0.8, (4, 3)) @ cell
    if translate:
        positions += [300.0, -100.0, 40.0]
    original_cell, original_positions = cell.copy(), positions.copy()
    cutoff = 2.6
    indices, shifts, unit_shifts, output_cell = get_neighborhood(
        positions, cutoff, pbc=pbc, cell=cell
    )
    actual = {tuple(edge) for edge in np.column_stack((indices.T, unit_shifts))}
    assert actual == _reference_edges(positions, original_cell, pbc, cutoff)
    np.testing.assert_array_equal(cell, original_cell)
    np.testing.assert_array_equal(positions, original_positions)
    if any(pbc):
        np.testing.assert_array_equal(output_cell, cell)
    np.testing.assert_allclose(shifts, unit_shifts @ output_cell)
    vectors = positions[indices[1]] - positions[indices[0]] + shifts
    assert np.all(np.linalg.norm(vectors, axis=1) < cutoff)


def test_molecule_search_cell_depends_on_extent_only():
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.2, -0.3]])
    reference = get_neighborhood(positions, 2.0)
    translated = get_neighborhood(positions + [100.0, -300.0, 50.0], 2.0)
    np.testing.assert_allclose(reference[3], np.diag(np.ptp(positions, axis=0) + 5))
    for near, far in zip(reference, translated, strict=True):
        np.testing.assert_allclose(near, far, atol=1e-12)


def test_free_search_direction_can_differ_from_nonperiodic_axis():
    # The third row is nonperiodic, but the free direction is x, not z.
    cell = np.array([[0.0, 2.5, 0.0], [0.0, 0.0, 3.0], [4.0, 0.0, 0.0]])
    pbc = (True, True, False)
    positions = np.array([[100.0, -50.0, 3.0], [100.5, -49.0, 3.2]])
    indices, _, units, _ = get_neighborhood(positions, 2.6, pbc=pbc, cell=cell)
    actual = {tuple(edge) for edge in np.column_stack((indices.T, units))}
    assert actual == _reference_edges(positions, cell, pbc, 2.6)


def test_slab_retains_physical_volume_under_translation():
    cell = np.diag([4.0, 4.0, 40.0])
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    for offset in (0.0, 100.0):
        _, _, _, returned = get_neighborhood(
            positions + [offset, 0, 0], 2.0, pbc=(True, True, False), cell=cell
        )
        np.testing.assert_array_equal(returned, np.diag([4.0, 4.0, 40.0]))
        np.testing.assert_array_equal(cell, np.diag([4.0, 4.0, 40.0]))


@pytest.mark.parametrize('pbc', [(True, False, False), (True, True, False)])
def test_missing_vacuum_rows_are_completed_orthogonally(pbc):
    cell = np.array([[1.5, 2.6, 0.2], [0.3, 2.8, -0.7], [0.0, 0.0, 0.0]])
    cell[~np.array(pbc)] = 0
    original = cell.copy()
    positions = np.array([[0.1, 0.2, 0.0], [0.8, 0.4, 0.3]])
    _, _, _, output = get_neighborhood(positions, 2.0, pbc=pbc, cell=cell)
    assert abs(np.linalg.det(output)) > 0
    np.testing.assert_array_equal(cell, original)
    np.testing.assert_array_equal(output[np.array(pbc)], cell[np.array(pbc)])
    for row in output[~np.array(pbc)]:
        np.testing.assert_allclose(cell[np.array(pbc)] @ row, 0, atol=1e-12)


def test_periodic_self_images_are_retained():
    indices, _, units, _ = get_neighborhood(
        np.zeros((1, 3)),
        1.05,
        pbc=(True, True, True),
        cell=np.eye(3),
    )
    assert indices.shape == (2, 6)
    assert not np.any(np.all(units == 0, axis=1))
