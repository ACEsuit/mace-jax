import numpy as np
from matscipy.neighbours import neighbour_list


def _orthogonal_residual(vector: np.ndarray, basis: list[np.ndarray]) -> np.ndarray:
    residual = np.array(vector, dtype=float, copy=True)
    for axis in basis:
        residual -= (residual @ axis) * axis
    return residual


def _aperiodic_search_directions(
    pbc: tuple[bool, bool, bool], cell: np.ndarray
) -> dict[int, np.ndarray]:
    """Build free directions orthogonal to the periodic lattice, as in Torch MACE.

    A Cartesian search-box axis is insufficient for a tilted slab or chain:
    periodic images can leave that box and acquire spurious nonperiodic shifts.
    """
    aperiodic = [dim for dim in range(3) if not pbc[dim]]
    if not aperiodic:
        return {}
    identity = np.eye(3)
    if not any(pbc):
        return {dim: identity[dim] for dim in aperiodic}

    spanned = []
    tolerance = 1e-8
    for dim in range(3):
        if pbc[dim]:
            residual = _orthogonal_residual(cell[dim], spanned)
            norm = np.linalg.norm(residual)
            if norm > tolerance * max(float(np.linalg.norm(cell[dim])), 1.0):
                spanned.append(residual / norm)

    directions = {}
    for dim in aperiodic:
        for candidate in (identity[dim], *identity):
            residual = _orthogonal_residual(candidate, spanned)
            norm = np.linalg.norm(residual)
            if norm > tolerance:
                directions[dim] = residual / norm
                spanned.append(directions[dim])
                break
    return directions


def get_neighborhood(
    positions: np.ndarray,  # [num_positions, 3]
    cutoff: float,
    pbc: tuple[bool, bool, bool] | None = None,
    cell: np.ndarray | None = None,  # [3, 3]
    true_self_interaction=False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if pbc is None:
        pbc = (False, False, False)

    if cell is None or cell.any() == np.zeros((3, 3)).any():
        cell = np.identity(3, dtype=float)

    assert len(pbc) == 3 and all(isinstance(i, (bool, np.bool_)) for i in pbc)
    assert cell.shape == (3, 3)

    # Match current Torch MACE: size the search box from atom extents, move the
    # atoms inside it, and keep its free axes orthogonal to the periodic span.
    # These copies also keep the caller's positions and physical cell intact.
    search_positions = np.array(positions, dtype=float, copy=True)
    extended_cell = np.array(cell, dtype=float, copy=True)
    for dim, direction in _aperiodic_search_directions(pbc, cell).items():
        projection = search_positions @ direction
        extent = projection.max() - projection.min()
        extended_cell[dim] = (extent + 2 * cutoff + 1) * direction
        search_positions -= (projection.min() - cutoff) * direction

    if any(pbc):
        # Slab/chain stress must use the physical volume. Only replace a missing
        # vacuum row, to avoid a degenerate cell for structures with zero vacuum.
        cell = np.array(cell, dtype=float, copy=True)
        for dim in range(3):
            if not pbc[dim] and not cell[dim].any():
                cell[dim] = extended_cell[dim]
    else:
        cell = extended_cell

    sender, receiver, unit_shifts = neighbour_list(
        quantities='ijS',
        pbc=pbc,
        cell=extended_cell,
        positions=search_positions,
        cutoff=cutoff,
        # self_interaction=True,  # we want edges from atom to itself in different periodic images
        # use_scaled_positions=False,  # positions are not scaled positions
    )

    if not true_self_interaction:
        # Eliminate self-edges that don't cross periodic boundaries
        true_self_edge = sender == receiver
        true_self_edge &= np.all(unit_shifts == 0, axis=1)
        keep_edge = ~true_self_edge

        # Note: after eliminating self-edges, it can be that no edges remain in this system
        sender = sender[keep_edge]
        receiver = receiver[keep_edge]
        unit_shifts = unit_shifts[keep_edge]

    # Build output
    edge_index = np.stack((sender, receiver))  # [2, n_edges]

    # From the docs: With the shift vector S, the distances D between atoms can be computed from
    # D = positions[j]-positions[i]+S.dot(cell)
    shifts = np.dot(unit_shifts, cell)  # [n_edges, 3]

    return edge_index, shifts, unit_shifts, cell
