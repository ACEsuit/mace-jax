from typing import Optional, Tuple

import ase.neighborlist
import numpy as np


def get_neighborhood(
    positions: np.ndarray,  # [num_positions, 3]
    cutoff: float,
    pbc: Optional[Tuple[bool, bool, bool]] = None,
    cell: Optional[np.ndarray] = None,  # [3, 3]
    true_self_interaction=False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if pbc is None:
        pbc = (False, False, False)

    if cell is None or (cell == 0.0).all():
        cell = np.identity(3, dtype=float)

    assert len(pbc) == 3 and all(isinstance(i, (bool, np.bool_)) for i in pbc)
    assert cell.shape == (3, 3)

    # Note (mario): I swapped senders and receivers here
    # j = senders, i = receivers instead of the other way around
    # such that the receivers are always in the central cell.
    # This is important to propagate message passing towards the center which can be useful in some cases.
    receivers, senders, senders_unit_shifts = ase.neighborlist.primitive_neighbor_list(
        quantities="ijS",
        pbc=pbc,
        cell=cell,
        positions=positions,
        cutoff=cutoff,
        self_interaction=true_self_interaction,
        use_scaled_positions=False,  # positions are not scaled positions
    )

    # From the docs: With the shift vector S, the distances D between atoms can be computed from
    # D = positions[j]-positions[i]+S.dot(cell)
    # Note (mario): this is done in the function get_edge_relative_vectors
    return senders, receivers, senders_unit_shifts
