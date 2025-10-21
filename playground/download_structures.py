#!/usr/bin/env python3
"""Download a few representative Materials Project structures.

The notebook in this folder expects three ready-to-use crystal structures.
This helper contacts the Materials Project REST API, retrieves conventional
cells with at least five atoms and multiple species, and stores them in
``material_structures.json`` for offline reuse.

Usage::

    MATERIALS_PROJECT_API_KEY="..." python download_structures.py

An API key can also be supplied via ``MP_API_KEY``.  Only ``pymatgen`` is
required; it ships with the slim REST client used below.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from pathlib import Path

from pymatgen.core import Structure
from pymatgen.ext.matproj import MPRester

# Materials Project entries we mirror locally (all multi-species, >=5 atoms).
TARGET_IDS: Iterable[str] = (
    'mp-1143',  # Al2O3
    'mp-3470',  # MgSiO3
    'mp-1960',  # Li2O
)


def _fetch_structure(mpr: MPRester, mp_id: str) -> tuple[Structure, str]:
    """Retrieve a pymatgen Structure and label for the requested MP id."""

    structure = mpr.get_structure_by_material_id(mp_id)
    if structure is None:
        raise ValueError(f'No structure returned for {mp_id}')

    label = structure.composition.reduced_formula or mp_id
    return structure, label


def _serialise_structure(structure: Structure) -> dict:
    """Convert a pymatgen Structure into a JSON-serialisable dictionary."""

    lattice = structure.lattice.matrix
    sites = []
    for site in structure.sites:
        sites.append(
            {
                'species': site.species_string,
                'xyz': list(map(float, site.coords)),
            }
        )

    return {
        'formula': structure.formula,
        'chemical_system': structure.chemical_system,
        'lattice': [list(map(float, row)) for row in lattice],
        'sites': sites,
    }


def main() -> None:
    api_key = os.environ.get('MATERIALS_PROJECT_API_KEY') or os.environ.get(
        'MP_API_KEY'
    )
    if not api_key:
        raise RuntimeError(
            'Materials Project API key not provided. Set MATERIALS_PROJECT_API_KEY or MP_API_KEY.'
        )

    output_path = Path(__file__).with_name('material_structures.json')

    records: list[dict] = []
    with MPRester(api_key) as mpr:
        for mp_id in TARGET_IDS:
            structure, label = _fetch_structure(mpr, mp_id)
            records.append(
                {
                    'mp_id': mp_id,
                    'label': label,
                    'structure': _serialise_structure(structure),
                }
            )

    output_path.write_text(json.dumps(records, indent=2))
    print(f'Saved {len(records)} structures to {output_path}')


if __name__ == '__main__':
    main()
