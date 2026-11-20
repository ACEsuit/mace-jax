"""Validate and extract explicitly configured model inputs from ASE metadata."""

import numpy as np

RESERVED_INPUTS = frozenset(
    {
        'positions',
        'numbers',
        'forces',
        'energy',
        'stress',
        'virials',
        'cell',
        'pbc',
        'node_attrs',
        'node_attrs_index',
        'edge_index',
        'shifts',
        'unit_shifts',
        'batch',
        'ptr',
        'head',
        'species',
        'node_type',
        'vectors',
        'natoms',
        'lammps_class',
    }
)


def extract_features(atoms, embedding_specs):
    """Return node and graph feature dictionaries with stable, validated shapes.

    ``key`` optionally maps a model feature name to an ASE metadata key. Missing
    features are errors: charge and spin states must not be silently guessed.
    """
    node_features, graph_features = {}, {}
    for name, spec in (embedding_specs or {}).items():
        if name in RESERVED_INPUTS:
            raise ValueError(
                f'Embedding feature {name!r} conflicts with a model input.'
            )
        per = spec.get('per', 'node')
        if per not in ('node', 'graph'):
            raise ValueError(f'Invalid embedding scope {per!r} for {name!r}.')
        key = spec.get('key', name)
        source = atoms.arrays if per == 'node' else atoms.info
        value = source.get(key, source.get(name))
        # Native MACE HDF5 files also store atom-wise properties in properties/.
        if value is None and per == 'node':
            value = atoms.info.get(key, atoms.info.get(name))
        if value is None:
            raise ValueError(
                f'Missing {per} embedding feature {name!r} (ASE key {key!r}).'
            )
        value = np.asarray(value)
        if value.dtype.kind not in 'biuf' or not np.all(np.isfinite(value)):
            raise ValueError(
                f'Embedding feature {name!r} must contain finite numeric values.'
            )
        if spec['type'] == 'categorical':
            expected = (len(atoms),) if per == 'node' else ()
            if value.shape == expected + (1,):
                value = value.reshape(expected)
            if value.shape != expected or np.any(value != np.floor(value)):
                raise ValueError(
                    f'Categorical feature {name!r} must have integer values and shape {expected}.'
                )
            # Promote before adding offsets: unsigned/narrow integer inputs can
            # overflow even though their eventual class index is valid.
            numeric = value.astype(np.float64)
            limits = np.iinfo(np.int32)
            if np.any(numeric < limits.min) or np.any(numeric > limits.max):
                raise ValueError(f'Categorical feature {name!r} must fit in int32.')
            indices = numeric + spec.get('offset', 0)
            if np.any(indices < 0) or np.any(indices >= spec['num_classes']):
                raise ValueError(
                    f'Categorical feature {name!r} is outside its configured class range.'
                )
            value = value.astype(np.int32)
        elif spec['type'] == 'continuous':
            width = int(spec.get('in_dim', 1))
            expected = (len(atoms), width) if per == 'node' else (width,)
            if width == 1 and value.shape == expected[:-1]:
                value = value.reshape(expected)
            if value.shape != expected:
                raise ValueError(
                    f'Continuous feature {name!r} must have shape {expected}, got {value.shape}.'
                )
        else:
            raise ValueError(f'Unknown embedding type {spec["type"]!r}.')
        (node_features if per == 'node' else graph_features)[name] = value
    return node_features or None, graph_features or None
