import itertools
import logging
import multiprocessing as mp
from collections import defaultdict, namedtuple
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import IO, NamedTuple

import ase.data
import ase.io
import jax
import jax.numpy as jnp
import jraph
import numpy as np
from roundmantissa import ceil_mantissa

from mace_jax.data.neighborhood import get_neighborhood


Vector = np.ndarray  # [3,]
Positions = np.ndarray  # [..., 3]
Forces = np.ndarray  # [..., 3]
Cell = np.ndarray  # [3,3]
Stress = np.ndarray  # [3,3]
Pbc = tuple  # (3,)

DEFAULT_CONFIG_TYPE = 'Default'
DEFAULT_CONFIG_TYPE_WEIGHTS = {DEFAULT_CONFIG_TYPE: 1.0}


@dataclass
class Configuration:
    atomic_numbers: np.ndarray
    positions: Positions  # Angstrom
    energy: float | None = None  # eV
    forces: Forces | None = None  # eV/Angstrom
    stress: Stress | None = None  # eV/Angstrom^3
    virials: Stress | None = None  # eV
    dipole: np.ndarray | None = None  # eÅ
    polarizability: np.ndarray | None = None  # eÅ²/V
    cell: Cell | None = None
    pbc: Pbc | None = None

    weight: float = 1.0  # weight of config in loss
    config_type: str | None = DEFAULT_CONFIG_TYPE  # config_type of config
    head: str = 'Default'


Configurations = list[Configuration]


def random_train_valid_split(
    items: Sequence, valid_num: int, seed: int
) -> tuple[list, list]:
    size = len(items)
    train_size = size - valid_num

    indices = list(range(size))
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    return (
        [items[i] for i in indices[:train_size]],
        [items[i] for i in indices[train_size:]],
    )


def _normalize_stress(stress: np.ndarray | None) -> np.ndarray | None:
    """Convert various stress layouts into a 3x3 matrix."""
    if stress is None:
        return None
    arr = np.asarray(stress)
    if arr.shape == (3, 3):
        return arr
    flat = arr.reshape(-1)
    if flat.size == 6:
        xx, yy, zz, yz, xz, xy = flat
        return np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])
    if flat.size == 9:
        return flat.reshape(3, 3)
    raise ValueError(
        f'Unsupported stress shape {arr.shape}; expected 3x3 or Voigt-like entries.'
    )


def config_from_atoms(
    atoms: ase.Atoms,
    energy_key='energy',
    forces_key='forces',
    stress_key='stress',
    config_type_weights: dict[str, float] = None,
    prefactor_stress: float = 1.0,
    remap_stress: np.ndarray = None,
    head_name: str | None = None,
    virials_key: str = 'virials',
    dipole_key: str = 'dipole',
    polarizability_key: str = 'polarizability',
) -> Configuration:
    """Convert ase.Atoms to Configuration"""
    if config_type_weights is None:
        config_type_weights = DEFAULT_CONFIG_TYPE_WEIGHTS

    energy = atoms.info.get(energy_key, None)  # eV
    if energy is None:
        try:
            energy = atoms.get_potential_energy()
        except Exception:
            energy = None
    if energy is None:
        energy = np.array(0.0)

    stress = atoms.info.get(stress_key, None)  # eV / Ang^3
    if stress is None:
        try:
            stress = atoms.get_stress()
        except Exception:
            stress = None

    if stress is not None:
        stress = prefactor_stress * _normalize_stress(stress)

        if remap_stress is not None:
            remap_stress = np.asarray(remap_stress)
            assert remap_stress.shape == (3, 3)
            assert remap_stress.dtype.kind == 'i'
            stress = stress.flatten()[remap_stress]

        # TODO(mario): fix this
        # make it traceless? because it seems that our formula is not valid for the trace
        # pressure = np.trace(stress)
        # stress -= pressure / 3.0 * np.eye(3)
    # else:
    # pressure = None

    forces = atoms.arrays.get(forces_key, None)  # eV / Ang
    if forces is None:
        try:
            forces = atoms.get_forces()
        except Exception:
            forces = None
    virials = atoms.info.get(virials_key, None)
    if virials is not None:
        virials = _normalize_stress(virials)
    dipole = atoms.info.get(dipole_key, None)
    if dipole is not None:
        dipole = np.asarray(dipole)
    polarizability = atoms.info.get(polarizability_key, None)
    if polarizability is not None:
        polarizability = np.asarray(polarizability)
    atomic_numbers = np.array(
        [ase.data.atomic_numbers[symbol] for symbol in atoms.symbols]
    )
    pbc = tuple(atoms.get_pbc())
    cell = np.array(atoms.get_cell())
    assert np.linalg.det(cell) >= 0.0
    config_type = atoms.info.get('config_type', 'Default')
    weight = config_type_weights.get(config_type, 1.0)
    head = head_name if head_name is not None else atoms.info.get('head', 'Default')
    return Configuration(
        atomic_numbers=atomic_numbers,
        positions=atoms.get_positions(),
        energy=energy,
        forces=forces,
        stress=stress,
        virials=virials,
        dipole=dipole,
        polarizability=polarizability,
        weight=weight,
        config_type=config_type,
        pbc=pbc,
        cell=cell,
        head=head,
    )


def test_config_types(
    test_configs: Configurations,
) -> list[tuple[str | None, list[Configuration]]]:
    """Split test set based on config_type-s"""
    test_by_ct = defaultdict(list)
    for conf in test_configs:
        test_by_ct[conf.config_type].append(conf)
    return list(test_by_ct.items())


def load_from_xyz(
    file_or_path: str | IO,
    config_type_weights: dict = None,
    energy_key: str = 'energy',
    forces_key: str = 'forces',
    stress_key: str = 'stress',
    virials_key: str = 'virials',
    dipole_key: str = 'dipole',
    polarizability_key: str = 'polarizability',
    extract_atomic_energies: bool = False,
    num_configs: int = None,
    prefactor_stress: float = 1.0,
    remap_stress: np.ndarray = None,
    head_name: str | None = None,
    no_data_ok: bool = False,
) -> tuple[dict[int, float], Configurations]:
    if num_configs is None:
        atoms_list = ase.io.read(file_or_path, format='extxyz', index=':')
    else:
        atoms_list = ase.io.read(file_or_path, format='extxyz', index=f':{num_configs}')
        if len(atoms_list) < num_configs:
            logging.warning(
                f'Only {len(atoms_list)} configurations found. Expected at least {num_configs}.'
            )

    if not isinstance(atoms_list, list):
        atoms_list = [atoms_list]

    atomic_energies_dict = {}
    if extract_atomic_energies:
        atoms_without_iso_atoms = []

        for idx, atoms in enumerate(atoms_list):
            if (
                len(atoms) == 1
                and getattr(atoms, 'config_type', None) == 'IsolatedAtom'
            ):
                if energy_key in atoms.info.keys():
                    atomic_energies_dict[atoms.get_atomic_numbers()[0]] = atoms.info[
                        energy_key
                    ]
                else:
                    logging.warning(
                        f"Configuration '{idx}' is marked as 'IsolatedAtom' "
                        'but does not contain an energy.'
                    )
            else:
                atoms_without_iso_atoms.append(atoms)

        if len(atomic_energies_dict) > 0:
            logging.info('Using isolated atom energies from training file')

        atoms_list = atoms_without_iso_atoms

    configs = [
        config_from_atoms(
            atoms,
            energy_key=energy_key,
            forces_key=forces_key,
            stress_key=stress_key,
            config_type_weights=config_type_weights,
            prefactor_stress=prefactor_stress,
            remap_stress=remap_stress,
            head_name=head_name,
            virials_key=virials_key,
            dipole_key=dipole_key,
            polarizability_key=polarizability_key,
        )
        for atoms in atoms_list
    ]
    return atomic_energies_dict, configs


class AtomicNumberTable:
    def __init__(self, zs: Sequence[int]):
        zs = [int(z) for z in zs]
        # unique
        assert len(zs) == len(set(zs))
        # sorted
        assert zs == sorted(zs)

        self.zs = zs

    def __len__(self) -> int:
        return len(self.zs)

    def __str__(self):
        return f'AtomicNumberTable: {tuple(s for s in self.zs)}'

    def index_to_z(self, index: int) -> int:
        return self.zs[index]

    def z_to_index(self, atomic_number: int) -> int:
        return self.zs.index(atomic_number)

    def z_to_index_map(self, max_atomic_number: int) -> np.ndarray:
        x = np.zeros(max_atomic_number + 1, dtype=np.int32)
        for i, z in enumerate(self.zs):
            x[z] = i
        return x


def get_atomic_number_table_from_zs(zs: Iterable[int]) -> AtomicNumberTable:
    return AtomicNumberTable(sorted(set(zs)))


def atomic_numbers_to_indices(
    atomic_numbers: np.ndarray, z_table: AtomicNumberTable
) -> np.ndarray:
    to_index_fn = np.vectorize(z_table.z_to_index)
    return to_index_fn(atomic_numbers)


def compute_average_E0s(
    graphs: list[jraph.GraphsTuple], z_table: AtomicNumberTable
) -> dict[int, float]:
    """
    Function to compute the average interaction energy of each chemical element
    returns dictionary of E0s
    """
    len_train = len(graphs)
    len_zs = len(z_table)
    A = np.zeros((len_train, len_zs))
    B = np.zeros(len_train)
    for i in range(len_train):
        energy = graphs[i].globals.energy
        B[i] = float(np.asarray(energy).reshape(-1)[0])
        for j, z in enumerate(z_table.zs):
            A[i, j] = np.count_nonzero(graphs[i].nodes.species == z)
    try:
        E0s = np.linalg.lstsq(A, B, rcond=None)[0]
        atomic_energies_dict = {}
        for i, z in enumerate(z_table.zs):
            atomic_energies_dict[z] = E0s[i]
    except np.linalg.LinAlgError:
        logging.warning(
            'Failed to compute E0s using least squares regression, using the same for all atoms'
        )
        atomic_energies_dict = {}
        for i, z in enumerate(z_table.zs):
            atomic_energies_dict[z] = 0.0
    return atomic_energies_dict


def compute_average_E0s_from_configs(
    configs: Configurations, z_table: AtomicNumberTable
) -> dict[int, float]:
    len_train = len(configs)
    len_zs = len(z_table)
    A = np.zeros((len_train, len_zs))
    B = np.zeros(len_train)
    for i, config in enumerate(configs):
        B[i] = float(np.asarray(config.energy).reshape(()))
        for j, z in enumerate(z_table.zs):
            A[i, j] = np.count_nonzero(config.atomic_numbers == z)
    try:
        E0s = np.linalg.lstsq(A, B, rcond=None)[0]
        atomic_energies_dict = {z: float(E0s[i]) for i, z in enumerate(z_table.zs)}
    except np.linalg.LinAlgError:
        logging.warning(
            'Failed to compute E0s using least squares regression, using the same for all atoms'
        )
        atomic_energies_dict = {z: 0.0 for z in z_table.zs}
    return atomic_energies_dict


def compute_average_E0s_from_species(
    graphs: list[jraph.GraphsTuple], num_species: int
) -> dict[int, float]:
    """
    Function to compute the average interaction energy of each chemical element
    returns dictionary of E0s
    """
    len_train = len(graphs)
    A = np.zeros((len_train, num_species))
    B = np.zeros(len_train)
    for i in range(len_train):
        energy = graphs[i].globals.energy
        B[i] = float(np.asarray(energy).reshape(-1)[0])
        for j in range(num_species):
            A[i, j] = np.count_nonzero(graphs[i].nodes.species == j)
    try:
        E0s = np.linalg.lstsq(A, B, rcond=None)[0]
    except np.linalg.LinAlgError:
        logging.warning(
            'Failed to compute E0s using least squares regression, using the same for all atoms'
        )
        E0s = np.zeros(num_species)
    return E0s


def save_configurations_as_HDF5(configurations: Configurations, _, h5_file) -> None:
    grp = h5_file.create_group('config_batch_0')
    for j, config in enumerate(configurations):
        subgroup = grp.create_group(f'config_{j}')
        subgroup['atomic_numbers'] = write_value(config.atomic_numbers)
        subgroup['positions'] = write_value(config.positions)
        subgroup['cell'] = write_value(config.cell)
        subgroup['pbc'] = write_value(config.pbc)
        subgroup['weight'] = write_value(config.weight)
        subgroup['config_type'] = write_value(config.config_type)
        if config.head is not None:
            subgroup['head'] = write_value(config.head)
        properties_subgrp = subgroup.create_group('properties')
        if config.energy is not None:
            properties_subgrp['energy'] = write_value(config.energy)
        if config.forces is not None:
            properties_subgrp['forces'] = write_value(config.forces)
        if config.stress is not None:
            properties_subgrp['stress'] = write_value(config.stress)
        if config.virials is not None:
            properties_subgrp['virials'] = write_value(config.virials)
        if config.dipole is not None:
            properties_subgrp['dipole'] = write_value(config.dipole)
        if config.polarizability is not None:
            properties_subgrp['polarizability'] = write_value(config.polarizability)
        weights_subgrp = subgroup.create_group('property_weights')
        for name, value in (
            ('energy', config.energy),
            ('forces', config.forces),
            ('stress', config.stress),
            ('virials', config.virials),
            ('dipole', config.dipole),
            ('polarizability', config.polarizability),
        ):
            weights_subgrp[name] = write_value(1.0 if value is not None else 0.0)


def write_value(value):
    return value if value is not None else 'None'


GraphNodes = namedtuple('Nodes', ['positions', 'forces', 'species'])
# Multiprocessing pickle support expects the class to be exported under its __name__.
Nodes = GraphNodes


class GraphEdges(NamedTuple):
    shifts: np.ndarray
    unit_shifts: np.ndarray | None = None


class GraphGlobals(NamedTuple):
    cell: np.ndarray
    energy: np.ndarray | None
    stress: np.ndarray | None
    weight: np.ndarray
    head: np.ndarray | None = None
    virials: np.ndarray | None = None
    dipole: np.ndarray | None = None
    polarizability: np.ndarray | None = None


def graph_from_configuration(
    config: Configuration,
    cutoff: float,
    z_table: AtomicNumberTable = None,
    head_to_index: dict[str, int] | None = None,
) -> jraph.GraphsTuple:
    (
        edge_index,
        shifts,
        unit_shifts,
        neighborhood_cell,
    ) = get_neighborhood(
        positions=config.positions, cutoff=cutoff, pbc=config.pbc, cell=config.cell
    )
    senders, receivers = edge_index
    cell = neighborhood_cell

    if z_table is None:
        species = config.atomic_numbers
    else:
        max_atomic_number = (
            int(np.max(config.atomic_numbers)) if config.atomic_numbers.size else 0
        )
        z_map = z_table.z_to_index_map(max_atomic_number=max(200, max_atomic_number))
        species = z_map[config.atomic_numbers]

    if head_to_index is None:
        head_index = 0
    else:
        if config.head not in head_to_index:
            raise KeyError(
                f"Unknown head '{config.head}'. Available heads: {tuple(head_to_index)}"
            )
        head_index = head_to_index[config.head]

    head_array = np.asarray(head_index, dtype=np.int32)

    return jraph.GraphsTuple(
        nodes=GraphNodes(
            positions=config.positions,
            forces=config.forces,
            species=species,
        ),
        edges=GraphEdges(shifts=shifts, unit_shifts=unit_shifts),
        globals=jax.tree_util.tree_map(
            lambda x: x[None, ...] if x is not None else None,
            GraphGlobals(
                cell=cell,
                energy=config.energy,
                stress=config.stress,
                weight=np.asarray(config.weight),
                head=head_array,
                virials=config.virials,
                dipole=config.dipole,
                polarizability=config.polarizability,
            ),
        ),
        receivers=receivers,
        senders=senders,
        n_edge=np.array([senders.shape[0]]),
        n_node=np.array([config.positions.shape[0]]),
    )


def _none_leaf(value):
    return value is None


def replicate_to_local_devices(tree):
    """Broadcast a pytree so the leading axis matches local device count."""
    device_count = jax.local_device_count()
    if device_count <= 1:
        return tree

    def _replicate(leaf):
        if leaf is None:
            return None
        arr = jnp.asarray(leaf)
        return jnp.broadcast_to(arr, (device_count,) + arr.shape)

    return jax.tree_util.tree_map(_replicate, tree, is_leaf=_none_leaf)


def unreplicate_from_local_devices(tree):
    """Strip a replicated leading axis (if present) from a pytree."""
    device_count = jax.local_device_count()
    if device_count <= 1:
        return tree

    host = jax.device_get(tree)
    if isinstance(host, (list, tuple)) and len(host) == device_count:
        return jax.tree_util.tree_map(lambda x: x[0], host, is_leaf=_none_leaf)

    def _maybe_collapse(leaf):
        if leaf is None:
            return None
        arr = np.asarray(leaf)
        if arr.ndim == 0 or arr.shape[0] != device_count:
            return leaf
        first = arr[0]
        if np.all(arr == first):
            return first
        return leaf

    return jax.tree_util.tree_map(_maybe_collapse, host, is_leaf=_none_leaf)


def prepare_single_batch(graph):
    """Cast a batched graph to device arrays, keeping None leaves."""

    def _to_device_array(x):
        if x is None:
            return None
        return jnp.asarray(x)

    return jax.tree_util.tree_map(_to_device_array, graph, is_leaf=_none_leaf)


def split_graphs_for_devices(graph, num_devices: int) -> list[jraph.GraphsTuple]:
    def _pad_graphs_to_multiple(graph, multiple):
        if multiple <= 1:
            return graph
        total = int(graph.n_node.shape[0])
        remainder = total % multiple
        if remainder == 0:
            return graph
        pad_graphs = multiple - remainder

        def _pad_global_value(value):
            if value is None:
                return None
            arr = np.asarray(value)
            if arr.ndim == 0:
                arr = np.broadcast_to(arr, (total,))
            pad_shape = (pad_graphs,) + arr.shape[1:]
            pad_vals = np.zeros(pad_shape, dtype=arr.dtype)
            return np.concatenate([arr, pad_vals], axis=0)

        pad_n_node = np.concatenate(
            [np.asarray(graph.n_node), np.zeros(pad_graphs, dtype=np.int32)]
        )
        pad_n_edge = np.concatenate(
            [np.asarray(graph.n_edge), np.zeros(pad_graphs, dtype=np.int32)]
        )

        globals_attr = graph.globals
        if globals_attr is None:
            globals_dict = None
        elif hasattr(globals_attr, 'items'):
            globals_dict = globals_attr.__class__()
            for key, value in globals_attr.items():
                globals_dict[key] = _pad_global_value(value)
        elif hasattr(globals_attr, '_fields'):
            padded = {
                key: _pad_global_value(value)
                for key, value in globals_attr._asdict().items()
            }
            globals_dict = globals_attr.__class__(**padded)
        else:
            globals_dict = _pad_global_value(globals_attr)

        return graph._replace(
            globals=globals_dict,
            n_node=pad_n_node,
            n_edge=pad_n_edge,
        )

    graph = _pad_graphs_to_multiple(graph, num_devices)
    total_graphs = int(graph.n_node.shape[0])
    if total_graphs % num_devices != 0:
        raise ValueError(
            'For multi-device execution, batch size must be divisible by the number of devices.'
        )
    per_device = total_graphs // num_devices
    return [_slice_graph(graph, i * per_device, per_device) for i in range(num_devices)]


def prepare_sharded_batch(graph, num_devices: int):
    """Prepare a micro-batch for ``jax.pmap`` execution."""

    def _ensure_graphs_tuple(item):
        if isinstance(item, jraph.GraphsTuple):
            return item
        if isinstance(item, Sequence) and not isinstance(item, (bytes, str)):
            return item[0] if len(item) == 1 else jraph.batch_np(item)
        raise TypeError('Expected a GraphsTuple or sequence of GraphsTuples.')

    def _pad_device_batches(device_graphs, targets=None):
        if not device_graphs:
            return device_graphs
        graph_counts = [int(np.asarray(g.n_node).shape[0]) for g in device_graphs]
        nodes_per_device = [int(np.sum(np.asarray(g.n_node))) for g in device_graphs]
        edges_per_device = [int(np.sum(np.asarray(g.n_edge))) for g in device_graphs]
        if (
            len(set(nodes_per_device)) == 1
            and len(set(edges_per_device)) == 1
            and len(set(graph_counts)) == 1
        ):
            return device_graphs
        if targets is None:
            max_nodes = max(nodes_per_device)
            max_edges = max(edges_per_device)
            max_graphs = max(graph_counts)
            target_n_node = max(max_nodes + 1, 1)
            target_n_edge = max_edges
            target_n_graph = max(2, max_graphs + 1)
        else:
            target_n_node, target_n_edge, target_n_graph = targets
        padded = []
        for g in device_graphs:
            if (
                int(np.sum(np.asarray(g.n_node))) < target_n_node
                or int(np.sum(np.asarray(g.n_edge))) < target_n_edge
                or int(np.asarray(g.n_node).shape[0]) < target_n_graph
            ):
                g = jraph.pad_with_graphs(
                    g,
                    n_node=target_n_node,
                    n_edge=target_n_edge,
                    n_graph=target_n_graph,
                )
            padded.append(g)
        return padded

    fixed_targets = None
    if isinstance(graph, jraph.GraphsTuple):
        total_nodes = int(np.sum(np.asarray(graph.n_node)))
        total_edges = int(np.sum(np.asarray(graph.n_edge)))
        fixed_targets = (max(total_nodes + 1, 1), total_edges, None)

    if isinstance(graph, Sequence) and not isinstance(graph, jraph.GraphsTuple):
        filtered = [g for g in graph if g is not None]
        if len(filtered) != num_devices:
            raise ValueError(
                f'Expected {num_devices} micro-batches for multi-device execution, got {len(filtered)}.'
            )
        device_graphs = [_ensure_graphs_tuple(item) for item in filtered]
    else:
        device_graphs = split_graphs_for_devices(graph, num_devices)

    if fixed_targets is not None:
        per_device_graphs = int(np.asarray(device_graphs[0].n_node).shape[0])
        fixed_targets = (
            fixed_targets[0],
            fixed_targets[1],
            max(2, per_device_graphs + 1),
        )
    device_graphs = _pad_device_batches(device_graphs, targets=fixed_targets)
    device_batches = [prepare_single_batch(g) for g in device_graphs]

    def _stack_or_none(*values):
        first = values[0]
        if first is None:
            return None
        return jnp.stack(values)

    return jax.tree_util.tree_map(_stack_or_none, *device_batches, is_leaf=_none_leaf)


_MP_WORKERS_SUPPORTED: bool | None = None


def supports_multiprocessing_workers() -> bool:
    """Return True if this environment can create ``multiprocessing`` locks."""
    global _MP_WORKERS_SUPPORTED
    if _MP_WORKERS_SUPPORTED is not None:
        return _MP_WORKERS_SUPPORTED

    try:
        ctx = mp.get_context('spawn')
        lock = ctx.Lock()
        lock.acquire()
        lock.release()
    except (OSError, PermissionError):
        _MP_WORKERS_SUPPORTED = False
    else:
        _MP_WORKERS_SUPPORTED = True
    return _MP_WORKERS_SUPPORTED


def stack_or_none(chunks):
    """Concatenate numpy arrays unless the list is empty."""
    if not chunks:
        return None
    return np.concatenate(chunks, axis=0)


def split_device_outputs(tree, num_devices: int):
    """Slice a replicated pytree into per-device host arrays."""
    host_tree = jax.tree_util.tree_map(
        lambda x: None if x is None else np.asarray(x),
        tree,
        is_leaf=lambda leaf: leaf is None,
    )
    slices = []
    for idx in range(num_devices):
        slices.append(
            jax.tree_util.tree_map(
                lambda x: None if x is None else x[idx],
                host_tree,
                is_leaf=lambda leaf: leaf is None,
            )
        )
    return slices


def iter_micro_batches(loader):
    """Flatten a loader that may emit lists of micro-batches."""
    for item in loader:
        if item is None:
            continue
        if isinstance(item, list):
            for sub in item:
                if sub is not None:
                    yield sub
        else:
            yield item


def take_chunk(iterator, size: int):
    """Collect up to ``size`` items from an iterator."""
    return list(itertools.islice(iterator, size))


def batched_iterator(
    iterator,
    size: int,
    *,
    remainder_action: Callable | None = None,
    drop_remainder: bool = True,
):
    """Yield fixed-size chunks from ``iterator``."""
    if size <= 0:
        raise ValueError('Chunk size must be positive.')
    while True:
        chunk = take_chunk(iterator, size)
        if len(chunk) < size:
            if chunk and remainder_action is not None:
                remainder_action(len(chunk), size)
            if chunk and not drop_remainder:
                yield chunk
            break
        yield chunk


def _slice_graph(graph: jraph.GraphsTuple, start_graph: int, count: int):
    start_graph = int(start_graph)
    count = int(count)
    n_node = np.asarray(graph.n_node)
    n_edge = np.asarray(graph.n_edge)

    graph_slice = slice(start_graph, start_graph + count)
    node_start = int(n_node[:start_graph].sum())
    node_end = int(node_start + n_node[graph_slice].sum())
    edge_start = int(n_edge[:start_graph].sum())
    edge_end = int(edge_start + n_edge[graph_slice].sum())

    def _slice_tree(tree, slc):
        return jax.tree_util.tree_map(
            lambda x: None if x is None else x[slc],
            tree,
        )

    nodes = _slice_tree(graph.nodes, slice(node_start, node_end))
    edges = _slice_tree(graph.edges, slice(edge_start, edge_end))

    senders = graph.senders[edge_start:edge_end] - node_start
    receivers = graph.receivers[edge_start:edge_end] - node_start

    globals_attr = graph.globals
    if globals_attr is None:
        globals_dict = None
    else:
        globals_dict = _slice_tree(globals_attr, graph_slice)
    n_node_slice = graph.n_node[graph_slice]
    n_edge_slice = graph.n_edge[graph_slice]

    return jraph.GraphsTuple(
        nodes=nodes,
        edges=edges,
        senders=senders,
        receivers=receivers,
        globals=globals_dict,
        n_node=n_node_slice,
        n_edge=n_edge_slice,
    )


def pad_graph_to_nearest_ceil_mantissa(
    graphs_tuple: jraph.GraphsTuple,
    *,
    n_mantissa_bits: int = 2,
    n_min_nodes: int = 1,
    n_min_edges: int = 1,
    n_min_graphs: int = 1,
    n_max_nodes: int = np.iinfo(np.int32).max,
    n_max_edges: int = np.iinfo(np.int32).max,
    n_max_graphs: int = np.iinfo(np.int32).max,
) -> jraph.GraphsTuple:
    """Pads a batched `GraphsTuple` to the nearest power of two.

    For example, if a `GraphsTuple` has 7 nodes, 5 edges and 3 graphs, this method
    would pad the `GraphsTuple` nodes and edges:
        7batch_sizedes --> 8 nodes (2^3)
        5 edges --> 8 edges (2^3)

    And since padding is accomplished using `jraph.pad_with_graphs`, an extra
    graph and node is added:
        8 nodes --> 9 nodes
        3 graphs --> 4 graphs

    Args:
        graphs_tuple: a batched `GraphsTuple` (can be batch size 1).

    Returns:
        A graphs_tuple batched to the nearest power of two.
    """
    n_nodes = graphs_tuple.n_node.sum()
    n_edges = len(graphs_tuple.senders)
    n_graphs = graphs_tuple.n_node.shape[0]

    pad_nodes_to = ceil_mantissa(n_nodes + 1, n_mantissa_bits)
    pad_edges_to = ceil_mantissa(n_edges, n_mantissa_bits)
    pad_graphs_to = ceil_mantissa(n_graphs + 1, n_mantissa_bits)

    pad_nodes_to = np.clip(pad_nodes_to, n_min_nodes, n_max_nodes)
    pad_edges_to = np.clip(pad_edges_to, n_min_edges, n_max_edges)
    pad_graphs_to = np.clip(pad_graphs_to, n_min_graphs, n_max_graphs)

    return jraph.pad_with_graphs(
        graphs_tuple, pad_nodes_to, pad_edges_to, pad_graphs_to
    )
