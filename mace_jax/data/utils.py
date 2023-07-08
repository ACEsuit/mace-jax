import logging
from collections import defaultdict, namedtuple
from dataclasses import dataclass
from random import shuffle
from typing import IO, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import ase.data
import ase.io
import jax
import jraph
import numpy as np
from roundmantissa import ceil_mantissa

from mace_jax.data.neighborhood import get_neighborhood

from .dynamically_batch import dynamically_batch

Vector = np.ndarray  # [3,]
Positions = np.ndarray  # [..., 3]
Forces = np.ndarray  # [..., 3]
Cell = np.ndarray  # [3,3]
Stress = np.ndarray  # [3,3]
Pbc = tuple  # (3,)

DEFAULT_CONFIG_TYPE = "Default"
DEFAULT_CONFIG_TYPE_WEIGHTS = {DEFAULT_CONFIG_TYPE: 1.0}


@dataclass
class Configuration:
    atomic_numbers: np.ndarray
    positions: Positions  # Angstrom
    energy: Optional[float] = None  # eV
    forces: Optional[Forces] = None  # eV/Angstrom
    stress: Optional[Stress] = None  # eV/Angstrom^3
    cell: Optional[Cell] = None
    pbc: Optional[Pbc] = None

    weight: float = 1.0  # weight of config in loss
    config_type: Optional[str] = DEFAULT_CONFIG_TYPE  # config_type of config


Configurations = List[Configuration]


def random_train_valid_split(
    items: Sequence, valid_num: int, seed: int
) -> Tuple[List, List]:
    size = len(items)
    train_size = size - valid_num

    indices = list(range(size))
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    return (
        [items[i] for i in indices[:train_size]],
        [items[i] for i in indices[train_size:]],
    )


def config_from_atoms(
    atoms: ase.Atoms,
    energy_key="energy",
    forces_key="forces",
    stress_key="stress",
    config_type_weights: Dict[str, float] = None,
    prefactor_stress: float = 1.0,
    remap_stress: np.ndarray = None,
) -> Configuration:
    """Convert ase.Atoms to Configuration"""
    if config_type_weights is None:
        config_type_weights = DEFAULT_CONFIG_TYPE_WEIGHTS

    energy = atoms.info.get(energy_key, None)  # eV
    stress = atoms.info.get(stress_key, None)  # eV / Ang^3

    if energy is None:
        energy = np.array(0.0)

    if stress is not None:
        stress = prefactor_stress * stress

        if remap_stress is not None:
            remap_stress = np.asarray(remap_stress)
            assert remap_stress.shape == (3, 3)
            assert remap_stress.dtype.kind == "i"
            stress = stress.flatten()[remap_stress]

        assert stress.shape == (3, 3)

        # TODO(mario): fix this
        # make it traceless? because it seems that our formula is not valid for the trace
        # pressure = np.trace(stress)
        # stress -= pressure / 3.0 * np.eye(3)
    # else:
    # pressure = None

    forces = atoms.arrays.get(forces_key, None)  # eV / Ang
    atomic_numbers = np.array(
        [ase.data.atomic_numbers[symbol] for symbol in atoms.symbols]
    )
    pbc = tuple(atoms.get_pbc())
    cell = np.array(atoms.get_cell())
    assert np.linalg.det(cell) >= 0.0
    config_type = atoms.info.get("config_type", "Default")
    weight = config_type_weights.get(config_type, 1.0)
    return Configuration(
        atomic_numbers=atomic_numbers,
        positions=atoms.get_positions(),
        energy=energy,
        forces=forces,
        stress=stress,
        weight=weight,
        config_type=config_type,
        pbc=pbc,
        cell=cell,
    )


def test_config_types(
    test_configs: Configurations,
) -> List[Tuple[Optional[str], List[Configuration]]]:
    """Split test set based on config_type-s"""
    test_by_ct = defaultdict(list)
    for conf in test_configs:
        test_by_ct[conf.config_type].append(conf)
    return list(test_by_ct.items())


def load_from_xyz(
    file_or_path: Union[str, IO],
    config_type_weights: Dict = None,
    energy_key: str = "energy",
    forces_key: str = "forces",
    stress_key: str = "stress",
    extract_atomic_energies: bool = False,
    num_configs: int = None,
    prefactor_stress: float = 1.0,
    remap_stress: np.ndarray = None,
) -> Tuple[Dict[int, float], Configurations]:
    if num_configs is None:
        atoms_list = ase.io.read(file_or_path, format="extxyz", index=":")
    else:
        atoms_list = ase.io.read(file_or_path, format="extxyz", index=f":{num_configs}")
        if len(atoms_list) < num_configs:
            logging.warning(
                f"Only {len(atoms_list)} configurations found. Expected at least {num_configs}."
            )

    if not isinstance(atoms_list, list):
        atoms_list = [atoms_list]

    atomic_energies_dict = {}
    if extract_atomic_energies:
        atoms_without_iso_atoms = []

        for idx, atoms in enumerate(atoms_list):
            if (
                len(atoms) == 1
                and getattr(atoms, "config_type", None) == "IsolatedAtom"
            ):
                if energy_key in atoms.info.keys():
                    atomic_energies_dict[atoms.get_atomic_numbers()[0]] = atoms.info[
                        energy_key
                    ]
                else:
                    logging.warning(
                        f"Configuration '{idx}' is marked as 'IsolatedAtom' "
                        "but does not contain an energy."
                    )
            else:
                atoms_without_iso_atoms.append(atoms)

        if len(atomic_energies_dict) > 0:
            logging.info("Using isolated atom energies from training file")

        atoms_list = atoms_without_iso_atoms

    configs = [
        config_from_atoms(
            atoms,
            energy_key,
            forces_key,
            stress_key,
            config_type_weights,
            prefactor_stress,
            remap_stress,
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
        return f"AtomicNumberTable: {tuple(s for s in self.zs)}"

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
    graphs: List[jraph.GraphsTuple], z_table: AtomicNumberTable
) -> Dict[int, float]:
    """
    Function to compute the average interaction energy of each chemical element
    returns dictionary of E0s
    """
    len_train = len(graphs)
    len_zs = len(z_table)
    A = np.zeros((len_train, len_zs))
    B = np.zeros(len_train)
    for i in range(len_train):
        B[i] = graphs[i].globals.energy
        for j, z in enumerate(z_table.zs):
            A[i, j] = np.count_nonzero(graphs[i].nodes.species == z)
    try:
        E0s = np.linalg.lstsq(A, B, rcond=None)[0]
        atomic_energies_dict = {}
        for i, z in enumerate(z_table.zs):
            atomic_energies_dict[z] = E0s[i]
    except np.linalg.LinAlgError:
        logging.warning(
            "Failed to compute E0s using least squares regression, using the same for all atoms"
        )
        atomic_energies_dict = {}
        for i, z in enumerate(z_table.zs):
            atomic_energies_dict[z] = 0.0
    return atomic_energies_dict


def compute_average_E0s_from_species(
    graphs: List[jraph.GraphsTuple], num_species: int
) -> Dict[int, float]:
    """
    Function to compute the average interaction energy of each chemical element
    returns dictionary of E0s
    """
    len_train = len(graphs)
    A = np.zeros((len_train, num_species))
    B = np.zeros(len_train)
    for i in range(len_train):
        B[i] = graphs[i].globals.energy
        for j in range(num_species):
            A[i, j] = np.count_nonzero(graphs[i].nodes.species == j)
    try:
        E0s = np.linalg.lstsq(A, B, rcond=None)[0]
    except np.linalg.LinAlgError:
        logging.warning(
            "Failed to compute E0s using least squares regression, using the same for all atoms"
        )
        E0s = np.zeros(num_species)
    return E0s


GraphNodes = namedtuple("Nodes", ["positions", "forces", "species"])
GraphEdges = namedtuple("Edges", ["shifts"])
GraphGlobals = namedtuple("Globals", ["cell", "energy", "stress", "weight"])


def graph_from_configuration(
    config: Configuration, cutoff: float, z_table: AtomicNumberTable = None
) -> jraph.GraphsTuple:
    senders, receivers, shifts = get_neighborhood(
        positions=config.positions, cutoff=cutoff, pbc=config.pbc, cell=config.cell
    )
    if z_table is None:
        species = config.atomic_numbers
    else:
        z_map = z_table.z_to_index_map(max_atomic_number=200)
        species = z_map[config.atomic_numbers]

    return jraph.GraphsTuple(
        nodes=GraphNodes(
            positions=config.positions,
            forces=config.forces,
            species=species,
        ),
        edges=GraphEdges(shifts=shifts),
        globals=jax.tree_util.tree_map(
            lambda x: x[None, ...],
            GraphGlobals(
                cell=config.cell,
                energy=config.energy,
                stress=config.stress,
                weight=np.asarray(config.weight),
            ),
        ),
        receivers=receivers,
        senders=senders,
        n_edge=np.array([senders.shape[0]]),
        n_node=np.array([config.positions.shape[0]]),
    )


class GraphDataLoader:
    def __init__(
        self,
        graphs: List[jraph.GraphsTuple],
        n_node: int,
        n_edge: int,
        n_graph: int,
        min_n_node: int = 1,
        min_n_edge: int = 1,
        min_n_graph: int = 1,
        shuffle: bool = True,
        n_mantissa_bits: Optional[int] = None,
    ):
        self.graphs = graphs
        self.n_node = n_node
        self.n_edge = n_edge
        self.n_graph = n_graph
        self.min_n_node = min_n_node
        self.min_n_edge = min_n_edge
        self.min_n_graph = min_n_graph
        self.shuffle = shuffle
        self.n_mantissa_bits = n_mantissa_bits
        self._length = None

        keep_graphs = [
            graph
            for graph in self.graphs
            if graph.n_node.sum() <= self.n_node - 1
            and graph.n_edge.sum() <= self.n_edge
        ]
        if len(keep_graphs) != len(self.graphs):
            logging.warning(
                f"Discarded {len(self.graphs) - len(keep_graphs)} graphs due to size constraints."
            )
        self.graphs = keep_graphs

    def __iter__(self):
        graphs = self.graphs.copy()  # this is a shallow copy
        if self.shuffle:
            shuffle(graphs)

        for batched_graph in dynamically_batch(
            graphs,
            n_node=self.n_node,
            n_edge=self.n_edge,
            n_graph=self.n_graph,
        ):
            if self.n_mantissa_bits is None:
                yield jraph.pad_with_graphs(
                    batched_graph, self.n_node, self.n_edge, self.n_graph
                )
            else:
                yield pad_graph_to_nearest_ceil_mantissa(
                    batched_graph,
                    n_mantissa_bits=self.n_mantissa_bits,
                    n_min_nodes=self.min_n_node,
                    n_min_edges=self.min_n_edge,
                    n_min_graphs=self.min_n_graph,
                    n_max_nodes=self.n_node,
                    n_max_edges=self.n_edge,
                    n_max_graphs=self.n_graph,
                )

    def __len__(self):
        if self.shuffle:
            raise NotImplementedError("Cannot compute length of shuffled data loader.")
        return self.approx_length()

    def approx_length(self):
        if self._length is None:
            self._length = 0
            for _ in self:
                self._length += 1
        return self._length

    def subset(self, i):
        graphs = self.graphs
        if isinstance(i, slice):
            graphs = graphs[i]
        if isinstance(i, int):
            graphs = graphs[:i]
        if isinstance(i, list):
            graphs = [graphs[j] for j in i]
        if isinstance(i, float):
            graphs = graphs[: int(len(graphs) * i)]

        return GraphDataLoader(
            graphs=graphs,
            n_node=self.n_node,
            n_edge=self.n_edge,
            n_graph=self.n_graph,
            min_n_node=self.min_n_node,
            min_n_edge=self.min_n_edge,
            min_n_graph=self.min_n_graph,
            shuffle=self.shuffle,
            n_mantissa_bits=self.n_mantissa_bits,
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
