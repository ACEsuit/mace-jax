"""Global device placement and coordinated iteration for fixed-shape graph batches."""

import hashlib
import itertools
import pickle
from collections.abc import Sequence

import jax
import jraph
import numpy as np
from jax.experimental import multihost_utils
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from mace_jax import data

AXIS_NAME = 'devices'


class DeviceMesh:
    """Map local graph batches onto all devices, keeping model state replicated."""

    def __init__(self):
        self.mesh = Mesh(np.asarray(jax.devices()), (AXIS_NAME,))
        self.local_devices = self.mesh.local_devices
        self.local_device_count = len(self.local_devices)
        local_mesh = Mesh(np.asarray(self.local_devices), (AXIS_NAME,))
        self.local_batch_sharding = NamedSharding(local_mesh, P(AXIS_NAME))
        self.batch_sharding = NamedSharding(self.mesh, P(AXIS_NAME))
        self.state_sharding = NamedSharding(self.mesh, P())

    def compile(self, fn, in_specs):
        """Compile a complete device step whose outputs are globally replicated."""
        # Stage the device body first so unused inputs are pruned before Shardy
        # lowers the manual region (avoids a compiler crash with JAX 0.11.2).
        # cuEquivariance's custom derivatives do not yet support VMA checking.
        return jax.jit(
            jax.shard_map(
                jax.jit(fn),
                mesh=self.mesh,
                in_specs=in_specs,
                out_specs=P(),
                check_vma=False,
            )
        )

    def replicate(self, tree, *, broadcast=False):
        """Place state without adding or guessing a replication dimension.

        Without broadcast, callers must supply identical values on every process.
        """
        if broadcast and jax.process_count() > 1:
            tree = multihost_utils.broadcast_one_to_all(tree)

        def place(leaf):
            if isinstance(leaf, jax.Array) and (
                not leaf.is_fully_addressable or leaf.sharding == self.state_sharding
            ):
                return jax.device_put(leaf, self.state_sharding)
            if not hasattr(leaf, 'shape'):
                leaf = np.asarray(leaf)
            # device_put to a global mesh rejects committed local arrays and
            # otherwise checks equality across hosts for every leaf. State is
            # already synchronized, so construct it from each process's copy.
            return jax.make_array_from_process_local_data(
                self.state_sharding, leaf, global_shape=leaf.shape
            )

        return jax.tree_util.tree_map(place, tree)

    def prepare_local_batch(self, graphs):
        """Prepare and transfer only local shards; safe to call from a prefetcher."""
        if len(graphs) != self.local_device_count:
            raise ValueError('Expected one microbatch per local device.')
        # Fixed-shape loader batches can go directly to their destination devices,
        # avoiding a device-zero stack followed by redistribution to other GPUs.
        shapes = [jax.tree_util.tree_map(lambda x: x.shape, graph) for graph in graphs]
        if all(shape == shapes[0] for shape in shapes):
            graph = jax.tree_util.tree_map(lambda *xs: np.stack(xs), *graphs)
        else:
            graph = data.prepare_sharded_batch(graphs, self.local_device_count)
        return jax.device_put(graph, self.local_batch_sharding)

    def global_batch(self, graph):
        """Describe local shards as a global array without gathering their data."""

        def convert(leaf):
            shape = (self.mesh.size,) + leaf.shape[1:]
            if (
                isinstance(leaf, jax.Array)
                and leaf.sharding == self.local_batch_sharding
            ):
                return jax.make_array_from_single_device_arrays(
                    shape,
                    self.batch_sharding,
                    [shard.data for shard in leaf.addressable_shards],
                )
            return jax.make_array_from_process_local_data(
                self.batch_sharding, leaf, global_shape=shape
            )

        return jax.tree_util.tree_map(convert, graph)


def _with_weights(graph):
    if graph.globals.weight is None:
        graph = graph._replace(
            globals=graph.globals._replace(
                weight=np.ones(graph.n_node.shape, dtype=np.float32)
            )
        )
    return graph


def empty_graph_like(graph):
    """Mask a valid graph's geometry, avoiding singular dummy model inputs."""
    return graph._replace(
        globals=graph.globals._replace(
            weight=np.zeros(graph.globals.weight.shape, graph.globals.weight.dtype)
        )
    )


def _micro_batches(iterator):
    for item in iterator:
        if isinstance(item, jraph.GraphsTuple):
            if item.n_node.ndim == 2:
                for index in range(item.n_node.shape[0]):
                    yield jax.tree_util.tree_map(lambda x: x[index], item)
            else:
                yield item
        elif isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
            yield from _micro_batches(item)
        else:
            raise TypeError('Expected graph batches or sequences of graph batches.')


def group_batches(iterator, group_size, *, pad_fn=empty_graph_like):
    """Keep complete loader batches, padding unused devices with ``pad_fn``."""
    if group_size < 1:
        raise ValueError('group_size must be positive.')
    micro_batches = _micro_batches(iterator)
    while chunk := list(itertools.islice(micro_batches, group_size)):
        chunk = [_with_weights(graph) for graph in chunk]
        chunk.extend(pad_fn(chunk[0]) for _ in range(group_size - len(chunk)))
        yield chunk


def iter_loader_batches(loader, **kwargs):
    """Route eager loader failures through the coordinated batch iterator too."""
    try:
        return loader.iter_batches(**kwargs)
    except Exception as exc:
        if jax.process_count() == 1:
            raise

        def failed_iterator(error=exc):
            raise error
            yield  # pragma: no cover

        return failed_iterator()


def _batch_signature(graph):
    leaves, tree = jax.tree_util.tree_flatten(graph)
    schema = (str(tree), [(x.shape[1:], str(x.dtype)) for x in leaves])
    digest = hashlib.sha256(repr(schema).encode()).digest()
    return np.frombuffer(digest, dtype=np.int32)


def _broadcast_template(graph, source):
    """Share a first batch only when a process has no local padding template."""
    is_source = jax.process_index() == source
    payload = pickle.dumps(jax.device_get(graph)) if is_source else b''
    size = multihost_utils.broadcast_one_to_all(
        np.asarray(len(payload), dtype=np.int32), is_source=is_source
    )
    buffer = (
        np.frombuffer(payload, dtype=np.uint8)
        if is_source
        else np.zeros(int(size), dtype=np.uint8)
    )
    buffer = multihost_utils.broadcast_one_to_all(buffer, is_source=is_source)
    return pickle.loads(np.asarray(buffer).tobytes())


def synchronize_batches(iterator, local_device_count):
    """Keep processes in lockstep, including exhausted or initially empty ranks.

    Items are (locally prepared graph, padding statistics). Run this iterator on
    the training thread, *after* local prefetching: its collectives must not race
    the model's collectives. Batch schemas must agree across processes.
    """
    if jax.process_count() == 1:
        yield from iterator
        return

    iterator = iter(iterator)
    template = None
    first_step = True
    while True:
        error = None
        try:
            item = next(iterator)
        except StopIteration:
            item = None
        except Exception as exc:
            error = exc
            item = None
        if item is not None:
            template = item
        state = np.zeros(9, dtype=np.int32)
        state[0] = -1 if error is not None else int(item is not None)
        if template is not None:
            state[1:] = _batch_signature(template[0])
        states = np.asarray(multihost_utils.process_allgather(state, tiled=False))
        failed = np.flatnonzero(states[:, 0] < 0)
        if failed.size:
            raise RuntimeError(
                f'Batch loading failed on processes {failed.tolist()}.'
            ) from error
        active = np.flatnonzero(states[:, 0] > 0)
        if not active.size:
            return
        signatures = states[np.any(states[:, 1:] != 0, axis=1), 1:]
        if not np.all(signatures == signatures[0]):
            raise ValueError(
                'All processes must use matching fixed batch shapes and dtypes.'
            )

        if first_step and active.size < jax.process_count():
            graph = _broadcast_template(
                None if template is None else template[0], int(active[0])
            )
            if template is None:
                graph = jax.tree_util.tree_map(
                    lambda x: np.repeat(x[:1], local_device_count, axis=0), graph
                )
                nodes = int(graph.nodes.positions.shape[1]) * local_device_count
                edges = int(graph.senders.shape[1]) * local_device_count
                graphs = int(graph.n_node.shape[1]) * local_device_count
                template = (graph, (nodes, edges, graphs, nodes, edges, graphs))
        first_step = False
        if item is None:
            graph, padding = template
            nodes, edges, graphs = padding[:3]
            item = (
                empty_graph_like(graph),
                (nodes, edges, graphs, nodes, edges, graphs),
            )
        yield item
