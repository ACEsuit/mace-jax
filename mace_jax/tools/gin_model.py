"""Gin-configurable model factory and graph-to-model adapters.

The model factory builds either a native Flax MACE module or wraps a converted
Torch checkpoint. The apply functions accept padded `GraphsTuple` batches and
return energy/force/stress predictions. Using padded, fixed-shape batches is
important for JAX/XLA: it lets the compiled model be reused across epochs
without recompiling for each new batch shape.
"""

import logging
from collections.abc import Callable, Mapping
from pathlib import Path

import e3nn_jax as e3nn
import gin
import jax
import jax.numpy as jnp
import jraph
import numpy as np

from mace_jax import data, modules, tools
from mace_jax.modules.blocks import RealAgnosticResidualInteractionBlock
from mace_jax.tools.dtype import default_dtype

gin.register(jax.nn.silu)
gin.register(jax.nn.relu)
gin.register(jax.nn.gelu)
gin.register(jnp.abs)
gin.register(jnp.tanh)
gin.register('identity')(lambda x: x)

gin.register('std_scaling')(tools.compute_mean_std_atomic_inter_energy)
gin.register('rms_forces_scaling')(tools.compute_mean_rms_energy_forces)


@gin.configurable
def constant_scaling(graphs, atomic_energies, *, mean=0.0, std=1.0):
    """Return fixed mean/std values for energy normalization.

    Args:
        graphs: Training graphs (unused; kept for API compatibility).
        atomic_energies: Per-species atomic energies (unused).
        mean: Constant mean energy offset to apply per graph.
        std: Constant scaling factor for energies/forces/stress.

    Returns:
        Tuple of (mean, std) consumed by `model()` to rescale predictions.
    """
    return mean, std


@gin.configurable
def bessel_basis(length, max_length, number: int):
    """Compute a radial basis using e3nn's Bessel expansion.

    Args:
        length: Edge lengths (distance) array.
        max_length: Cutoff radius used by the basis.
        number: Number of basis functions.

    Returns:
        Radial basis values with shape derived from `length` and `number`.
    """
    return e3nn.bessel(length, number, max_length)


@gin.configurable
def soft_envelope(
    length, max_length, arg_multiplicator: float = 2.0, value_at_origin: float = 1.2
):
    """Compute a smooth cutoff envelope for radial features.

    Args:
        length: Edge lengths (distance) array.
        max_length: Cutoff radius.
        arg_multiplicator: Controls how quickly the envelope decays.
        value_at_origin: Value of the envelope at zero distance.

    Returns:
        Envelope values matching `length` shape, used to dampen features near cutoff.
    """
    return e3nn.soft_envelope(
        length,
        max_length,
        arg_multiplicator=arg_multiplicator,
        value_at_origin=value_at_origin,
    )


@gin.configurable
def polynomial_envelope(length, max_length, degree0: int, degree1: int):
    """Compute a polynomial cutoff envelope for radial features.

    Args:
        length: Edge lengths (distance) array.
        max_length: Cutoff radius.
        degree0: Inner polynomial degree.
        degree1: Outer polynomial degree.

    Returns:
        Envelope values used by the MACE radial basis pipeline.
    """
    return e3nn.poly_envelope(degree0, degree1, max_length)(length)


@gin.configurable
def u_envelope(length, max_length, p: int):
    """Compute a U-shaped polynomial envelope parameterized by p.

    Args:
        length: Edge lengths (distance) array.
        max_length: Cutoff radius.
        p: Envelope smoothness parameter.

    Returns:
        Envelope values for tapering radial basis contributions.
    """
    return e3nn.poly_envelope(p - 1, 2, max_length)(length)


@gin.configurable
def _graph_to_data(
    graph: jraph.GraphsTuple, *, num_species: int
) -> dict[str, jnp.ndarray]:
    """Convert a (possibly padded) graph into the MACE data dictionary.

    Args:
        graph: Input `jraph.GraphsTuple`, potentially padded.
        num_species: Size of the atomic-number vocabulary used for one-hot encoding.

    Returns:
        Dict of arrays matching the MACE module inputs (positions, edges, shifts,
        batch/ptr indexing, optional head indices, and cell data).

    This adapter is called by both the native Flax MACE apply function and the
    Torch-to-JAX converted apply function.
    """
    positions = jnp.asarray(graph.nodes.positions, dtype=default_dtype())
    shifts = jnp.asarray(graph.edges.shifts, dtype=positions.dtype)
    cell = jnp.asarray(graph.globals.cell, dtype=positions.dtype)

    species = jnp.asarray(graph.nodes.species, dtype=jnp.int32)
    senders = jnp.asarray(graph.senders, dtype=jnp.int32)
    receivers = jnp.asarray(graph.receivers, dtype=jnp.int32)

    # Per-node mask derived from graph_mask.
    node_mask = jraph.get_node_padding_mask(graph).astype(positions.dtype)

    # Build one-hot node attributes (zeroed for padded nodes).
    node_attrs = jax.nn.one_hot(
        species,
        num_classes=num_species,
        dtype=positions.dtype,
    )
    node_attrs = node_attrs * node_mask[:, None]

    # Batch indices and pointer array defining graph segment boundaries.
    graph_indices = jnp.arange(graph.n_node.shape[0], dtype=jnp.int32)
    batch = jnp.repeat(
        graph_indices, graph.n_node, total_repeat_length=positions.shape[0]
    )
    ptr_counts = graph.n_node.astype(jnp.int32)
    ptr = jnp.concatenate(
        [
            jnp.array([0], dtype=jnp.int32),
            jnp.cumsum(ptr_counts),
        ]
    )

    data_dict: dict[str, jnp.ndarray] = {
        'positions': positions,
        'node_attrs': node_attrs,
        'edge_index': jnp.stack([senders, receivers], axis=0),
        'shifts': shifts,
        'batch': batch,
        'ptr': ptr,
        'cell': cell,
    }

    unit_shifts = getattr(graph.edges, 'unit_shifts', None)
    if unit_shifts is None:
        unit_shifts = jnp.zeros(shifts.shape, dtype=positions.dtype)
    else:
        unit_shifts = jnp.asarray(unit_shifts, dtype=positions.dtype)
    data_dict['unit_shifts'] = unit_shifts

    # Optional per-graph head information (for multi-head outputs).
    head_attr = getattr(graph.globals, 'head', None)
    if head_attr is not None:
        data_dict['head'] = jnp.asarray(head_attr, dtype=jnp.int32).reshape(-1)

    return data_dict


@gin.configurable
def model(
    *,
    r_max: float,
    atomic_energies_dict: dict[int, float] = None,
    train_graphs: list[jraph.GraphsTuple] = None,
    initialize_seed: int | None = None,
    scaling: Callable = None,
    atomic_energies: str | np.ndarray | dict[int, float] = None,
    avg_num_neighbors: float = 'average',
    avg_r_min: float = None,
    num_species: int = None,
    num_interactions=3,
    path_normalization='path',
    gradient_normalization='path',
    learnable_atomic_energies=False,
    radial_basis: Callable[[jnp.ndarray], jnp.ndarray] = bessel_basis,
    radial_envelope: Callable[[jnp.ndarray], jnp.ndarray] = soft_envelope,
    torch_checkpoint: str | None = None,
    torch_head: str | None = None,
    torch_param_dtype: str | None = None,
    **kwargs,
):
    """Construct a MACE model and return an apply function plus initial params.

    This gin-configurable factory is used by training/evaluation entry points to
    build a Flax MACE module or wrap a converted Torch checkpoint. It resolves
    dataset-derived defaults (species, atomic energies, neighbor statistics) and
    optional output normalization.

    Args:
        r_max: Cutoff radius for neighbor interactions.
        atomic_energies_dict: Optional per-species atomic energies from preprocessing.
        train_graphs: Sample graphs used to infer species and statistics.
        initialize_seed: PRNG seed used to initialize parameters (optional).
        scaling: Callable that returns (mean, std) for output rescaling.
        atomic_energies: How to populate atomic energies ('average', 'isolated_atom',
            'zero', or explicit array/dict).
        avg_num_neighbors: Precomputed or 'average' neighbor count for normalization.
        avg_r_min: Precomputed or 'average' minimum neighbor distance.
        num_species: Explicit number of species; inferred if not provided.
        num_interactions: Number of interaction blocks in the MACE model.
        path_normalization: e3nn path normalization setting.
        gradient_normalization: e3nn gradient normalization setting.
        learnable_atomic_energies: Flag for learnable atomic energies (unsupported).
        radial_basis: Callable to compute radial basis features.
        radial_envelope: Callable to compute radial envelope values.
        torch_checkpoint: Optional Torch checkpoint path to convert and wrap.
        torch_head: Optional head name to select from Torch multi-head models.
        torch_param_dtype: Optional dtype override ('float32' or 'float64') for Torch params.
        **kwargs: Remaining MACE module constructor arguments.

    Returns:
        Tuple of (apply_fn, params_bundle, num_interactions). The apply function
        accepts a `jraph.GraphsTuple` and returns energy/force/stress predictions.
        `params_bundle` includes parameters (and optional config state), and
        `num_interactions` reflects the number of interaction blocks used.
    """
    z_table = kwargs.get('z_table', None)
    if z_table is not None:
        max_z = max(z_table.zs)
        required_species = max_z + 1
        if num_species is None:
            logging.info(
                'num_species not specified; inferring %s from dataset (max atomic number %s)',
                required_species,
                max_z,
            )
            num_species = required_species
        elif num_species <= max_z:
            logging.warning(
                'num_species=%s too small for dataset (max atomic number %s); expanding to %s',
                num_species,
                max_z,
                required_species,
            )
            num_species = required_species

    if torch_checkpoint is not None:
        import torch  # noqa: PLC0415
        from mace.tools.scripts_utils import extract_config_mace_model  # noqa: PLC0415

        from mace_jax.cli import mace_jax_from_torch  # noqa: PLC0415

        checkpoint_path = Path(torch_checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Torch checkpoint '{checkpoint_path}' does not exist."
            )

        logging.info('Loading Torch foundation model from %s', checkpoint_path)
        bundle = torch.load(checkpoint_path, map_location='cpu')
        torch_model = (
            bundle['model']
            if isinstance(bundle, dict) and 'model' in bundle
            else bundle
        )
        torch_model.eval()

        if torch_head is not None:
            heads = getattr(torch_model, 'heads', None)
            if not heads:
                raise ValueError(
                    f'Torch model has no heads attribute; cannot select head {torch_head!r}'
                )
            if torch_head not in heads:
                raise ValueError(
                    f'Head {torch_head!r} not found in Torch model heads {heads!r}'
                )
            logging.info(
                'Selected Torch head %s from Torch model heads %s', torch_head, heads
            )

        config = extract_config_mace_model(torch_model)
        if 'error' in config:
            raise RuntimeError(
                f'Failed to extract Torch configuration: {config["error"]}'
            )
        config['torch_model_class'] = torch_model.__class__.__name__

        logging.info('Converting Torch model to JAX representation')
        jax_module, variables, _ = mace_jax_from_torch.convert_model(
            torch_model, config
        )

        # Separate trainable params from config/state collections.
        config_state = None
        if isinstance(variables, dict) or hasattr(variables, 'get'):
            config_state = variables.get('config') or variables.get('constants')
            params = variables.get('params', variables)
        else:
            params = variables

        if torch_param_dtype is not None:
            if torch_param_dtype not in {'float64', 'float32'}:
                raise ValueError(
                    f'Unsupported torch_param_dtype={torch_param_dtype!r}; expected float64 or float32'
                )
            target_dtype = (
                jnp.float64 if torch_param_dtype == 'float64' else jnp.float32
            )
            params = jax.tree_util.tree_map(
                lambda x: x.astype(target_dtype) if isinstance(x, jnp.ndarray) else x,
                params,
            )
            if config_state is not None:
                config_state = jax.tree_util.tree_map(
                    lambda x: x.astype(target_dtype)
                    if isinstance(x, jnp.ndarray)
                    and jnp.issubdtype(x.dtype, jnp.inexact)
                    else x,
                    config_state,
                )

        torch_atomic_numbers = tuple(int(z) for z in config['atomic_numbers'])
        num_species_local = len(torch_atomic_numbers)
        torch_num_interactions = int(config.get('num_interactions', num_interactions))

        logging.info(
            'Loaded Torch foundation with %s atomic species and %s interaction blocks',
            num_species_local,
            torch_num_interactions,
        )

        def apply_fn(
            parameters,
            graph: jraph.GraphsTuple,
            *,
            compute_force: bool = True,
            compute_stress: bool = False,
        ) -> dict[str, jnp.ndarray]:
            """Apply the converted Torch model to a `GraphsTuple` input."""
            variables_local = parameters
            if config_state is not None and not (
                isinstance(parameters, dict) and 'config' in parameters
            ):
                variables_local = {'params': parameters, 'config': config_state}

            data_dict = _graph_to_data(graph, num_species=num_species_local)
            if torch_head is not None:
                head_names = config.get('heads') or []
                if torch_head not in head_names:
                    raise ValueError(
                        f'Head {torch_head!r} not present in Torch configuration heads {head_names!r}'
                    )
                head_index = head_names.index(torch_head)
                data_dict['head'] = jnp.asarray([head_index], dtype=jnp.int32)
            return jax_module.apply(
                variables_local,
                data_dict,
                compute_force=compute_force,
                compute_stress=compute_stress,
            )

        params_bundle = (
            {'params': params, 'config': config_state} if config_state else params
        )
        return apply_fn, params_bundle, torch_num_interactions

    passed_z_table = kwargs.pop('z_table', None)
    z_table = passed_z_table
    if z_table is None:
        if train_graphs is None:
            z_table = None
        else:
            z_table = data.get_atomic_number_table_from_zs(
                z for graph in train_graphs for z in graph.nodes.species
            )
    else:
        logging.info('Using provided z_table.')
    logging.info(f'z_table= {z_table}')

    if z_table is not None:
        max_z = max(z_table.zs)
        required_species = max_z + 1
        if num_species is None or num_species <= max_z:
            if num_species is None:
                logging.info(
                    'num_species not specified; inferring %s from dataset (max atomic number %s)',
                    required_species,
                    max_z,
                )
            else:
                logging.warning(
                    'num_species=%s too small for dataset (max atomic number %s); expanding to %s',
                    num_species,
                    max_z,
                    required_species,
                )
            num_species = required_species

    if avg_num_neighbors == 'average':
        avg_num_neighbors = tools.compute_avg_num_neighbors(train_graphs)
        logging.info(
            f'Compute the average number of neighbors: {avg_num_neighbors:.3f}'
        )
    else:
        logging.info(f'Use the average number of neighbors: {avg_num_neighbors:.3f}')

    if avg_r_min == 'average':
        avg_r_min = tools.compute_avg_min_neighbor_distance(train_graphs)
        logging.info(f'Compute the average min neighbor distance: {avg_r_min:.3f}')
    elif avg_r_min is None:
        logging.info('Do not normalize the radial basis (avg_r_min=None)')
    else:
        logging.info(f'Use the average min neighbor distance: {avg_r_min:.3f}')

    if atomic_energies is None:
        if atomic_energies_dict is None or len(atomic_energies_dict) == 0:
            atomic_energies = 'average'
        else:
            atomic_energies = 'isolated_atom'

    if atomic_energies == 'average':
        atomic_energies_dict = data.compute_average_E0s(train_graphs, z_table)
        logging.info(
            f'Computed average Atomic Energies using least squares: {atomic_energies_dict}'
        )
        atomic_energies = np.array(
            [atomic_energies_dict.get(z, 0.0) for z in range(num_species)]
        )
    elif atomic_energies == 'isolated_atom':
        logging.info(
            f'Using atomic energies from isolated atoms in the dataset: {atomic_energies_dict}'
        )
        atomic_energies = np.array(
            [atomic_energies_dict.get(z, 0.0) for z in range(num_species)]
        )
    elif atomic_energies == 'zero':
        logging.info('Not using atomic energies')
        atomic_energies = np.zeros(num_species)
    elif isinstance(atomic_energies, np.ndarray):
        logging.info(
            f'Use Atomic Energies that are provided: {atomic_energies.tolist()}'
        )
        if atomic_energies.shape != (num_species,):
            logging.error(
                f'atomic_energies.shape={atomic_energies.shape} != (num_species={num_species},)'
            )
            raise ValueError
    elif isinstance(atomic_energies, dict):
        atomic_energies_dict = atomic_energies
        logging.info(f'Use Atomic Energies that are provided: {atomic_energies_dict}')
        atomic_energies = np.array(
            [atomic_energies_dict.get(z, 0.0) for z in range(num_species)]
        )
    else:
        raise ValueError(f'atomic_energies={atomic_energies} is not supported')

    # check that num_species is consistent with the dataset
    if z_table is None:
        if train_graphs is not None:
            for graph in train_graphs:
                if not np.all(graph.nodes.species < num_species):
                    raise ValueError(
                        f'max(graph.nodes.species)={np.max(graph.nodes.species)} >= num_species={num_species}'
                    )
    else:
        if max(z_table.zs) >= num_species:
            raise ValueError(
                f'max(z_table.zs)={max(z_table.zs)} >= num_species={num_species}'
            )

    if scaling is None:
        mean, std = 0.0, 1.0
    else:
        mean, std = scaling(train_graphs, atomic_energies)
        logging.info(
            f'Scaling with {scaling.__qualname__}: mean={mean:.2f}, std={std:.2f}'
        )

    if learnable_atomic_energies:
        raise NotImplementedError(
            'learnable_atomic_energies is not supported by the Flax-based gin model.'
        )

    kwargs.update(
        dict(
            r_max=r_max,
            avg_num_neighbors=avg_num_neighbors,
            num_interactions=num_interactions,
            avg_r_min=avg_r_min,
            num_species=num_species,
            radial_basis=radial_basis,
            radial_envelope=radial_envelope,
        )
    )
    logging.info(f'Create MACE with parameters {kwargs}')
    kwargs.setdefault(
        'atomic_numbers',
        tuple(z_table.zs) if z_table is not None else tuple(range(num_species)),
    )
    kwargs.setdefault('atomic_energies', atomic_energies)

    num_species_bound = kwargs.pop('num_species', None)
    if num_species_bound is None:
        num_elements = len(kwargs['atomic_numbers'])
    else:
        num_elements = num_species_bound
    kwargs.setdefault('num_elements', num_elements)

    kwargs.pop('avg_r_min', None)
    kwargs.pop('radial_basis', None)
    kwargs.pop('radial_envelope', None)

    def _ensure_irreps(value):
        """Coerce configuration values into e3nn.Irreps when possible."""
        if value is None:
            return None
        if isinstance(value, e3nn.Irreps):
            return value
        if isinstance(value, str):
            return e3nn.Irreps(value)
        return value

    for irreps_key in ('hidden_irreps', 'MLP_irreps', 'edge_irreps'):
        if irreps_key in kwargs:
            kwargs[irreps_key] = _ensure_irreps(kwargs[irreps_key])

    kwargs.setdefault('interaction_cls', RealAgnosticResidualInteractionBlock)
    kwargs.setdefault('interaction_cls_first', RealAgnosticResidualInteractionBlock)

    mace_module = modules.MACE(**kwargs)
    config_state = None

    def apply_fn(
        params,
        graph: jraph.GraphsTuple,
        *,
        compute_force: bool = True,
        compute_stress: bool = False,
    ) -> dict[str, jnp.ndarray]:
        """Apply the MACE module to a (possibly padded) graph.

        Args:
            params: Trainable parameters (and optional config state).
            graph: Input batch as a `jraph.GraphsTuple`.
            compute_force: Whether to compute forces from the energy prediction.
            compute_stress: Whether to compute stress/virials from the energy prediction.

        Returns:
            Dict with `energy`, `forces`, and `stress` arrays (masked for padding).
        """
        e3nn.config('path_normalization', path_normalization)
        e3nn.config('gradient_normalization', gradient_normalization)

        variables_local = params
        if config_state is not None and not (
            isinstance(params, dict) and 'config' in params
        ):
            variables_local = {'params': params, 'config': config_state}

        graph_mask = jraph.get_graph_padding_mask(graph).astype(default_dtype())
        graph_mask_bool = graph_mask > 0.0
        node_mask = jraph.get_node_padding_mask(graph).astype(default_dtype())
        node_mask_bool = node_mask > 0.0
        edge_mask = jraph.get_edge_padding_mask(graph).astype(default_dtype())
        edge_mask_bool = edge_mask > 0.0

        def _sanitize(array, mask, expand_dims=0):
            """Fill padded entries with a valid sentinel so masked values are stable."""
            if array is None:
                return None
            if array.shape[0] == 0:
                return array
            fill = jnp.broadcast_to(array[:1], array.shape)
            if expand_dims > 0:
                broadcast_mask = mask.reshape(mask.shape + (1,) * expand_dims)
            else:
                broadcast_mask = mask
            return jnp.where(broadcast_mask, array, fill)

        sanitized_nodes = graph.nodes.__class__(
            positions=_sanitize(graph.nodes.positions, node_mask_bool, expand_dims=1),
            forces=_sanitize(graph.nodes.forces, node_mask_bool, expand_dims=1)
            if getattr(graph.nodes, 'forces', None) is not None
            else None,
            species=_sanitize(graph.nodes.species, node_mask_bool),
        )
        sanitized_edges = graph.edges.__class__(
            shifts=_sanitize(graph.edges.shifts, edge_mask_bool, expand_dims=1),
            unit_shifts=_sanitize(
                graph.edges.unit_shifts, edge_mask_bool, expand_dims=1
            )
            if getattr(graph.edges, 'unit_shifts', None) is not None
            else None,
        )
        sanitized_globals = graph.globals.__class__(
            cell=_sanitize(graph.globals.cell, graph_mask_bool, expand_dims=2)
            if getattr(graph.globals, 'cell', None) is not None
            else None,
            energy=_sanitize(graph.globals.energy, graph_mask_bool)
            if getattr(graph.globals, 'energy', None) is not None
            else None,
            stress=_sanitize(graph.globals.stress, graph_mask_bool, expand_dims=2)
            if getattr(graph.globals, 'stress', None) is not None
            else None,
            weight=graph.globals.weight,
            head=_sanitize(graph.globals.head, graph_mask_bool)
            if getattr(graph.globals, 'head', None) is not None
            else None,
            virials=_sanitize(graph.globals.virials, graph_mask_bool, expand_dims=2)
            if getattr(graph.globals, 'virials', None) is not None
            else None,
            dipole=_sanitize(graph.globals.dipole, graph_mask_bool, expand_dims=1)
            if getattr(graph.globals, 'dipole', None) is not None
            else None,
            polarizability=_sanitize(
                graph.globals.polarizability, graph_mask_bool, expand_dims=2
            )
            if getattr(graph.globals, 'polarizability', None) is not None
            else None,
        )
        sanitized_graph = graph._replace(
            nodes=sanitized_nodes, edges=sanitized_edges, globals=sanitized_globals
        )

        data_dict = _graph_to_data(sanitized_graph, num_species=num_species)
        outputs = mace_module.apply(
            variables_local,
            data_dict,
            compute_force=compute_force,
            compute_stress=compute_stress,
        )

        # Apply optional rescaling consistent with the historical Haiku version.
        num_nodes = graph.n_node.astype(default_dtype())
        energy = outputs['energy']
        if energy is not None:
            energy = std * jnp.nan_to_num(energy) + mean * num_nodes
            energy = energy * graph_mask

        forces = outputs['forces']
        if forces is not None:
            forces = std * jnp.nan_to_num(forces)
            forces = forces * node_mask[:, None]

        stress = outputs['stress']
        if stress is not None:
            stress = std * jnp.nan_to_num(stress)
            stress = stress * graph_mask[:, None, None]

        return {
            'energy': energy,
            'forces': forces,
            'stress': stress,
        }

    params = None
    if initialize_seed is not None and train_graphs:
        example_graph = train_graphs[0]
        example_data = _graph_to_data(example_graph, num_species=num_species)
        variables = mace_module.init(jax.random.PRNGKey(initialize_seed), example_data)
        if isinstance(variables, dict) or hasattr(variables, 'get'):
            params = variables.get('params', variables)
            config_state = variables.get('config')
        else:
            params = variables

    params_bundle = (
        {'params': params, 'config': config_state} if config_state else params
    )
    return apply_fn, params_bundle, num_interactions
