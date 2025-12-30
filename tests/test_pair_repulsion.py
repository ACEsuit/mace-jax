from pathlib import Path

import flax.core as flax_core
import jax
import jax.numpy as jnp
import numpy as np
from ase.io import read
from e3nn_jax import Irreps

from mace_jax import modules
from mace_jax.data.utils import (
    Configuration,
    config_from_atoms,
    graph_from_configuration,
)
from mace_jax.modules import utils as modules_utils
from mace_jax.tools import gin_model


def _make_configuration(distance: float) -> Configuration:
    positions = np.array([[0.0, 0.0, 0.0], [distance, 0.0, 0.0]], dtype=float)
    return Configuration(
        atomic_numbers=np.array([29, 29], dtype=int),
        positions=positions,
        energy=np.array(0.0),
        forces=np.zeros_like(positions),
        stress=np.zeros((3, 3)),
        cell=np.eye(3) * 20.0,
        pbc=(False, False, False),
    )


def _graph_to_data(distance: float, cutoff: float = 6.0) -> dict[str, jnp.ndarray]:
    config = _make_configuration(distance)
    graph = graph_from_configuration(config, cutoff=cutoff)
    return gin_model._graph_to_data(graph, num_species=1)


def _build_mace_model(
    *,
    pair_repulsion: bool,
    distance_transform: str = 'None',
    avg_num_neighbors: float = 1.0,
):
    return modules.ScaleShiftMACE(
        r_max=6.0,
        num_bessel=2,
        num_polynomial_cutoff=2,
        max_ell=1,
        interaction_cls=modules.interaction_classes[
            'RealAgnosticResidualInteractionBlock'
        ],
        interaction_cls_first=modules.interaction_classes[
            'RealAgnosticResidualInteractionBlock'
        ],
        num_interactions=1,
        num_elements=1,
        hidden_irreps=Irreps('1x0e'),
        MLP_irreps=Irreps('1x0e'),
        atomic_energies=np.zeros((1,), dtype=np.float64),
        avg_num_neighbors=avg_num_neighbors,
        atomic_numbers=(29,),
        correlation=1,
        gate=None,
        pair_repulsion=pair_repulsion,
        distance_transform=distance_transform,
        atomic_inter_scale=np.asarray(1.0),
        atomic_inter_shift=np.asarray(0.0),
    )


def _fill_params(params, value: float):
    """Recursively fill numeric leaves with a constant for deterministic comparisons."""
    if isinstance(params, dict):
        return {k: _fill_params(v, value) for k, v in params.items()}
    if isinstance(params, (list, tuple)):
        return type(params)(_fill_params(v, value) for v in params)
    if isinstance(params, (np.ndarray, jnp.ndarray)):
        return jnp.ones_like(params) * value
    return params


def _energy_from_model(model, params, data) -> float:
    outputs = model.apply(
        params,
        data,
        compute_force=False,
        compute_stress=False,
    )
    return float(np.asarray(outputs['energy'])[0])


def test_pair_repulsion_raises_close_contact_energy():
    """Pair repulsion should penalise close-contact Cu dimers compared to baseline."""
    close_data = _graph_to_data(1.0)
    far_data = _graph_to_data(3.0)

    model_no_rep = _build_mace_model(pair_repulsion=False)
    params_no_rep = model_no_rep.init(jax.random.PRNGKey(0), close_data)

    model_with_rep = _build_mace_model(pair_repulsion=True)
    params_with_rep = model_with_rep.init(jax.random.PRNGKey(0), close_data)

    energy_close_no = _energy_from_model(model_no_rep, params_no_rep, close_data)
    energy_far_no = _energy_from_model(model_no_rep, params_no_rep, far_data)

    energy_close_with = _energy_from_model(model_with_rep, params_with_rep, close_data)
    energy_far_with = _energy_from_model(model_with_rep, params_with_rep, far_data)

    baseline_delta = energy_close_no - energy_far_no
    rep_delta = energy_close_with - energy_far_with

    assert rep_delta > baseline_delta + 1e-3
    assert energy_close_with > energy_far_with


def test_pair_repulsion_contributes_to_interaction_energy():
    """Pair repulsion should raise the interaction energy for close contacts."""
    close_data = _graph_to_data(1.0)
    far_data = _graph_to_data(3.0)

    base_kwargs = dict(distance_transform='None')

    model_no_rep = _build_mace_model(pair_repulsion=False, **base_kwargs)
    params_no_rep_close = model_no_rep.init(jax.random.PRNGKey(0), close_data)
    params_no_rep_far = model_no_rep.init(jax.random.PRNGKey(1), far_data)

    model_with_rep = _build_mace_model(pair_repulsion=True, **base_kwargs)
    params_with_rep_close = model_with_rep.init(jax.random.PRNGKey(0), close_data)
    params_with_rep_far = model_with_rep.init(jax.random.PRNGKey(1), far_data)

    out_close_no = model_no_rep.apply(
        params_no_rep_close, close_data, compute_force=False, compute_stress=False
    )
    out_far_no = model_no_rep.apply(
        params_no_rep_far, far_data, compute_force=False, compute_stress=False
    )
    out_close_rep = model_with_rep.apply(
        params_with_rep_close, close_data, compute_force=False, compute_stress=False
    )
    out_far_rep = model_with_rep.apply(
        params_with_rep_far, far_data, compute_force=False, compute_stress=False
    )

    inter_close_no = float(np.asarray(out_close_no['interaction_energy'])[0])
    inter_far_no = float(np.asarray(out_far_no['interaction_energy'])[0])
    inter_close_rep = float(np.asarray(out_close_rep['interaction_energy'])[0])
    inter_far_rep = float(np.asarray(out_far_rep['interaction_energy'])[0])

    # Pair repulsion should elevate interaction energy at close distance, but not far.
    assert inter_close_rep > inter_close_no
    assert inter_far_rep >= inter_far_no  # far case should be negligible or equal


def test_distance_transform_changes_radial_embedding():
    """Agnesi distance transform should alter radial embeddings on a real aspirin graph."""
    from mace_jax.modules.blocks import RadialEmbeddingBlock  # noqa: PLC0415

    dataset_path = (
        Path(__file__).resolve().parents[1] / 'data' / 'rmd17_aspirin_train.xyz'
    )
    atoms = read(dataset_path.as_posix(), index=0)
    config = config_from_atoms(atoms)
    graph = graph_from_configuration(config, cutoff=3.0)
    unique_species = np.unique(config.atomic_numbers)
    num_species = len(unique_species)
    data = gin_model._graph_to_data(graph, num_species=num_species)

    positions = data['positions']
    edge_index = data['edge_index']
    shifts = data['shifts']
    node_attrs = data['node_attrs']
    atomic_numbers = jnp.asarray(unique_species, dtype=jnp.int32)

    _, lengths = modules_utils.get_edge_vectors_and_lengths(
        positions=positions,
        edge_index=edge_index,
        shifts=shifts,
        normalize=False,
    )

    key = jax.random.PRNGKey(42)

    block_none = RadialEmbeddingBlock(
        r_max=3.0,
        num_bessel=3,
        num_polynomial_cutoff=2,
        radial_type='bessel',
        distance_transform='None',
        apply_cutoff=True,
    )
    params_none = block_none.init(
        key,
        lengths,
        node_attrs,
        edge_index,
        atomic_numbers,
    )
    out_none, _ = block_none.apply(
        params_none,
        lengths,
        node_attrs,
        edge_index,
        atomic_numbers,
    )

    block_agn = RadialEmbeddingBlock(
        r_max=3.0,
        num_bessel=3,
        num_polynomial_cutoff=2,
        radial_type='bessel',
        distance_transform='Agnesi',
        apply_cutoff=True,
    )
    params_agn = block_agn.init(
        key,
        lengths,
        node_attrs,
        edge_index,
        atomic_numbers,
    )
    out_agn, _ = block_agn.apply(
        params_agn,
        lengths,
        node_attrs,
        edge_index,
        atomic_numbers,
    )

    assert out_none.shape == out_agn.shape
    assert not np.allclose(np.asarray(out_none), np.asarray(out_agn))
