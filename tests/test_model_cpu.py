"""Small end-to-end Torch conversion checks that do not require CUDA."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from e3nn import o3
from mace.modules import blocks, models

from mace_jax.cli.mace_jax_from_torch import convert_model
from mace_jax.tools.import_from_torch import extract_config_mace_model


@pytest.mark.parametrize('radial_type', ['bessel', 'gaussian'])
def test_converted_multihead_model_matches_torch_on_cpu(radial_type):
    torch.manual_seed(10)
    config = dict(
        r_max=3.0,
        num_bessel=3,
        num_polynomial_cutoff=5,
        max_ell=1,
        interaction_cls=blocks.RealAgnosticResidualInteractionBlock,
        interaction_cls_first=blocks.RealAgnosticInteractionBlock,
        num_interactions=2,
        num_elements=2,
        hidden_irreps=o3.Irreps('2x0e+2x1o'),
        MLP_irreps=o3.Irreps('2x0e'),
        atomic_energies=np.array([[-1.0, -2.0], [-3.0, -4.0]]),
        avg_num_neighbors=2.0,
        atomic_numbers=[1, 8],
        correlation=2,
        gate=torch.nn.functional.silu,
        atomic_inter_scale=[1.2, 0.8],
        atomic_inter_shift=[0.3, -0.2],
        use_reduced_cg=False,
        heads=['first', 'second'],
        radial_type=radial_type,
    )
    model = models.ScaleShiftMACE(**config)
    config = extract_config_mace_model(model)
    assert config['num_bessel'] == 3
    assert config['radial_type'] == radial_type
    if radial_type == 'gaussian':
        assert not hasattr(model.radial_embedding.bessel_fn, 'bessel_weights')
    config['torch_model_class'] = 'ScaleShiftMACE'
    graphdef, state, _ = convert_model(model, config)
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.2, 0.0], [0.2, 1.0, 0.1]])
    edges = np.array([[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]])
    data = {
        'positions': np.concatenate([positions, 1.1 * positions]),
        'node_attrs': np.tile([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]], (2, 1)),
        'edge_index': np.concatenate([edges, edges + 3], axis=1),
        'shifts': np.zeros((12, 3)),
        'unit_shifts': np.zeros((12, 3)),
        'cell': np.tile(4 * np.eye(3)[None], (2, 1, 1)),
        'batch': np.repeat(np.arange(2), 3),
        'ptr': np.array([0, 3, 6]),
        'head': np.array([0, 1]),
    }
    expected = model(
        {key: torch.tensor(value) for key, value in data.items()}, compute_stress=True
    )

    @jax.jit
    def predict(state, data):
        return graphdef.apply(state)(data, compute_stress=True)[0]

    actual = predict(state, {key: jnp.asarray(value) for key, value in data.items()})
    for key in ('energy', 'forces', 'stress', 'node_energy'):
        np.testing.assert_allclose(
            actual[key],
            expected[key].detach().numpy(),
            rtol=1e-5,
            atol=1e-7,
            err_msg=key,
        )
