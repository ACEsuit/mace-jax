import json

import numpy as np
from flax import nnx, serialization

from mace_jax.nnx_utils import state_to_serializable_dict
from mace_jax.tools import bundle as bundle_tools
from mace_jax.tools import model_builder


def _small_config():
    return {
        'r_max': 4.0,
        'num_bessel': 2,
        'num_polynomial_cutoff': 2,
        'max_ell': 1,
        'interaction_cls': 'RealAgnosticInteractionBlock',
        'interaction_cls_first': 'RealAgnosticInteractionBlock',
        'num_interactions': 1,
        'num_elements': 2,
        'hidden_irreps': '2x0e',
        'edge_irreps': None,
        'MLP_irreps': '2x0e',
        'atomic_numbers': [1, 8],
        'atomic_energies': [0.0, 0.0],
        'avg_num_neighbors': 1.0,
        'correlation': 1,
        'radial_type': 'bessel',
        'pair_repulsion': False,
        'distance_transform': None,
        'use_so3': False,
        'use_reduced_cg': True,
        'use_agnostic_product': False,
        'use_last_readout_only': False,
        'use_embedding_readout': False,
        'gate': 'silu',
        'apply_cutoff': True,
    }


def test_bundle_roundtrip(tmp_path):
    config = _small_config()
    config, _, _ = model_builder._normalize_atomic_config(config)
    model = model_builder._build_jax_model(config, rngs=nnx.Rngs(0))
    graphdef, state = nnx.split(model)
    params_payload = state_to_serializable_dict(state)

    config_path = tmp_path / 'config.json'
    params_path = tmp_path / 'params.msgpack'
    config_path.write_text(json.dumps(config))
    params_path.write_bytes(serialization.to_bytes(params_payload))

    bundle = bundle_tools.load_model_bundle(str(tmp_path), dtype='float64')

    assert '_normalize2mom_consts_var' in bundle.params
    template = model_builder._prepare_template_data(config)

    expected, _ = graphdef.apply(state)(
        template,
        compute_force=False,
        compute_stress=False,
    )
    actual, _ = bundle.graphdef.apply(bundle.params)(
        template,
        compute_force=False,
        compute_stress=False,
    )

    np.testing.assert_allclose(
        np.asarray(actual['energy']),
        np.asarray(expected['energy']),
        rtol=1e-6,
        atol=1e-6,
    )
