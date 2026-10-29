import json
import pickle

import jax
import numpy as np
import pytest
from flax import nnx, serialization

from mace_jax.adapters.e3nn.math import _normalize_activation as normalization
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


@pytest.mark.parametrize(
    'file_format', ['bundle', 'checkpoint_bytes', 'checkpoint_dict']
)
@pytest.mark.parametrize('legacy', [False, True])
def test_gated_bundle_preserves_saved_normalization(
    tmp_path, monkeypatch, file_format, legacy
):
    config = _small_config()
    config.update(
        interaction_cls='RealAgnosticResidualNonLinearInteractionBlock',
        interaction_cls_first='RealAgnosticResidualNonLinearInteractionBlock',
        num_interactions=2,
        hidden_irreps='2x0e + 2x1o',
        radial_MLP=[4],
    )
    # Reproduce the old loader's JAX fallback independently of the migration,
    # including when another imported model has changed the global registry.
    monkeypatch.delitem(normalization._CONST_OVERRIDES, 'sigmoid', raising=False)
    legacy_sigmoid = float(
        normalization.normalize2mom(jax.nn.sigmoid)._normalize2mom_const
    )
    config['normalize2mom_consts'] = {
        'silu': 1.7,
        'swish': 1.7,
        'sigmoid': legacy_sigmoid if legacy else 1.9,
    }
    model = model_builder._build_jax_model(config, rngs=nnx.Rngs(0))
    graphdef, state = nnx.split(model)
    data = model_builder._prepare_template_data(config)
    expected, _ = graphdef.apply(state)(data, compute_force=True, compute_stress=True)
    payload = state_to_serializable_dict(state)
    if legacy:
        payload['_normalize2mom_consts_var'].pop('sigmoid')
    # The saved state must also work when model_config has no normalization data.
    config.pop('normalize2mom_consts')
    if file_format == 'bundle':
        (tmp_path / 'config.json').write_text(json.dumps(config))
        (tmp_path / 'params.msgpack').write_bytes(serialization.to_bytes(payload))
        path = tmp_path
    else:
        path = tmp_path / 'model.ckpt'
        if file_format == 'checkpoint_bytes':
            payload = serialization.to_bytes(payload)
        path.write_bytes(pickle.dumps({'model_config': config, 'state': payload}))

    normalization.register_normalize2mom_const('sigmoid', 3.0)
    bundle = bundle_tools.load_model_bundle(str(path), dtype='float64')
    actual, _ = bundle.graphdef.apply(bundle.params)(
        data, compute_force=True, compute_stress=True
    )
    for key in ('energy', 'forces', 'stress'):
        np.testing.assert_allclose(actual[key], expected[key], rtol=1e-10, atol=1e-10)
    assert bundle.config['normalize2mom_consts']['sigmoid'] == pytest.approx(
        legacy_sigmoid if legacy else 1.9, rel=1e-12
    )


def test_bundle_still_rejects_missing_weights(tmp_path):
    config = _small_config()
    model = model_builder._build_jax_model(config, rngs=nnx.Rngs(0))
    payload = state_to_serializable_dict(nnx.state(model))
    payload['_normalize2mom_consts_var'].pop('sigmoid')
    payload.pop('node_embedding')
    (tmp_path / 'config.json').write_text(json.dumps(config))
    (tmp_path / 'params.msgpack').write_bytes(serialization.to_bytes(payload))
    with pytest.raises(ValueError, match='node_embedding'):
        bundle_tools.load_model_bundle(str(tmp_path), dtype='float64')
