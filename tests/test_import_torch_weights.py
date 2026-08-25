from __future__ import annotations

import types

import jax.numpy as jnp
import pytest
import torch
from flax import nnx

from mace_jax.cli import mace_jax_from_torch as jax_from_torch
from mace_jax.nnx_config import ConfigVar
from mace_jax.nnx_utils import state_to_pure_dict
from mace_jax.tools import model_builder
from mace_jax.tools.import_from_torch import _extract_norm_consts


def _patch_common(monkeypatch, jax_model):
    monkeypatch.setattr(
        jax_from_torch, '_build_jax_model', lambda config, **kwargs: jax_model
    )
    monkeypatch.setattr(
        jax_from_torch, '_prepare_template_data', lambda config: {'dummy': 1}
    )


class _DummyJaxModel(nnx.Module):
    def __init__(
        self, use_reduced_cg: bool, import_impl, *, rngs: nnx.Rngs | None = None
    ):
        self.use_reduced_cg = use_reduced_cg
        self._import_impl = import_impl
        self.w = nnx.Param(jnp.ones((1,), dtype=jnp.float32))
        self._normalize2mom_consts_var = ConfigVar(
            {
                'sigmoid': jnp.asarray(0.0, dtype=jnp.float32),
                'silu': jnp.asarray(0.0, dtype=jnp.float32),
                'swish': jnp.asarray(0.0, dtype=jnp.float32),
            }
        )

    def import_from_torch(self, torch_model, variables):
        return self._import_impl(torch_model, variables)


def test_convert_model_rejects_reduced_cg_mismatch(monkeypatch):
    dummy_jax = _DummyJaxModel(
        use_reduced_cg=True,
        import_impl=lambda torch_model, variables: variables,
    )
    _patch_common(monkeypatch, dummy_jax)
    torch_model = types.SimpleNamespace(use_reduced_cg=False)

    with pytest.raises(ValueError, match='use_reduced_cg'):
        jax_from_torch.convert_model(torch_model, {})


def test_convert_model_detects_unimported_parameters(monkeypatch):
    def _return_nan(_, variables):
        return variables  # NaNs remain -> trigger check

    dummy_jax = _DummyJaxModel(use_reduced_cg=True, import_impl=_return_nan)
    _patch_common(monkeypatch, dummy_jax)
    torch_model = types.SimpleNamespace(use_reduced_cg=True)

    with pytest.raises(ValueError, match='still NaN'):
        jax_from_torch.convert_model(torch_model, {})


def test_convert_model_success(monkeypatch):
    def _populate_params(_, variables):
        variables['w'] = jnp.array([42.0], dtype=jnp.float32)
        return variables

    dummy_jax = _DummyJaxModel(use_reduced_cg=True, import_impl=_populate_params)
    _patch_common(monkeypatch, dummy_jax)
    torch_model = types.SimpleNamespace(use_reduced_cg=True)

    graphdef, state, template = jax_from_torch.convert_model(torch_model, {'cfg': 1})

    assert graphdef is not None
    assert template == {'dummy': 1}
    params = state_to_pure_dict(state)
    assert jnp.array_equal(params['w'], jnp.array([42.0], dtype=jnp.float32))


def test_extracts_plain_gated_block_activation_constants():
    gate_constant = 1.8467055342154763

    class _PlainGate(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self._act_scalar = torch.nn.functional.silu
            self._scalar_cst = float(_extract_norm_consts()['silu'])
            self._act_gate = torch.sigmoid
            self._gate_cst = gate_constant

    constants = _extract_norm_consts(_PlainGate())

    assert constants['sigmoid'] == gate_constant
    assert constants['silu'] == constants['swish']


def test_convert_model_exports_checkpoint_gate_constant(monkeypatch):
    gate_constant = 1.8467055342154763

    class _PlainGate(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.use_reduced_cg = True
            self._act_gate = torch.sigmoid
            self._gate_cst = gate_constant

    def _populate_params(_, variables):
        variables['w'] = jnp.array([42.0], dtype=jnp.float32)
        return variables

    dummy_jax = _DummyJaxModel(
        use_reduced_cg=True,
        import_impl=_populate_params,
    )
    _patch_common(monkeypatch, dummy_jax)
    config = {}

    jax_from_torch.convert_model(_PlainGate(), config)

    assert config['normalize2mom_consts']['sigmoid'] == pytest.approx(gate_constant)


def test_rejects_conflicting_model_activation_constants():
    class _Gate(torch.nn.Module):
        def __init__(self, constant):
            super().__init__()
            self._act_gate = torch.sigmoid
            self._gate_cst = constant

    model = torch.nn.ModuleList([_Gate(1.0), _Gate(2.0)])

    with pytest.raises(ValueError, match='Conflicting normalize2mom constants'):
        _extract_norm_consts(model)


def test_build_jax_model_forwards_mh1_specific_config(monkeypatch):
    captured = {}

    class _SentinelModel:
        pass

    def _fake_scaleshiftmace(*, rngs, **kwargs):
        del rngs
        captured.update(kwargs)
        return _SentinelModel()

    monkeypatch.setattr(model_builder, 'ScaleShiftMACE', _fake_scaleshiftmace)

    config = {
        'r_max': 5.0,
        'num_bessel': 8,
        'num_polynomial_cutoff': 5,
        'max_ell': 3,
        'interaction_cls': 'RealAgnosticResidualNonLinearInteractionBlock',
        'interaction_cls_first': 'RealAgnosticResidualNonLinearInteractionBlock',
        'num_interactions': 2,
        'hidden_irreps': '512x0e + 512x1o',
        'MLP_irreps': '16x0e',
        'atomic_numbers': [1, 6],
        'atomic_energies': [0.0, 0.0],
        'avg_num_neighbors': 12.0,
        'correlation': 3,
        'radial_type': 'bessel',
        'distance_transform': 'Agnesi',
        'use_so3': False,
        'use_reduced_cg': False,
        'use_edge_irreps_first': True,
        'use_agnostic_product': True,
        'use_last_readout_only': False,
        'use_embedding_readout': False,
        'edge_irreps': '128x0e + 128x1o',
        'readout_cls': 'NonLinearReadoutBlock',
        'gate': 'silu',
        'heads': [
            'matpes_r2scan',
            'mp_pbe_refit_add',
            'spice_wB97M',
            'oc20_usemppbe',
            'omol',
            'omat_pbe',
        ],
        'atomic_inter_scale': 1.0,
        'atomic_inter_shift': 0.0,
    }

    model = model_builder._build_jax_model(config)

    assert isinstance(model, _SentinelModel)
    assert captured['use_edge_irreps_first'] is True
    assert captured['heads'] == tuple(config['heads']) or captured['heads'] == config['heads']
