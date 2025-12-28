from __future__ import annotations

import types

import jax.numpy as jnp
import pytest

from mace_jax.cli import mace_jax_from_torch as jax_from_torch


def _patch_common(monkeypatch, jax_model):
    monkeypatch.setattr(jax_from_torch, '_build_jax_model', lambda config: jax_model)
    monkeypatch.setattr(
        jax_from_torch, '_prepare_template_data', lambda config: {'dummy': 1}
    )
    monkeypatch.setattr(jax_from_torch.jax.random, 'PRNGKey', lambda seed: 0)


class _DummyJaxModel:
    def __init__(self, use_reduced_cg: bool, import_impl):
        self.use_reduced_cg = use_reduced_cg
        self._import_impl = import_impl

    def init(self, rng, template):
        del rng, template
        return {'params': {'w': jnp.ones((1,), dtype=jnp.float32)}}

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
        variables['params']['w'] = jnp.array([42.0], dtype=jnp.float32)
        return variables

    dummy_jax = _DummyJaxModel(use_reduced_cg=True, import_impl=_populate_params)
    _patch_common(monkeypatch, dummy_jax)
    torch_model = types.SimpleNamespace(use_reduced_cg=True)

    model, variables, template = jax_from_torch.convert_model(torch_model, {'cfg': 1})

    assert model is dummy_jax
    assert template == {'dummy': 1}
    assert jnp.array_equal(
        variables['params']['w'], jnp.array([42.0], dtype=jnp.float32)
    )
