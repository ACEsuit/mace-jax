from __future__ import annotations

import types

import jax.numpy as jnp
import pytest
from flax import nnx

from mace_jax.cli import mace_jax_from_torch as jax_from_torch
from mace_jax.nnx_utils import state_to_pure_dict


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
