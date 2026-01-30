import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mace_jax.modules import utils as models_utils
from mace_jax.tools import lammps_exchange


class _DummyLAMMPS:
    """Minimal LAMMPS stub for forward-exchange tests."""

    def __init__(self):
        self.calls = 0

    def forward_exchange(self, src, dst, vec_len):
        self.calls += 1
        np.copyto(dst, src[::-1])


def test_forward_exchange_requires_lammps_interface():
    feats = jnp.arange(6, dtype=jnp.float64).reshape(3, 2)

    class _Noop:
        pass

    with pytest.raises(AttributeError):
        _ = lammps_exchange.forward_exchange(feats, _Noop())


def test_apply_lammps_exchange_forwards_to_lammps():
    feats = jnp.array([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float64)
    dummy = _DummyLAMMPS()

    @jax.jit
    def _wrapped(x):
        return lammps_exchange.forward_exchange(x, dummy)

    result = _wrapped(feats)
    expected = np.asarray([[3.0, 4.0], [1.0, 2.0]])
    np.testing.assert_allclose(np.asarray(result), expected)
    assert dummy.calls == 1


def test_prepare_graph_carries_lammps_metadata():
    batch = {
        'node_attrs': jnp.eye(2, dtype=jnp.float64),
        'vectors': jnp.ones((3, 3), dtype=jnp.float64),
        'batch': jnp.array([0, 0], dtype=jnp.int32),
        'edge_index': jnp.array([[0, 1, 0], [1, 0, 1]], dtype=jnp.int32),
        'natoms': (2, 1),
        'ptr': jnp.array([0, 2], dtype=jnp.int32),
        'positions': jnp.zeros((2, 3), dtype=jnp.float64),
        'shifts': jnp.zeros((3, 3), dtype=jnp.float64),
        'cell': jnp.zeros((2, 3, 3), dtype=jnp.float64),
    }
    dummy = _DummyLAMMPS()
    ctx = models_utils.prepare_graph(batch, lammps_mliap=True, lammps_class=dummy)
    assert ctx.is_lammps
    assert ctx.interaction_kwargs.lammps_class is dummy
    assert ctx.interaction_kwargs.lammps_natoms == (2, 1)


def test_forward_exchange_prefers_ffi_when_available(monkeypatch):
    feats = jnp.array([[1.0, 0.0], [0.0, 1.0]], dtype=jnp.float32)

    class _PairHandleStub:
        pass

    lammps_stub = _PairHandleStub()
    lammps_stub._mace_jax_pair_handle = 42

    monkeypatch.setattr(lammps_exchange, '_HAS_LAMMPS_JAX', True)

    calls: dict[str, tuple] = {}

    def _fake_forward(x, handle):
        calls['args'] = (np.asarray(x), handle)
        return x + 1.0

    monkeypatch.setattr(lammps_exchange, '_ffi_forward_exchange', _fake_forward)

    out = lammps_exchange.forward_exchange(feats, lammps_stub)
    np.testing.assert_allclose(np.asarray(out), np.asarray(feats + 1.0))
    assert calls['args'][1] == 42
