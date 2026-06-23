"""Regression tests for the two bugs.

(1) ``AgnesiTransform.__call__`` indexes the ``atomic_numbers`` argument
    (a Python list arriving from the model config) with a traced JAX index.
    Inside a traced region (``jax.jit`` / ``pmap`` / ``shard_map``) that
    Python-list ``__getitem__`` calls ``__array__`` on the tracer and raises
    ``TracerArrayConversionError``.

(2) ``mace_jax.cli.mace_jax_from_torch._serialize_for_json`` references
    ``torch.Tensor`` / ``torch.device`` / ``torch.dtype`` / ``torch.Size``,
    but the module only imports ``torch`` *inside* ``main()``. Calling the
    function from anywhere else (e.g. ``remote_handler.scripts.mace.
    mace_utils.convert_torch_model_to_jax_bundle``) raises ``NameError``.


Reproduce via:
```bash
conda create --name mace_jax_fix python=3.13
conda activate mace_jax_fix
pip install -e .[jax-cuda12,torch-cuda12,test]
pytest test/test_bugs.py
```
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest


def test_agnesi_transform_traced_atomic_numbers():
    """The transform must work when its caller passes ``atomic_numbers`` as a
    Python list (which is what ``mace_jax/modules/models.py`` does) and
    ``node_attrs_index`` as a traced array (which is what happens under
    ``jit``/``pmap``).
    """
    from mace_jax.modules.radial import AgnesiTransform

    transform = AgnesiTransform(trainable=False)

    # Two atoms in a single edge, two distinct species (Z=50 Sn, Z=8 O).
    edge_lengths = jnp.array([[2.5]], dtype=jnp.float32)
    node_attrs = jnp.array([[1.0, 0.0], [0.0, 1.0]], dtype=jnp.float32)
    edge_index = jnp.array([[0], [1]], dtype=jnp.int32)
    node_attrs_index = jnp.array([0, 1], dtype=jnp.int32)

    # ``atomic_numbers`` arrives as a Python list / tuple from the config --
    # NOT a jnp.array. This is the exact shape of the bug.
    atomic_numbers = [50, 8]

    @jax.jit
    def f(x, na, ei, idx):
        return transform(
            x, na, ei,
            atomic_numbers=atomic_numbers,
            node_attrs_index=idx,
        )

    out = f(edge_lengths, node_attrs, edge_index, node_attrs_index)
    assert out.shape == (1, 1)
    assert jnp.isfinite(out).all()


def test_serialize_for_json_torch_namespace():
    """The serializer must be callable as a library function (i.e. without
    going through ``main()``, which is the only place ``torch`` is imported).
    """
    torch = pytest.importorskip("torch")
    from mace_jax.cli.mace_jax_from_torch import _serialize_for_json

    # The torch.Tensor branch (line ~103) is the most direct trigger; if torch
    # is bound in the module namespace this returns a list, otherwise NameError.
    result = _serialize_for_json(torch.tensor([1.0, 2.0, 3.0]))
    assert result == [1.0, 2.0, 3.0]

    # dtype branch -- separate code path that ALSO needs ``torch``.
    assert _serialize_for_json(torch.float32) == "float32"
