"""Regression tests for the fixed bugs.

(1) The radial transforms (``AgnesiTransform`` / ``SoftTransform`` /
    ``ZBLBasis``) index the array attributes ``_atomic_numbers`` and
    ``_covalent_radii`` with traced species indices. Both attributes are plain
    arrays (not ``nnx.Param``): a freshly built model holds them as *JAX*
    arrays (which tolerate tracer indexing), but after a model is serialized
    and reloaded through the bundle loader they come back as *NumPy* arrays.
    Indexing a NumPy array with a traced index inside a traced region
    (``jax.jit`` / ``pmap`` / ``shard_map``) raises
    ``TracerArrayConversionError``. The fix coerces both with ``jnp.asarray``.

    Note: ``mace_jax/modules/models.py`` passes ``self._atomic_numbers`` (a
    JAX/NumPy array), never a Python list -- so this only reproduces through
    the full build -> serialize -> reload -> ``jit`` path below.

(2) ``mace_jax.cli.mace_jax_from_torch._serialize_for_json`` references
    ``torch.Tensor`` / ``torch.device`` / ``torch.dtype`` / ``torch.Size``,
    but the module only imported ``torch`` *inside* ``main()``. Calling the
    function from anywhere else (e.g. ``remote_handler.scripts.mace.
    mace_utils.convert_torch_model_to_jax_bundle``) raised ``NameError``.


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


# ---------------------------------------------------------------------------
# (1) The covalent-radii transforms (``AgnesiTransform`` / ``SoftTransform`` /
#     ``ZBLBasis``) index the array attributes ``_atomic_numbers`` and
#     ``_covalent_radii`` with traced species indices. In a freshly built model
#     both are *JAX* arrays, so the indexing happens to work; after a model is
#     serialized and reloaded through the real bundle loader they come back as
#     *NumPy* arrays, and indexing a NumPy array with a traced index under
#     ``jit``/``vmap``/``pmap`` raises ``TracerArrayConversionError``
#     (radial.py: ``atomic_numbers[...]`` and ``covalent_radii[Z_u]``).
#
# This only reproduces through the full build -> serialize -> reload -> ``jit``
# path below; a directly constructed transform never produces the NumPy state.
# The test fails on the pre-fix commits and passes once both attributes are
# coerced with ``jnp.asarray``.
# ---------------------------------------------------------------------------


def _small_config(**overrides):
    config = {
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
    config.update(overrides)
    return config


@pytest.mark.parametrize(
    'overrides',
    [
        pytest.param({'distance_transform': 'Agnesi'}, id='agnesi'),
        pytest.param({'distance_transform': 'Soft'}, id='soft'),
        pytest.param({'pair_repulsion': True}, id='zbl-pair-repulsion'),
    ],
)
def test_covalent_radii_transform_survives_reload_and_jit(tmp_path, overrides):
    """A reloaded model whose radial block uses the covalent-radii transforms
    must be callable under ``jax.jit``.

    The reload is what makes ``_covalent_radii`` a NumPy array (the state that
    direct construction never produces), so this is the only way to reach the
    ``TracerArrayConversionError`` the fix addresses.
    """
    import json

    from flax import nnx, serialization

    from mace_jax.nnx_utils import state_to_serializable_dict
    from mace_jax.tools import bundle as bundle_tools
    from mace_jax.tools import model_builder

    config = _small_config(**overrides)
    config, _, _ = model_builder._normalize_atomic_config(config)

    model = model_builder._build_jax_model(config, rngs=nnx.Rngs(0))
    _, state = nnx.split(model)
    payload = state_to_serializable_dict(state)

    (tmp_path / 'config.json').write_text(json.dumps(config))
    (tmp_path / 'params.msgpack').write_bytes(serialization.to_bytes(payload))

    bundle = bundle_tools.load_model_bundle(str(tmp_path), dtype='float64')
    template = model_builder._prepare_template_data(config)

    @jax.jit
    def energy(data):
        out, _ = bundle.graphdef.apply(bundle.params)(
            data, compute_force=False, compute_stress=False
        )
        return out['energy']

    result = energy(template)
    assert jnp.isfinite(result).all()
