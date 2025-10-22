import flax.linen as fnn
import jax
import jax.numpy as jnp
import numpy as np
import torch
from e3nn_jax import Irreps
from mace.modules.blocks import (
    LinearNodeEmbeddingBlock as TorchLinearNodeEmbeddingBlock,
)
from mace.modules.blocks import (
    ScaleShiftBlock as TorchScaleShiftBlock,
)

from mace_jax.adapters.flax.torch import init_from_torch
from mace_jax.modules.blocks import (
    LinearNodeEmbeddingBlock as FlaxLinearNodeEmbeddingBlock,
)
from mace_jax.modules.blocks import (
    ScaleShiftBlock as FlaxScaleShiftBlock,
)
from mace_jax.modules.utils import (
    add_output_interface,
    compute_forces_and_stress,
)


def _np(array):
    if isinstance(array, torch.Tensor):
        return array.detach().cpu().numpy()
    return np.asarray(array)


class TestScaleShiftImport:
    def test_matches_torch(self):
        scale = np.array([1.25, -0.35], dtype=np.float32)
        shift = np.array([0.5, -0.2], dtype=np.float32)

        torch_module = TorchScaleShiftBlock(scale=scale, shift=shift).float().eval()

        flax_module = FlaxScaleShiftBlock(scale=scale, shift=shift)
        flax_module, variables = init_from_torch(
            flax_module,
            torch_module,
            jax.random.PRNGKey(0),
            jnp.asarray([0.0, 0.0], dtype=jnp.float32),
            jnp.asarray([0, 1], dtype=jnp.int32),
        )

        x_np = np.array([0.8, -1.2], dtype=np.float32)
        head_np = np.array([0, 1], dtype=np.int32)
        torch_out = torch_module(
            torch.tensor(x_np, dtype=torch.float32),
            torch.tensor(head_np, dtype=torch.long),
        )
        flax_out = flax_module.apply(
            variables,
            jnp.asarray(x_np),
            jnp.asarray(head_np),
        )

        np.testing.assert_allclose(_np(flax_out), _np(torch_out), atol=1e-6, rtol=1e-6)


class TestLinearEmbeddingImport:
    def test_matches_torch(self):
        irreps_in = Irreps('1x0e + 1x1o')
        irreps_out = Irreps('1x0e + 1x1o')

        torch_module = TorchLinearNodeEmbeddingBlock(
            str(irreps_in), str(irreps_out)
        ).float()
        torch_module.eval()

        flax_module = FlaxLinearNodeEmbeddingBlock(
            irreps_in=irreps_in, irreps_out=irreps_out
        )

        rng = jax.random.PRNGKey(123)
        x_np = (
            np.random.default_rng(0)
            .standard_normal((5, irreps_in.dim))
            .astype(np.float32)
        )

        flax_module, variables = init_from_torch(
            flax_module,
            torch_module,
            rng,
            jnp.asarray(x_np),
        )

        torch_out = torch_module(torch.tensor(x_np, dtype=torch.float32))
        flax_out = flax_module.apply(variables, jnp.asarray(x_np))

        np.testing.assert_allclose(_np(flax_out), _np(torch_out), atol=1e-5, rtol=1e-5)


@add_output_interface
class _HookeModel(fnn.Module):
    """Simple quadratic energy to exercise get_outputs."""

    stiffness: float = 2.0

    def __call__(self, data):
        energy = 0.5 * self.stiffness * jnp.sum(data['positions'] ** 2, axis=-1)
        return jnp.array([jnp.sum(energy)])


class TestOutputInterface:
    @staticmethod
    def _graph():
        return {
            'positions': jnp.array([[0.3, -0.2, 0.1]], dtype=jnp.float32),
            'cell': jnp.eye(3, dtype=jnp.float32).reshape(1, 3, 3),
            'unit_shifts': jnp.zeros((0, 3), dtype=jnp.float32),
            'edge_index': jnp.zeros((2, 0), dtype=jnp.int32),
            'batch': jnp.array([0], dtype=jnp.int32),
            'ptr': jnp.array([0, 1], dtype=jnp.int32),
            'shifts': jnp.zeros((0, 3), dtype=jnp.float32),
        }

    def test_forces_and_stress_paths(self):
        model = _HookeModel(stiffness=1.5)
        data = self._graph()

        variables = model.init(jax.random.PRNGKey(0), data)

        outputs = model.apply(variables, data, compute_force=True, compute_stress=False)
        expected_forces = -model.stiffness * data['positions']
        np.testing.assert_allclose(
            _np(outputs['forces']), _np(expected_forces), atol=1e-6, rtol=1e-6
        )
        assert outputs['stress'] is None

        outputs_stress = model.apply(
            variables, data, compute_force=False, compute_stress=True
        )
        assert outputs_stress['forces'] is not None
        stress = _np(outputs_stress['stress'])

        def energy_core(pos, shifts=None, cell_override=None):
            new_data = dict(data)
            new_data['positions'] = pos
            if shifts is not None:
                new_data['shifts'] = shifts
            if cell_override is not None:
                new_data['cell'] = cell_override
            return model.apply(variables, new_data, method=model._energy_fn)

        forces_direct, stress_direct = compute_forces_and_stress(
            energy_core,
            data['positions'],
            data['cell'],
            data['unit_shifts'],
            data['edge_index'],
            data['batch'],
            int(data['ptr'].shape[0] - 1),
        )

        np.testing.assert_allclose(
            _np(outputs_stress['forces']),
            _np(forces_direct),
            atol=1e-6,
            rtol=1e-6,
        )
        np.testing.assert_allclose(
            stress,
            _np(stress_direct),
            atol=1e-6,
            rtol=1e-6,
        )

    def test_jitted_apply_handles_optional_outputs(self):
        model = _HookeModel(stiffness=1.5)
        data = self._graph()
        variables = model.init(jax.random.PRNGKey(0), data)

        compiled_apply = jax.jit(model.apply)

        energy_only = compiled_apply(
            variables,
            data,
            compute_force=False,
            compute_stress=False,
        )
        assert 'forces_mask' in energy_only
        assert 'stress_mask' in energy_only
        assert not bool(np.asarray(energy_only['forces_mask']))
        assert not bool(np.asarray(energy_only['stress_mask']))
        assert energy_only['forces'].shape == data['positions'].shape

        full_outputs = compiled_apply(
            variables,
            data,
            compute_force=True,
            compute_stress=True,
        )
        assert bool(np.asarray(full_outputs['forces_mask']))
        assert bool(np.asarray(full_outputs['stress_mask']))
        expected_forces = -model.stiffness * data['positions']
        np.testing.assert_allclose(
            _np(full_outputs['forces']),
            _np(expected_forces),
            atol=1e-6,
            rtol=1e-6,
        )
