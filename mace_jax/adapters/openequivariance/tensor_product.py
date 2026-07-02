"""Strict OpenEquivariance tensor-product adapters."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from e3nn_jax import Irreps
from flax import nnx

from mace_jax.adapters.nnx.torch import nxx_auto_import_from_torch

from .problem import build_tp_problem, weight_permutation


@nxx_auto_import_from_torch(allow_missing_mapper=True)
class TensorProduct(nnx.Module):
    """OpenEquivariance TP, optionally fused with graph aggregation."""

    def __init__(
        self,
        irreps_in1: Irreps,
        irreps_in2: Irreps,
        irreps_out: Irreps,
        shared_weights: bool = False,
        internal_weights: bool = False,
        instructions=None,
        *,
        conv_fusion: bool = True,
        layout: str = 'mul_ir',
        group: str = 'O3_e3nn',
        rngs: nnx.Rngs | None = None,
        _unsafe_allow_standalone_for_diagnostics: bool = False,
    ) -> None:
        if not conv_fusion and not _unsafe_allow_standalone_for_diagnostics:
            raise ValueError(
                'Standalone OpenEquivariance tensor products are currently '
                'unsupported because the OpenEquivariance 0.6.8 backward path '
                'is unsafe for general multi-problem execution.'
            )
        if internal_weights and not shared_weights:
            raise ValueError(
                'TensorProduct requires shared_weights=True when internal_weights=True'
            )
        self.irreps_in1 = Irreps(irreps_in1)
        self.irreps_in2 = Irreps(irreps_in2)
        self.irreps_out = Irreps(irreps_out)
        self.shared_weights = bool(shared_weights)
        self.internal_weights = bool(internal_weights)
        self.conv_fusion = bool(conv_fusion)
        self.layout = layout
        allowed_modes = frozenset({'uvu'}) if self.conv_fusion else None
        self.problem, oeq_jax, self.dtype, self.instructions = build_tp_problem(
            self.irreps_in1,
            self.irreps_in2,
            self.irreps_out,
            instructions,
            shared_weights=self.shared_weights,
            layout=layout,
            group=group,
            allowed_modes=allowed_modes,
        )
        try:
            if self.conv_fusion:
                self.operator = oeq_jax.TensorProductConv(
                    self.problem, deterministic=False, requires_jvp=True
                )
                self._conv_method = 'openeq_atomic'
            else:
                self.operator = oeq_jax.TensorProduct(self.problem)
                self._conv_method = 'openeq_tp'
        except Exception as exc:
            raise RuntimeError(
                'OpenEquivariance could not compile the requested CUDA tensor product.'
            ) from exc
        self.weight_numel = int(self.problem.weight_numel)
        self.weight_permutation = weight_permutation(
            self.operator,
            self.weight_numel,
            shared_weights=self.shared_weights,
        )
        self.expects_projected_weights = bool(
            not self.shared_weights and not self.internal_weights
        )
        self.projection_permutation = (
            self.weight_permutation if self.expects_projected_weights else None
        )
        if self.internal_weights:
            if rngs is None:
                raise ValueError('rngs is required when internal_weights=True')
            # Parameters deliberately remain in canonical e3nn order.
            self.weight = nnx.Param(
                jax.random.normal(rngs(), (1, self.weight_numel), dtype=self.dtype)
            )
        else:
            self.weight = None

    def _weights(self, weights, batch_size: int, *, already_reordered: bool):
        if self.internal_weights:
            if weights is not None:
                raise ValueError('weights must be None when internal_weights=True')
            weights = self.weight
        elif weights is None:
            raise ValueError('OpenEquivariance requires explicit weights.')
        weights = jnp.asarray(weights)
        if weights.dtype != self.dtype:
            raise TypeError(
                f'OpenEquivariance weights use {weights.dtype}, expected {self.dtype}; '
                'mixed dtypes are unsupported.'
            )
        if weights.ndim == 1:
            weights = weights[None, :]
        if weights.ndim != 2 or weights.shape[-1] != self.weight_numel:
            raise ValueError(
                f'Expected weights shape (*, {self.weight_numel}), got {weights.shape}.'
            )
        if self.shared_weights:
            if weights.shape[0] != 1:
                raise ValueError('Shared OpenEquivariance weights must have shape (1, W).')
            weights = weights[0]
        elif weights.shape[0] != batch_size:
            raise ValueError('Unshared OpenEquivariance weights require one row per item.')
        if not already_reordered:
            weights = jnp.take(weights, jnp.asarray(self.weight_permutation), axis=-1)
        return weights

    def __call__(
        self,
        x1,
        x2,
        weights=None,
        edge_index=None,
        *,
        weights_are_openeq_order: bool = False,
    ):
        x1, x2 = jnp.asarray(x1), jnp.asarray(x2)
        if x1.dtype != self.dtype or x2.dtype != self.dtype:
            raise TypeError(
                'OpenEquivariance inputs must all use the configured dtype '
                f'{self.dtype}; received {x1.dtype} and {x2.dtype}.'
            )
        if self.conv_fusion:
            if edge_index is None:
                raise ValueError('OpenEquivariance convolution requires edge_index.')
            edge_index = jnp.asarray(edge_index)
            if edge_index.ndim != 2 or edge_index.shape[0] != 2:
                raise ValueError(
                    f'edge_index must have shape (2, num_edges); got {edge_index.shape}.'
                )
            if x2.shape[0] != edge_index.shape[1]:
                raise ValueError('Edge attributes and edge_index must have equal edge counts.')
            if x2.shape[0] == 0:
                return jnp.zeros((x1.shape[0], self.irreps_out.dim), dtype=self.dtype)
            reordered = self._weights(
                weights,
                edge_index.shape[1],
                already_reordered=weights_are_openeq_order,
            )
            rows = edge_index[1].astype(jnp.int32)
            cols = edge_index[0].astype(jnp.int32)
            return self.operator.forward(x1, x2, reordered, rows, cols)

        if x1.shape[0] != x2.shape[0]:
            raise ValueError('Standalone OpenEquivariance TP inputs need equal batch sizes.')
        if x1.shape[0] == 0:
            return jnp.zeros((0, self.irreps_out.dim), dtype=self.dtype)
        reordered = self._weights(
            weights, x1.shape[0], already_reordered=weights_are_openeq_order
        )
        return self.operator.forward(x1, x2, reordered)


def _tensor_product_import_from_torch(cls, torch_module, variables):
    if getattr(torch_module, 'internal_weights', False) and getattr(
        torch_module, 'weight_numel', 0
    ):
        value = torch_module.weight.detach().cpu().numpy()
        variables['weight'] = jnp.asarray(value.reshape(1, -1))
    return variables


TensorProduct.import_from_torch = classmethod(_tensor_product_import_from_torch)
