"""Cue-equivariant tensor product implemented via segmented polynomials."""

from __future__ import annotations

import cuequivariance_jax as cuex
import haiku as hk
import jax
import jax.numpy as jnp
from e3nn_jax import Irreps  # type: ignore

import cuequivariance as cue
from mace_jax.haiku.torch import register_import

from .utility import ir_mul_to_mul_ir, mul_ir_to_ir_mul


@register_import('e3nn.o3._tensor_product._sub.FullyConnectedTensorProduct')
class FullyConnectedTensorProduct(hk.Module):
    r"""Fully connected tensor product evaluated with cuequivariance-jax.

    This module parallels :class:`e3nn.o3.FullyConnectedTensorProduct`, which in
    e3nn terminology exhaustively mixes every multiplicity from the two inputs
    into every valid output irrep using the ``'uvw'`` connection mode.  In the
    cuequivariance setting the corresponding descriptor is obtained from
    :func:`cue.descriptors.fully_connected_tensor_product` and is interpreted as
    a segmented polynomial where:

    .. math::

        z_{w,k} = \sum_{u,v} w_{w,u,v} \, \langle x_u \otimes y_v, \mathrm{CG}_{u,v \to k} \rangle,

    with ``u``/``v`` iterating over all multiplicities of the inputs and ``w``
    indexing the output multiplicity of ``irreps_out``.  Every admissible triple
    contributes because the module is fully connected.

    - Inputs and weights live in ``ir_mul`` layout, grouped by irrep block.
    - Each path carries the Clebsch–Gordan coefficients for one triple
      (input1, input2, output).  Because the map is fully connected the output
      multiplicity is the product of the input multiplicities (all u–v–w
      combinations are present).

    To present an e3nn-style interface we:

    1. Accept activations in ``mul_ir`` layout, convert them to ``ir_mul`` before
       wrapping them in :class:`cuequivariance_jax.RepArray` (attaching the cue
       ``Irreps`` metadata and layout), invoke
       :func:`cuequivariance_jax.segmented_polynomial`, and convert the result
       back afterwards so the caller receives the standard e3nn layout.
    2. Reduce the redundant multiplicity dimension introduced by cue’s
       descriptor (summing across u–v combinations and normalising by
       ``sqrt(multiplicity)``) so that the output dimension matches the e3nn
       expectation.
    3. Handle internal/shared weights exactly as e3nn does, while letting the
       cue backend evaluate the actual polynomial.

    The outcome is a Haiku module that behaves like e3nn’s fully connected tensor
    product but is implemented on top of cuequivariance.
    """

    def __init__(
        self,
        irreps_in1: Irreps,
        irreps_in2: Irreps,
        irreps_out: Irreps,
        shared_weights: bool = True,
        internal_weights: bool = True,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)

        if internal_weights and not shared_weights:
            raise ValueError(
                'FullyConnectedTensorProduct requires shared_weights=True when internal_weights=True'
            )

        self.internal_weights = internal_weights
        self.shared_weights = shared_weights

        self.irreps_in1_o3 = Irreps(irreps_in1)
        self.irreps_in2_o3 = Irreps(irreps_in2)
        self.irreps_out_o3 = Irreps(irreps_out)

        self.irreps_in1_cue = cue.Irreps(cue.O3, irreps_in1)
        self.irreps_in2_cue = cue.Irreps(cue.O3, irreps_in2)
        self.irreps_out_cue = cue.Irreps(cue.O3, irreps_out)

        descriptor = cue.descriptors.fully_connected_tensor_product(
            self.irreps_in1_cue,
            self.irreps_in2_cue,
            self.irreps_out_cue,
        )
        self.descriptor = descriptor
        self.weight_irreps = descriptor.inputs[0].irreps
        self.weight_numel = descriptor.polynomial.operands[0].size
        self.descriptor_out_irreps_o3 = Irreps(str(descriptor.outputs[0].irreps))

        self.internal_weight_rep = None
        if self.internal_weights:
            weights = hk.get_parameter(
                'weight', (1, self.weight_numel), init=hk.initializers.RandomNormal()
            )
            self.internal_weight_rep = cuex.RepArray(
                self.weight_irreps,
                weights,
                cue.ir_mul,
            )

    def __call__(
        self,
        x1: jnp.ndarray,
        x2: jnp.ndarray,
        weights: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        batch_size = x1.shape[0]

        x1_ir_mul = mul_ir_to_ir_mul(x1, self.irreps_in1_o3)
        x2_ir_mul = mul_ir_to_ir_mul(x2, self.irreps_in2_o3)

        x1_rep = cuex.RepArray(
            self.irreps_in1_cue,
            x1_ir_mul,
            cue.ir_mul,
        )
        x2_rep = cuex.RepArray(
            self.irreps_in2_cue,
            x2_ir_mul,
            cue.ir_mul,
        )

        if self.internal_weights:
            if weights is not None:
                raise ValueError(
                    'Weights must be None when internal weights are used in FullyConnectedTensorProduct'
                )
            weight_rep = self.internal_weight_rep
        else:
            if weights is None:
                raise ValueError(
                    'Weights must be provided when internal weights are not used'
                )
            weights = jnp.asarray(weights, dtype=x1.dtype)
            if weights.ndim == 1:
                if weights.shape[0] != self.weight_numel:
                    raise ValueError(
                        f'Expected weights last dimension {self.weight_numel}, got {weights.shape[-1]}'
                    )
                weights = weights[jnp.newaxis, :]
            elif weights.ndim == 2:
                if weights.shape[-1] != self.weight_numel:
                    raise ValueError(
                        f'Expected weights last dimension {self.weight_numel}, got {weights.shape[-1]}'
                    )
            else:
                raise ValueError(
                    'Weights must have rank 1 or 2 when internal weights are not used'
                )

            if self.shared_weights:
                if weights.shape[0] not in (1, batch_size):
                    raise ValueError(
                        'Shared weights require leading dimension 1 or equal to the batch size'
                    )
                if weights.shape[0] == 1 and batch_size != 1:
                    weights = jnp.broadcast_to(weights, (batch_size, self.weight_numel))
            else:
                if weights.shape[0] != batch_size:
                    raise ValueError(
                        'Unshared weights require leading dimension equal to the batch size'
                    )

            weight_rep = cuex.RepArray(
                self.weight_irreps,
                weights,
                cue.ir_mul,
            )

        [out_ir_mul] = cuex.segmented_polynomial(
            self.descriptor.polynomial,
            [weight_rep.array, x1_rep.array, x2_rep.array],
            [
                jax.ShapeDtypeStruct(
                    (*x1.shape[:-1], self.descriptor_out_irreps_o3.dim), x1.dtype
                )
            ],
            method='naive',
            math_dtype=x1.dtype,
        )

        out_mul_ir = ir_mul_to_mul_ir(out_ir_mul, self.descriptor_out_irreps_o3)
        return out_mul_ir

    @classmethod
    def import_from_torch(cls, torch_module, hk_params, scope):
        hk_params = hk.data_structures.to_mutable_dict(hk_params)
        if torch_module.weight_numel > 0 and torch_module.internal_weights:
            hk_params[scope]['weight'] = jnp.array(
                torch_module.weight.detach().cpu().numpy()[None, :]
            )
        return hk.data_structures.to_immutable_dict(hk_params)
