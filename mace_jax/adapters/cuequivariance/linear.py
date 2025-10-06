"""Cue-equivariant linear layer."""

from __future__ import annotations

import cuequivariance_jax as cuex
import haiku as hk
import jax.numpy as jnp
from e3nn_jax import Irreps  # type: ignore

import cuequivariance as cue
from mace_jax.haiku.torch import register_import

from .utility import ir_mul_to_mul_ir, mul_ir_to_ir_mul


@register_import('e3nn.o3._linear.Linear')
class Linear(hk.Module):
    r"""Linear transform evaluated with cuequivariance-jax.

    Mathematically the module realises :math:`y = W x`, where ``x`` carries the
    input irreps specified by ``irreps_in`` and ``W`` is constrained so that
    each output irrep mixes only the compatible input irreps.  The resulting
    vector ``y`` is organised according to ``irreps_out``.  Cuequivariance
    expresses such linear maps as segmented polynomials evaluated on a
    descriptor produced by :func:`cue.descriptors.linear`.  Following the same
    terminology as for the tensor product adapter:

    - Inputs are expected in ``ir_mul`` order with one segment per irrep block.
    - The descriptor stores a single *path* per pair of input and output
      irreps, together with the Clebsch–Gordan coefficients that realise the
      linear map as a polynomial.

    By contrast :mod:`e3nn` exposes :class:`e3nn.o3.Linear`, whose public API
    works with e3nn ``Irreps`` objects, stores tensors in ``mul_ir`` layout
    (multiplicity-major), and controls internal or shared weights directly on
    the module instance.

    This adapter constructs the cue descriptor once, converts the incoming
    activations from ``mul_ir`` to the ``ir_mul`` layout that the cue backend
    expects, wraps them in :class:`cuequivariance_jax.RepArray` (which pairs the
    array with its cue ``Irreps`` metadata and layout flag), delegates evaluation
    to :func:`cuequivariance_jax.equivariant_polynomial` (the linear-specialised
    segmented-polynomial evaluator), and then converts the result back to
    ``mul_ir`` so callers observe the standard e3nn layout.  Weight handling
    mirrors e3nn semantics: internal weights live as Haiku parameters, while
    external weights are validated and passed through to the backend.
    """

    def __init__(
        self,
        irreps_in: Irreps,
        irreps_out: Irreps,
        *,
        shared_weights: bool | None = None,
        internal_weights: bool | None = None,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)

        if shared_weights is False and internal_weights is None:
            internal_weights = False

        if shared_weights is None:
            shared_weights = True

        if internal_weights is None:
            internal_weights = True

        self.internal_weights = internal_weights
        self.shared_weights = shared_weights

        self.irreps_in_o3 = Irreps(irreps_in)
        self.irreps_out_o3 = Irreps(irreps_out)
        self.irreps_in_cue = cue.Irreps(cue.O3, irreps_in)
        self.irreps_out_cue = cue.Irreps(cue.O3, irreps_out)

        descriptor = cue.descriptors.linear(self.irreps_in_cue, self.irreps_out_cue)
        self.descriptor = descriptor
        self.weight_irreps = descriptor.inputs[0].irreps
        self.weight_numel = descriptor.polynomial.operands[0].size

        self.internal_weight_rep = None
        if self.internal_weights:
            weights = hk.get_parameter(
                'weight', (self.weight_numel,), init=hk.initializers.RandomNormal()
            )
            self.internal_weight_rep = cuex.RepArray(
                self.weight_irreps,
                weights,
                cue.ir_mul,
            )

    def __call__(
        self,
        x: jnp.ndarray,
        weights: jnp.ndarray = None,
    ) -> jnp.ndarray:
        x_ir_mul = mul_ir_to_ir_mul(x, self.irreps_in_o3)
        x_rep = cuex.RepArray(
            self.irreps_in_cue,
            jnp.asarray(x_ir_mul),
            cue.ir_mul,
        )

        if self.internal_weights:
            if weights is not None:
                raise ValueError(
                    'Weights must be None when internal_weights=True in Linear'
                )
            weight_rep = self.internal_weight_rep
        else:
            if self.shared_weights:
                if weights is None:
                    raise ValueError(
                        'Weights must be provided when internal_weights=False and shared_weights=True in Linear'
                    )
                if weights.shape[-1] != self.weight_numel:
                    raise ValueError(
                        f'Expected weights last dimension {self.weight_numel}, got {weights.shape[-1]}'
                    )
                weight_rep = cuex.RepArray(
                    self.weight_irreps,
                    weights,
                    cue.ir_mul,
                )
            else:
                weight_rep = None

        output_rep = cuex.equivariant_polynomial(
            self.descriptor,
            [weight_rep, x_rep],
            math_dtype=jnp.float64,
            method='naive',
        )

        out_ir_mul = output_rep.array
        out_mul_ir = ir_mul_to_mul_ir(out_ir_mul, self.irreps_out_o3)
        return out_mul_ir

    @classmethod
    def import_from_torch(cls, torch_module, hk_params, scope):
        hk_params = hk.data_structures.to_mutable_dict(hk_params)
        if scope not in hk_params:
            hk_params[scope] = {}

        if hasattr(torch_module, 'weight') and torch_module.weight is not None:
            hk_params[scope]['weight'] = jnp.array(
                torch_module.weight.detach().cpu().numpy().reshape(-1)
            )
        if hasattr(torch_module, 'bias') and torch_module.bias is not None:
            hk_params[scope]['bias'] = jnp.array(
                torch_module.bias.detach().cpu().numpy().reshape(-1)
            )
        return hk.data_structures.to_immutable_dict(hk_params)
