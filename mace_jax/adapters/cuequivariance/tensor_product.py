"""Cue-equivariant tensor product implemented via segmented polynomials."""

from __future__ import annotations

import cuequivariance_jax as cuex
import haiku as hk
import jax
import jax.numpy as jnp
from e3nn_jax import Irreps  # type: ignore

import cuequivariance as cue
from mace_jax.haiku.torch import register_import

from .utility import collapse_ir_mul_segments, ir_mul_to_mul_ir, mul_ir_to_ir_mul


def _expected_channelwise_instructions(
    irreps_in1: Irreps, irreps_in2: Irreps, target_irreps: Irreps
) -> tuple[Irreps, list[tuple[int, int, int, str, bool, float]]]:
    """Return the sorted output irreps and instruction list for channel-wise TP."""

    collected: list[tuple[int, Irreps]] = []
    instructions: list[tuple[int, int, int, str, bool, float]] = []
    for i_in1, (mul_in1, ir_in1) in enumerate(irreps_in1):
        for i_in2, (_, ir_in2) in enumerate(irreps_in2):
            for ir_out in ir_in1 * ir_in2:
                if ir_out in target_irreps:
                    idx = len(collected)
                    collected.append((mul_in1, ir_out))
                    instructions.append((i_in1, i_in2, idx, 'uvu', True, 1.0))

    irreps_out = Irreps(collected)
    irreps_out_sorted, perm, _ = irreps_out.sort()
    remapped_instructions = [
        (i_in1, i_in2, perm[i_out], mode, has_weight, path_weight)
        for i_in1, i_in2, i_out, mode, has_weight, path_weight in instructions
    ]
    remapped_instructions.sort(key=lambda item: item[2])
    return irreps_out_sorted, remapped_instructions


def _normalise_instruction(inst) -> tuple[int, int, int, str, bool, float]:
    if len(inst) == 5:
        i1, i2, i_out, mode, has_weight = inst
        path_weight = 1.0
    elif len(inst) == 6:
        i1, i2, i_out, mode, has_weight, path_weight = inst
    else:
        raise ValueError(
            'TensorProduct instructions must have length 5 or 6, '
            f'got length {len(inst)}'
        )
    return (
        int(i1),
        int(i2),
        int(i_out),
        str(mode),
        bool(has_weight),
        float(path_weight),
    )


@register_import('e3nn.o3._tensor_product._tensor_product.TensorProduct')
@register_import(
    'cuequivariance_torch.operations.tp_channel_wise.ChannelWiseTensorProduct'
)
class TensorProduct(hk.Module):
    r"""Channel-wise tensor product evaluated with cuequivariance-jax.

    Given two inputs ``x`` and ``y`` carrying irreps ``irreps_in1`` and
    ``irreps_in2``, and a learned weight tensor ``w`` shaped according to the
    channel-wise instructions, this module evaluates

    .. math::

        z_{u,k} = \sum_{v} w_{u,v} \,\langle x_u \otimes y_v, \mathrm{CG}_{u,v \to k}\rangle,

    where the indices ``u``/``v`` enumerate multiplicities of the two inputs,
    ``k`` enumerates the output multiplicities consistent with the Clebsch–
    Gordan (CG) rules, and the inner product contracts the irrep components with
    the CG coefficients supplied by the descriptor.  The output ``z`` is then
    rearranged into ``mul_ir`` layout matching :mod:`e3nn`.

    **Terminology.** In the cuequivariance world a *descriptor* produced by
    :func:`cue.descriptors.channelwise_tensor_product` contains:

    - *Operands* – arrays laid out in ``ir_mul`` order (irrep components are the
      slow axis, multiplicity is the fast axis). Each operand is split into
      *segments*, one per irrep block; the descriptor records the segment sizes.
    - *Paths* – metadata describing how to combine specific segments of the
      operands.  Each path corresponds to a weighted contribution that mixes the
      multiplicity indices of the inputs.  Evaluating all paths is sometimes
      colloquially called a *contraction*.
    - A *segmented polynomial representation* – the tensor product is expressed
      as a polynomial where the operands provide the variables and the path
      coefficients encode the Clebsch–Gordan data.  Backends such as
      :mod:`cuequivariance_jax` evaluate this polynomial via
      :func:`cuequivariance_jax.segmented_polynomial`.

    **Contrast with e3nn.** :mod:`e3nn` offers object-oriented modules like
    :class:`e3nn.o3.TensorProduct`.  Users provide ``Irreps`` objects plus a list
    of instructions ``(i_in1, i_in2, i_out, mode, has_weight[, path_weight])``.
    Multiplicities are captured directly in the ``Irreps`` entries, tensors are
    stored in ``mul_ir`` layout (multiplicity blocks followed by the irrep
    components), and the module manages weight sharing semantics.

    **What this adapter does.** Given e3nn-style inputs and configuration we:

    1. Build the cue descriptor mirroring the same tensor product.  This gives
       us the segmented-polynomial view and the cue-specific ``ir_mul`` layout.
    2. Map inputs from e3nn's ``mul_ir`` layout into ``ir_mul`` before calling
       the backend, and convert the result back afterwards so callers continue to
       see e3nn-compatible shapes.
    3. When cue expands multiplicities differently (for example, with ``'uvu'``
       instructions the descriptor treats multiplicities as the product of the
       input counts) we *collapse output segments*: we reshape the result into
       ``(ir_dim, mul_in1, mul_in2)``, sum over the redundant axis (a reduction
       of the segmented polynomial), normalise by ``sqrt(multiplicity)`` so that
       norms stay comparable to e3nn, and finally flatten back to
       ``ir_dim * mul_out``.
    4. Manage shared or internal weights using the same conventions as
       :class:`e3nn.o3.TensorProduct` while delegating the numeric evaluation to
       :func:`cuequivariance_jax.segmented_polynomial`.

    The result is a Haiku module with the familiar e3nn API whose computations
    are carried out by cuequivariance.  Internally we wrap each operand in a
    :class:`cuequivariance_jax.RepArray`, which couples the raw ``ir_mul`` array
    with the corresponding cue ``Irreps`` object and layout tag.  This is the
    canonical entry point for the JAX backend and makes the descriptor/array
    pairing explicit before the segmented polynomial is evaluated.
    """

    def __init__(
        self,
        irreps_in1: Irreps,
        irreps_in2: Irreps,
        irreps_out: Irreps,
        shared_weights: bool = False,
        internal_weights: bool = False,
        instructions=None,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)

        if internal_weights and not shared_weights:
            raise ValueError(
                'TensorProduct requires shared_weights=True when internal_weights=True'
            )
        self.shared_weights = shared_weights
        self.internal_weights = internal_weights

        self.irreps_in1_o3 = Irreps(irreps_in1)
        self.irreps_in2_o3 = Irreps(irreps_in2)
        self.irreps_out_o3 = Irreps(irreps_out)

        self.irreps_in1_cue = cue.Irreps(cue.O3, irreps_in1)
        self.irreps_in2_cue = cue.Irreps(cue.O3, irreps_in2)
        self.irreps_out_cue = cue.Irreps(cue.O3, irreps_out)

        descriptor = cue.descriptors.channelwise_tensor_product(
            self.irreps_in1_cue, self.irreps_in2_cue, self.irreps_out_cue
        )
        self.descriptor = descriptor
        self.weight_irreps = descriptor.inputs[0].irreps
        self.weight_numel = descriptor.polynomial.operands[0].size
        self.descriptor_out_irreps_o3 = Irreps(str(descriptor.outputs[0].irreps))
        self.output_segment_shapes = tuple(descriptor.polynomial.operands[-1].segments)

        expected_irreps, expected_instructions = _expected_channelwise_instructions(
            self.irreps_in1_o3, self.irreps_in2_o3, self.irreps_out_o3
        )
        if expected_irreps != self.irreps_out_o3:
            raise ValueError(
                'TensorProduct irreps_out is incompatible with channel-wise descriptor'
            )

        if instructions is not None:
            normalised = [_normalise_instruction(inst) for inst in instructions]
            if normalised != expected_instructions:
                raise ValueError(
                    'TensorProduct only supports channel-wise "uvu" instructions '
                    'matching those returned by e3nn; received '
                    f'{instructions!r}'
                )

    def __call__(
        self,
        x1: jnp.ndarray,
        x2: jnp.ndarray,
        weights: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        batch_size = x1.shape[0]

        if self.internal_weights:
            if weights is not None:
                raise ValueError(
                    'TensorProduct uses internal weights; weights argument must be None'
                )
            parameter = hk.get_parameter(
                'weight',
                shape=(1, self.weight_numel),
                dtype=x1.dtype,
                init=hk.initializers.RandomNormal(),
            )
            weight_tensor = parameter
        else:
            if weights is None:
                raise ValueError(
                    'TensorProduct requires explicit weights when internal_weights=False'
                )
            weight_tensor = jnp.asarray(weights, dtype=x1.dtype)

        if weight_tensor.ndim == 1:
            weight_tensor = weight_tensor[jnp.newaxis, :]
        elif weight_tensor.ndim != 2:
            raise ValueError(
                f'Weights must have rank 1 or 2, got rank {weight_tensor.ndim}'
            )

        if weight_tensor.shape[-1] != self.weight_numel:
            raise ValueError(
                f'Expected weights last dimension {self.weight_numel}, got {weight_tensor.shape[-1]}'
            )

        if self.shared_weights:
            if weight_tensor.shape[0] not in (1, batch_size):
                raise ValueError(
                    'Shared weights require leading dimension 1 or equal to the batch size'
                )
            if weight_tensor.shape[0] == 1 and batch_size != 1:
                weight_tensor = jnp.broadcast_to(
                    weight_tensor, (batch_size, self.weight_numel)
                )
        else:
            if weight_tensor.shape[0] != batch_size:
                raise ValueError(
                    'Unshared weights require leading dimension equal to the batch size'
                )

        x1_ir_mul = mul_ir_to_ir_mul(x1, self.irreps_in1_o3)
        x2_ir_mul = mul_ir_to_ir_mul(x2, self.irreps_in2_o3)

        x1_rep = cuex.RepArray(
            self.irreps_in1_cue,
            jnp.asarray(x1_ir_mul),
            cue.ir_mul,
        )
        x2_rep = cuex.RepArray(
            self.irreps_in2_cue,
            jnp.asarray(x2_ir_mul),
            cue.ir_mul,
        )
        weight_rep = cuex.RepArray(
            self.weight_irreps,
            weight_tensor,
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
        out_ir_mul = collapse_ir_mul_segments(
            out_ir_mul,
            self.descriptor_out_irreps_o3,
            self.irreps_out_o3,
            self.output_segment_shapes,
        )
        out_mul_ir = ir_mul_to_mul_ir(out_ir_mul, self.irreps_out_o3)
        return out_mul_ir

    @classmethod
    def import_from_torch(cls, torch_module, hk_params, scope):
        hk_params = hk.data_structures.to_mutable_dict(hk_params)
        if torch_module.weight_numel > 0 and torch_module.internal_weights:
            weight_np = torch_module.weight.detach().cpu().numpy()
            module_path = (
                f'{torch_module.__class__.__module__}.{torch_module.__class__.__name__}'
            )
            if module_path.startswith('cuequivariance_torch'):
                if weight_np.ndim == 1:
                    weight_np = weight_np[None, :]
            else:
                weight_np = weight_np.reshape(1, -1)
            hk_params.setdefault(scope, {})
            hk_params[scope]['weight'] = jnp.array(weight_np)
        return hk.data_structures.to_immutable_dict(hk_params)
