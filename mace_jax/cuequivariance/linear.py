"""Cue-equivariant linear layer."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Optional, Sequence, Union

import haiku as hk
import jax
import jax.numpy as jnp
from e3nn_jax import Irreps

from mace_jax.cuequivariance.tensor_product import TensorProduct
from mace_jax.e3nn._linear import Instruction as LinearInstruction
from mace_jax.e3nn._tensor_product._instruction import Instruction as TPInstruction
from mace_jax.haiku.torch import register_import


@dataclass
class _BiasSlice:
    i_out: int
    start: int
    stop: int
    path_weight: float
    path_shape: tuple[int, ...]


@register_import('mace.modules.linear.Linear')
class Linear(hk.Module):
    """Cue-backed O(3)-equivariant linear map.

    This mirrors :class:`mace_jax.e3nn._linear.Linear` but evaluates the weighted
    paths via the CuEquivariance tensor product backend.
    """

    def __init__(
        self,
        irreps_in: Irreps,
        irreps_out: Irreps,
        *,
        f_in: Optional[int] = None,
        f_out: Optional[int] = None,
        internal_weights: Optional[bool] = None,
        shared_weights: Optional[bool] = None,
        instructions: Optional[Sequence[tuple[int, int]]] = None,
        biases: Union[bool, Sequence[bool]] = False,
        path_normalization: str = 'element',
        cueq_config=None,
        name: Optional[str] = None,
    ) -> None:
        if f_in is not None or f_out is not None:
            raise NotImplementedError('Cue Linear does not implement f_in / f_out.')

        super().__init__(name=name)

        self.irreps_in = Irreps(irreps_in)
        self.irreps_out = Irreps(irreps_out)
        self.cueq_config = cueq_config

        # ------------------------------------------------------------------
        # Build instructions (mirrors mace_jax.e3nn._linear.Linear)
        # ------------------------------------------------------------------
        if instructions is None:
            instructions = [
                (i_in, i_out)
                for i_in, (_, ir_in) in enumerate(self.irreps_in)
                for i_out, (_, ir_out) in enumerate(self.irreps_out)
                if ir_in == ir_out
            ]

        lin_instructions: list[LinearInstruction] = [
            LinearInstruction(
                i_in=i_in,
                i_out=i_out,
                path_shape=(self.irreps_in[i_in].mul, self.irreps_out[i_out].mul),
                path_weight=1.0,
            )
            for i_in, i_out in instructions
        ]

        def alpha(ins: LinearInstruction) -> float:
            if path_normalization == 'element':
                accumulator = sum(
                    self.irreps_in[i.i_in].mul for i in lin_instructions if i.i_out == ins.i_out
                )
            elif path_normalization == 'path':
                accumulator = sum(
                    self.irreps_in[i.i_in].mul for i in lin_instructions if i.i_out == ins.i_out
                )
            else:
                raise ValueError(f'Unsupported path_normalization {path_normalization!r}.')
            return 1.0 if accumulator == 0 else accumulator

        lin_instructions = [
            LinearInstruction(
                i_in=ins.i_in,
                i_out=ins.i_out,
                path_shape=ins.path_shape,
                path_weight=alpha(ins) ** (-0.5),
            )
            for ins in lin_instructions
        ]

        if isinstance(biases, bool):
            biases = [biases and ir.is_scalar() for _, ir in self.irreps_out]
        else:
            biases = list(biases)
        assert len(biases) == len(self.irreps_out)

        for i_out, (bias_flag, mul_ir) in enumerate(zip(biases, self.irreps_out)):
            if bias_flag:
                lin_instructions.append(
                    LinearInstruction(
                        i_in=-1,
                        i_out=i_out,
                        path_shape=(mul_ir.dim,),
                        path_weight=1.0,
                    )
                )

        if shared_weights is False and internal_weights is None:
            internal_weights = False
        if shared_weights is None:
            shared_weights = True
        if internal_weights is None:
            internal_weights = shared_weights
        assert shared_weights or not internal_weights

        self.shared_weights = shared_weights
        self.internal_weights = internal_weights

        # Instructions using weights (i_in != -1) -> evaluate via TensorProduct
        weight_instructions = [ins for ins in lin_instructions if ins.i_in != -1]
        bias_instructions = [ins for ins in lin_instructions if ins.i_in == -1]

        tp_instructions = [
            TPInstruction(
                i_in1=ins.i_in,
                i_in2=0,
                i_out=ins.i_out,
                connection_mode='uvw',
                has_weight=True,
                path_weight=ins.path_weight,
                path_shape=(
                    self.irreps_in[ins.i_in].mul,
                    1,
                    self.irreps_out[ins.i_out].mul,
                ),
            )
            for ins in weight_instructions
        ]

        tp_name = None if name is None else f'{name}_tp'
        self._tp = TensorProduct(
            self.irreps_in,
            Irreps('1x0e'),
            self.irreps_out,
            instructions=tp_instructions,
            shared_weights=self.shared_weights,
            internal_weights=False,
            cueq_config=cueq_config,
            name=tp_name,
        )
        self.weight_numel = self._tp.weight_numel

        # Bias bookkeeping
        offset = 0
        self.bias_numel = sum(prod(ins.path_shape) for ins in bias_instructions)
        self._bias_slices: list[_BiasSlice] = []
        for ins in bias_instructions:
            size = prod(ins.path_shape)
            self._bias_slices.append(
                _BiasSlice(
                    i_out=ins.i_out,
                    start=offset,
                    stop=offset + size,
                    path_weight=ins.path_weight,
                    path_shape=ins.path_shape,
                )
            )
            offset += size

        # Pre-compute output slices for bias broadcasting
        self._out_slices = list(self.irreps_out.slices())

    def __call__(
        self,
        x: jnp.ndarray,
        w: Optional[jnp.ndarray] = None,
        b: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        ones = jnp.ones(x.shape[:-1] + (1,), dtype=x.dtype)

        if self.weight_numel > 0:
            if w is None:
                if not self.internal_weights:
                    raise RuntimeError(
                        'Weights must be provided when internal_weights=False.'
                    )
                w = hk.get_parameter(
                    'weight',
                    (self.weight_numel,),
                    init=hk.initializers.RandomNormal(),
                )
            else:
                w = jnp.asarray(w)
                if self.shared_weights:
                    if w.shape != (self.weight_numel,):
                        raise ValueError(
                            f'Invalid weight shape {w.shape}; expected {(self.weight_numel,)}.'
                        )
                else:
                    if w.shape[-1] != self.weight_numel:
                        raise ValueError(
                            f'Invalid weight shape {w.shape}; last dim must be {self.weight_numel}.'
                        )
        out = self._tp(x, ones, w)

        if self.bias_numel > 0:
            if b is None:
                if not self.internal_weights:
                    raise RuntimeError(
                        'Biases must be provided when internal_weights=False.'
                    )
                b = hk.get_parameter('bias', (self.bias_numel,), init=jnp.zeros)
            else:
                b = jnp.asarray(b)
                if b.shape != (self.bias_numel,):
                    raise ValueError(
                        f'Invalid bias shape {b.shape}; expected {(self.bias_numel,)}.'
                    )

            flat_out = out.reshape(-1, self.irreps_out.dim)
            for info in self._bias_slices:
                bias_vec = info.path_weight * b[info.start : info.stop]
                mul_ir_out = self.irreps_out[info.i_out]
                bias_vec = bias_vec.reshape(mul_ir_out.dim)
                sl = self._out_slices[info.i_out]
                flat_out = flat_out.at[:, sl].add(bias_vec)
            out = flat_out.reshape(out.shape)

        return out

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
