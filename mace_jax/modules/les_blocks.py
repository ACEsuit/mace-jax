"""Equivariant tensor readouts used by MACELES."""

import jax.nn as jnn
import jax.numpy as jnp
from e3nn_jax import Irreps
from flax import nnx

from mace_jax.adapters.nnx.torch import nxx_auto_import_from_torch, nxx_register_module
from mace_jax.modules.wrapper_ops import Linear, TransposeIrrepsLayoutWrapper
from mace_jax.tools.dtype import default_dtype


class _TensorReadout(nnx.Module):
    def __init__(self, irreps_in, equivariance_config=None, *, rngs):
        self.irreps_in = Irreps(irreps_in)
        self.linear = Linear(
            self.irreps_in,
            self.irreps_in,
            equivariance_config=equivariance_config,
            rngs=rngs,
        )
        self.to_mul_ir = TransposeIrrepsLayoutWrapper(
            self.irreps_in,
            source=getattr(equivariance_config, 'layout', 'mul_ir'),
            target='mul_ir',
            equivariance_config=equivariance_config,
        )
        self.slices = {
            key: tuple(
                (s.start, s.stop)
                for (mul, ir), s in zip(self.irreps_in, self.irreps_in.slices())
                if str(ir) == key
            )
            for key in ('0e', '1o', '1e')
        }
        self.n_scalar = self.irreps_in.count('0e')
        self.n_1o = self.irreps_in.count('1o')
        self.n_1e = self.irreps_in.count('1e')
        if not self.n_scalar or not self.n_1o + self.n_1e:
            raise ValueError('Tensor readouts require scalar and vector channels.')

    def collect(self, y, ir):
        blocks = [
            y[:, s:e].reshape(len(y), -1, 1 if ir == '0e' else 3)
            for s, e in self.slices[ir]
        ]
        result = jnp.concatenate(blocks, axis=1)
        return result[..., 0] if ir == '0e' else result


@nxx_register_module('mace.modules.blocks.LinearLesReadoutBlock')
@nxx_auto_import_from_torch(allow_missing_mapper=True)
class LinearLesReadoutBlock(_TensorReadout):
    def __init__(self, irreps_in, make_w_pos=True, equivariance_config=None, *, rngs):
        super().__init__(irreps_in, equivariance_config, rngs=rngs)
        self.make_w_pos = make_w_pos
        for ir, count in (('1o', self.n_1o), ('1e', self.n_1e)):
            setattr(
                self,
                'scalar_to_weight_' + ir,
                nnx.Linear(self.n_scalar, count, param_dtype=default_dtype(), rngs=rngs)
                if count
                else None,
            )

    def __call__(self, x, heads=None):
        y = self.linear(x)
        if self.to_mul_ir is not None:
            y = self.to_mul_ir(y)
        s = self.collect(y, '0e')
        result = jnp.zeros((len(x), 3, 3), dtype=x.dtype)
        for ir in ('1o', '1e'):
            readout = getattr(self, 'scalar_to_weight_' + ir)
            if readout is not None:
                w = readout(s)
                if self.make_w_pos and ir == '1o':
                    w = w**2
                v = self.collect(y, ir)
                result += jnp.einsum('nc,nci,ncj->nij', w, v, v)
        return result


@nxx_register_module('mace.modules.blocks.NonLinearLesReadoutBlock')
@nxx_auto_import_from_torch(allow_missing_mapper=True)
class NonLinearLesReadoutBlock(_TensorReadout):
    def __init__(self, irreps_in, hidden_dim=32, equivariance_config=None, *, rngs):
        super().__init__(irreps_in, equivariance_config, rngs=rngs)
        if not self.n_1o:
            raise ValueError('Nonlinear tensor readouts require 1o channels.')
        self.mlp = nnx.List(
            [
                nnx.Linear(
                    self.n_scalar, hidden_dim, param_dtype=default_dtype(), rngs=rngs
                ),
                jnn.silu,
                nnx.Linear(
                    hidden_dim, self.n_1o**2, param_dtype=default_dtype(), rngs=rngs
                ),
            ]
        )

    def __call__(self, x, heads=None):
        y = self.linear(x)
        if self.to_mul_ir is not None:
            y = self.to_mul_ir(y)
        s, v = self.collect(y, '0e'), self.collect(y, '1o')
        for layer in self.mlp:
            s = layer(s)
        matrix = s.reshape(len(x), self.n_1o, self.n_1o)
        matrix = (matrix + matrix.swapaxes(-1, -2)) / 2
        result = jnp.einsum('ncd,nci,ndj->nij', matrix, v, v)
        return (result + result.swapaxes(-1, -2)) / 2
