"""PolarMACE field-dependent blocks, matching the Torch MACE architecture."""

import jax
import jax.numpy as jnp
from e3nn_jax import Irreps
from flax import nnx

from mace_jax.adapters.e3nn import nn
from mace_jax.adapters.nnx.torch import nxx_auto_import_from_torch, nxx_register_module
from mace_jax.modules.blocks import LinearReadoutBlock
from mace_jax.modules.irreps_tools import tp_out_irreps_with_instructions
from mace_jax.modules.radial import RadialMLP
from mace_jax.modules.wrapper_ops import Linear
from mace_jax.nnx_config import ConfigVar
from mace_jax.tools.dtype import default_dtype


@nxx_register_module('mace.modules.blocks.GeneralNonLinearBiasReadoutBlock')
@nxx_auto_import_from_torch(allow_missing_mapper=True)
class GeneralNonLinearBiasReadoutBlock(nnx.Module):
    def __init__(self, irreps_in, MLP_irreps, gate, irreps_out, *, rngs):
        hidden, output = Irreps(MLP_irreps), Irreps(irreps_out)
        scalars = Irreps(
            [(mul, ir) for mul, ir in hidden if ir.l == 0 and ir in output]
        )
        gated = Irreps([(mul, ir) for mul, ir in hidden if ir.l > 0 and ir in output])
        gates = Irreps([(mul, '0e') for mul, _ in gated])
        self.equivariant_nonlin = nn.Gate(
            scalars,
            [gate or jax.nn.silu] * len(scalars),
            gates,
            [jax.nn.sigmoid] * len(gates),
            gated,
        )
        nonlinear = self.equivariant_nonlin.irreps_in.simplify()
        self.linear_1 = Linear(irreps_in, nonlinear, rngs=rngs)
        self.linear_mid = Linear(hidden, nonlinear, biases=True, rngs=rngs)
        self.linear_2 = Linear(hidden, output, biases=True, rngs=rngs)

    def __call__(self, x):
        x = self.equivariant_nonlin(self.linear_1(x))
        x = self.equivariant_nonlin(self.linear_mid(x))
        value = self.linear_2(x)
        return getattr(value, 'array', value)


@nxx_register_module('mace.modules.field_blocks.MultiLayerFeatureMixer')
@nxx_auto_import_from_torch(allow_missing_mapper=True)
class MultiLayerFeatureMixer(nnx.Module):
    def __init__(self, irreps, num_interactions, *, rngs):
        self.linears = nnx.List(
            [Linear(irreps, irreps, rngs=rngs) for _ in range(num_interactions)]
        )

    def __call__(self, features):
        return sum(linear(x) for linear, x in zip(self.linears, features))


@nxx_register_module('mace.modules.field_blocks.EnvironmentDependentSpinSourceBlock')
@nxx_auto_import_from_torch(allow_missing_mapper=True)
class EnvironmentDependentSpinSourceBlock(LinearReadoutBlock):
    def __init__(self, irreps, max_l, *, rngs):
        sh = Irreps.spherical_harmonics(max_l)
        super().__init__(irreps, sh + sh, rngs=rngs)


@nxx_register_module(
    'mace.modules.field_blocks.AgnosticChargeBiasedLinearPotentialEmbedding'
)
@nxx_auto_import_from_torch(allow_missing_mapper=True)
class AgnosticChargeBiasedLinearPotentialEmbedding(nnx.Module):
    def __init__(self, potential, features, charges, *, rngs):
        self.potential_linear = Linear(potential, features, rngs=rngs)
        self.node_feats_linear = Linear(features, features, rngs=rngs)
        self.charge_embedding = Linear(charges, features, rngs=rngs)

    def __call__(self, potential, features, charges):
        return (
            self.potential_linear(potential)
            + self.node_feats_linear(features)
            + self.charge_embedding(charges)
        )


@nxx_register_module('mace.modules.field_blocks.SparseUvuTensorProduct')
@nxx_auto_import_from_torch(allow_missing_mapper=True)
class SparseUvuTensorProduct(nnx.Module):
    def __init__(self, left, right, output, instructions, *, rngs):
        self.irreps_out = output
        normalizations = [
            sum(right[j].mul for _, j, k, _, _ in instructions if k == out)
            for out in range(len(output))
        ]
        paths, offset = [], 0
        for i, j, k, mode, trainable in instructions:
            m1, ir1 = left[i]
            m2, ir2 = right[j]
            mo, iro = output[k]
            if mode != 'uvu' or not trainable or mo != m1:
                raise ValueError('Unsupported sparse tensor product path.')
            dot = iro.l == 0 and ir1 == ir2
            if not dot and not (ir2.l == 0 and iro == ir1):
                raise ValueError(
                    'Sparse tensor products require dot or scalar modulation paths.'
                )
            paths.append(
                (
                    left.slices()[i],
                    right.slices()[j],
                    output.slices()[k],
                    offset,
                    m1,
                    m2,
                    ir1.dim,
                    dot,
                    (iro.dim / normalizations[k] / ir1.dim) ** 0.5,
                )
            )
            offset += m1 * m2
        self.paths = tuple(paths)
        self.weight = nnx.Param(
            jax.random.normal(rngs.params(), (offset,), dtype=default_dtype())
        )
        self.output_mask = ConfigVar(jnp.ones(output.dim, dtype=default_dtype()))

    def __call__(self, x, y):
        result = jnp.zeros((len(x), self.irreps_out.dim), dtype=x.dtype)
        for a, b, c, offset, m1, m2, dim, dot, factor in self.paths:
            weight = self.weight[...][offset : offset + m1 * m2].reshape(m1, m2)
            left = x[:, a].reshape(len(x), m1, dim)
            if dot:
                right = y[:, b].reshape(len(y), m2, dim)
                value = jnp.einsum('nui,uv,nvi->nu', left, weight, right)
            else:
                value = left * jnp.einsum('nv,uv->nu', y[:, b], weight)[:, :, None]
            result = result.at[:, c].add(value.reshape(len(x), -1) * factor)
        return result * self.output_mask[...]


def sparse_tp(left, right, output, rngs, invariant=False):
    _, instructions = tp_out_irreps_with_instructions(left, right, output)
    if invariant:
        instructions = [
            (i, j, 0, mode, trainable) for i, j, _, mode, trainable in instructions
        ]
    return SparseUvuTensorProduct(left, right, output, instructions, rngs=rngs)


@nxx_register_module('mace.modules.field_blocks.AgnosticEmbeddedOneBodyVariableUpdate')
@nxx_auto_import_from_torch(allow_missing_mapper=True)
class AgnosticEmbeddedOneBodyVariableUpdate(nnx.Module):
    def __init__(
        self, attrs, features, potential, charges, field_norm_factor=0.02, *, rngs
    ):
        invar = Irreps(f'{features.count("0e")}x0e')
        self.field_norm_factor = ConfigVar(
            jnp.asarray(field_norm_factor, dtype=default_dtype())
        )
        self.potential_embedding = AgnosticChargeBiasedLinearPotentialEmbedding(
            potential, features, charges, rngs=rngs
        )
        self.source_embedding = Linear(attrs, invar, rngs=rngs)
        self.dot_products = sparse_tp(features, features, invar, rngs, invariant=True)
        self.nonlinearity = RadialMLP([2 * invar.dim, 64, 64, 64, invar.dim], rngs=rngs)
        self.tp_out = sparse_tp(features, invar, features, rngs)
        hidden = (
            (32 * Irreps.spherical_harmonics(charges.lmax)).sort().irreps.simplify()
        )
        self.readout = GeneralNonLinearBiasReadoutBlock(
            features, hidden, jax.nn.silu, charges + Irreps('2x0e'), rngs=rngs
        )

    def __call__(self, attrs, features, potential, charges):
        mixed = self.potential_embedding(potential, features, charges)
        invariant = self.dot_products(features, mixed)
        hidden = self.nonlinearity(
            jnp.concatenate((invariant, self.source_embedding(attrs)), axis=-1)
        )
        return self.readout(self.tp_out(features, hidden))


@nxx_register_module('mace.modules.field_blocks.OneBodyMLPFieldReadout')
@nxx_auto_import_from_torch(allow_missing_mapper=True)
class OneBodyMLPFieldReadout(nnx.Module):
    def __init__(self, features, potential, charges, *, rngs):
        invar = Irreps(f'{features.count("0e")}x0e')
        self.linear_up_q = Linear(charges, features, biases=True, rngs=rngs)
        self.linear_up_v = Linear(potential, features, biases=True, rngs=rngs)
        self.dot_products_q = sparse_tp(features, features, invar, rngs, invariant=True)
        self.dot_products_v = sparse_tp(features, features, invar, rngs, invariant=True)
        self.mlp = RadialMLP([2 * invar.dim, 128, 128, 128, 1], rngs=rngs)

    def __call__(self, features, potential, initial, current):
        invariant = jnp.concatenate(
            (
                self.dot_products_q(features, self.linear_up_q(initial + current)),
                self.dot_products_v(features, self.linear_up_v(potential)),
            ),
            axis=-1,
        )
        return self.mlp(invariant).squeeze(-1)
