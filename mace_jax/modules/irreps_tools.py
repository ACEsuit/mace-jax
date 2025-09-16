###########################################################################################
# Elementary tools for handling irreducible representations
# Authors: Ilyes Batatia, Gregor Simm
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

from typing import Optional, Tuple

import haiku as hk
import jax.numpy as jnp
from e3nn_jax import Irreps

from mace_jax.modules.wrapper_ops import CuEquivarianceConfig


# Based on mir-group/nequip
def tp_out_irreps_with_instructions(
    irreps1: Irreps, irreps2: Irreps, target_irreps: Irreps
) -> Tuple[Irreps, list]:
    trainable = True

    # Collect possible irreps and their instructions
    irreps_out_list: list[Tuple[int, Irreps]] = []
    instructions = []
    for i, (mul, ir_in) in enumerate(irreps1):
        for j, (_, ir_edge) in enumerate(irreps2):
            for ir_out in ir_in * ir_edge:  # | l1 - l2 | <= l <= l1 + l2
                if ir_out in target_irreps:
                    k = len(irreps_out_list)  # instruction index
                    irreps_out_list.append((mul, ir_out))
                    instructions.append((i, j, k, 'uvu', trainable))

    # We sort the output irreps of the tensor product so that we can simplify them
    # when they are provided to the second o3.Linear
    irreps_out = Irreps(irreps_out_list)
    irreps_out, permut, _ = irreps_out.sort()

    # Permute the output indexes of the instructions to match the sorted irreps:
    instructions = [
        (i_in1, i_in2, permut[i_out], mode, train)
        for i_in1, i_in2, i_out, mode, train in instructions
    ]

    instructions = sorted(instructions, key=lambda x: x[2])

    return irreps_out, instructions


def linear_out_irreps(irreps: Irreps, target_irreps: Irreps) -> Irreps:
    # Assuming simplified irreps
    irreps_mid = []
    for _, ir_in in irreps:
        found = False

        for mul, ir_out in target_irreps:
            if ir_in == ir_out:
                irreps_mid.append((mul, ir_out))
                found = True
                break

        if not found:
            raise RuntimeError(f'{ir_in} not in {target_irreps}')

    return Irreps(irreps_mid)


class reshape_irreps(hk.Module):
    """
    Haiku module that reshapes a flat tensor according to an Irreps specification.
    """

    def __init__(
        self,
        irreps: Irreps,
        cueq_config: Optional['CuEquivarianceConfig'] = None,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self.irreps = Irreps(irreps)
        self.cueq_config = cueq_config
        self.dims = []
        self.muls = []
        for mul, ir in self.irreps:
            d = ir.dim
            self.dims.append(d)
            self.muls.append(mul)

    def __call__(self, tensor: jnp.ndarray) -> jnp.ndarray:
        ix = 0
        out = []
        batch = tensor.shape[0]

        for mul, d in zip(self.muls, self.dims):
            field = tensor[:, ix : ix + mul * d]
            ix += mul * d

            if self.cueq_config is not None:
                if self.cueq_config.layout_str == 'mul_ir':
                    field = jnp.reshape(field, (batch, mul, d))
                else:  # "ir_mul"
                    field = jnp.reshape(field, (batch, d, mul))
            else:
                field = jnp.reshape(field, (batch, mul, d))

            out.append(field)

        if self.cueq_config is not None:
            if self.cueq_config.layout_str == 'mul_ir':
                return jnp.concatenate(out, axis=-1)
            else:  # "ir_mul"
                return jnp.concatenate(out, axis=-2)
        return jnp.concatenate(out, axis=-1)


def mask_head(x: jnp.ndarray, head: int, num_heads: int) -> jnp.ndarray:
    """
    Mask all but one attention head.

    Parameters
    ----------
    x : jnp.ndarray
        Shape (batch, features).
    head : int
        Index of the head to keep.
    num_heads : int
        Total number of heads.

    Returns
    -------
    jnp.ndarray
        Same shape as `x`, but with only the selected head kept.
    """
    batch, features = x.shape
    head_dim = features // num_heads

    # Create a mask of shape (batch, head_dim, num_heads)
    mask = jnp.zeros((batch, head_dim, num_heads), dtype=x.dtype)
    idx = jnp.arange(batch)

    # Set the given head to 1 for all positions
    mask = mask.at[idx, :, head].set(1)

    # Rearrange to (batch, num_heads, head_dim) then flatten
    mask = jnp.transpose(mask, (0, 2, 1)).reshape(x.shape)

    return x * mask
