import haiku as hk
import jax
import jax.numpy as jnp

from mace_jax.haiku.torch import (
    auto_import_from_torch,
    register_import,
)


@register_import('torch.nn.modules.activation.SiLU')
@auto_import_from_torch(separator='~')
class SiLU(hk.Module):
    """Wrapper so SiLU shows up as a submodule (like in Torch)."""

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return jax.nn.silu(x)
