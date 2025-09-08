###########################################################################################
# Radial basis and cutoff
# Authors: Ilyes Batatia, Gregor Simm
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

import jax.numpy as jnp
import haiku as hk
import numpy as np
from typing import Optional

from .special import chebyshev_polynomial_t


class BesselBasis(hk.Module):
    """
    Equation (7) from the paper.
    JAX/Haiku version of BesselBasis.
    """

    def __init__(
        self,
        r_max: float,
        num_basis: int = 8,
        trainable: bool = False,
        name: str = None,
    ):
        super().__init__(name=name)
        self.r_max_val = float(r_max)
        self.num_basis = num_basis
        self.trainable = trainable

        # Precompute prefactor (constant)
        self.prefactor = jnp.sqrt(2.0 / r_max)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # Compute default bessel weights
        init_bessel = (
            np.pi
            / self.r_max_val
            * jnp.linspace(1.0, self.num_basis, self.num_basis, dtype=x.dtype)
        )

        if self.trainable:
            bessel_weights = hk.get_parameter(
                "bessel_weights",
                shape=init_bessel.shape,
                dtype=x.dtype,
                init=lambda *_: init_bessel,
            )
        else:
            bessel_weights = init_bessel

        numerator = jnp.sin(bessel_weights * x)  # [..., num_basis]
        return self.prefactor * (numerator / x)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(r_max={self.r_max_val}, "
            f"num_basis={self.num_basis}, trainable={self.trainable})"
        )


class ChebychevBasis(hk.Module):
    """
    JAX/Haiku version of ChebychevBasis (Equation 7).
    """

    def __init__(self, r_max: float, num_basis: int = 8, name: Optional[str] = None):
        super().__init__(name=name)
        self.num_basis = num_basis
        self.r_max = r_max

        # Precompute n values [1..num_basis]
        self.n = jnp.arange(1, num_basis + 1)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            x: shape [..., 1] or [...], radial distances normalized to [-1, 1].

        Returns:
            shape [..., num_basis]
        """
        x = jnp.broadcast_to(x, x.shape[:-1] + (self.num_basis,))
        return chebyshev_polynomial_t(x, self.n)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(r_max={self.r_max}, num_basis={self.num_basis})"
        )
