###########################################################################################
# Radial basis and cutoff
# Authors: Ilyes Batatia, Gregor Simm
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

import logging
from typing import Optional

import ase
import haiku as hk
import jax.numpy as jnp
import numpy as np

from mace_jax.tools.scatter import scatter_sum

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


class GaussianBasis(hk.Module):
    """
    Gaussian basis functions (Haiku version).

    Parameters
    ----------
    r_max : float
        Maximum radius.
    num_basis : int, default=128
        Number of Gaussian basis functions.
    trainable : bool, default=False
        Whether the Gaussian centers (weights) are trainable.
    """

    def __init__(
        self,
        r_max: float,
        num_basis: int = 128,
        trainable: bool = False,
        name: str = None,
    ):
        super().__init__(name=name)
        self.r_max = r_max
        self.num_basis = num_basis
        self.trainable = trainable

        # Precompute coefficient
        self.coeff = -0.5 / (r_max / (num_basis - 1)) ** 2

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # Initialize Gaussian centers (like torch.linspace)
        init_gaussian_weights = jnp.linspace(
            start=0.0, stop=self.r_max, num=self.num_basis, dtype=jnp.float32
        )

        if self.trainable:
            gaussian_weights = hk.get_parameter(
                "gaussian_weights",
                shape=init_gaussian_weights.shape,
                init=lambda *_: init_gaussian_weights,
            )
        else:
            # Non-trainable "buffer" -> constant
            gaussian_weights = hk.get_state(
                "gaussian_weights",
                shape=init_gaussian_weights.shape,
                dtype=jnp.float32,
                init=lambda *_: init_gaussian_weights,
            )
            # Ensure it stays constant
            hk.set_state("gaussian_weights", gaussian_weights)

        # Apply Gaussian basis transform
        x = x[..., None] - gaussian_weights  # expand along basis dimension
        return jnp.exp(self.coeff * jnp.square(x))


class PolynomialCutoff(hk.Module):
    """Polynomial cutoff function that goes from 1 to 0 as x goes from 0 to r_max.
    Equation (8) -- TODO: from where?
    """

    def __init__(self, r_max: float, p: int = 6, name: str = None):
        super().__init__(name=name)

        # Store as non-trainable constants (buffers in PyTorch)
        self.r_max = hk.get_state(
            "r_max",
            shape=(),
            dtype=jnp.float32,
            init=lambda *_: jnp.array(r_max, dtype=jnp.float32),
        )
        self.p = hk.get_state(
            "p",
            shape=(),
            dtype=jnp.int32,
            init=lambda *_: jnp.array(p, dtype=jnp.int32),
        )

        # Ensure they stay constant
        hk.set_state("r_max", self.r_max)
        hk.set_state("p", self.p)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.calculate_envelope(x, self.r_max, self.p)

    @staticmethod
    def calculate_envelope(
        x: jnp.ndarray, r_max: jnp.ndarray, p: jnp.ndarray
    ) -> jnp.ndarray:
        r_over_r_max = x / r_max
        envelope = (
            1.0
            - ((p + 1.0) * (p + 2.0) / 2.0) * jnp.power(r_over_r_max, p)
            + p * (p + 2.0) * jnp.power(r_over_r_max, p + 1)
            - (p * (p + 1.0) / 2.0) * jnp.power(r_over_r_max, p + 2)
        )
        return envelope * (x < r_max)

    def __repr__(self):
        return f"{self.__class__.__name__}(p={int(self.p)}, r_max={float(self.r_max)})"


class ZBLBasis(hk.Module):
    """Implementation of the Ziegler-Biersack-Littmark (ZBL) potential
    with a polynomial cutoff envelope (Haiku version).
    """

    def __init__(self, p: int = 6, trainable: bool = False, name: str = None, **kwargs):
        super().__init__(name=name)

        if "r_max" in kwargs:
            logging.warning(
                "r_max is deprecated. r_max is determined from the covalent radii."
            )

        # Constants (non-trainable buffers in PyTorch)
        self.c = jnp.array([0.1818, 0.5099, 0.2802, 0.02817], dtype=jnp.float32)
        self.p = jnp.array(p, dtype=jnp.int32)
        self.covalent_radii = jnp.array(ase.data.covalent_radii, dtype=jnp.float32)

        # Parameters (trainable or frozen)
        if trainable:
            self.a_exp = hk.get_parameter(
                "a_exp", shape=(), init=lambda *_: jnp.array(0.300, dtype=jnp.float32)
            )
            self.a_prefactor = hk.get_parameter(
                "a_prefactor",
                shape=(),
                init=lambda *_: jnp.array(0.4543, dtype=jnp.float32),
            )
        else:
            self.a_exp = jnp.array(0.300, dtype=jnp.float32)
            self.a_prefactor = jnp.array(0.4543, dtype=jnp.float32)

    def __call__(
        self,
        x: jnp.ndarray,
        node_attrs: jnp.ndarray,
        edge_index: jnp.ndarray,
        atomic_numbers: jnp.ndarray,
    ) -> jnp.ndarray:
        # edge_index: (2, num_edges)
        sender, receiver = edge_index

        # Convert one-hot node_attrs to atomic numbers
        node_atomic_numbers = atomic_numbers[jnp.argmax(node_attrs, axis=1)][..., None]
        Z_u = node_atomic_numbers[sender].astype(jnp.int32)
        Z_v = node_atomic_numbers[receiver].astype(jnp.int32)

        # Screening length a
        a = (
            self.a_prefactor
            * 0.529
            / (jnp.power(Z_u, self.a_exp) + jnp.power(Z_v, self.a_exp))
        )
        r_over_a = x / a

        # Screening function φ(r/a)
        phi = (
            self.c[0] * jnp.exp(-3.2 * r_over_a)
            + self.c[1] * jnp.exp(-0.9423 * r_over_a)
            + self.c[2] * jnp.exp(-0.4028 * r_over_a)
            + self.c[3] * jnp.exp(-0.2016 * r_over_a)
        )

        # Pairwise ZBL potential
        v_edges = (14.3996 * Z_u * Z_v) / x * phi

        # Smooth cutoff
        r_max = self.covalent_radii[Z_u] + self.covalent_radii[Z_v]
        envelope = PolynomialCutoff.calculate_envelope(x, r_max, self.p)
        v_edges = 0.5 * v_edges * envelope

        # Aggregate edge potentials per receiver node
        V_ZBL = scatter_sum(
            v_edges.squeeze(-1), receiver, num_segments=node_attrs.shape[0]
        )
        return V_ZBL

    def __repr__(self):
        return f"{self.__class__.__name__}(c={self.c})"


class AgnesiTransform(hk.Module):
    """Agnesi transform - see section on Radial transformations in
    ACEpotentials.jl, JCP 2023 (https://doi.org/10.1063/5.0158783).
    """

    def __init__(
        self,
        q: float = 0.9183,
        p: float = 4.5791,
        a: float = 1.0805,
        trainable: bool = False,
        name: str = None,
    ):
        super().__init__(name=name)

        # Store constants (as JAX arrays)
        self.covalent_radii = jnp.array(ase.data.covalent_radii, dtype=jnp.float32)

        if trainable:
            self.a = hk.get_parameter(
                "a", shape=(), init=lambda *_: jnp.array(a, dtype=jnp.float32)
            )
            self.q = hk.get_parameter(
                "q", shape=(), init=lambda *_: jnp.array(q, dtype=jnp.float32)
            )
            self.p = hk.get_parameter(
                "p", shape=(), init=lambda *_: jnp.array(p, dtype=jnp.float32)
            )
        else:
            self.a = jnp.array(a, dtype=jnp.float32)
            self.q = jnp.array(q, dtype=jnp.float32)
            self.p = jnp.array(p, dtype=jnp.float32)

    def __call__(
        self,
        x: jnp.ndarray,
        node_attrs: jnp.ndarray,
        edge_index: jnp.ndarray,
        atomic_numbers: jnp.ndarray,
    ) -> jnp.ndarray:
        # edge_index: (2, num_edges)
        sender, receiver = edge_index

        # Convert one-hot node_attrs to atomic numbers
        node_atomic_numbers = atomic_numbers[jnp.argmax(node_attrs, axis=1)][..., None]
        Z_u = node_atomic_numbers[sender].astype(jnp.int32)
        Z_v = node_atomic_numbers[receiver].astype(jnp.int32)

        # Reference distance
        r_0 = 0.5 * (self.covalent_radii[Z_u] + self.covalent_radii[Z_v])
        r_over_r_0 = x / r_0

        # Agnesi transform
        numerator = self.a * jnp.power(r_over_r_0, self.q)
        denominator = 1.0 + jnp.power(r_over_r_0, self.q - self.p)
        return 1.0 / (1.0 + numerator / denominator)

    def __repr__(self):
        return f"{self.__class__.__name__}(a={float(self.a):.4f}, q={float(self.q):.4f}, p={float(self.p):.4f})"
