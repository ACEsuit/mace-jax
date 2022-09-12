import e3nn_jax as e3nn
import haiku as hk
import jax.numpy as jnp


class BesselBasis(hk.Module):
    """
    Klicpera, J.; Groß, J.; Günnemann, S. Directional Message Passing for Molecular Graphs; ICLR 2020.
    Equation (7)
    """

    def __init__(self, r_max: float, num_basis: int = 8):
        super().__init__()

        self.r_max = r_max
        self.num_basis = num_basis

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:  # [..., 1]
        n = jnp.arange(1, self.num_basis + 1)
        x = x[..., None]
        x_nonzero = jnp.where(x == 0, 1, x)
        return jnp.sqrt(2 / self.r_max) * jnp.where(
            x == 0,
            n * jnp.pi / self.r_max,
            jnp.sin(n * jnp.pi / self.r_max * x_nonzero) / x_nonzero,
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(r_max={self.r_max}, num_basis={self.num_basis}"
        )


class PolynomialCutoff(hk.Module):
    """
    Klicpera, J.; Groß, J.; Günnemann, S. Directional Message Passing for Molecular Graphs; ICLR 2020.
    Equation (8)
    """

    def __init__(self, r_max: float, p: int = 6):
        super().__init__()
        self.r_max = r_max
        self.p = p

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return e3nn.poly_envelope(self.p - 1, 2)(x / self.r_max) * (x < self.r_max)

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p}, r_max={self.r_max})"
