from typing import Callable, Optional

import haiku as hk
import jax.numpy as jnp
from e3nn_jax import Irreps

from ..math import normalize2mom


class Activation(hk.Module):
    """Scalar activation function for equivariant features.

    Odd scalar inputs require activation functions with a defined parity (odd or even).

    Parameters
    ----------
    irreps_in : Irreps
        Representation of the input.
    acts : List of Callable or None
        List of activation functions; `None` if non-scalar or identity.
    """

    def __init__(
        self,
        irreps_in: Irreps,
        acts: list[Optional[Callable]],
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self.irreps_in = Irreps(irreps_in)

        if len(self.irreps_in) != len(acts):
            raise ValueError(
                f'Irreps in and number of activation functions does not match: {len(acts), (irreps_in, acts)}'
            )

        # normalize the second moment
        acts = [normalize2mom(act) if act is not None else None for act in acts]

        irreps_out = []
        for (mul, (l_in, p_in)), act in zip(self.irreps_in, acts):
            if act is not None:
                if l_in != 0:
                    raise ValueError(
                        'Activation: cannot apply an activation function to a non-scalar input.'
                    )

                # parity check using a sample vector
                x = jnp.linspace(0, 10, 256)
                a1, a2 = act(x), act(-x)
                if jnp.max(jnp.abs(a1 - a2)) < 1e-5:
                    p_act = 1
                elif jnp.max(jnp.abs(a1 + a2)) < 1e-5:
                    p_act = -1
                else:
                    p_act = 0

                p_out = p_act if p_in == -1 else p_in
                irreps_out.append((mul, (0, p_out)))

                if p_out == 0:
                    raise ValueError(
                        'Activation: the parity is violated! The input scalar is odd but the activation is neither even nor odd.'
                    )
            else:
                irreps_out.append((mul, (l_in, p_in)))

        self.irreps_out = Irreps(irreps_out)
        self.acts = acts
        self.paths = [
            (mul, (l, p), act) for (mul, (l, p)), act in zip(self.irreps_in, self.acts)
        ]

    def __repr__(self) -> str:
        acts_str = ''.join(['x' if a is not None else ' ' for a in self.acts])
        return f'{self.__class__.__name__} [{acts_str}] ({self.irreps_in} -> {self.irreps_out})'

    def __call__(self, features: jnp.ndarray, axis: int = -1) -> jnp.ndarray:
        """Evaluate the activation function.

        Parameters
        ----------
        features : jnp.ndarray
            Tensor of shape (..., channels)
        axis : int
            Axis along which the irreps are stored.

        Returns
        -------
        jnp.ndarray
            Tensor with the same shape as input.
        """
        output = []
        index = 0
        for mul, (l, _), act in self.paths:
            ir_dim = 2 * l + 1
            if act is not None:
                output.append(
                    act(jnp.take(features, jnp.arange(index, index + mul), axis=axis))
                )
            else:
                output.append(
                    jnp.take(
                        features, jnp.arange(index, index + mul * ir_dim), axis=axis
                    )
                )
            index += mul * ir_dim

        if len(output) > 1:
            return jnp.concatenate(output, axis=axis)
        elif len(output) == 1:
            return output[0]
        else:
            return jnp.zeros_like(features)
