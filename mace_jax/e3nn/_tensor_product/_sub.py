from typing import Optional

from e3nn_jax import Irreps

from mace_jax.haiku.torch import register_import

from ._tensor_product import TensorProduct


@register_import('e3nn.o3._tensor_product._sub.FullyConnectedTensorProduct')
class FullyConnectedTensorProduct(TensorProduct):
    r"""Fully-connected weighted tensor product

    All the possible paths allowed by :math:`|l_1 - l_2| \leq l_{out} \leq l_1 + l_2` are made.
    The output is a sum on different paths:

    .. math::

        z_w = \sum_{u,v} w_{uvw} x_u \otimes y_v + \cdots \text{other paths}

    where :math:`u,v,w` are the indices of the multiplicities.

    Parameters
    ----------
    irreps_in1 : `Irreps`
        representation of the first input

    irreps_in2 : `Irreps`
        representation of the second input

    irreps_out : `Irreps`
        representation of the output

    irrep_normalization : {'component', 'norm'}
        see `TensorProduct`

    path_normalization : {'element', 'path'}
        see `TensorProduct`

    internal_weights : bool
        see `TensorProduct`

    shared_weights : bool
        see `TensorProduct`
    """

    def __init__(
        self,
        irreps_in1,
        irreps_in2,
        irreps_out,
        irrep_normalization: str = None,
        path_normalization: str = None,
        name: Optional[str] = None,
        **kwargs,
    ):
        irreps_in1 = Irreps(irreps_in1)
        irreps_in2 = Irreps(irreps_in2)
        irreps_out = Irreps(irreps_out)

        instructions = [
            (i_1, i_2, i_out, 'uvw', True, 1.0)
            for i_1, (_, ir_1) in enumerate(irreps_in1)
            for i_2, (_, ir_2) in enumerate(irreps_in2)
            for i_out, (_, ir_out) in enumerate(irreps_out)
            if ir_out in ir_1 * ir_2
        ]
        super().__init__(
            irreps_in1=irreps_in1,
            irreps_in2=irreps_in2,
            irreps_out=irreps_out,
            instructions=instructions,
            irrep_normalization=irrep_normalization,
            path_normalization=path_normalization,
            name=name,
            **kwargs,
        )
