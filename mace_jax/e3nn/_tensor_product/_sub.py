import operator
from functools import reduce
from typing import Iterator, Optional

import haiku as hk
import jax.numpy as jnp
from e3nn_jax import Irrep, Irreps

from ._tensor_product import TensorProduct


class FullyConnectedTensorProduct(hk.Module):
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
        **kwargs,
    ):
        super().__init__()
        irreps_in1 = Irreps(irreps_in1)
        irreps_in2 = Irreps(irreps_in2)
        irreps_out = Irreps(irreps_out)

        instructions = [
            (i_1, i_2, i_out, "uvw", True, 1.0)
            for i_1, (_, ir_1) in enumerate(irreps_in1)
            for i_2, (_, ir_2) in enumerate(irreps_in2)
            for i_out, (_, ir_out) in enumerate(irreps_out)
            if ir_out in ir_1 * ir_2
        ]

        self.tp = TensorProduct(
            irreps_in1=irreps_in1,
            irreps_in2=irreps_in2,
            irreps_out=irreps_out,
            instructions=instructions,
            irrep_normalization=irrep_normalization,
            path_normalization=path_normalization,
            **kwargs,
        )

    def __call__(self, x, y, weights=None):
        return self.tp(x, y, weights)


class ElementwiseTensorProduct(hk.Module):
    r"""Elementwise connected tensor product.

    .. math::

        z_u = x_u \otimes y_u

    where :math:`u` runs over the irreps. Note that there are no weights.
    The output representation is determined by the two input representations.

    Parameters
    ----------
    irreps_in1 : `Irreps`
        representation of the first input

    irreps_in2 : `Irreps`
        representation of the second input

    filter_ir_out : iterator of `Irrep`, optional
        filter to select only specific `Irrep`s of the output

    irrep_normalization : {'component', 'norm'}
        see `TensorProduct`

    Examples
    --------
    Elementwise scalar product

    >>> ElementwiseTensorProduct("5x1o + 5x1e", "10x1e", ["0e", "0o"])
    ElementwiseTensorProduct(5x1o+5x1e x 10x1e -> 5x0o+5x0e | 10 paths | 0 weights)
    """

    def __init__(
        self,
        irreps_in1,
        irreps_in2,
        filter_ir_out=None,
        irrep_normalization: str = None,
        **kwargs,
    ):
        super().__init__()
        irreps_in1 = Irreps(irreps_in1).simplify()
        irreps_in2 = Irreps(irreps_in2).simplify()

        if filter_ir_out is not None:
            try:
                filter_ir_out = [Irrep(ir) for ir in filter_ir_out]
            except ValueError:
                raise ValueError(
                    f"filter_ir_out (={filter_ir_out}) must be an iterable of Irrep"
                )

        assert irreps_in1.num_irreps == irreps_in2.num_irreps

        irreps_in1 = list(irreps_in1)
        irreps_in2 = list(irreps_in2)

        # Align multiplicities
        i = 0
        while i < len(irreps_in1):
            mul_1, ir_1 = irreps_in1[i]
            mul_2, ir_2 = irreps_in2[i]

            if mul_1 < mul_2:
                irreps_in2[i] = (mul_1, ir_2)
                irreps_in2.insert(i + 1, (mul_2 - mul_1, ir_2))

            if mul_2 < mul_1:
                irreps_in1[i] = (mul_2, ir_1)
                irreps_in1.insert(i + 1, (mul_1 - mul_2, ir_1))
            i += 1

        out = []
        instructions = []
        for i, ((mul, ir_1), (mul_2, ir_2)) in enumerate(zip(irreps_in1, irreps_in2)):
            assert mul == mul_2
            for ir in ir_1 * ir_2:
                if filter_ir_out is not None and ir not in filter_ir_out:
                    continue
                i_out = len(out)
                out.append((mul, ir))
                instructions.append((i, i, i_out, "uuu", False))

        self.tp = TensorProduct(
            irreps_in1=irreps_in1,
            irreps_in2=irreps_in2,
            irreps_out=out,
            instructions=instructions,
            irrep_normalization=irrep_normalization,
            **kwargs,
        )

    def __call__(self, x, y, weights=None):
        return self.tp(x, y, weights)


class FullTensorProduct(hk.Module):
    r"""Full tensor product between two irreps.

    .. math::

        z_{uv} = x_u \otimes y_v

    where :math:`u` and :math:`v` run over the irreps. Note that there are no weights.
    The output representation is determined by the two input representations.

    Parameters
    ----------
    irreps_in1 : `Irreps`
        representation of the first input

    irreps_in2 : `Irreps`
        representation of the second input

    filter_ir_out : iterator of `Irrep`, optional
        filter to select only specific `Irrep`s of the output

    irrep_normalization : {'component', 'norm'}
        see `TensorProduct`
    """

    def __init__(
        self,
        irreps_in1: Irreps,
        irreps_in2: Irreps,
        filter_ir_out: Optional[Iterator[Irrep]] = None,
        irrep_normalization: str = None,
        **kwargs,
    ):
        super().__init__()
        irreps_in1 = Irreps(irreps_in1).simplify()
        irreps_in2 = Irreps(irreps_in2).simplify()

        if filter_ir_out is not None:
            try:
                filter_ir_out = [Irrep(ir) for ir in filter_ir_out]
            except ValueError:
                raise ValueError(
                    f"filter_ir_out (={filter_ir_out}) must be an iterable of Irrep"
                )

        out = []
        instr = []
        for i_1, (mul_1, ir_1) in enumerate(irreps_in1):
            for i_2, (mul_2, ir_2) in enumerate(irreps_in2):
                for ir_out in ir_1 * ir_2:
                    if filter_ir_out is not None and ir_out not in filter_ir_out:
                        continue
                    i_out = len(out)
                    out.append((mul_1 * mul_2, ir_out))
                    instr.append((i_1, i_2, i_out, "uvuv", False))

        out = Irreps(out)
        out, p, _ = out.sort()
        instr = [
            (i_1, i_2, p[i_out], mode, train) for i_1, i_2, i_out, mode, train in instr
        ]

        self.tp = TensorProduct(
            irreps_in1=irreps_in1,
            irreps_in2=irreps_in2,
            irreps_out=out,
            instructions=instr,
            irrep_normalization=irrep_normalization,
            **kwargs,
        )

    def __call__(self, x, y, weights=None):
        return self.tp(x, y, weights)


def _square_instructions_full(irreps_in, filter_ir_out=None, irrep_normalization=None):
    """Generate instructions for square tensor product.

    Parameters
    ----------
    irreps_in : `Irreps`
        representation of the input

    filter_ir_out : iterator of `Irrep`, optional
        filter to select only specific `Irrep` of the output

    irrep_normalization : {'component', 'norm', 'none'}
        see `e3nn.o3.TensorProduct`

    Returns
    -------
    irreps_out : `Irreps`
        representation of the output

    instr : list of tuple
        list of instructions

    """
    # pylint: disable=too-many-nested-blocks
    irreps_out = []
    instr = []
    for i_1, (mul_1, ir_1) in enumerate(irreps_in):
        for i_2, (mul_2, ir_2) in enumerate(irreps_in):
            for ir_out in ir_1 * ir_2:
                if filter_ir_out is not None and ir_out not in filter_ir_out:
                    continue

                if irrep_normalization == "component":
                    alpha = ir_out.dim
                if irrep_normalization == "norm":
                    alpha = ir_1.dim * ir_2.dim
                if irrep_normalization == "none":
                    alpha = 1

                if i_1 < i_2:
                    i_out = len(irreps_out)
                    irreps_out.append((mul_1 * mul_2, ir_out))
                    instr += [(i_1, i_2, i_out, "uvuv", False, alpha)]
                elif i_1 == i_2:
                    i = i_1
                    mul = mul_1

                    if mul > 1:
                        i_out = len(irreps_out)
                        irreps_out.append((mul * (mul - 1) // 2, ir_out))
                        instr += [(i, i, i_out, "uvu<v", False, alpha)]

                    if ir_out.l % 2 == 0:
                        if irrep_normalization == "component":
                            if ir_out.l == 0:
                                alpha = ir_out.dim / (ir_1.dim + 2)
                            else:
                                alpha = ir_out.dim / 2
                        if irrep_normalization == "norm":
                            if ir_out.l == 0:
                                alpha = ir_out.dim * ir_1.dim
                            else:
                                alpha = ir_1.dim * (ir_1.dim + 2) / 2

                        i_out = len(irreps_out)
                        irreps_out.append((mul, ir_out))
                        instr += [(i, i, i_out, "uuu", False, alpha)]

    irreps_out = Irreps(irreps_out)
    irreps_out, p, _ = irreps_out.sort()

    instr = [
        (i_1, i_2, p[i_out], mode, train, alpha)
        for i_1, i_2, i_out, mode, train, alpha in instr
    ]

    return irreps_out, instr


def _square_instructions_fully_connected(
    irreps_in, irreps_out, irrep_normalization=None
):
    """Generate instructions for square tensor product.

    Parameters
    ----------
    irreps_in : `Irreps`
        representation of the input

    irreps_out : `Irreps`
        representation of the output

    irrep_normalization : {'component', 'norm', 'none'}
        see `e3nn.o3.TensorProduct`

    Returns
    -------
    instr : list of tuple
        list of instructions
    """
    # pylint: disable=too-many-nested-blocks
    instr = []
    for i_1, (mul_1, ir_1) in enumerate(irreps_in):
        for i_2, (_mul_2, ir_2) in enumerate(irreps_in):
            for i_out, (_mul_out, ir_out) in enumerate(irreps_out):
                if ir_out in ir_1 * ir_2:
                    if irrep_normalization == "component":
                        alpha = ir_out.dim
                    if irrep_normalization == "norm":
                        alpha = ir_1.dim * ir_2.dim
                    if irrep_normalization == "none":
                        alpha = 1

                    if i_1 < i_2:
                        instr += [(i_1, i_2, i_out, "uvw", True, alpha)]
                    elif i_1 == i_2:
                        i = i_1
                        mul = mul_1

                        if mul > 1:
                            instr += [(i, i, i_out, "u<vw", True, alpha)]

                        if ir_out.l % 2 == 0:
                            if irrep_normalization == "component":
                                if ir_out.l == 0:
                                    alpha = ir_out.dim / (ir_1.dim + 2)
                                else:
                                    alpha = ir_out.dim / 2
                            if irrep_normalization == "norm":
                                if ir_out.l == 0:
                                    alpha = ir_out.dim * ir_1.dim
                                else:
                                    alpha = ir_1.dim * (ir_1.dim + 2) / 2

                            instr += [(i, i, i_out, "uuw", True, alpha)]

    return instr


def prod(shape):
    return reduce(operator.mul, shape, 1)


class TensorSquare(hk.Module):
    r"""Compute the square tensor product of a tensor and reduce it in irreps.

    If `irreps_out` is given, this operation is fully connected.
    If `irreps_out` is not given, the operation has no parameter and is like a full tensor product.
    """

    def __init__(
        self,
        irreps_in: Irreps,
        irreps_out: Optional[Irreps] = None,
        filter_ir_out: Optional[Iterator[Irrep]] = None,
        irrep_normalization: Optional[str] = None,
        name: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(name=name)
        if irrep_normalization is None:
            irrep_normalization = "component"
        assert irrep_normalization in ["component", "norm", "none"]

        self.irreps_in = Irreps(irreps_in).simplify()
        self.irreps_out = None  # will set later
        self._instr = None
        self._tp = None
        self._irrep_normalization = irrep_normalization
        self._kwargs = kwargs

        if filter_ir_out is not None:
            try:
                filter_ir_out = [Irrep(ir) for ir in filter_ir_out]
            except ValueError as exc:
                raise ValueError(
                    f"Error constructing filter_ir_out irrep: {exc}"
                ) from exc

        if irreps_out is None:
            irreps_out, instr = _square_instructions_full(
                self.irreps_in, filter_ir_out, irrep_normalization
            )
        else:
            if filter_ir_out is not None:
                raise ValueError(
                    "Both `irreps_out` and `filter_ir_out` are not None, this is ambiguous."
                )
            irreps_out = Irreps(irreps_out).simplify()
            instr = _square_instructions_fully_connected(
                self.irreps_in, irreps_out, irrep_normalization
            )

        self.irreps_out = irreps_out
        self._instr = instr

        # instantiate the internal TensorProduct (JAX version)
        self._tp = TensorProduct(
            self.irreps_in,
            self.irreps_in,
            self.irreps_out,
            self._instr,
            irrep_normalization="none",
            **kwargs,
        )

    def __call__(self, x: jnp.ndarray, weight: Optional[jnp.ndarray] = None):
        """Forward pass: compute x ⊗ x, optionally with weights."""
        return self._tp(x, x, weight)

    def __repr__(self) -> str:
        try:
            weight_info = (
                self._tp.total_weight_numel
                if self._tp is not None
                else "not initialized"
            )
        except AttributeError:
            weight_info = "not initialized"

        npath = sum(jnp.prod(jnp.array(i.path_shape)) for i in self._instr)
        return (
            f"{self.__class__.__name__}"
            f"({self.irreps_in} -> {self.irreps_out.simplify()} | "
            f"{npath} paths | {weight_info} weights)"
        )
