"""
Wrapper class for o3.Linear that optionally uses cuet.Linear
"""

import dataclasses
from typing import Callable, Optional

import jax.numpy as jnp
from e3nn_jax import Irreps

from mace_jax.e3nn import _linear
from mace_jax.e3nn import _tensor_product as _tp
from mace_jax.modules.symmetric_contraction import SymmetricContraction
from mace_jax.tools.cg import O3_e3nn
from mace_jax.tools.scatter import scatter_sum

try:
    import cuequivariance as cue
    import cuequivariance_jax as cuex

    CUEX_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    CUEX_AVAILABLE = False


@dataclasses.dataclass
class CuEquivarianceConfig:
    """Configuration for cuequivariance acceleration"""

    enabled: bool = False
    layout: str = 'mul_ir'  # One of: mul_ir, ir_mul
    layout_str: str = 'mul_ir'
    group: str = 'O3'
    optimize_all: bool = False  # Set to True to enable all optimizations
    optimize_linear: bool = False
    optimize_channelwise: bool = False
    optimize_symmetric: bool = False
    optimize_fctp: bool = False
    conv_fusion: bool = False  # Set to True to enable conv fusion

    def __post_init__(self):
        if self.enabled and CUEX_AVAILABLE:
            self.layout_str = self.layout
            self.layout = getattr(cue, self.layout)
            self.group = (
                O3_e3nn if self.group == 'O3_e3nn' else getattr(cue, self.group)
            )
        if not CUEX_AVAILABLE:
            self.enabled = False


class Linear:
    """Returns either a cuet.Linear or o3.Linear based on config"""

    def __new__(
        cls,
        irreps_in: Irreps,
        irreps_out: Irreps,
        shared_weights: bool = True,
        internal_weights: bool = True,
        cueq_config: Optional[CuEquivarianceConfig] = None,
        name: Optional[str] = None,
    ):
        if (
            CUEX_AVAILABLE
            and cueq_config is not None
            and cueq_config.enabled
            and (cueq_config.optimize_all or cueq_config.optimize_linear)
        ):
            raise NotImplementedError(
                'cuex.Linear is not available in cuequivariance-jax.'
            )

        return _linear.Linear(
            irreps_in,
            irreps_out,
            shared_weights=shared_weights,
            internal_weights=internal_weights,
            name=name,
        )


def with_scatter_sum(conv_fn: Callable) -> Callable:
    """
    Wrap a convolution-like function with scatter_sum aggregation.

    Args:
        conv_fn: a function with signature
            conv_fn(node_feats[sender], edge_attrs, tp_weights) -> messages

    Returns:
        A wrapped function with signature:
            (node_feats, edge_attrs, tp_weights, edge_index) -> aggregated messages
    """

    def wrapped(
        node_feats: jnp.ndarray,
        edge_attrs: jnp.ndarray,
        tp_weights: jnp.ndarray,
        edge_index: jnp.ndarray,
    ) -> jnp.ndarray:
        sender = edge_index[0]
        receiver = edge_index[1]
        num_nodes = node_feats.shape[0]

        mji = conv_fn(node_feats[sender], edge_attrs, tp_weights)
        message = scatter_sum(src=mji, index=receiver, dim=0, dim_size=num_nodes)
        return message

    return wrapped


def with_cueq_conv_fusion(
    conv_fn: Callable, num_segments: int, num_operands: int
) -> Callable:
    """
    Wrap a ConvTensorProduct-like function to use conv fusion.

    Args:
        conv_fn: callable implementing the fused cuEQ conv, typically conv_fn(inputs, senders, nodes, receivers).
        num_segments: number of segments in buffer (analogous to conv_tp.m.buffer_num_segments[0]).
        num_operands: operand extent (analogous to conv_tp.m.operand_extent).

    Returns:
        A wrapped function with signature:
            (node_feats, edge_attrs, tp_weights, edge_index) -> tensor
    """
    weight_numel = num_segments * num_operands

    def wrapped(
        node_feats: jnp.ndarray,
        edge_attrs: jnp.ndarray,
        tp_weights: jnp.ndarray,
        edge_index: jnp.ndarray,
    ) -> jnp.ndarray:
        sender = edge_index[0]
        receiver = edge_index[1]
        # conv_fn is expected to return a tuple/list, like original_forward(...)[0]
        return conv_fn(
            [tp_weights, node_feats, edge_attrs],
            {1: sender},
            {0: node_feats},
            {0: receiver},
        )[0]

    wrapped.weight_numel = weight_numel
    return wrapped


class TensorProduct:
    """Wrapper around o3.TensorProduct / cuequivariance_jax.segmented_polynomial"""

    def __new__(
        cls,
        irreps_in1: Irreps,
        irreps_in2: Irreps,
        irreps_out: Irreps,
        instructions=None,
        shared_weights: bool = False,
        internal_weights: bool = False,
        cueq_config=None,
        name: Optional[str] = None,
    ):
        # --- Case 1: CuEquivariance backend ---
        if cueq_config is not None and cueq_config.enabled:
            # build polynomial descriptor
            poly_desc = (
                cuex.descriptors.channelwise_tensor_product(
                    cuex.Irreps(cueq_config.group, irreps_in1),
                    cuex.Irreps(cueq_config.group, irreps_in2),
                    cuex.Irreps(cueq_config.group, irreps_out),
                )
                .flatten_coefficient_modes()
                .squeeze_modes()
                .polynomial
            )

            def forward(tp_weights, node_feats, edge_attrs, sender, receiver):
                """Mimic fused forward in JAX"""
                num_nodes = node_feats.shape[0]
                output_shape = (num_nodes, irreps_out.dim)

                return cuex.segmented_polynomial(
                    polynomial=poly_desc,
                    inputs=[tp_weights, node_feats, edge_attrs],
                    outputs_shape_dtype=(output_shape, jnp.float32),
                    indices={1: sender},  # like message passing index
                    math_dtype=jnp.float32,
                    precision='highest',
                )

            return forward

        # --- Default: fallback to e3nn_jax.TensorProduct ---
        return _tp.TensorProduct(
            irreps_in1,
            irreps_in2,
            irreps_out,
            instructions=instructions,
            shared_weights=shared_weights,
            internal_weights=internal_weights,
            name=name,
        )


def FullyConnectedTensorProduct(
    irreps_in1: Irreps,
    irreps_in2: Irreps,
    irreps_out: Irreps,
    shared_weights: bool = True,
    internal_weights: bool = True,
    cueq_config: Optional[CuEquivarianceConfig] = None,
    name: Optional[str] = None,
):
    """
    Wrapper around o3.FullyConnectedTensorProduct (JAX version).
    If cuequivariance is available and enabled, would return an accelerated version.
    Otherwise, defaults to e3nn_jax.o3.FullyConnectedTensorProduct.
    """
    if (
        CUEX_AVAILABLE
        and cueq_config is not None
        and cueq_config.enabled
        and (cueq_config.optimize_all or cueq_config.optimize_fctp)
    ):
        # No JAX cuet binding available (PyTorch only).
        raise NotImplementedError(
            'cuex.FullyConnectedTensorProduct is not available in JAX.'
        )

    # Default: e3nn_jax implementation
    return _tp.FullyConnectedTensorProduct(
        irreps_in1,
        irreps_in2,
        irreps_out,
        shared_weights=shared_weights,
        internal_weights=internal_weights,
        name=name,
    )


def SymmetricContractionWrapper(
    irreps_in: Irreps,
    irreps_out: Irreps,
    correlation: int,
    num_elements: Optional[int] = None,
    cueq_config: Optional['CuEquivarianceConfig'] = None,
    use_reduced_cg: bool = True,
    name: Optional[str] = None,
):
    """
    JAX implementation of SymmetricContraction.

    Notes:
    - In PyTorch, cuet.SymmetricContraction can accelerate this.
    - In JAX, cuequivariance_jax does NOT expose SymmetricContraction, so we
      always build it ourselves from tensor products.
    """

    return SymmetricContraction(
        irreps_in=irreps_in,
        irreps_out=irreps_out,
        correlation=correlation,
        num_elements=num_elements,
        use_reduced_cg=use_reduced_cg,
        name=name,
    )


class TransposeIrrepsLayoutWrapper:
    """Wrapper around cuex.TransposeIrrepsLayout"""

    def __new__(
        cls,
        irreps: Irreps,
        source: str,
        target: str,
        cueq_config: Optional[CuEquivarianceConfig] = None,
    ):
        return None
