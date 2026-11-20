"""Spin-orbit message passing and magnetic product bases."""

import jax
import jax.numpy as jnp
from e3nn_jax import Irreps

from mace_jax.adapters.e3nn import nn
from mace_jax.adapters.nnx.torch import nxx_auto_import_from_torch, nxx_register_module
from mace_jax.modules.blocks import EquivariantProductBasisBlock, InteractionBlock
from mace_jax.modules.irreps_tools import (
    reshape_irreps,
    tp_out_irreps_with_instructions,
)
from mace_jax.modules.wrapper_ops import (
    FullyConnectedTensorProduct,
    Linear,
    TensorProduct,
)
from mace_jax.tools.scatter import scatter_sum


def _tensor_product(left, right, target, config, rngs):
    mid, instructions = tp_out_irreps_with_instructions(left, right, target)
    return TensorProduct(
        left,
        right,
        mid,
        instructions=instructions,
        shared_weights=False,
        internal_weights=False,
        equivariance_config=config,
        rngs=rngs,
    )


def _weights(widths, tp, rngs, act=None):
    return nn.FullyConnectedNet(
        hs=[*widths, tp.weight_numel],
        act=act,
        output_permutation=getattr(tp, 'projection_permutation', None),
        rngs=rngs,
    )


def _apply_tp(tp, left, right, weights):
    options = (
        {'weights_are_openeq_order': True}
        if getattr(tp, 'expects_projected_weights', False)
        else {}
    )
    return tp(left, right, weights, **options)


class _MagneticInteraction(InteractionBlock):
    residual = False

    def __init__(
        self, *, magmom_node_inv_feats_irreps, magmom_node_attrs_irreps, **kwargs
    ):
        self.magmom_node_inv_feats_irreps = Irreps(magmom_node_inv_feats_irreps)
        self.magmom_node_attrs_irreps = Irreps(magmom_node_attrs_irreps)
        super().__init__(**kwargs)
        if hasattr(self, 'conv_fusion'):
            raise ValueError('Magnetic interactions require unfused tensor products.')

    def _setup(self, rngs):
        config = self.equivariance_config
        self.linear_up = Linear(
            self.node_feats_irreps,
            self.node_feats_irreps,
            equivariance_config=config,
            rngs=rngs,
        )
        self.conv_tp = _tensor_product(
            self.node_feats_irreps,
            self.edge_attrs_irreps,
            self.target_irreps,
            config,
            rngs,
        )
        self.magmom_conv_tp = _tensor_product(
            self.conv_tp.irreps_out,
            self.magmom_node_attrs_irreps,
            self.target_irreps,
            config,
            rngs,
        )
        width = (
            self.edge_feats_irreps.num_irreps
            + self.magmom_node_inv_feats_irreps.num_irreps
        )
        self.conv_tp_weights = _weights(
            [width, *self.radial_MLP], self.conv_tp, rngs, jax.nn.silu
        )
        self.conv_tp_weights_magmom = _weights([width], self.magmom_conv_tp, rngs)
        self.magmom_linear = Linear(
            self.magmom_conv_tp.irreps_out,
            self.target_irreps,
            equivariance_config=config,
            rngs=rngs,
        )
        self.density_fn = nn.FullyConnectedNet(
            [self.edge_feats_irreps.num_irreps, 1], act=jax.nn.silu, rngs=rngs
        )
        self.reshape = reshape_irreps(self.target_irreps, equivariance_config=config)
        if self.residual:
            self.skip_tp = FullyConnectedTensorProduct(
                self.node_feats_irreps,
                self.node_attrs_irreps,
                self.hidden_irreps,
                equivariance_config=config,
                rngs=rngs,
            )
        else:
            self.magmom_skip_tp = FullyConnectedTensorProduct(
                self.target_irreps,
                self.node_attrs_irreps,
                self.target_irreps,
                equivariance_config=config,
                rngs=rngs,
            )

    def __call__(
        self,
        node_attrs,
        node_feats,
        edge_attrs,
        edge_feats,
        edge_index,
        magmom_node_inv_feats,
        magmom_node_attrs,
        cutoff=None,
        n_real=None,
        first_layer=False,
    ):
        if n_real is not None:
            raise ValueError('Magnetic interactions do not yet support ghost exchange.')
        sender, receiver = edge_index
        n = len(node_feats)
        sc = self.skip_tp(node_feats, node_attrs) if self.residual else None
        features = jnp.concatenate((edge_feats, magmom_node_inv_feats[sender]), axis=-1)
        weights = self.conv_tp_weights(features)
        density = jnp.tanh(self.density_fn(edge_feats) ** 2)
        if cutoff is not None:
            weights *= cutoff
            density *= cutoff
        message = _apply_tp(
            self.conv_tp, self.linear_up(node_feats)[sender], edge_attrs, weights
        )
        message = _apply_tp(
            self.magmom_conv_tp,
            message,
            magmom_node_attrs[sender],
            self.conv_tp_weights_magmom(features),
        )
        density = scatter_sum(density, receiver, dim=0, dim_size=n)
        message = self.magmom_linear(
            scatter_sum(message, receiver, dim=0, dim_size=n)
        ) / (density + 1)
        if not self.residual:
            message = self.magmom_skip_tp(message, node_attrs)
        return self.reshape(message), sc


@nxx_register_module(
    'mace.modules.blocks.MagneticRealAgnosticSpinOrbitCoupledDensityInteractionBlock'
)
@nxx_auto_import_from_torch(allow_missing_mapper=True)
class MagneticRealAgnosticSpinOrbitCoupledDensityInteractionBlock(_MagneticInteraction):
    pass


@nxx_register_module(
    'mace.modules.blocks.MagneticRealAgnosticResidueSpinOrbitCoupledDensityInteractionBlock'
)
@nxx_auto_import_from_torch(allow_missing_mapper=True)
class MagneticRealAgnosticResidueSpinOrbitCoupledDensityInteractionBlock(
    _MagneticInteraction
):
    residual = True


@nxx_register_module('mace.modules.blocks.EquivariantProductBasisWithSelfMagmomBlock')
@nxx_auto_import_from_torch(allow_missing_mapper=True)
class EquivariantProductBasisWithSelfMagmomBlock(EquivariantProductBasisBlock):
    def __init__(
        self, *, magmom_node_inv_feats_irreps, magmom_node_attrs_irreps, rngs, **kwargs
    ):
        super().__init__(rngs=rngs, **kwargs)
        config = self.equivariance_config
        self.linear_ori = self.linear
        self.conv_tp = _tensor_product(
            self.target_irreps,
            Irreps(magmom_node_attrs_irreps),
            self.target_irreps,
            config,
            rngs,
        )
        self.conv_tp_weights = _weights(
            [Irreps(magmom_node_inv_feats_irreps).dim, 64, 64, 64],
            self.conv_tp,
            rngs,
            jax.nn.silu,
        )
        self.linear = Linear(
            self.conv_tp.irreps_out,
            self.target_irreps,
            equivariance_config=config,
            rngs=rngs,
        )

    def __call__(
        self,
        node_feats,
        sc,
        node_attrs,
        magmom_node_inv_feats,
        magmom_node_attrs,
        node_attrs_index=None,
    ):
        features = self._contract(node_feats, node_attrs, node_attrs_index)
        magnetic = _apply_tp(
            self.conv_tp,
            features,
            magmom_node_attrs,
            self.conv_tp_weights(magmom_node_inv_feats),
        )
        result = self.linear(magnetic) + self.linear_ori(features)
        return result + sc if self.use_sc and sc is not None else result
