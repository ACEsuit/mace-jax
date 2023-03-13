import e3nn as e3nn_torch
import e3nn_jax as e3nn
import haiku as hk
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import torch

from mace_jax.modules import MACE


def load_torch_model(model: torch.nn.Module,):
    num_bessel = len(model.radial_embedding.bessel_fn.bessel_weights)
    num_polynomial_cutoff = int(model.radial_embedding.cutoff_fn.p.item())
    max_ell = model.spherical_harmonics._lmax
    num_interactions = model.num_interactions.item()
    num_species = model.node_embedding.linear.irreps_in.count(e3nn_torch.o3.Irrep(0, 1))
    hidden_irreps = str(model.interactions[0].hidden_irreps)
    readout_mlp_irreps = model.readouts[-1].hidden_irreps
    avg_num_neighbors = model.interactions[0].avg_num_neighbors
    correlation = model.products[0].symmetric_contractions.contractions[0].correlation
    atomic_energies = model.atomic_energies_fn.atomic_energies.numpy()
    residual_first = (
        model.interactions[0].__class__.__name__
        == "RealAgnosticResidualInteractionBlock"
    )

    # check if model has scale_shift layer
    mean = 0.0
    std = 1.0
    if hasattr(model, "scale_shift"):
        mean = model.scale_shift.shift
        std = model.scale_shift.scale

    config_torch = dict(
        num_bessel=num_bessel,
        num_polynomial_cutoff=num_polynomial_cutoff,
        max_ell=max_ell,
        num_interactions=num_interactions,
        num_species=num_species,
        hidden_irreps=hidden_irreps,
        readout_mlp_irreps=readout_mlp_irreps,
        avg_num_neighbors=avg_num_neighbors,
        correlation=correlation,
        residual_first=residual_first,
        mean=mean,
        std=std,
        atomic_energies=atomic_energies,
    )

    @hk.without_apply_rng
    @hk.transform
    def jax_model(
        vectors: jnp.ndarray,  # [n_edges, 3]
        node_specie: jnp.ndarray,  # [n_nodes, #scalar_features]
        senders: jnp.ndarray,  # [n_edges]
        receivers: jnp.ndarray,  # [n_edges]
    ):
        e3nn.config("path_normalization", "path")
        e3nn.config("gradient_normalization", "path")
        mace = MACE(
            r_max=float(model.r_max),
            radial_basis=lambda r, r_max: e3nn.bessel(r, num_bessel, r_max),
            radial_envelope=lambda r, r_max: e3nn.poly_envelope(
                num_polynomial_cutoff - 1, 2, r_max
            )(r),
            max_ell=max_ell,
            num_interactions=num_interactions,
            num_species=num_species,
            hidden_irreps=hidden_irreps,
            readout_mlp_irreps=readout_mlp_irreps,
            avg_num_neighbors=avg_num_neighbors,
            correlation=correlation,
            output_irreps="0e",
            symmetric_tensor_product_basis=False,
            torch_style=True,
            residual_first=residual_first,
        )
        contributions = mace(
            vectors, node_specie, senders, receivers
        )  # [n_nodes, num_interactions, 0e]
        contributions = contributions.array[:, :, 0]  # [n_nodes, num_interactions]
        node_energies = jnp.sum(contributions, axis=1)  # [n_nodes, ]

        node_energies = mean + std * node_energies

        atomic_energies_ = jnp.asarray(atomic_energies)
        node_energies += atomic_energies_[node_specie]  # [n_nodes, ]
        return node_energies

    params = jax.jit(jax_model.init)(
        jax.random.PRNGKey(0),
        jnp.zeros((1, 3)),
        jnp.array([16]),
        jnp.array([0]),
        jnp.array([0]),
    )
    params_from_torch = create_jax_params(model, config_torch)
    assert jax.tree_util.tree_structure(params) == jax.tree_util.tree_structure(
        params_from_torch
    )
    return jax_model.apply, params_from_torch


def linear_torch_to_jax(linear):
    return {
        f"w[{ins.i_in},{ins.i_out}] {linear.irreps_in[ins.i_in]},{linear.irreps_out[ins.i_out]}": jnp.asarray(
            w.data
        )
        for i, ins, w in linear.weight_views(yield_instruction=True)
    }


def skip_tp_torch_to_jax(tp):
    return {
        f"w[{ins.i_in1},{ins.i_out}] {tp.irreps_in1[ins.i_in1]},{tp.irreps_out[ins.i_out]}": jnp.moveaxis(
            jnp.asarray(w.data), 1, 0
        )
        for i, ins, w in tp.weight_views(yield_instruction=True)
    }


def create_jax_params(model: torch.nn.Module, config_torch: dict):
    params = {
        "mace/~/linear_node_embedding_block": {
            "embeddings": (
                model.node_embedding.linear.weight.detach().numpy().reshape((1, -1))
            )
        },
    }
    num_interactions = config_torch["num_interactions"]
    correlation = config_torch["correlation"]
    residual_first = config_torch["residual_first"]
    if not residual_first:
        params["mace/layer_0/skip_tp_first"] = skip_tp_torch_to_jax(
            model.interactions[0].skip_tp
        )

    for i in range(num_interactions):
        if i != 0 or residual_first:
            params[f"mace/layer_{i}/skip_tp"] = skip_tp_torch_to_jax(
                model.interactions[i].skip_tp
            )
        params[f"mace/layer_{i}/interaction_block/linear_up"] = linear_torch_to_jax(
            model.interactions[i].linear_up
        )
        params[f"mace/layer_{i}/interaction_block/linear_down"] = linear_torch_to_jax(
            model.interactions[i].linear
        )

        params[
            f"mace/layer_{i}/interaction_block/message_passing_convolution/multi_layer_perceptron/linear_0"
        ] = {
            "w": (model.interactions[i].conv_tp_weights.layer0.weight.detach().numpy())
        }

        params[
            f"mace/layer_{i}/interaction_block/message_passing_convolution/multi_layer_perceptron/linear_1"
        ] = {
            "w": (model.interactions[i].conv_tp_weights.layer1.weight.detach().numpy())
        }

        params[
            f"mace/layer_{i}/interaction_block/message_passing_convolution/multi_layer_perceptron/linear_2"
        ] = {
            "w": (model.interactions[i].conv_tp_weights.layer2.weight.detach().numpy())
        }

        params[
            f"mace/layer_{i}/interaction_block/message_passing_convolution/multi_layer_perceptron/linear_3"
        ] = {
            "w": (model.interactions[i].conv_tp_weights.layer3.weight.detach().numpy())
        }

        product_params = {}
        irreps_out = model.products[i].linear.irreps_in
        for corr in range(correlation - 1, -1, -1):
            for j, (mul, ir) in enumerate(irreps_out):
                if corr == correlation - 1:
                    product_params[f"w{corr + 1}_{ir}"] = (
                        model.products[i]
                        .symmetric_contractions.contractions[j]
                        .weights_max.detach()
                        .numpy()
                    )

                else:
                    product_params[f"w{corr + 1}_{ir}"] = (
                        model.products[i]
                        .symmetric_contractions.contractions[j]
                        .weights[corr]
                        .detach()
                        .numpy()
                    )

        params[
            f"mace/layer_{i}/equivariant_product_basis_block/~/symmetric_contraction"
        ] = product_params
        params[
            f"mace/layer_{i}/equivariant_product_basis_block/linear"
        ] = linear_torch_to_jax(model.products[i].linear)

        if i != num_interactions - 1:
            params[f"mace/layer_{i}/linear_readout_block/linear"] = linear_torch_to_jax(
                model.readouts[i].linear
            )
        else:
            params[
                f"mace/layer_{i}/non_linear_readout_block/linear"
            ] = linear_torch_to_jax(model.readouts[i].linear_1)
            params[
                f"mace/layer_{i}/non_linear_readout_block/linear_1"
            ] = linear_torch_to_jax(model.readouts[i].linear_2)

    return params

