import ast
import logging
import os
from typing import Dict

import e3nn_jax as e3nn
import haiku as hk
import jax
import jax.numpy as jnp
import jraph
import numpy as np
import optax
import torch.nn.functional
from utils import create_error_table, get_dataset_from_xyz

import mace_jax
from mace_jax import data, modules, tools
from mace_jax.tools import (
    flatten_dict,
    get_batched_padded_graph_tuples,
    torch_geometric,
    unflatten_dict,
)
from jax_md import space, partition
from jax_md.partition import neighbor_list


def main() -> None:
    args = tools.build_default_arg_parser().parse_args()
    tag = tools.get_tag(name=args.name, seed=args.seed)

    # Setup
    jax.config.update("jax_debug_nans", args.debug_nans)
    tools.set_seeds(args.seed)
    tools.setup_logger(level=args.log_level, tag=tag, directory=args.log_dir)
    try:
        logging.info(f"MACE version: {mace_jax.__version__}")
    except AttributeError:
        logging.info("Cannot find MACE version, please install MACE via pip")
    logging.info(f"Configuration: {args}")
    # device = tools.init_device(args.device)
    tools.set_default_dtype(args.default_dtype)

    try:
        config_type_weights = ast.literal_eval(args.config_type_weights)
        assert isinstance(config_type_weights, dict)
    except Exception as e:  # pylint: disable=W0703
        logging.warning(
            f"Config type weights not specified correctly ({e}), using Default"
        )
        config_type_weights = {"Default": 1.0}

    # prepare periodic system for simulation
    disp, shift = space.periodic(args.box_size)
    # Data preparation
    collections, atomic_energies_dict = get_dataset_from_xyz(
        train_path=args.train_file,
        valid_path=args.valid_file,
        valid_fraction=args.valid_fraction,
        config_type_weights=config_type_weights,
        test_path=args.test_file,
        seed=args.seed,
        energy_key=args.energy_key,
        forces_key=args.forces_key,
    )

    logging.info(
        f"Total number of configurations: train={len(collections.train)}, valid={len(collections.valid)}, "
        f"tests=[{', '.join([name + ': ' + str(len(test_configs)) for name, test_configs in collections.tests])}]"
    )

    # Atomic number table
    # yapf: disable
    z_table = tools.get_atomic_number_table_from_zs(
        z
        for configs in (collections.train, collections.valid)
        for config in configs
        for z in config.atomic_numbers
    )
    # yapf: enable
    logging.info(z_table)
    if atomic_energies_dict is None or len(atomic_energies_dict) == 0:
        if args.E0s is not None:
            logging.info(
                "Atomic Energies not in training file, using command line argument E0s"
            )
            if args.E0s.lower() == "average":
                logging.info(
                    "Computing average Atomic Energies using least squares regression"
                )
                atomic_energies_dict = data.compute_average_E0s(
                    collections.train, z_table
                )
            else:
                try:
                    atomic_energies_dict = ast.literal_eval(args.E0s)
                    assert isinstance(atomic_energies_dict, dict)
                except Exception as e:
                    raise RuntimeError(
                        f"E0s specified invalidly, error {e} occured"
                    ) from e
        else:
            raise RuntimeError(
                "E0s not found in training file and not specified in command line"
            )
    atomic_energies: np.ndarray = np.array(
        [atomic_energies_dict[z] for z in z_table.zs]
    )
    logging.info(f"Atomic energies: {atomic_energies.tolist()}")

    train_loader = torch_geometric.dataloader.DataLoader(
        dataset=[
            data.AtomicData.from_config(config, z_table=z_table, cutoff=args.r_max)
            for config in collections.train
        ],
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )
    valid_loader = torch_geometric.dataloader.DataLoader(
        dataset=[
            data.AtomicData.from_config(config, z_table=z_table, cutoff=args.r_max)
            for config in collections.valid
        ],
        batch_size=args.valid_batch_size,
        shuffle=False,
        drop_last=False,
    )

    loss_fn = modules.WeightedEnergyForcesLoss(
        energy_weight=args.energy_weight, forces_weight=args.forces_weight
    )
    logging.info(loss_fn)

    if args.compute_avg_num_neighbors:
        args.avg_num_neighbors = modules.compute_avg_num_neighbors(train_loader)
    logging.info(f"Average number of neighbors: {args.avg_num_neighbors:.3f}")

    # Build model
    logging.info("Building model")
    model_config = dict(
        r_max=args.r_max,
        num_bessel=args.num_radial_basis,
        num_deriv_in_zero=args.num_cutoff_basis - 1,
        num_deriv_in_one=2,
        max_ell=args.max_ell,
        interaction_cls=modules.interaction_classes[args.interaction],
        num_interactions=args.num_interactions,
        hidden_irreps=e3nn.Irreps(args.hidden_irreps),
        atomic_energies=atomic_energies,
        avg_num_neighbors=args.avg_num_neighbors,
        epsilon=args.epsilon,
        disp_fn=disp,
    )

    def initialize_nbr_list(graphs: jraph.GraphsTuple):
        # initialize a jax-md neighbour list to allow for efficient simulation

        neighbour_fn = partition.neighbor_list(
            disp, args.box_size, args.r_max, format=partition.Sparse, mask_self=True
        )
        return neighbour_fn.allocate(graphs.nodes.positions)

    model: hk.Transformed

    @hk.without_apply_rng
    @hk.transform
    def model(
        graph: jraph.GraphsTuple, nbr_list: neighbor_list
    ) -> Dict[str, jnp.ndarray]:
        if args.model == "MACE":
            if args.scaling == "no_scaling":
                std = 1.0
                logging.info("No scaling selected")
            else:
                mean, std = modules.scaling_classes[args.scaling](
                    train_loader, atomic_energies
                )
            mace = modules.ScaleShiftMACE(
                **model_config,
                correlation=args.correlation,
                gate=modules.gate_dict[args.gate],
                interaction_cls_first=modules.interaction_classes[
                    "RealAgnosticInteractionBlock"
                ],
                MLP_irreps=e3nn.Irreps(args.MLP_irreps),
                atomic_inter_scale=std,
                atomic_inter_shift=0.0,
            )
        elif args.model == "ScaleShiftMACE":
            mean, std = modules.scaling_classes[args.scaling](
                train_loader, atomic_energies
            )
            mace = modules.ScaleShiftMACE(
                **model_config,
                correlation=args.correlation,
                gate=modules.gate_dict[args.gate],
                interaction_cls_first=modules.interaction_classes[
                    args.interaction_first
                ],
                MLP_irreps=e3nn.Irreps(args.MLP_irreps),
                atomic_inter_scale=std,
                atomic_inter_shift=mean,
            )
        else:
            raise RuntimeError(f"Unknown model: '{args.model}'")

        return mace(graph, nbr_list)

    # This needs to take initial positions from the dataloader

    # initialize with jax-md neighbour list
    init_positions = get_batched_padded_graph_tuples(next(iter(train_loader)))
    nbr_list = initialize_nbr_list(init_positions)
    params = jax.jit(model.init)(jax.random.PRNGKey(0), init_positions, nbr_list)
    # Optimizer

    def weight_decay_mask(params):
        params = flatten_dict(params)
        mask = {
            k: "linear_2" in k
            and "multi_layer_perceptron" not in k
            or "symmetric_contraction" in k
            for k, _ in params.items()
        }
        return unflatten_dict(mask)

    logger = tools.MetricsLogger(directory=args.results_dir, tag=tag + "_train")

    gradient_transform: optax.GradientTransformation = None
    gradient_transform = optax.chain(
        optax.clip_by_global_norm(
            np.inf if args.clip_grad is None else args.clip_grad
        ),  # Clip by the gradient by the global norm.
        optax.scale_by_adam(),  # Use the updates from adam.
        optax.add_decayed_weights(args.weight_decay, mask=weight_decay_mask),
        optax.scale_by_schedule(
            optax.exponential_decay(
                init_value=args.lr,
                transition_steps=4 * args.max_num_epochs // 5,
                decay_rate=args.lr_factor,
            )
        ),  # Use the learning rate from the scheduler.),
        optax.scale(-1.0),  # Gradient descent.
    )

    optimizer_state = gradient_transform.init(params)

    checkpoint_handler = tools.CheckpointHandler(
        directory=args.checkpoints_dir, tag=tag, keep=args.keep_checkpoints
    )

    start_epoch = 0
    # if args.restart_latest:
    #     opt_start_epoch = checkpoint_handler.load_latest(
    #         state=tools.CheckpointState(model, optimizer, lr_scheduler), device=device
    #     )
    #     if opt_start_epoch is not None:
    #         start_epoch = opt_start_epoch

    logging.info(model)
    logging.info(f"Number of parameters: {tools.count_parameters(params)}")
    logging.info(f"Optimizer: {gradient_transform}")

    params, optimizer_state = tools.train(
        model=jax.jit(model.apply),
        params=params,
        loss_fn=loss_fn,
        train_loader=train_loader,
        valid_loader=valid_loader,
        gradient_transform=gradient_transform,
        optimizer_state=optimizer_state,
        checkpoint_handler=checkpoint_handler,
        eval_interval=args.eval_interval,
        start_epoch=start_epoch,
        max_num_epochs=args.max_num_epochs,
        logger=logger,
        patience=args.patience,
        swa=None,
        ema_decay=args.ema_decay,
        max_grad_norm=args.clip_grad,
        log_errors=args.error_table,
        nbr_list=nbr_list,
    )

    # epoch = checkpoint_handler.load_latest(
    #     state=tools.CheckpointState(model, optimizer, lr_scheduler), device=device
    # )
    # logging.info(f"Loaded model from epoch {epoch}")

    # Evaluation on test datasets
    logging.info("Computing metrics for training, validation, and test sets")

    all_collections = [
        ("train", collections.train),
        ("valid", collections.valid),
    ] + collections.tests

    table = create_error_table(
        args.error_table,
        all_collections,
        z_table,
        args.r_max,
        args.valid_batch_size,
        model,
        loss_fn,
        params,
    )

    logging.info("\n" + str(table))

    # Save entire model
    model_path = os.path.join(args.checkpoints_dir, tag + ".model")
    logging.info(f"Saving model to {model_path}")
    if args.save_cpu:
        model = model.to("cpu")
    torch.save(model, model_path)

    logging.info("Done")


if __name__ == "__main__":
    main()
