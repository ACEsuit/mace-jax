import logging
from typing import Dict, Iterable, List, Sequence, Tuple

import gin
import jax
import numpy as np
from tqdm import tqdm

from mace_jax import data
from mace_jax.tools import gin_model


def _ensure_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    return [str(value)]


def _load_pseudolabel_predictor(
    checkpoint: str,
    *,
    torch_head: str | None,
    torch_param_dtype: str | None,
    r_max: float,
):
    logging.info('Loading Torch checkpoint %s for pseudolabeling', checkpoint)
    predictor_fn, params, _ = gin_model.model(
        r_max=r_max,
        torch_checkpoint=checkpoint,
        torch_head=torch_head,
        torch_param_dtype=torch_param_dtype,
    )

    def _predict(graph):
        return predictor_fn(params, graph, compute_force=True, compute_stress=True)

    return _predict


def _apply_predictions_to_config(
    cfg: data.Configuration,
    output: dict,
) -> None:
    if output.get('energy') is not None:
        energy = np.asarray(jax.device_get(output['energy'])).reshape(-1)[0]
        cfg.energy = np.asarray(energy)
    if output.get('forces') is not None:
        cfg.forces = np.asarray(jax.device_get(output['forces']))
    if output.get('stress') is not None:
        stress = np.asarray(jax.device_get(output['stress']))
        cfg.stress = np.asarray(stress.reshape(-1, 3, 3)[0])


def _maybe_apply_pseudolabels(
    *,
    head_cfg: Dict,
    train_configs: List[data.Configuration],
    valid_configs: List[data.Configuration],
    replay_configs: List[data.Configuration],
    head_to_index: Dict[str, int],
    r_max: float,
) -> None:
    checkpoint = head_cfg.get('pseudolabel_checkpoint')
    if not checkpoint:
        return

    targets = head_cfg.get('pseudolabel_targets', 'train')
    if targets not in {'train', 'valid', 'all', 'replay'}:
        raise ValueError(
            f"Unsupported pseudolabel_targets='{targets}'. Expected one of train, valid, replay, all."
        )
    if targets == 'train':
        configs = train_configs
    elif targets == 'valid':
        configs = valid_configs
    elif targets == 'all':
        configs = train_configs + valid_configs
    else:
        configs = replay_configs

    if not configs:
        logging.info(
            "No configurations available for pseudolabel target '%s'; skipping.",
            targets,
        )
        return

    predictor = _load_pseudolabel_predictor(
        checkpoint,
        torch_head=head_cfg.get('pseudolabel_head'),
        torch_param_dtype=head_cfg.get('pseudolabel_param_dtype'),
        r_max=r_max,
    )

    logging.info(
        'Applying pseudolabels from %s to %d configurations (%s target).',
        checkpoint,
        len(configs),
        targets,
    )
    for cfg in configs:
        graph = data.graph_from_configuration(
            cfg, cutoff=r_max, head_to_index=head_to_index
        )
        output = predictor(graph)
        _apply_predictions_to_config(cfg, output)


def _load_configs_from_path(
    path: str,
    *,
    config_type_weights: Dict,
    energy_key: str,
    forces_key: str,
    prefactor_stress: float,
    remap_stress: np.ndarray | None,
    head_name: str,
    num_configs: int | None,
    extract_atomic_energies: bool,
    allow_unlabeled: bool,
):
    loader_kwargs = dict(
        config_type_weights=config_type_weights,
        energy_key=energy_key,
        forces_key=forces_key,
        prefactor_stress=prefactor_stress,
        remap_stress=remap_stress,
        head_name=head_name,
        num_configs=num_configs,
    )
    lower_path = path.lower()
    if lower_path.endswith(('.h5', '.hdf5')):
        return data.load_from_hdf5(
            path,
            no_data_ok=allow_unlabeled,
            **loader_kwargs,
        )
    return data.load_from_xyz(
        file_or_path=path,
        extract_atomic_energies=extract_atomic_energies,
        no_data_ok=allow_unlabeled,
        **loader_kwargs,
    )


@gin.configurable
def datasets(
    *,
    r_max: float,
    train_path: str,
    config_type_weights: Dict = None,
    train_num: int = None,
    valid_path: str = None,
    valid_fraction: float = None,
    valid_num: int = None,
    test_path: str = None,
    test_num: int = None,
    seed: int = 1234,
    energy_key: str = 'energy',
    forces_key: str = 'forces',
    n_node: int = 1,
    n_edge: int = 1,
    n_graph: int = 1,
    min_n_node: int = 1,
    min_n_edge: int = 1,
    min_n_graph: int = 1,
    n_mantissa_bits: int = 1,
    prefactor_stress: float = 1.0,
    remap_stress: np.ndarray = None,
    heads: Sequence[str] = ('Default',),
    head_configs: Dict[str, Dict] | None = None,
) -> Tuple[
    data.GraphDataLoader,
    data.GraphDataLoader,
    data.GraphDataLoader,
    Dict[int, float],
    float,
]:
    """Load training and test dataset from xyz file"""

    head_names = tuple(heads) if heads else ('Default',)
    head_to_index = {name: idx for idx, name in enumerate(head_names)}

    def _merge_atomic_energies(target: Dict[int, float], source: Dict[int, float]):
        for atomic_number, value in source.items():
            if atomic_number in target and not np.isclose(target[atomic_number], value):
                logging.warning(
                    'Conflicting isolated atom energies for Z=%s: keeping %s, ignoring %s',
                    atomic_number,
                    target[atomic_number],
                    value,
                )
                continue
            target[atomic_number] = value

    if head_configs:
        train_configs: list[data.Configuration] = []
        valid_configs: list[data.Configuration] = []
        test_configs: list[data.Configuration] = []
        atomic_energies_dict: Dict[int, float] = {}

        for head_name in head_names:
            head_cfg = head_configs.get(head_name, {})
            head_train_paths = _ensure_list(head_cfg.get('train_path', train_path))
            if not head_train_paths:
                raise ValueError(
                    f"Head '{head_name}' does not define a train_path and no global train_path was provided."
                )

            head_ct_weights = head_cfg.get('config_type_weights', config_type_weights)
            head_train_num = head_cfg.get('train_num', train_num)
            head_valid_path = head_cfg.get('valid_path', valid_path)
            head_valid_fraction = head_cfg.get('valid_fraction', valid_fraction)
            head_valid_num = head_cfg.get('valid_num', valid_num)
            head_test_path = head_cfg.get('test_path', test_path)
            head_test_num = head_cfg.get('test_num', test_num)
            head_seed = seed + head_to_index[head_name]
            allow_unlabeled = bool(head_cfg.get('allow_unlabeled', False))
            head_energy_key = head_cfg.get('energy_key', energy_key)
            head_forces_key = head_cfg.get('forces_key', forces_key)
            head_prefactor_stress = head_cfg.get('prefactor_stress', prefactor_stress)
            head_remap_stress = head_cfg.get('remap_stress', remap_stress)

            head_all_train: list[data.Configuration] = []
            for head_train_path in head_train_paths:
                head_atomic, head_chunk = _load_configs_from_path(
                    head_train_path,
                    config_type_weights=head_ct_weights,
                    energy_key=head_energy_key,
                    forces_key=head_forces_key,
                    prefactor_stress=head_prefactor_stress,
                    remap_stress=head_remap_stress,
                    head_name=head_name,
                    num_configs=head_train_num,
                    extract_atomic_energies=True,
                    allow_unlabeled=allow_unlabeled,
                )
                _merge_atomic_energies(atomic_energies_dict, head_atomic)
                head_all_train.extend(head_chunk)
                logging.info(
                    "Loaded %s training configurations for head '%s' from '%s'",
                    len(head_chunk),
                    head_name,
                    head_train_path,
                )

            if head_valid_path is not None:
                _, head_valid = _load_configs_from_path(
                    head_valid_path,
                    config_type_weights=head_ct_weights,
                    energy_key=head_energy_key,
                    forces_key=head_forces_key,
                    prefactor_stress=head_prefactor_stress,
                    remap_stress=head_remap_stress,
                    head_name=head_name,
                    num_configs=None,
                    extract_atomic_energies=False,
                    allow_unlabeled=allow_unlabeled,
                )
                head_train = head_all_train
                logging.info(
                    "Loaded %s validation configurations for head '%s' from '%s'",
                    len(head_valid),
                    head_name,
                    head_valid_path,
                )
            elif head_valid_fraction is not None:
                logging.info(
                    "Head '%s': using random %.1f%% of training set for validation",
                    head_name,
                    100 * head_valid_fraction,
                )
                head_train, head_valid = data.random_train_valid_split(
                    head_all_train,
                    int(len(head_all_train) * head_valid_fraction),
                    head_seed,
                )
            elif head_valid_num is not None:
                logging.info(
                    "Head '%s': using %s random configs for validation",
                    head_name,
                    head_valid_num,
                )
                head_train, head_valid = data.random_train_valid_split(
                    head_all_train,
                    head_valid_num,
                    head_seed,
                )
            else:
                head_train = head_all_train
                head_valid = []

            if head_test_path is not None:
                _, head_test = _load_configs_from_path(
                    head_test_path,
                    config_type_weights=head_ct_weights,
                    energy_key=head_energy_key,
                    forces_key=head_forces_key,
                    prefactor_stress=head_prefactor_stress,
                    remap_stress=head_remap_stress,
                    head_name=head_name,
                    num_configs=head_test_num,
                    extract_atomic_energies=False,
                    allow_unlabeled=allow_unlabeled,
                )
                logging.info(
                    "Loaded %s test configurations for head '%s' from '%s'",
                    len(head_test),
                    head_name,
                    head_test_path,
                )
            else:
                head_test = []

            replay_paths = _ensure_list(head_cfg.get('replay_paths'))
            replay_allow_unlabeled = bool(
                head_cfg.get('replay_allow_unlabeled', allow_unlabeled)
            )
            replay_configs: list[data.Configuration] = []
            for replay_path in replay_paths:
                _, replay_chunk = _load_configs_from_path(
                    replay_path,
                    config_type_weights=head_ct_weights,
                    energy_key=energy_key,
                    forces_key=forces_key,
                    prefactor_stress=prefactor_stress,
                    remap_stress=remap_stress,
                    head_name=head_name,
                    num_configs=head_cfg.get('replay_num', None),
                    extract_atomic_energies=False,
                    allow_unlabeled=replay_allow_unlabeled,
                )
                replay_configs.extend(replay_chunk)
                logging.info(
                    "Loaded %s replay configurations for head '%s' from '%s'",
                    len(replay_chunk),
                    head_name,
                    replay_path,
                )

            replay_weight = head_cfg.get('replay_weight', 1.0)
            if replay_weight != 1.0:
                for cfg in replay_configs:
                    cfg.weight *= float(replay_weight)

            head_train.extend(replay_configs)
            _maybe_apply_pseudolabels(
                head_cfg=head_cfg,
                train_configs=head_train,
                valid_configs=head_valid,
                replay_configs=replay_configs,
                head_to_index=head_to_index,
                r_max=r_max,
            )

            train_configs.extend(head_train)
            valid_configs.extend(head_valid)
            test_configs.extend(head_test)
    else:
        atomic_energies_dict, all_train_configs = _load_configs_from_path(
            train_path,
            config_type_weights=config_type_weights,
            energy_key=energy_key,
            forces_key=forces_key,
            prefactor_stress=prefactor_stress,
            remap_stress=remap_stress,
            head_name=head_names[0],
            num_configs=train_num,
            extract_atomic_energies=True,
            allow_unlabeled=False,
        )
        logging.info(
            f"Loaded {len(all_train_configs)} training configurations from '{train_path}'"
        )

        if valid_path is not None:
            _, valid_configs = _load_configs_from_path(
                valid_path,
                config_type_weights=config_type_weights,
                energy_key=energy_key,
                forces_key=forces_key,
                prefactor_stress=prefactor_stress,
                remap_stress=remap_stress,
                head_name=head_names[0],
                num_configs=None,
                extract_atomic_energies=False,
                allow_unlabeled=False,
            )
            logging.info(
                f"Loaded {len(valid_configs)} validation configurations from '{valid_path}'"
            )
            train_configs = all_train_configs
        elif valid_fraction is not None:
            logging.info(
                f'Using random {100 * valid_fraction}% of training set for validation'
            )
            train_configs, valid_configs = data.random_train_valid_split(
                all_train_configs, int(len(all_train_configs) * valid_fraction), seed
            )
        elif valid_num is not None:
            logging.info(f'Using random {valid_num} configurations for validation')
            train_configs, valid_configs = data.random_train_valid_split(
                all_train_configs, valid_num, seed
            )
        else:
            logging.info('No validation set')
            train_configs = all_train_configs
            valid_configs = []
        del all_train_configs

        if test_path is not None:
            _, test_configs = _load_configs_from_path(
                test_path,
                config_type_weights=config_type_weights,
                energy_key=energy_key,
                forces_key=forces_key,
                prefactor_stress=prefactor_stress,
                remap_stress=remap_stress,
                head_name=head_names[0],
                num_configs=test_num,
                extract_atomic_energies=False,
                allow_unlabeled=False,
            )
            logging.info(
                f"Loaded {len(test_configs)} test configurations from '{test_path}'"
            )
        else:
            test_configs = []

    logging.info(
        f'Total number of configurations: '
        f'train={len(train_configs)}, '
        f'valid={len(valid_configs)}, '
        f'test={len(test_configs)}'
    )

    def _graphs(configs: list[data.Configuration]):
        return [
            data.graph_from_configuration(c, cutoff=r_max, head_to_index=head_to_index)
            for c in tqdm(configs)
        ]

    train_loader = data.GraphDataLoader(
        graphs=_graphs(train_configs),
        n_node=n_node,
        n_edge=n_edge,
        n_graph=n_graph,
        min_n_node=min_n_node,
        min_n_edge=min_n_edge,
        min_n_graph=min_n_graph,
        n_mantissa_bits=n_mantissa_bits,
        shuffle=True,
        heads=head_names,
    )
    valid_loader = data.GraphDataLoader(
        graphs=_graphs(valid_configs),
        n_node=n_node,
        n_edge=n_edge,
        n_graph=n_graph,
        min_n_node=min_n_node,
        min_n_edge=min_n_edge,
        min_n_graph=min_n_graph,
        n_mantissa_bits=n_mantissa_bits,
        shuffle=False,
        heads=head_names,
    )
    test_loader = data.GraphDataLoader(
        graphs=_graphs(test_configs),
        n_node=n_node,
        n_edge=n_edge,
        n_graph=n_graph,
        min_n_node=min_n_node,
        min_n_edge=min_n_edge,
        min_n_graph=min_n_graph,
        n_mantissa_bits=n_mantissa_bits,
        shuffle=False,
        heads=head_names,
    )
    return train_loader, valid_loader, test_loader, atomic_energies_dict, r_max
