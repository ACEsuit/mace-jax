import numpy as np
import torch
import torch.nn.functional
from e3nn import o3
from scipy.spatial.transform import Rotation as R

from mace_jax import data, modules, tools
from mace_jax.tools import torch_geometric

torch.set_default_dtype(torch.float64)
config = data.Configuration(
    atomic_numbers=np.array([8, 1, 1]),
    positions=np.array(
        [
            [0.0, -2.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    ),
    forces=np.array(
        [
            [0.0, -1.3, 0.0],
            [1.0, 0.2, 0.0],
            [0.0, 1.1, 0.3],
        ]
    ),
    energy=-1.5,
)
table = tools.AtomicNumberTable([1, 8])


def test_mace():
    # Create MACE model
    atomic_energies = np.array([1.0, 3.0], dtype=float)
    model_config = dict(
        r_max=5,
        num_bessel=8,
        num_deriv_in_one=5,
        num_deriv_in_zero=2,
        max_ell=2,
        interaction_cls=modules.interaction_classes["AgnosticResidualInteractionBlock"],
        interaction_cls_first=modules.interaction_classes[
            "AgnosticResidualInteractionBlock"
        ],
        num_interactions=5,
        num_elements=2,
        hidden_irreps=o3.Irreps("32x0e + 32x1o"),
        readout_mlp_irreps=o3.Irreps("16x0e"),
        gate=torch.nn.functional.silu,
        atomic_energies=atomic_energies,
        avg_num_neighbors=8,
        atomic_numbers=table.zs,
        correlation=3,
    )
    model = modules.MACE(**model_config)

    # Created the rotated environment
    rot = R.from_euler("z", 60, degrees=True).as_matrix()
    positions_rotated = np.array(rot @ config.positions.T).T
    config_rotated = data.Configuration(
        atomic_numbers=np.array([8, 1, 1]),
        positions=positions_rotated,
        forces=np.array(
            [
                [0.0, -1.3, 0.0],
                [1.0, 0.2, 0.0],
                [0.0, 1.1, 0.3],
            ]
        ),
        energy=-1.5,
    )

    atomic_data = data.AtomicData.from_config(config, z_table=table, cutoff=3.0)
    atomic_data2 = data.AtomicData.from_config(
        config_rotated, z_table=table, cutoff=3.0
    )

    data_loader = torch_geometric.dataloader.DataLoader(
        dataset=[atomic_data, atomic_data2],
        batch_size=2,
        shuffle=True,
        drop_last=False,
    )
    batch = next(iter(data_loader))

    output = model(batch, training=True)
    assert torch.allclose(output["energy"][0], output["energy"][1])
