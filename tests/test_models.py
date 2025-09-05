from ase import neighborlist
import e3nn_jax as e3nn
import haiku as hk
import jax
import numpy as np
from e3nn_jax.utils import assert_equivariant

from mace_jax.modules import MACE


def test_mace():
    @hk.without_apply_rng
    @hk.transform
    def model(node_specie, positions, senders, receivers, receivers_unit_shifts):
        vectors = (positions[receivers] + receivers_unit_shifts @ cell) - positions[
            senders
        ]
        return MACE(
            r_max=5.0,
            radial_basis=lambda r, r_max: e3nn.bessel(r, 8, r_max),
            radial_envelope=lambda r, r_max: e3nn.poly_envelope(5 - 1, 2, r_max)(r),
            max_ell=2,
            num_interactions=5,
            num_species=2,
            hidden_irreps=e3nn.Irreps("32x0e"),
            readout_mlp_irreps=e3nn.Irreps("16x0e"),
            avg_num_neighbors=8.0,
            correlation=3,
            output_irreps="0e",
            symmetric_tensor_product_basis=False,
        )(vectors, node_specie, senders, receivers)

    # Define input
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.0, 0.4, 0.0],
            [0.0, 0.3, 0.3],
        ]
    )
    node_specie = np.arange(4) % 1
    cell = np.identity(3)

    senders, receivers, receivers_unit_shifts = neighborlist.primitive_neighbor_list(
        quantities="ijS",
        pbc=(True, True, False),
        cell=cell,
        positions=positions,
        cutoff=2.0,
    )

    w = model.init(
        jax.random.PRNGKey(0),
        node_specie,
        positions,
        senders,
        receivers,
        receivers_unit_shifts,
    )

    def wrapper(x):
        node_specie = np.arange(4) % 1
        cell = np.identity(3)

        senders, receivers, receivers_unit_shifts = (
            neighborlist.primitive_neighbor_list(
                quantities="ijS",
                pbc=(True, True, False),
                cell=cell,
                positions=np.array(x.array),
                cutoff=2.0,
            )
        )

        y = model.apply(
            w, node_specie, positions, senders, receivers, receivers_unit_shifts
        )
        return y

    x = e3nn.normal("1o", jax.random.PRNGKey(1), (2,))
    assert_equivariant(wrapper, jax.random.PRNGKey(1), x)


if __name__ == "__main__":
    test_mace()
