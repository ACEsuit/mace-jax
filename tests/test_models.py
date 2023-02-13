from collections import namedtuple

import e3nn_jax as e3nn
import haiku as hk
import jax
import jax.numpy as jnp
import jraph
import numpy as np
from e3nn_jax.util import assert_equivariant

from mace_jax import modules
from mace_jax.modules import MACE, SymmetricContraction


def test_symmetric_contraction():
    x = e3nn.normal("0e + 0o + 1o + 1e + 2e + 2o", jax.random.PRNGKey(0), (32, 128))
    y = jax.random.normal(jax.random.PRNGKey(1), (32, 4))

    model = hk.without_apply_rng(
        hk.transform(lambda x, y: SymmetricContraction(3, ["0e", "1o", "2e"])(x, y))
    )
    w = model.init(jax.random.PRNGKey(2), x, y)

    assert_equivariant(
        lambda x: model.apply(w, x, y), jax.random.PRNGKey(3), args_in=(x,)
    )


# TODO fix this test
def test_mace():
    atomic_energies = np.array([1.0, 3.0], dtype=float)

    @hk.without_apply_rng
    @hk.transform
    def model(graph):
        return MACE(
            r_max=5,
            num_bessel=8,
            num_deriv_in_zero=5,
            num_deriv_in_one=2,
            max_ell=2,
            interaction_cls=modules.interaction_classes[
                "AgnosticResidualInteractionBlock"
            ],
            interaction_cls_first=modules.interaction_classes[
                "AgnosticResidualInteractionBlock"
            ],
            num_interactions=5,
            hidden_irreps=e3nn.Irreps("32x0e"),
            readout_mlp_irreps=e3nn.Irreps("16x0e"),
            gate=jax.nn.silu,
            atomic_energies=atomic_energies,
            avg_num_neighbors=8,
            correlation=3,
        )(graph)

    Node = namedtuple("Node", ["positions", "attrs"])
    Edge = namedtuple("Edge", ["shifts"])
    Globals = namedtuple("Globals", ["cell"])

    graph = jraph.GraphsTuple(
        nodes=Node(
            positions=jnp.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
            attrs=jax.nn.one_hot(jnp.array([0, 1]), 2),
        ),
        edges=Edge(shifts=jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])),
        globals=Globals(cell=None),
        senders=jnp.array([0, 1]),
        receivers=jnp.array([1, 0]),
        n_edge=jnp.array([2]),
        n_node=jnp.array([2]),
    )

    w = model.init(jax.random.PRNGKey(0), graph)

    def wrapper(positions):
        graph = jraph.GraphsTuple(
            nodes=Node(
                positions=positions.array,
                attrs=jax.nn.one_hot(jnp.array([0, 1]), 2),
            ),
            edges=Edge(shifts=jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])),
            globals=Globals(cell=None),
            senders=jnp.array([0, 1]),
            receivers=jnp.array([1, 0]),
            n_edge=jnp.array([2]),
            n_node=jnp.array([2]),
        )
        energy = model.apply(w, graph)["energy"]
        return e3nn.IrrepsArray("0e", energy)

    positions = e3nn.normal("1o", jax.random.PRNGKey(1), (2,))
    assert_equivariant(wrapper, jax.random.PRNGKey(1), args_in=(positions,))


if __name__ == "__main__":
    test_mace()
    # test_symmetric_contraction()
