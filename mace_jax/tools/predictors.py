import warnings
from typing import Callable, Dict

import jax.numpy as jnp
import jraph


def predict_energy_forces_stress(
    model: Callable[[jraph.GraphsTuple], Dict[str, jnp.ndarray]],
    graph: jraph.GraphsTuple,
) -> Dict[str, jnp.ndarray]:
    """Compatibility wrapper returning ``model(graph)``.

    The previous Haiku-based implementation required explicit gradients with
    respect to positions.  The Flax port already provides energy/forces/stress
    directly, so this helper now simply delegates to ``model``.  Third-party
    callers can continue to depend on the function name without changing their
    code.
    """

    warnings.warn(
        "predict_energy_forces_stress now expects a model that operates on "
        "GraphsTuple inputs and returns a dict with energy/forces/stress.",
        DeprecationWarning,
        stacklevel=2,
    )
    return model(graph)
