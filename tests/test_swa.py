import jax
import jax.numpy as jnp
import jraph
import numpy as np
import optax
import pytest

from mace_jax.tools.train import SWAConfig
from mace_jax.tools.train import train as jax_train

torch = pytest.importorskip('torch')  # pragma: no cover - optional dependency
from torch.optim.swa_utils import AveragedModel  # noqa: I001


jax.config.update('jax_enable_x64', True)


def _make_graph(target: float) -> jraph.GraphsTuple:
    """Build a padded graph carrying the desired regression target."""
    base = jraph.GraphsTuple(
        nodes=jnp.zeros((1, 1), dtype=jnp.float64),
        senders=jnp.array([0], dtype=jnp.int32),
        receivers=jnp.array([0], dtype=jnp.int32),
        edges=jnp.zeros((1, 1), dtype=jnp.float64),
        n_node=jnp.array([1], dtype=jnp.int32),
        n_edge=jnp.array([1], dtype=jnp.int32),
        globals={'target': jnp.array([target], dtype=jnp.float64)},
    )
    padded = jraph.pad_with_graphs(
        base,
        n_node=2,
        n_edge=2,
        n_graph=2,
    )
    # Ensure padded graph has neutral target (already zeros by default but explicit for clarity).
    padded_globals = padded.globals
    padded_targets = jnp.asarray(padded_globals['target'], dtype=jnp.float64)
    padded_targets = padded_targets.at[1].set(0.0)
    padded = padded._replace(globals={'target': padded_targets})
    return padded


class _SequentialGraphLoader:
    """Tiny loader that mimics GraphDataLoader ordering without full padding.

    The real trainer expects `train_loader` to yield a potentially infinite
    iterator (because it caches iterables in `interval_loader`). Using a simple
    list would terminate the training loop after the first interval, so this
    loader cycles through the fixed targets indefinitely, mirroring the
    behaviour of PyTorch's DataLoader in the reference MACE test.
    """

    def __init__(self, targets: list[float]):
        self._targets = list(targets)
        self._index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if not self._targets:
            raise StopIteration
        graph = _make_graph(self._targets[self._index])
        self._index = (self._index + 1) % len(self._targets)
        return graph

    def approx_length(self):
        return 1


def _scalar_loss_fn(params, graph: jraph.GraphsTuple) -> jnp.ndarray:
    """Simple squared error loss between the parameter and per-graph target."""
    graph_mask = jraph.get_graph_padding_mask(graph).astype(jnp.float64)
    targets = graph.globals['target']
    prediction = params['w'][0]
    preds = jnp.full_like(targets, prediction)
    diff = (preds - targets) * graph_mask
    return 0.5 * diff * diff


def _run_jax_swa(targets, learning_rate, swa_cfg):
    params = {'w': jnp.zeros((1,), dtype=jnp.float64)}
    gradient_transform = optax.sgd(learning_rate=learning_rate)
    optimizer_state = gradient_transform.init(params)
    loader = _SequentialGraphLoader(targets)

    trainer = jax_train(
        params=params,
        total_loss_fn=_scalar_loss_fn,
        train_loader=loader,
        gradient_transform=gradient_transform,
        optimizer_state=optimizer_state,
        progress_bar=False,
        swa_config=swa_cfg,
    )

    final_eval_params = None
    target_intervals = len(targets)
    for interval, _, _, eval_params in trainer:
        if interval == target_intervals:
            final_eval_params = eval_params
            break

    assert final_eval_params is not None
    return jax.tree_util.tree_map(
        lambda x: np.asarray(x, dtype=np.float64), final_eval_params
    )


def _run_torch_swa(targets, learning_rate, start_interval):
    class _TorchScalar(torch.nn.Module):  # pragma: no cover - simple helper
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.zeros(1, dtype=torch.float64))

        def forward(self, target_tensor):
            diff = self.w - target_tensor
            return 0.5 * diff.pow(2)

    model = _TorchScalar()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    swa_model = AveragedModel(model)

    for interval, target in enumerate(targets):
        target_tensor = torch.tensor([target], dtype=torch.float64)
        loss = model(target_tensor)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if interval >= start_interval:
            swa_model.update_parameters(model)

    return swa_model.module.w.detach().cpu().numpy()


def test_swa_matches_torch_average():
    """Ensure SWA averaging matches the Torch helper on a toy scalar problem.

    This mirrors the logic used in `mace`'s unit tests: both frameworks train a
    single learnable scalar against a deterministic sequence of targets and
    switch to SWA after a fixed interval. We only compare the averaged
    parameters—not the entire training loop—because the optimizers and padding
    semantics differ substantially between the full JAX and Torch code paths.
    """
    targets = [1.0, -0.5, 0.25, -1.25]
    learning_rate = 0.3
    start_interval = 1
    min_snapshots = 2

    swa_cfg = SWAConfig(
        start_interval=start_interval,
        update_interval=1,
        min_snapshots_for_eval=min_snapshots,
    )

    jax_params = _run_jax_swa(targets, learning_rate, swa_cfg)
    torch_params = _run_torch_swa(targets, learning_rate, start_interval)

    np.testing.assert_allclose(jax_params['w'], torch_params, rtol=1e-6, atol=1e-6)
