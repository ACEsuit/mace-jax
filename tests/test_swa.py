import jax
import jax.numpy as jnp
import jraph
import numpy as np
import optax
import pytest

from mace_jax.data import (
    AtomicNumberTable,
    Configuration,
    HDF5Dataset,
    StreamingDatasetSpec,
    StreamingGraphDataLoader,
)
from mace_jax.tools.train import SWAConfig
from mace_jax.tools.train import train as jax_train
from tests.conftest import _write_hdf5_from_configs

torch = pytest.importorskip('torch')  # pragma: no cover - optional dependency
from torch.optim.swa_utils import AveragedModel  # noqa: I001

jax.config.update('jax_enable_x64', True)


@pytest.fixture(autouse=True)
def _force_single_device(monkeypatch):
    """Keep unit tests deterministic on multi-device hosts."""
    monkeypatch.setattr(jax, 'local_device_count', lambda: 1)


def _make_streaming_loader(tmp_path, targets: list[float]) -> StreamingGraphDataLoader:
    configs = [
        Configuration(
            atomic_numbers=np.array([1], dtype=np.int32),
            positions=np.zeros((1, 3), dtype=np.float64),
            energy=float(target),
            forces=np.zeros((1, 3), dtype=np.float64),
            stress=np.zeros((3, 3), dtype=np.float64),
        )
        for target in targets
    ]
    dataset_path = tmp_path / 'swa_targets.h5'
    _write_hdf5_from_configs(dataset_path, configs)
    dataset = HDF5Dataset(dataset_path, mode='r')
    return StreamingGraphDataLoader(
        datasets=[dataset],
        dataset_specs=[StreamingDatasetSpec(path=dataset_path)],
        z_table=AtomicNumberTable([1]),
        r_max=2.5,
        n_node=None,
        n_edge=None,
        head_to_index={'Default': 0},
        num_workers=0,
        pad_graphs=2,
    )


def _scalar_loss_fn(params, graph: jraph.GraphsTuple) -> jnp.ndarray:
    """Simple squared error loss between the parameter and per-graph target."""
    graph_mask = jraph.get_graph_padding_mask(graph).astype(jnp.float64)
    targets = jnp.asarray(graph.globals.energy, dtype=jnp.float64)
    prediction = params['w'][0]
    preds = jnp.full_like(targets, prediction)
    diff = (preds - targets) * graph_mask
    return 0.5 * diff * diff


def _run_jax_swa(tmp_path, targets, learning_rate, swa_cfg, epochs):
    params = {'w': jnp.zeros((1,), dtype=jnp.float64)}
    gradient_transform = optax.sgd(learning_rate=learning_rate)
    optimizer_state = gradient_transform.init(params)
    loader = _make_streaming_loader(tmp_path, targets)

    try:
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
        for interval, _, _, eval_params in trainer:
            if interval == epochs:
                final_eval_params = eval_params
                break
    finally:
        loader.close()

    assert final_eval_params is not None
    return jax.tree_util.tree_map(
        lambda x: np.asarray(x, dtype=np.float64), final_eval_params
    )


def _run_torch_swa(targets, learning_rate, start_interval, epochs):
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

    for epoch in range(epochs):
        for target in targets:
            target_tensor = torch.tensor([target], dtype=torch.float64)
            loss = model(target_tensor)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        if epoch >= start_interval:
            swa_model.update_parameters(model)

    return swa_model.module.w.detach().cpu().numpy()


def test_swa_matches_torch_average(tmp_path):
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

    epochs = len(targets)
    jax_params = _run_jax_swa(tmp_path, targets, learning_rate, swa_cfg, epochs)
    torch_params = _run_torch_swa(targets, learning_rate, start_interval, epochs)

    np.testing.assert_allclose(jax_params['w'], torch_params, rtol=1e-6, atol=1e-6)
