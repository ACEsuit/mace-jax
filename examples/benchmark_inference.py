from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Any

import ase
import jax
import jax.numpy as jnp
import numpy as np
import torch
from ase.build import bulk
from mace.calculators import foundations_models
from mace.data.atomic_data import AtomicData
from mace.data.utils import config_from_atoms
from mace.tools import torch_geometric
from mace.tools.multihead_tools import AtomicNumberTable
from mace.tools.scripts_utils import extract_config_mace_model
from mace.tools.torch_geometric.batch import Batch

from mace_jax.cli.mace_jax_from_torch import convert_model
from mace_jax.modules.wrapper_ops import CuEquivarianceConfig
from mace_jax.tools.device import configure_torch_runtime, get_torch_device


@dataclass
class BenchmarkResult:
    mean: float
    std: float
    min_time: float


def load_foundation_model(
    source: str = 'mp',
    variant: str | None = None,
    device: Any | str = 'cpu',
) -> torch.nn.Module:
    """Return a pretrained Torch MACE foundation model on the specified device."""

    loader_kwargs: dict[str, Any] = {'device': device}
    source_lower = source.lower()
    if source_lower in {'mp', 'off', 'omol'}:
        loader = getattr(foundations_models, f'mace_{source_lower}')
        if variant is not None:
            loader_kwargs['model'] = variant
    elif source_lower == 'anicc':
        loader = foundations_models.mace_anicc
        if variant is not None:
            loader_kwargs['model_path'] = variant
    else:
        raise ValueError(
            "Unknown foundation source. Expected one of {'mp', 'off', 'anicc', 'omol'}."
        )

    try:
        model = loader(return_raw_model=True, **loader_kwargs)
    except Exception:  # pragma: no cover - loader API fallback
        calculator = loader(return_raw_model=False, **loader_kwargs)
        model = getattr(calculator, 'model', None)
        if model is None:
            models = getattr(calculator, 'models', None)
            if models:
                model = models[0]
        if model is None:
            raise

    return model.float().eval()


def extract_foundation_metadata(
    torch_model: torch.nn.Module,
) -> tuple[dict[str, Any], AtomicNumberTable, float]:
    config = extract_config_mace_model(torch_model)
    config['torch_model_class'] = torch_model.__class__.__name__
    atomic_numbers = tuple(int(z) for z in config['atomic_numbers'])
    z_table = AtomicNumberTable(atomic_numbers)
    cutoff = float(config['r_max'])
    return config, z_table, cutoff


def build_example_atoms(symbol: str = 'Si', repeat: int = 2) -> ase.Atoms:
    """Construct a simple crystalline structure for benchmarking."""
    atoms = bulk(symbol, 'diamond', a=5.43)
    return atoms.repeat((repeat, repeat, repeat))


def batch_to_jax(batch: Batch) -> dict[str, jnp.ndarray]:
    converted: dict[str, Any] = {}
    for key in batch.keys:
        value = batch[key]
        if isinstance(value, torch.Tensor):
            converted[key] = jnp.asarray(value.detach().cpu().numpy())
        else:
            converted[key] = value
    return converted


def prepare_batches(
    torch_model: torch.nn.Module, atoms: ase.Atoms, device: Any
) -> tuple[Batch, dict[str, jnp.ndarray], dict[str, Any]]:
    config, z_table, cutoff = extract_foundation_metadata(torch_model)
    config_atoms = config_from_atoms(atoms)
    config_atoms.pbc = [bool(x) for x in config_atoms.pbc]
    atomic_data = AtomicData.from_config(
        config_atoms,
        z_table=z_table,
        cutoff=cutoff,
    )
    batch_torch = torch_geometric.batch.Batch.from_data_list([atomic_data])
    batch_torch = batch_torch.to(device)
    batch_jax = batch_to_jax(batch_torch)
    return batch_torch, batch_jax, config


def run_torch_inference(
    model: torch.nn.Module,
    batch: Batch,
    device: Any,
    *,
    repeats: int,
    warmup: int,
    compute_force: bool,
    compute_stress: bool,
) -> tuple[BenchmarkResult, dict[str, torch.Tensor]]:
    grad_ctx = torch.enable_grad if (compute_force or compute_stress) else torch.no_grad

    for _ in range(warmup):
        with grad_ctx():
            model(batch, compute_force=compute_force, compute_stress=compute_stress)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)

    timings: list[float] = []
    outputs: dict[str, torch.Tensor] | None = None
    for _ in range(repeats):
        start = time.perf_counter()
        with grad_ctx():
            outputs = model(
                batch,
                compute_force=compute_force,
                compute_stress=compute_stress,
            )
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        timings.append(time.perf_counter() - start)

    assert outputs is not None
    arr = np.array(timings)
    stats = BenchmarkResult(
        mean=float(arr.mean()), std=float(arr.std()), min_time=float(arr.min())
    )
    return stats, outputs


def run_jax_inference(
    jax_model,
    variables,
    batch_jax: dict[str, jnp.ndarray],
    *,
    repeats: int,
    warmup: int,
    compute_force: bool,
    compute_stress: bool,
) -> tuple[BenchmarkResult, dict[str, Any]]:
    apply_fn = jax.jit(
        lambda params, data: jax_model.apply(
            params,
            data,
            compute_force=compute_force,
            compute_stress=compute_stress,
        )
    )

    for _ in range(warmup):
        outputs = apply_fn(variables, batch_jax)
        jax.block_until_ready(outputs['energy'])

    timings: list[float] = []
    outputs: dict[str, Any] | None = None
    for _ in range(repeats):
        start = time.perf_counter()
        outputs = apply_fn(variables, batch_jax)
        jax.block_until_ready(outputs['energy'])
        timings.append(time.perf_counter() - start)

    assert outputs is not None
    arr = np.array(timings)
    stats = BenchmarkResult(
        mean=float(arr.mean()), std=float(arr.std()), min_time=float(arr.min())
    )
    return stats, outputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Benchmark Torch vs JAX MACE inference.'
    )
    parser.add_argument(
        '--foundation', default='mp', help='Foundation source (mp, off, anicc, omol).'
    )
    parser.add_argument(
        '--variant', default='medium-mpa-0', help='Foundation model variant.'
    )
    parser.add_argument(
        '--symbol', default='Si', help='Element symbol for the benchmark crystal.'
    )
    parser.add_argument(
        '--repeat', type=int, default=2, help='Supercell repeat along each axis.'
    )
    parser.add_argument(
        '--repeats', type=int, default=10, help='Number of timed runs for each backend.'
    )
    parser.add_argument(
        '--warmup', type=int, default=3, help='Number of warmup runs before timing.'
    )
    parser.add_argument(
        '--disable-forces',
        action='store_true',
        help='Skip force computation in the benchmark.',
    )
    parser.add_argument(
        '--disable-stress',
        action='store_true',
        help='Skip stress computation in the benchmark.',
    )
    parser.add_argument(
        '--cue-conv-fusion',
        action='store_true',
        help='Enable cuequivariance conv fusion in the converted JAX model.',
    )
    args = parser.parse_args()

    compute_force = not args.disable_forces
    compute_stress = not args.disable_stress

    torch_device = configure_torch_runtime(get_torch_device(), deterministic=False)
    print(f'Torch device: {torch_device}')
    print(f'JAX devices: {[f"{dev.platform}:{dev.id}" for dev in jax.devices()]}')
    print(f'Computing forces: {compute_force} | stresses: {compute_stress}')

    torch_model = load_foundation_model(args.foundation, args.variant, device='cpu')
    torch_model = torch_model.to(torch_device)

    atoms = build_example_atoms(args.symbol, args.repeat)
    batch_torch, batch_jax, config = prepare_batches(torch_model, atoms, torch_device)

    cue_config: CuEquivarianceConfig | None = None
    if args.cue_conv_fusion:
        # Leave ``enabled`` false so only the tensor-product path switches to cue
        # for conv fusion while symmetric contractions remain on pure JAX, matching
        # the behaviour of the Torch wrapper.
        cue_config = CuEquivarianceConfig(
            enabled=False,
            optimize_channelwise=True,
            conv_fusion=True,
            layout='mul_ir',
        )

    jax_model, variables, _ = convert_model(
        torch_model,
        config,
        cueq_config=cue_config,
    )

    torch_stats, torch_outputs = run_torch_inference(
        torch_model,
        batch_torch,
        torch_device,
        repeats=args.repeats,
        warmup=args.warmup,
        compute_force=compute_force,
        compute_stress=compute_stress,
    )
    print('Torch inference (per call):')
    print(
        f'  mean = {torch_stats.mean * 1e3:.2f} ms  std = {torch_stats.std * 1e3:.2f} ms  min = {torch_stats.min_time * 1e3:.2f} ms'
    )

    jax_stats, jax_outputs = run_jax_inference(
        jax_model,
        variables,
        batch_jax,
        repeats=args.repeats,
        warmup=args.warmup,
        compute_force=compute_force,
        compute_stress=compute_stress,
    )
    print('JAX (jitted) inference (per call):')
    print(
        f'  mean = {jax_stats.mean * 1e3:.2f} ms  std = {jax_stats.std * 1e3:.2f} ms  min = {jax_stats.min_time * 1e3:.2f} ms'
    )

    torch_energy = torch_outputs['energy'].detach().cpu().numpy()[0]
    jax_energy = float(np.asarray(jax_outputs['energy'])[0])
    print(f'Energy difference |Torch - JAX|: {abs(torch_energy - jax_energy):.6e} eV')

    if compute_force:
        torch_forces = torch_outputs.get('forces')
        jax_forces = jax_outputs.get('forces')
        if torch_forces is not None and jax_forces is not None:
            torch_forces_np = torch_forces.detach().cpu().numpy()
            jax_forces_np = np.asarray(jax_forces)
            diff = torch_forces_np - jax_forces_np
            print('Force difference:')
            print(
                f'  max |ΔF| = {np.abs(diff).max():.6e} eV/Å  '
                f'RMSE = {np.sqrt(np.mean(diff**2)):.6e} eV/Å'
            )
        else:
            print('Force outputs unavailable for comparison.')

    if compute_stress:
        torch_stress = torch_outputs.get('stress')
        jax_stress = jax_outputs.get('stress')
        if torch_stress is not None and jax_stress is not None:
            torch_stress_np = torch_stress.detach().cpu().numpy()
            jax_stress_np = np.asarray(jax_stress)
            diff = torch_stress_np - jax_stress_np
            print('Stress difference:')
            print(
                f'  max |Δσ| = {np.abs(diff).max():.6e} eV/Å³  '
                f'RMSE = {np.sqrt(np.mean(diff**2)):.6e} eV/Å³'
            )
        else:
            print('Stress outputs unavailable for comparison.')


if __name__ == '__main__':
    main()
