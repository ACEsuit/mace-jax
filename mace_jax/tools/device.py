from __future__ import annotations

import logging
import os
from typing import Any

import jax

try:
    import torch
except ImportError:  # pragma: no cover - torch not installed
    torch = None  # type: ignore[assignment]

_GPU_PLATFORMS: tuple[str, ...] = ('cuda', 'gpu', 'rocm')
_JAX_RUNTIME_INITIALIZED = False


def get_torch_device(
    *,
    prefer_gpu: bool = True,
    prefer_mps: bool = True,
) -> str:
    """Return the preferred Torch device, defaulting to GPU when available."""

    if torch is None:  # pragma: no cover - torch optional dependency
        raise RuntimeError('PyTorch is not installed in this environment.')

    if prefer_gpu and torch.cuda.is_available():
        return 'cuda'

    if prefer_gpu and prefer_mps:
        mps = getattr(torch.backends, 'mps', None)
        if mps is not None and getattr(mps, 'is_available', lambda: False)():
            return 'mps'

    return 'cpu'


def configure_torch_runtime(
    device: Any | str | None = None,
    *,
    deterministic: bool = False,
) -> Any:
    """Configure PyTorch for the requested device and optional determinism."""

    if torch is None:  # pragma: no cover - torch optional dependency
        raise RuntimeError('PyTorch is not installed in this environment.')

    if device is None:
        device = get_torch_device()

    torch_device = torch.device(device)

    if torch_device.type == 'cuda':
        if deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.use_deterministic_algorithms(True)
    elif torch_device.type == 'mps':
        pass  # no runtime knobs for MPS currently

    return torch_device


def select_jax_device(*, prefer_gpu: bool = True):
    """Pick a JAX device, preferring GPUs when available."""

    devices = tuple(jax.devices())
    if prefer_gpu:
        for dev in devices:
            if dev.platform in _GPU_PLATFORMS:
                return dev
    return next(iter(devices))


def runtime_device_summary(torch_device: Any | None) -> dict[str, object]:
    """Return a snapshot describing the Torch and JAX runtime devices."""

    jax_devices = [f'{dev.platform}:{dev.id}' for dev in jax.devices()]
    summary: dict[str, object] = {'jax_devices': jax_devices}

    if torch is not None:
        summary.update({
            'torch_device': str(torch_device) if torch_device is not None else None,
            'torch_cuda_available': torch.cuda.is_available(),
        })
    else:  # pragma: no cover - torch optional dependency
        summary['torch_available'] = False

    return summary


def _env_or(value: Any | None, env_key: str, *, cast=int):
    if value is not None:
        return value
    raw = os.environ.get(env_key)
    if raw is None:
        return None
    try:
        return cast(raw)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return None


def _resolve_coordinator(address: str | None, port: int | None) -> str | None:
    if not address:
        return None
    if ':' in address:
        return address
    return f'{address}:{port or 12345}'


def initialize_jax_runtime(
    *,
    device: str | None = None,
    distributed: bool = False,
    process_count: int | None = None,
    process_index: int | None = None,
    coordinator_address: str | None = None,
    coordinator_port: int | None = None,
) -> None:
    """Configure JAX runtime platform and optionally initialize distributed mode."""

    global _JAX_RUNTIME_INITIALIZED  # pylint: disable=global-statement

    if device and device not in {'auto', 'default'}:
        normalized = device.lower()
        jax.config.update('jax_platform_name', normalized)

    if not distributed:
        return

    if not hasattr(jax, 'distributed'):  # pragma: no cover - very old jax
        raise RuntimeError('This JAX build does not expose jax.distributed.initialize.')

    if _JAX_RUNTIME_INITIALIZED:
        return

    process_count = _env_or(process_count, 'JAX_PROCESS_COUNT')
    process_index = _env_or(process_index, 'JAX_PROCESS_INDEX')
    coordinator_from_env = os.environ.get('JAX_COORDINATOR_ADDRESS')
    coordinator_address = coordinator_address or coordinator_from_env
    coordinator = _resolve_coordinator(coordinator_address, coordinator_port)

    init_kwargs: dict[str, Any] = {}
    if coordinator is not None:
        init_kwargs['coordinator_address'] = coordinator
    if process_count is not None:
        init_kwargs['num_processes'] = process_count
    if process_index is not None:
        init_kwargs['process_id'] = process_index

    try:
        jax.distributed.initialize(**init_kwargs)
    except RuntimeError as exc:  # pragma: no cover - initialization race
        if 'already initialized' not in str(exc):
            raise
    else:
        _JAX_RUNTIME_INITIALIZED = True
        logging.info(
            'Initialized JAX distributed runtime (process %s/%s).',
            jax.process_index(),
            jax.process_count(),
        )


__all__ = [
    'configure_torch_runtime',
    'get_torch_device',
    'initialize_jax_runtime',
    'runtime_device_summary',
    'select_jax_device',
]
