from __future__ import annotations

from typing import Any

import jax

try:
    import torch
except ImportError:  # pragma: no cover - torch not installed
    torch = None  # type: ignore[assignment]

_GPU_PLATFORMS: tuple[str, ...] = ('cuda', 'gpu', 'rocm')


def get_torch_device(
    *,
    prefer_gpu: bool = True,
    prefer_mps: bool = True,
) -> Any:
    """Return the preferred Torch device, defaulting to GPU when available."""

    if torch is None:  # pragma: no cover - torch optional dependency
        raise RuntimeError('PyTorch is not installed in this environment.')

    if prefer_gpu and torch.cuda.is_available():
        return torch.device('cuda')

    if prefer_gpu and prefer_mps:
        mps = getattr(torch.backends, 'mps', None)
        if mps is not None and getattr(mps, 'is_available', lambda: False)():
            return torch.device('mps')

    return torch.device('cpu')


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
        torch.cuda.set_device(torch_device)
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


__all__ = [
    'configure_torch_runtime',
    'get_torch_device',
    'runtime_device_summary',
    'select_jax_device',
]
