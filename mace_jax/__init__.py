from __future__ import annotations

import warnings

from .__version__ import __version__

# Suppress noisy torch.load warnings emitted by e3nn when TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD is set.
warnings.filterwarnings(
    'ignore',
    message='Environment variable TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD detected.*',
    module='e3nn.o3._wigner',
)

__all__ = [
    '__version__',
]
