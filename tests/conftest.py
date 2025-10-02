import warnings

import torch
from jax import config as jax_config

# Apply warning filters globally before any tests run
warnings.filterwarnings('ignore', category=DeprecationWarning, module='haiku')

# Register safe globals for torch before imports in test files
torch.serialization.add_safe_globals([slice])

# Set default dtype for JAX and Torch
jax_config.update('jax_enable_x64', True)
torch.set_default_dtype(torch.float64)
