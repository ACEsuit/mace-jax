from __future__ import annotations

from typing import Any


def _load_foundation_modules():
    from mace.calculators import foundations_models
    from mace.calculators.foundations_models import mace_mp_names

    return foundations_models, mace_mp_names


def get_mace_mp_names():
    _, mace_mp_names = _load_foundation_modules()
    return mace_mp_names


def load_foundation_torch_model(
    *,
    source: str,
    model: str | None = None,
    device: str | 'torch.device' = 'cpu',
    default_dtype: str | None = None,
) -> torch.nn.Module:
    """Load a Torch foundation model and return the raw torch.nn.Module.

    This centralizes all interactions with mace.calculators so the rest of
    mace-jax can avoid importing the torch calculator API directly.
    """

    import torch

    foundations_models, _ = _load_foundation_modules()
    source = source.lower()
    if source not in {'mp', 'off', 'anicc', 'omol'}:
        raise ValueError(
            "Unknown foundation source. Supported values are 'mp', 'off', 'anicc', 'omol'."
        )

    loader_kwargs: dict[str, Any] = {'device': device}
    if default_dtype is not None:
        loader_kwargs['default_dtype'] = default_dtype

    if source in {'mp', 'off', 'omol'}:
        loader = getattr(
            foundations_models, f'mace_{"mp" if source == "mp" else source}'
        )
        if model is not None:
            loader_kwargs['model'] = model
    else:  # anicc
        loader = foundations_models.mace_anicc
        if model is not None:
            loader_kwargs['model_path'] = model

    try:
        torch_model = loader(return_raw_model=True, **loader_kwargs)
    except Exception:
        calc = loader(return_raw_model=False, **loader_kwargs)
        torch_model = getattr(calc, 'model', None)
        if torch_model is None:
            models_attr = getattr(calc, 'models', None)
            if models_attr:
                torch_model = models_attr[0]
        if torch_model is None:
            raise

    return torch_model
