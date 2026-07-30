"""
Wrapper class for o3.Linear that optionally uses cuet.Linear
"""

import dataclasses
import warnings

import cuequivariance as cue
import jax.numpy as jnp
from e3nn_jax import Irreps  # type: ignore
from flax import nnx

from mace_jax.adapters.cuequivariance import (
    FullyConnectedTensorProduct as CueFullyConnectedTensorProduct,
)
from mace_jax.adapters.cuequivariance import Linear as CueLinear
from mace_jax.adapters.cuequivariance import (
    SymmetricContraction as CueSymmetricContraction,
)
from mace_jax.adapters.cuequivariance import TensorProduct as CueTensorProduct
from mace_jax.tools.cg import O3_e3nn

_SUPPORTED_CUE_GROUPS = {"O3", "O3_e3nn"}
_SUPPORTED_BACKENDS = {"jax", "cueq", "openeq"}
_SUPPORTED_LAYOUTS = {"mul_ir", "ir_mul"}
_OPTIMIZATION_FIELDS = (
    "optimize_all",
    "optimize_linear",
    "optimize_channelwise",
    "optimize_symmetric",
    "optimize_fctp",
    "conv_fusion",
)


def _group_name(group_value: object | None) -> str | None:
    if group_value is None:
        return None
    if isinstance(group_value, str):
        return group_value
    return getattr(group_value, "__name__", None) or str(group_value)


def _legacy_get(config: object, name: str, default: object = None) -> object:
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def _layout_name(layout_value: object | None) -> str | None:
    if layout_value is None:
        return None
    raw = layout_value if isinstance(layout_value, str) else (
        getattr(layout_value, "__name__", None) or str(layout_value)
    )
    return {
        "(mul,irrep)": "mul_ir",
        "(irrep,mul)": "ir_mul",
    }.get(raw, raw)


def _validate_cue_group(group_value: object | None, *, context: str) -> None:
    name = _group_name(group_value)
    if name is None:
        return
    if name not in _SUPPORTED_CUE_GROUPS:
        raise ValueError(
            f"{context} only supports the O3 or O3_e3nn groups; "
            f"received {group_value!r}."
        )


def _resolve_cue_group(cueq_config: object | None) -> object:
    if cueq_config is None:
        return cue.O3
    group_value = getattr(cueq_config, "group", None)
    if group_value is None:
        return cue.O3
    if isinstance(group_value, str):
        if group_value == "O3_e3nn":
            return O3_e3nn
        try:
            return getattr(cue, group_value)
        except AttributeError as exc:
            raise ValueError(
                f"Unsupported cuequivariance group {group_value!r}."
            ) from exc
    return group_value


def _legacy_config_active(config: object | None) -> bool:
    if config is None:
        return False
    return bool(
        getattr(config, "enabled", False)
        or any(bool(getattr(config, field, False)) for field in _OPTIMIZATION_FIELDS)
    )


def _validate_openeq_options(
    *,
    group: object | None,
    optimize_linear: bool,
    optimize_symmetric: bool,
    optimize_fctp: bool,
) -> None:
    unsupported = []
    if optimize_linear:
        unsupported.append("optimize_linear")
    if optimize_symmetric:
        unsupported.append("optimize_symmetric")
    if optimize_fctp:
        unsupported.append(
            "optimize_fctp (OpenEquivariance 0.6.8 backward is not safe for general FCTP execution)"
        )
    if unsupported:
        joined = ", ".join(unsupported)
        raise ValueError(
            "OpenEquivariance does not accelerate these operations: "
            f"{joined}."
        )
    if _group_name(group) != "O3_e3nn":
        raise ValueError(
            "OpenEquivariance v1 requires group O3_e3nn; "
            f"received {group!r}."
        )


@dataclasses.dataclass
class _LegacyOpeneqConfig:
    enabled: bool = False
    group: str = "O3_e3nn"
    optimize_all: bool = False
    optimize_linear: bool = False
    optimize_channelwise: bool = False
    optimize_symmetric: bool = False
    optimize_fctp: bool = False
    conv_fusion: bool = False

    def __post_init__(self) -> None:
        _validate_openeq_options(
            group=self.group,
            optimize_linear=self.optimize_linear,
            optimize_symmetric=self.optimize_symmetric,
            optimize_fctp=self.optimize_fctp,
        )

    @property
    def active(self) -> bool:
        return _legacy_config_active(self)

    @property
    def channelwise_fusion(self) -> bool:
        return bool(
            self.enabled
            and self.conv_fusion
            and (self.optimize_all or self.optimize_channelwise)
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "enabled": bool(self.enabled),
            "group": "O3_e3nn",
            "optimize_all": bool(self.optimize_all),
            "optimize_linear": bool(self.optimize_linear),
            "optimize_channelwise": bool(self.optimize_channelwise),
            "optimize_symmetric": bool(self.optimize_symmetric),
            "optimize_fctp": bool(self.optimize_fctp),
            "conv_fusion": bool(self.conv_fusion),
        }


@dataclasses.dataclass
class _LegacyCueqConfig:
    enabled: bool = False
    layout: str | None = None
    layout_str: str | None = None
    group: object = "O3"
    optimize_all: bool = False
    optimize_linear: bool = False
    optimize_channelwise: bool = False
    optimize_symmetric: bool = False
    optimize_fctp: bool = False
    conv_fusion: bool = False

    def __post_init__(self) -> None:
        if self.layout is None and self.layout_str is None:
            self.layout_str = "mul_ir"
        elif self.layout_str is None:
            self.layout_str = str(self.layout)
        elif self.layout is not None and str(self.layout) != self.layout_str:
            raise ValueError(
                "cueq_config has conflicting layout and layout_str values."
            )
        if self.layout_str not in _SUPPORTED_LAYOUTS:
            raise ValueError(
                "cueq_config layout must be mul_ir or ir_mul; "
                f"received {self.layout_str!r}."
            )
        self.layout = self.layout_str
        _validate_cue_group(self.group, context="cueq_config")
        if self.enabled:
            self.group = _resolve_cue_group(self)

    @property
    def active(self) -> bool:
        return _legacy_config_active(self)

    def to_dict(self) -> dict[str, object]:
        return {
            "enabled": bool(self.enabled),
            "group": _group_name(self.group) or "O3",
            "optimize_all": bool(self.optimize_all),
            "optimize_linear": bool(self.optimize_linear),
            "optimize_channelwise": bool(self.optimize_channelwise),
            "optimize_symmetric": bool(self.optimize_symmetric),
            "optimize_fctp": bool(self.optimize_fctp),
            "conv_fusion": bool(self.conv_fusion),
        }


def _coerce_cueq_config(
    config: object | None,
) -> _LegacyCueqConfig | None:
    if config is None:
        return None
    if isinstance(config, _LegacyCueqConfig):
        return config
    layout = _layout_name(_legacy_get(config, "layout", None))
    layout_str = _layout_name(_legacy_get(config, "layout_str", None))
    if isinstance(config, dict):
        if layout is not None and layout_str is not None and layout != layout_str:
            raise ValueError("cueq_config has conflicting layout and layout_str values.")
        normalized_layout = layout_str or layout
    else:
        # Torch config objects keep both aliases but do not normalize them;
        # layout_str is the serialized field used by mace-torch.
        normalized_layout = layout_str or layout
    return _LegacyCueqConfig(
        enabled=bool(_legacy_get(config, "enabled", False)),
        layout=normalized_layout,
        layout_str=normalized_layout,
        group=_legacy_get(config, "group", "O3"),
        optimize_all=bool(_legacy_get(config, "optimize_all", False)),
        optimize_linear=bool(_legacy_get(config, "optimize_linear", False)),
        optimize_channelwise=bool(
            _legacy_get(config, "optimize_channelwise", False)
        ),
        optimize_symmetric=bool(_legacy_get(config, "optimize_symmetric", False)),
        optimize_fctp=bool(_legacy_get(config, "optimize_fctp", False)),
        conv_fusion=bool(_legacy_get(config, "conv_fusion", False)),
    )


def _coerce_openeq_config(
    config: object | None,
) -> _LegacyOpeneqConfig | None:
    if config is None:
        return None
    if isinstance(config, _LegacyOpeneqConfig):
        return config
    return _LegacyOpeneqConfig(
        enabled=bool(_legacy_get(config, "enabled", False)),
        group=_legacy_get(config, "group", "O3_e3nn"),
        optimize_all=bool(_legacy_get(config, "optimize_all", False)),
        optimize_linear=bool(_legacy_get(config, "optimize_linear", False)),
        optimize_channelwise=bool(
            _legacy_get(config, "optimize_channelwise", False)
        ),
        optimize_symmetric=bool(_legacy_get(config, "optimize_symmetric", False)),
        optimize_fctp=bool(_legacy_get(config, "optimize_fctp", False)),
        conv_fusion=bool(_legacy_get(config, "conv_fusion", False)),
    )


@dataclasses.dataclass(init=False)
class EquivarianceConfig:
    """Backend-neutral equivariance acceleration configuration."""

    backend: str = "jax"
    layout: str = "mul_ir"
    group: str | None = None
    optimize_all: bool = False
    optimize_linear: bool = False
    optimize_channelwise: bool = False
    optimize_symmetric: bool = False
    optimize_fctp: bool = False
    conv_fusion: bool = False

    def __init__(
        self,
        layout: str | None = None,
        backend: str | None = "jax",
        group: object | None = None,
        optimize_all: bool | None = None,
        optimize_linear: bool | None = None,
        optimize_channelwise: bool | None = None,
        optimize_symmetric: bool | None = None,
        optimize_fctp: bool | None = None,
        conv_fusion: bool | None = None,
        cueq_config: object | None = None,
        openeq_config: object | None = None,
    ) -> None:
        cue_cfg = _coerce_cueq_config(cueq_config)
        openeq_cfg = _coerce_openeq_config(openeq_config)
        if cue_cfg is not None or openeq_cfg is not None:
            warnings.warn(
                "Nested cueq_config/openeq_config is deprecated; use EquivarianceConfig.backend instead.",
                DeprecationWarning,
                stacklevel=2,
            )

        requested_backend = "jax" if backend is None else str(backend)
        if requested_backend not in _SUPPORTED_BACKENDS:
            raise ValueError(
                "backend must be one of jax, cueq, or openeq; "
                f"received {requested_backend!r}."
            )

        active_backends: list[tuple[str, object | None]] = []
        if requested_backend != "jax":
            active_backends.append((requested_backend, None))
        if cue_cfg is not None and cue_cfg.active:
            active_backends.append(("cueq", cue_cfg))
        if openeq_cfg is not None and openeq_cfg.active:
            active_backends.append(("openeq", openeq_cfg))

        selected = {name for name, _ in active_backends}
        if len(selected) > 1:
            raise ValueError(
                "EquivarianceConfig selects multiple acceleration backends; choose exactly one of jax, cueq, or openeq."
            )
        self.backend = next(iter(selected), "jax")

        source_config = None
        for name, config in active_backends:
            if name == self.backend and config is not None:
                source_config = config
                break

        if layout is None and cue_cfg is not None:
            layout = cue_cfg.layout_str
        self.layout = layout or "mul_ir"
        if self.layout not in _SUPPORTED_LAYOUTS:
            raise ValueError(
                f"layout must be either mul_ir or ir_mul; received {self.layout!r}."
            )

        if group is None and source_config is not None:
            group = getattr(source_config, "group", None)
        if group is None:
            if self.backend == "cueq":
                group = "O3"
            elif self.backend == "openeq":
                group = "O3_e3nn"
        self.group = _group_name(group)

        self.optimize_all = self._resolve_flag(
            "optimize_all", optimize_all, source_config
        )
        self.optimize_linear = self._resolve_flag(
            "optimize_linear", optimize_linear, source_config
        )
        self.optimize_channelwise = self._resolve_flag(
            "optimize_channelwise", optimize_channelwise, source_config
        )
        self.optimize_symmetric = self._resolve_flag(
            "optimize_symmetric", optimize_symmetric, source_config
        )
        self.optimize_fctp = self._resolve_flag(
            "optimize_fctp", optimize_fctp, source_config
        )
        self.conv_fusion = self._resolve_flag(
            "conv_fusion", conv_fusion, source_config
        )

        if self.backend == "cueq":
            _validate_cue_group(self.group, context="EquivarianceConfig")
        elif self.backend == "openeq":
            _validate_openeq_options(
                group=self.group,
                optimize_linear=self.optimize_linear,
                optimize_symmetric=self.optimize_symmetric,
                optimize_fctp=self.optimize_fctp,
            )

    @staticmethod
    def _resolve_flag(
        name: str,
        explicit: bool | None,
        source_config: object | None,
    ) -> bool:
        if explicit is not None:
            return bool(explicit)
        if source_config is not None:
            return bool(getattr(source_config, name, False))
        return False

    @property
    def enabled(self) -> bool:
        return self.backend != "jax"

    @property
    def layout_str(self) -> str:
        return self.layout

    @property
    def cueq_config(self) -> _LegacyCueqConfig | None:
        if self.backend != "cueq":
            return None
        return _LegacyCueqConfig(
            enabled=True,
            layout=self.layout,
            group=self.group or "O3",
            optimize_all=self.optimize_all,
            optimize_linear=self.optimize_linear,
            optimize_channelwise=self.optimize_channelwise,
            optimize_symmetric=self.optimize_symmetric,
            optimize_fctp=self.optimize_fctp,
            conv_fusion=self.conv_fusion,
        )

    @property
    def openeq_config(self) -> _LegacyOpeneqConfig | None:
        if self.backend != "openeq":
            return None
        return _LegacyOpeneqConfig(
            enabled=True,
            group=self.group or "O3_e3nn",
            optimize_all=self.optimize_all,
            optimize_linear=self.optimize_linear,
            optimize_channelwise=self.optimize_channelwise,
            optimize_symmetric=self.optimize_symmetric,
            optimize_fctp=self.optimize_fctp,
            conv_fusion=self.conv_fusion,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "layout": self.layout,
            "group": self.group,
            "optimize_all": bool(self.optimize_all),
            "optimize_linear": bool(self.optimize_linear),
            "optimize_channelwise": bool(self.optimize_channelwise),
            "optimize_symmetric": bool(self.optimize_symmetric),
            "optimize_fctp": bool(self.optimize_fctp),
            "conv_fusion": bool(self.conv_fusion),
        }


def resolve_equivariance_config(
    equivariance_config: EquivarianceConfig | dict[str, object] | None = None,
    *,
    cueq_config: object | None = None,
) -> EquivarianceConfig | None:
    if equivariance_config is not None and cueq_config is not None:
        raise ValueError(
            "Specify only equivariance_config; cueq_config is a deprecated alias."
        )
    if equivariance_config is not None:
        if isinstance(equivariance_config, EquivarianceConfig):
            return equivariance_config
        return EquivarianceConfig(**equivariance_config)
    if cueq_config is None:
        return None

    warnings.warn(
        "cueq_config is deprecated; pass equivariance_config instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Nested cueq_config/openeq_config is deprecated.*",
            category=DeprecationWarning,
        )
        if isinstance(cueq_config, dict):
            legacy = dict(cueq_config)
            nested_openeq = legacy.pop("openeq_config", None)
            return EquivarianceConfig(cueq_config=legacy, openeq_config=nested_openeq)
        return EquivarianceConfig(cueq_config=cueq_config)


def _resolve_wrapper_equivariance_config(
    equivariance_config: EquivarianceConfig | dict[str, object] | None = None,
    *,
    cueq_config: object | None = None,
) -> EquivarianceConfig | None:
    return resolve_equivariance_config(equivariance_config, cueq_config=cueq_config)


class Linear:
    """Returns either a cuet.Linear or o3.Linear based on config"""

    def __new__(
        cls,
        irreps_in: Irreps,
        irreps_out: Irreps,
        shared_weights: bool = True,
        internal_weights: bool = True,
        equivariance_config: EquivarianceConfig | dict[str, object] | None = None,
        cueq_config: object | None = None,
        rngs: nnx.Rngs | None = None,
    ):
        equivariance_config = _resolve_wrapper_equivariance_config(
            equivariance_config, cueq_config=cueq_config
        )
        cueq_config = (
            equivariance_config.cueq_config if equivariance_config is not None else None
        )
        group_value = getattr(cueq_config, 'group', None) if cueq_config else None
        _validate_cue_group(group_value, context='Linear')
        group = _resolve_cue_group(cueq_config) if cueq_config else None
        layout = (
            equivariance_config.layout if equivariance_config is not None else 'mul_ir'
        )

        linear_kwargs = dict(
            shared_weights=shared_weights,
            internal_weights=internal_weights,
            layout=layout,
        )
        if group is not None:
            linear_kwargs['group'] = group

        return CueLinear(
            irreps_in,
            irreps_out,
            rngs=rngs,
            **linear_kwargs,
        )


class TensorProduct:
    """Wrapper around o3.TensorProduct / cuequivariance_jax.segmented_polynomial"""

    def __new__(
        cls,
        irreps_in1: Irreps,
        irreps_in2: Irreps,
        irreps_out: Irreps,
        instructions=None,
        shared_weights: bool = False,
        internal_weights: bool = False,
        equivariance_config: EquivarianceConfig | dict[str, object] | None = None,
        cueq_config: object | None = None,
        rngs: nnx.Rngs | None = None,
    ):
        conv_fusion = False
        equivariance_config = _resolve_wrapper_equivariance_config(
            equivariance_config, cueq_config=cueq_config
        )
        cueq_config = (
            equivariance_config.cueq_config if equivariance_config is not None else None
        )
        openeq_config = (
            equivariance_config.openeq_config
            if equivariance_config is not None
            else None
        )
        use_openeq = bool(
            openeq_config is not None and openeq_config.channelwise_fusion
        )
        if use_openeq:
            # Keep this import lazy: OpenEquivariance is an optional CUDA backend.
            from mace_jax.adapters.openequivariance import (
                TensorProduct as OpenEqTensorProduct,
            )

            return OpenEqTensorProduct(
                irreps_in1,
                irreps_in2,
                irreps_out,
                instructions=instructions,
                shared_weights=shared_weights,
                internal_weights=internal_weights,
                conv_fusion=openeq_config.conv_fusion,
                layout=equivariance_config.layout,
                group=openeq_config.group,
                rngs=rngs,
            )
        group_value = getattr(cueq_config, 'group', None) if cueq_config else None
        _validate_cue_group(group_value, context='TensorProduct')
        group = _resolve_cue_group(cueq_config) if cueq_config else None
        if cueq_config is not None:
            conv_fusion = bool(getattr(cueq_config, 'conv_fusion', False))
        tp_kwargs = dict(
            instructions=instructions,
            shared_weights=shared_weights,
            internal_weights=internal_weights,
            conv_fusion=conv_fusion,
        )
        if group is not None:
            tp_kwargs['group'] = group

        return CueTensorProduct(
            irreps_in1,
            irreps_in2,
            irreps_out,
            rngs=rngs,
            **tp_kwargs,
        )


def FullyConnectedTensorProduct(
    irreps_in1: Irreps,
    irreps_in2: Irreps,
    irreps_out: Irreps,
    shared_weights: bool = True,
    internal_weights: bool = True,
    equivariance_config: EquivarianceConfig | dict[str, object] | None = None,
    cueq_config: object | None = None,
    rngs: nnx.Rngs | None = None,
):
    """
    Wrapper around o3.FullyConnectedTensorProduct (JAX version).
    When CuEquivariance acceleration is requested, this raises since a JAX binding
    is not yet available; otherwise defaults to the e3nn_jax implementation.
    """
    equivariance_config = _resolve_wrapper_equivariance_config(
        equivariance_config, cueq_config=cueq_config
    )
    cueq_config = (
        equivariance_config.cueq_config if equivariance_config is not None else None
    )
    use_cue = (
        cueq_config is not None
        and getattr(cueq_config, 'enabled', False)
        and (
            getattr(cueq_config, 'optimize_all', False)
            or getattr(cueq_config, 'optimize_symmetric', False)
        )
    )
    # conv_fusion can be toggled independently (enabled stays False) so that the
    # tensor product backend switches to cue while symmetric contraction remains
    # on the pure-JAX implementation, matching the Torch wrapper semantics.
    group_value = getattr(cueq_config, 'group', None) if cueq_config else None
    if use_cue or cueq_config is not None:
        _validate_cue_group(group_value, context='FullyConnectedTensorProduct')
    group = _resolve_cue_group(cueq_config) if cueq_config else None

    fctp_kwargs = dict(
        shared_weights=shared_weights,
        internal_weights=internal_weights,
    )
    if group is not None:
        fctp_kwargs['group'] = group

    return CueFullyConnectedTensorProduct(
        irreps_in1,
        irreps_in2,
        irreps_out,
        rngs=rngs,
        **fctp_kwargs,
    )


def SymmetricContractionWrapper(
    irreps_in: Irreps,
    irreps_out: Irreps,
    correlation: int,
    num_elements: int | None = None,
    equivariance_config: EquivarianceConfig | dict[str, object] | None = None,
    cueq_config: object | None = None,
    use_reduced_cg: bool = True,
    rngs: nnx.Rngs | None = None,
):
    """
    JAX implementation of SymmetricContraction powered by cuequivariance-jax.
    """

    equivariance_config = _resolve_wrapper_equivariance_config(
        equivariance_config, cueq_config=cueq_config
    )
    cueq_config = (
        equivariance_config.cueq_config if equivariance_config is not None else None
    )
    use_cue = cueq_config is not None and getattr(cueq_config, 'enabled', False)

    group_value = getattr(cueq_config, 'group', None) if cueq_config else None
    if cueq_config is not None:
        _validate_cue_group(group_value, context='SymmetricContraction')
    group = _resolve_cue_group(cueq_config) if cueq_config else None
    if equivariance_config is not None and equivariance_config.layout not in {
        'mul_ir',
        'ir_mul',
    }:
        raise ValueError(
            f"Unsupported equivariance layout '{equivariance_config.layout}'."
        )

    input_layout = (
        equivariance_config.layout if equivariance_config is not None else 'mul_ir'
    )

    sc_kwargs = dict(
        correlation=correlation,
        num_elements=num_elements,
        use_reduced_cg=use_reduced_cg,
        input_layout=input_layout,
    )
    if group is not None:
        sc_kwargs['group'] = group

    return CueSymmetricContraction(
        irreps_in=irreps_in,
        irreps_out=irreps_out,
        rngs=rngs,
        **sc_kwargs,
    )


class TransposeIrrepsLayoutWrapper:
    """Wrapper around cuex.TransposeIrrepsLayout"""

    def __new__(
        cls,
        irreps: Irreps,
        source: str,
        target: str,
        equivariance_config: EquivarianceConfig | dict[str, object] | None = None,
        cueq_config: object | None = None,
    ):
        equivariance_config = _resolve_wrapper_equivariance_config(
            equivariance_config, cueq_config=cueq_config
        )
        # These boundary adapters are needed only while the model-wide
        # representation is ir_mul.  A mul_ir model already matches the
        # canonical layout consumed and produced by e3nn nonlinearities.
        if equivariance_config is None or equivariance_config.layout != 'ir_mul':
            return None

        source = source.lower()
        target = target.lower()

        if source == target:
            return _IdentityTranspose()

        if {source, target} != {'mul_ir', 'ir_mul'}:
            raise ValueError(
                "TransposeIrrepsLayoutWrapper only supports conversions between 'mul_ir' and 'ir_mul' layouts"
                f' (got source={source!r}, target={target!r}).'
            )

        return _IrrepsLayoutTranspose(irreps=Irreps(irreps), swap_to=target)


class _IdentityTranspose:
    def __call__(self, tensor: jnp.ndarray) -> jnp.ndarray:
        return tensor


class _IrrepsLayoutTranspose:
    def __init__(self, *, irreps: Irreps, swap_to: str) -> None:
        self.irreps = irreps
        self.swap_to = swap_to

    def __call__(self, tensor: jnp.ndarray) -> jnp.ndarray:
        leading_shape = tensor.shape[:-1]
        offset = 0
        pieces = []

        for mul, ir in self.irreps:
            dim = ir.dim
            block_size = mul * dim
            if block_size == 0:
                continue

            segment = tensor[..., offset : offset + block_size]
            offset += block_size

            if self.swap_to == 'ir_mul':
                segment = segment.reshape(leading_shape + (mul, dim))
                segment = jnp.swapaxes(segment, -2, -1)
            else:  # target is 'mul_ir'
                segment = segment.reshape(leading_shape + (dim, mul))
                segment = jnp.swapaxes(segment, -2, -1)

            segment = segment.reshape(leading_shape + (block_size,))
            pieces.append(segment)

        if not pieces:
            return tensor

        return jnp.concatenate(pieces, axis=-1)
