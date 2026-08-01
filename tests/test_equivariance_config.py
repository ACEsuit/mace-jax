import gin
import pytest
from e3nn_jax import Irreps

from mace_jax.cli import mace_jax_train as train_cli
from mace_jax.modules.wrapper_ops import (
    EquivarianceConfig,
    TransposeIrrepsLayoutWrapper,
    resolve_equivariance_config,
)


class LegacyCueqConfig:
    def __init__(self, **kwargs):
        self.enabled = kwargs.get("enabled", False)
        self.layout = kwargs.get("layout", None)
        self.layout_str = kwargs.get("layout_str", None)
        self.group = kwargs.get("group", "O3")
        self.optimize_all = kwargs.get("optimize_all", False)
        self.optimize_linear = kwargs.get("optimize_linear", False)
        self.optimize_channelwise = kwargs.get("optimize_channelwise", False)
        self.optimize_symmetric = kwargs.get("optimize_symmetric", False)
        self.optimize_fctp = kwargs.get("optimize_fctp", False)
        self.conv_fusion = kwargs.get("conv_fusion", False)


class LatestMaceOEQConfig:
    def __init__(self, **kwargs):
        self.enabled = kwargs.get("enabled", False)
        self.optimize_all = kwargs.get("optimize_all", False)
        self.optimize_channelwise = kwargs.get("optimize_channelwise", False)
        self.conv_fusion = kwargs.get("conv_fusion", "atomic")


def test_equivariance_config_canonical_round_trip():
    config = EquivarianceConfig(
        backend="cueq",
        layout="ir_mul",
        group="O3_e3nn",
        optimize_all=True,
        conv_fusion=True,
    )

    payload = config.to_dict()

    assert EquivarianceConfig(**payload).to_dict() == payload
    assert payload == {
        "backend": "cueq",
        "layout": "ir_mul",
        "group": "O3_e3nn",
        "optimize_all": True,
        "optimize_linear": False,
        "optimize_channelwise": False,
        "optimize_symmetric": False,
        "optimize_fctp": False,
        "conv_fusion": True,
    }
    assert "cueq_config" not in payload
    assert "openeq_config" not in payload


def test_legacy_nested_cueq_config_is_migrated():
    legacy = {
        "layout": "ir_mul",
        "layout_str": "ir_mul",
        "enabled": True,
        "optimize_channelwise": True,
        "conv_fusion": True,
    }

    with pytest.deprecated_call(match="cueq_config.*deprecated"):
        config = resolve_equivariance_config(cueq_config=legacy)

    assert config.backend == "cueq"
    assert config.layout == "ir_mul"
    assert config.cueq_config.conv_fusion
    assert config.cueq_config.optimize_channelwise


def test_legacy_openeq_config_is_migrated():
    with pytest.deprecated_call(match="Nested cueq_config"):
        config = EquivarianceConfig(
            openeq_config={
                "enabled": True,
                "optimize_all": True,
                "conv_fusion": True,
            }
        )

    assert config.backend == "openeq"
    assert config.group == "O3_e3nn"
    assert config.openeq_config.enabled
    assert config.openeq_config.channelwise_fusion


def test_latest_mace_disabled_openeq_config_stays_inactive():
    with pytest.deprecated_call(match="openeq_config.*deprecated"):
        config = resolve_equivariance_config(
            openeq_config=LatestMaceOEQConfig()
        )

    assert config.backend == "jax"
    assert config.openeq_config is None


def test_latest_mace_enabled_openeq_config_is_migrated():
    with pytest.deprecated_call(match="openeq_config.*deprecated"):
        config = resolve_equivariance_config(
            openeq_config=LatestMaceOEQConfig(
                enabled=True, optimize_all=True, conv_fusion="atomic"
            )
        )

    assert config.backend == "openeq"
    assert config.group == "O3_e3nn"
    assert config.conv_fusion is True
    assert config.openeq_config.channelwise_fusion


def test_legacy_cueq_config_object_preserves_layout_alias():
    legacy = LegacyCueqConfig(layout="ir_mul", conv_fusion=True)

    assert legacy.layout == "ir_mul"
    assert legacy.layout_str is None

    with pytest.deprecated_call(match="cueq_config.*deprecated"):
        config = resolve_equivariance_config(cueq_config=legacy)

    assert config.backend == "cueq"
    assert config.layout == "ir_mul"
    assert config.cueq_config.conv_fusion


def test_legacy_cueq_config_accepts_layout_str_alias():
    legacy = LegacyCueqConfig(layout="mul_ir", layout_str="ir_mul")

    with pytest.deprecated_call(match="cueq_config.*deprecated"):
        config = resolve_equivariance_config(cueq_config=legacy)

    assert config.backend == 'jax'
    assert config.layout == 'ir_mul'
    assert config.layout_str == 'ir_mul'


def test_old_and_new_model_configs_are_mutually_exclusive():
    with pytest.raises(ValueError, match="Specify only equivariance_config"):
        resolve_equivariance_config(
            EquivarianceConfig(), cueq_config=LegacyCueqConfig()
        )


def test_invalid_and_conflicting_layouts_are_rejected():
    with pytest.raises(ValueError, match="layout must be"):
        EquivarianceConfig(layout="invalid")
    with pytest.deprecated_call():
        with pytest.raises(ValueError, match="conflicting layout"):
            resolve_equivariance_config(
                cueq_config={"layout": "mul_ir", "layout_str": "ir_mul"}
            )


def test_cli_binds_neutral_config_and_deprecated_alias():
    gin.clear_config()
    args, _ = train_cli.parse_args(
        ["--enable-openeq", "--equivariance-layout", "ir_mul"]
    )
    train_cli.apply_cli_overrides(args)
    config = gin.query_parameter(
        "mace_jax.tools.gin_model.model.equivariance_config"
    )
    assert isinstance(config, EquivarianceConfig)
    assert config.backend == "openeq"
    assert config.layout == "ir_mul"
    assert config.openeq_config.enabled
    gin.clear_config()

    args, _ = train_cli.parse_args(["--cueq-layout", "ir_mul"])
    with pytest.deprecated_call(match="--cueq-layout"):
        train_cli.apply_cli_overrides(args)
    config = gin.query_parameter(
        "mace_jax.tools.gin_model.model.equivariance_config"
    )
    assert config.backend == "jax"
    assert config.layout == "ir_mul"
    gin.clear_config()


def test_cli_rejects_conflicting_backend_layout_aliases():
    gin.clear_config()
    args, _ = train_cli.parse_args(
        ["--cueq-layout", "mul_ir", "--openeq-layout", "ir_mul"]
    )
    with pytest.raises(ValueError, match="Conflicting equivariance layouts"):
        train_cli.apply_cli_overrides(args)
    gin.clear_config()


def test_cli_rejects_conflicting_backend_requests():
    gin.clear_config()
    args, _ = train_cli.parse_args(["--enable-cueq", "--enable-openeq"])
    with pytest.raises(ValueError, match="Conflicting equivariance backends"):
        train_cli.apply_cli_overrides(args)
    gin.clear_config()


def test_mul_ir_config_does_not_insert_nonlinearity_transposes():
    transpose = TransposeIrrepsLayoutWrapper(
        Irreps("2x1o"),
        source="ir_mul",
        target="mul_ir",
        equivariance_config=EquivarianceConfig(layout="mul_ir"),
    )
    assert transpose is None

    transpose = TransposeIrrepsLayoutWrapper(
        Irreps("2x1o"),
        source="ir_mul",
        target="mul_ir",
        equivariance_config=EquivarianceConfig(layout="ir_mul"),
    )
    assert transpose is not None
