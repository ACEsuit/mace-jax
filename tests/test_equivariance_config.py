import gin
import pytest

from mace_jax.cli import mace_jax_train as train_cli
from mace_jax.modules.wrapper_ops import (
    CuEquivarianceConfig,
    EquivarianceConfig,
    OpenEquivarianceConfig,
    resolve_equivariance_config,
    TransposeIrrepsLayoutWrapper,
)
from e3nn_jax import Irreps


def test_equivariance_config_canonical_round_trip():
    config = EquivarianceConfig(
        layout='ir_mul',
        cueq_config=CuEquivarianceConfig(optimize_fctp=True),
        openeq_config=OpenEquivarianceConfig(optimize_channelwise=True),
    )

    payload = config.to_dict()

    assert EquivarianceConfig(**payload).to_dict() == payload
    assert payload == {
        'layout': 'ir_mul',
        'cueq_config': config.cueq_config.to_dict(),
        'openeq_config': config.openeq_config.to_dict(),
    }
    assert 'layout' not in payload['cueq_config']
    assert 'layout' not in payload['openeq_config']


def test_legacy_nested_cueq_config_is_migrated():
    legacy = {
        'layout': 'ir_mul',
        'layout_str': 'ir_mul',
        'enabled': False,
        'conv_fusion': True,
        'openeq_config': {
            'enabled': True,
            'optimize_all': True,
            'conv_fusion': False,
        },
    }

    with pytest.deprecated_call(match='cueq_config is deprecated'):
        config = resolve_equivariance_config(cueq_config=legacy)

    assert config.layout == 'ir_mul'
    assert config.cueq_config.conv_fusion
    assert config.openeq_config.enabled


def test_old_and_new_model_configs_are_mutually_exclusive():
    with pytest.raises(ValueError, match='Specify only equivariance_config'):
        resolve_equivariance_config(
            EquivarianceConfig(), cueq_config=CuEquivarianceConfig()
        )


def test_invalid_and_conflicting_layouts_are_rejected():
    with pytest.raises(ValueError, match='layout must be'):
        EquivarianceConfig(layout='invalid')
    with pytest.deprecated_call():
        with pytest.raises(ValueError, match='conflicting layout'):
            resolve_equivariance_config(
                cueq_config={'layout': 'mul_ir', 'layout_str': 'ir_mul'}
            )


def test_cli_binds_neutral_config_and_deprecated_alias():
    gin.clear_config()
    args, _ = train_cli.parse_args(
        ['--enable-openeq', '--equivariance-layout', 'ir_mul']
    )
    train_cli.apply_cli_overrides(args)
    config = gin.query_parameter(
        'mace_jax.tools.gin_model.model.equivariance_config'
    )
    assert isinstance(config, EquivarianceConfig)
    assert config.layout == 'ir_mul'
    assert config.openeq_config.enabled
    gin.clear_config()

    args, _ = train_cli.parse_args(['--cueq-layout', 'ir_mul'])
    with pytest.deprecated_call(match='--cueq-layout'):
        train_cli.apply_cli_overrides(args)
    config = gin.query_parameter(
        'mace_jax.tools.gin_model.model.equivariance_config'
    )
    assert config.layout == 'ir_mul'
    gin.clear_config()


def test_cli_rejects_conflicting_backend_layout_aliases():
    gin.clear_config()
    args, _ = train_cli.parse_args(
        ['--cueq-layout', 'mul_ir', '--openeq-layout', 'ir_mul']
    )
    with pytest.raises(ValueError, match='Conflicting equivariance layouts'):
        train_cli.apply_cli_overrides(args)
    gin.clear_config()


def test_mul_ir_config_does_not_insert_nonlinearity_transposes():
    transpose = TransposeIrrepsLayoutWrapper(
        Irreps('2x1o'),
        source='ir_mul',
        target='mul_ir',
        equivariance_config=EquivarianceConfig(layout='mul_ir'),
    )
    assert transpose is None

    transpose = TransposeIrrepsLayoutWrapper(
        Irreps('2x1o'),
        source='ir_mul',
        target='mul_ir',
        equivariance_config=EquivarianceConfig(layout='ir_mul'),
    )
    assert transpose is not None
