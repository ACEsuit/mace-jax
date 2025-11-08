import json
from pathlib import Path

import gin
import pytest
import torch

from mace_jax import tools
from mace_jax.cli import mace_jax_train as train_cli
from mace_jax.cli import mace_jax_train_plot as train_plot
from mace_jax.tools import gin_functions
from mace_jax.tools.train import SWAConfig


class _DummyTorchModel(torch.nn.Module):
    def __init__(self, r_max=4.0):
        super().__init__()
        self.r_max = torch.tensor(r_max)


@pytest.mark.slow
def test_run_train_cli_dry_run(tmp_path, monkeypatch):
    results_dir = tmp_path / 'results'
    torch_ckpt = tmp_path / 'dummy.pt'
    torch_ckpt.write_text('checkpoint', encoding='utf-8')
    train_xyz = Path(__file__).resolve().parent / 'test_data' / 'simple.xyz'

    args = [
        '--dry-run',
        '-b',
        'mace_jax.tools.gin_functions.flags.debug=False',
        '-b',
        "mace_jax.tools.gin_functions.flags.dtype='float32'",
        '-b',
        'mace_jax.tools.gin_functions.flags.seed=0',
        '-b',
        f"mace_jax.tools.gin_functions.logs.directory='{results_dir}'",
        '--torch-checkpoint',
        str(torch_ckpt),
        '--torch-head',
        'Surface',
        '--torch-param-dtype',
        'float32',
        '--train-path',
        str(train_xyz),
        '--valid-path',
        'None',
        '--r-max',
        '3.5',
    ]

    train_cli.main(args)

    gin_files = list(Path(results_dir).glob('*.gin'))
    assert gin_files, 'Expected operative gin config to be written during dry run.'
    operative = gin_files[0].read_text(encoding='utf-8')
    assert 'mace_jax.tools.gin_model.model.torch_checkpoint' in operative
    assert 'dummy.pt' in operative
    assert 'mace_jax.tools.gin_datasets.datasets.train_path' in operative


def test_plot_train_cli_generates_output(tmp_path):
    metrics_path = tmp_path / 'example.metrics'
    metrics_path.write_text(
        '\n'.join([
            '{"interval": 1, "mode": "eval", "loss": 0.5, "mae_e": 0.2, "mae_f": 0.3}',
            '{"interval": 1, "mode": "eval_train", "loss": 0.4}',
            '{"interval": 2, "mode": "eval", "loss": 0.3, "mae_e": 0.1, "mae_f": 0.2}',
            '{"interval": 2, "mode": "eval_train", "loss": 0.35}',
        ]),
        encoding='utf-8',
    )

    train_plot.main([
        '--path',
        str(tmp_path),
        '--output-format',
        'png',
    ])

    output_plot = tmp_path / 'example.png'
    assert output_plot.exists(), 'Expected plot file to be generated.'


def test_cli_head_overrides_bindings(tmp_path):
    gin.clear_config()
    head_cfg_path = tmp_path / 'heads.json'
    head_cfg = {
        'Default': {'train_path': 'a.xyz'},
        'Surface': {'train_path': 'b.xyz'},
    }
    head_cfg_path.write_text(json.dumps(head_cfg), encoding='utf-8')

    args, _ = train_cli.parse_args([
        '--heads',
        'Default',
        'Surface',
        '--head-config',
        str(head_cfg_path),
    ])
    train_cli.apply_cli_overrides(args)

    assert gin.query_parameter('mace_jax.tools.gin_model.model.heads') == (
        'Default',
        'Surface',
    )
    assert gin.query_parameter('mace_jax.tools.gin_datasets.datasets.heads') == (
        'Default',
        'Surface',
    )
    assert gin.query_parameter('mace_jax.tools.gin_datasets.datasets.head_configs') == {
        'Default': {'train_path': 'a.xyz'},
        'Surface': {'train_path': 'b.xyz'},
    }
    gin.clear_config()


def test_cli_heads_literal_sets_configs():
    gin.clear_config()
    head_literal = (
        "{'Default': {'train_path': 'a.xyz'}, 'Surface': {'train_path': 'b.xyz'}}"
    )
    args, _ = train_cli.parse_args([
        '--heads_config',
        head_literal,
    ])
    train_cli.apply_cli_overrides(args)

    assert gin.query_parameter('mace_jax.tools.gin_model.model.heads') == (
        'Default',
        'Surface',
    )
    assert gin.query_parameter('mace_jax.tools.gin_datasets.datasets.head_configs') == {
        'Default': {'train_path': 'a.xyz'},
        'Surface': {'train_path': 'b.xyz'},
    }
    gin.clear_config()


def test_cli_foundation_model_helper(tmp_path, monkeypatch):
    gin.clear_config()

    monkeypatch.setattr(
        train_cli,
        '_load_foundation_model',
        lambda *_, **__: _DummyTorchModel(),
    )
    args, _ = train_cli.parse_args([
        '--foundation_model',
        'small',
        '--foundation_head',
        'Surface',
    ])
    train_cli.apply_cli_overrides(args)

    checkpoint = gin.query_parameter('mace_jax.tools.gin_model.model.torch_checkpoint')
    assert checkpoint.endswith('.pt')
    assert Path(checkpoint).exists()
    assert gin.query_parameter('mace_jax.tools.gin_model.model.torch_head') == 'Surface'
    gin.clear_config()
    train_cli._cleanup_foundation_artifacts()


def test_cli_foundation_conflict(tmp_path):
    gin.clear_config()
    dummy_ckpt = tmp_path / 'model.pt'
    dummy_ckpt.write_text('torch', encoding='utf-8')
    args, _ = train_cli.parse_args([
        '--foundation_model',
        str(dummy_ckpt),
        '--torch-checkpoint',
        str(dummy_ckpt),
    ])
    with pytest.raises(
        ValueError, match='either --torch-checkpoint or --foundation-model'
    ):
        train_cli.apply_cli_overrides(args)
    gin.clear_config()


def test_cli_multiheads_adjusts_defaults(monkeypatch):
    gin.clear_config()
    monkeypatch.setattr(
        train_cli,
        '_load_foundation_model',
        lambda *_, **__: _DummyTorchModel(),
    )
    args, _ = train_cli.parse_args([
        '--foundation_model',
        'small',
        '--multiheads_finetuning',
    ])
    train_cli.apply_cli_overrides(args)
    assert gin.query_parameter('mace_jax.tools.gin_functions.optimizer.lr') == 1e-4
    assert (
        gin.query_parameter('mace_jax.tools.gin_functions.train.ema_decay') == 0.99999
    )
    gin.clear_config()


def test_cli_multiheads_force_override(monkeypatch):
    gin.clear_config()
    monkeypatch.setattr(
        train_cli,
        '_load_foundation_model',
        lambda *_, **__: _DummyTorchModel(),
    )
    args, _ = train_cli.parse_args([
        '--foundation_model',
        'small',
        '--multiheads_finetuning',
        '--force_mh_ft_lr',
        '--lr',
        '0.005',
        '--ema-decay',
        '0.9',
    ])
    train_cli.apply_cli_overrides(args)
    assert gin.query_parameter('mace_jax.tools.gin_functions.optimizer.lr') == 0.005
    assert gin.query_parameter('mace_jax.tools.gin_functions.train.ema_decay') == 0.9
    gin.clear_config()


def test_cli_multiheads_without_foundation():
    gin.clear_config()
    args, _ = train_cli.parse_args([
        '--multiheads_finetuning',
    ])
    train_cli.apply_cli_overrides(args)
    assert gin.query_parameter('mace_jax.tools.gin_functions.optimizer.lr') == 1e-4
    assert (
        gin.query_parameter('mace_jax.tools.gin_functions.train.ema_decay') == 0.99999
    )
    gin.clear_config()


def test_cli_sets_runtime_and_training_controls(tmp_path):
    gin.clear_config()
    train_xyz = Path(__file__).resolve().parent / 'test_data' / 'simple.xyz'
    args, _ = train_cli.parse_args([
        '--name',
        'cli-test',
        '--log_dir',
        str(tmp_path),
        '--seed',
        '321',
        '--default_dtype',
        'float32',
        '--debug',
        '--train_file',
        str(train_xyz),
        '--valid_file',
        'None',
        '--energy_key',
        'E',
        '--forces_key',
        'F',
        '--clip_grad',
        '0.5',
        '--ema',
        '--swa',
        '--start_swa',
        '2',
        '--swa_every',
        '1',
        '--swa_min_snapshots',
        '2',
        '--wandb',
        '--wandb_project',
        'mace-jax-tests',
        '--wandb_tag',
        'unit',
        '--optimizer',
        'amsgrad',
        '--lr',
        '0.02',
        '--weight-decay',
        '1e-4',
        '--scheduler',
        'exponential',
        '--lr_scheduler_gamma',
        '0.5',
    ])
    train_cli.apply_cli_overrides(args)

    assert gin.query_parameter('mace_jax.tools.gin_functions.logs.name') == 'cli-test'
    assert gin.query_parameter('mace_jax.tools.gin_functions.logs.directory') == str(
        tmp_path
    )
    assert gin.query_parameter('mace_jax.tools.gin_functions.flags.seed') == 321
    assert gin.query_parameter('mace_jax.tools.gin_functions.flags.dtype') == 'float32'
    assert gin.query_parameter('mace_jax.tools.gin_functions.flags.debug') is True
    assert gin.query_parameter(
        'mace_jax.tools.gin_datasets.datasets.train_path'
    ) == str(train_xyz)
    assert (
        gin.query_parameter('mace_jax.tools.gin_datasets.datasets.valid_path') is None
    )
    assert gin.query_parameter('mace_jax.tools.gin_datasets.datasets.energy_key') == 'E'
    assert gin.query_parameter('mace_jax.tools.gin_datasets.datasets.forces_key') == 'F'
    assert (
        gin.query_parameter('mace_jax.tools.gin_functions.train.max_grad_norm') == 0.5
    )
    assert gin.query_parameter('mace_jax.tools.gin_functions.train.ema_decay') == 0.99
    swa_cfg = gin.query_parameter('mace_jax.tools.gin_functions.train.swa_config')
    assert isinstance(swa_cfg, SWAConfig)
    assert gin.query_parameter('mace_jax.tools.gin_functions.wandb_run.enabled') is True
    assert (
        gin.query_parameter('mace_jax.tools.gin_functions.wandb_run.project')
        == 'mace-jax-tests'
    )
    assert gin.query_parameter('mace_jax.tools.gin_functions.wandb_run.tags') == (
        'unit',
    )
    assert (
        gin.query_parameter('mace_jax.tools.gin_functions.optimizer.algorithm')
        == tools.scale_by_amsgrad
    )
    assert gin.query_parameter('mace_jax.tools.gin_functions.optimizer.lr') == 0.02
    assert (
        gin.query_parameter('mace_jax.tools.gin_functions.optimizer.weight_decay')
        == 1e-4
    )
    assert (
        gin.query_parameter('mace_jax.tools.gin_functions.optimizer.scheduler')
        == gin_functions.exponential_decay
    )
    assert (
        gin.query_parameter('mace_jax.tools.gin_functions.exponential_decay.decay_rate')
        == 0.5
    )
    gin.clear_config()
