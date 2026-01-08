import json
import logging
from pathlib import Path

import gin
import pytest
import torch

from mace_jax import modules, tools
from mace_jax.cli import mace_jax_train as train_cli
from mace_jax.cli import mace_jax_train_plot as train_plot
from mace_jax.tools import gin_functions
from mace_jax.tools.train import SWAConfig


class _DummyTorchModel(torch.nn.Module):
    def __init__(self, r_max=4.0):
        super().__init__()
        self.r_max = torch.tensor(r_max)


@pytest.mark.slow
def test_run_train_cli_writes_operative_config(tmp_path, monkeypatch, simple_hdf5_path):
    gin.clear_config()
    results_dir = tmp_path / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)
    torch_ckpt = tmp_path / 'dummy.pt'
    torch_ckpt.write_text('checkpoint', encoding='utf-8')
    train_hdf5 = simple_hdf5_path

    args = [
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
        str(train_hdf5),
        '--valid-path',
        'None',
        '--r-max',
        '3.5',
    ]

    parsed, remaining = train_cli.parse_args(args)
    train_cli.configure_gin(parsed, remaining)
    train_cli.apply_cli_overrides(parsed)
    operative = gin.operative_config_str()
    train_cli._write_operative_config(
        directory=str(results_dir),
        tag='test',
        operative_config=operative,
        is_primary=True,
    )

    gin_files = list(Path(results_dir).glob('*.gin'))
    assert gin_files, 'Expected operative gin config to be written.'
    operative = gin_files[0].read_text(encoding='utf-8')
    assert 'mace_jax.tools.gin_model.model.torch_checkpoint' in operative
    assert 'dummy.pt' in operative
    assert 'mace_jax.tools.gin_datasets.datasets.train_path' in operative
    gin.clear_config()


def test_plot_train_cli_generates_output(tmp_path):
    metrics_path = tmp_path / 'example.metrics'
    metrics_path.write_text(
        '\n'.join(
            [
                '{"interval": 1, "mode": "eval", "loss": 0.5, "mae_e": 0.2, "mae_f": 0.3}',
                '{"interval": 1, "mode": "train", "loss": 0.4}',
                '{"interval": 2, "mode": "eval", "loss": 0.3, "mae_e": 0.1, "mae_f": 0.2}',
                '{"interval": 2, "mode": "train", "loss": 0.35}',
            ]
        ),
        encoding='utf-8',
    )

    train_plot.main(
        [
            '--path',
            str(tmp_path),
            '--output-format',
            'png',
        ]
    )

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

    args, _ = train_cli.parse_args(
        [
            '--heads',
            'Default',
            'Surface',
            '--head-config',
            str(head_cfg_path),
        ]
    )
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
    args, _ = train_cli.parse_args(
        [
            '--heads_config',
            head_literal,
        ]
    )
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


def test_cli_pt_head_from_cli(tmp_path):
    gin.clear_config()
    pt_file = tmp_path / 'pt.xyz'
    pt_file.write_text('', encoding='utf-8')
    args, _ = train_cli.parse_args(
        [
            '--pt_train_file',
            str(pt_file),
        ]
    )
    train_cli.apply_cli_overrides(args)
    head_cfgs = gin.query_parameter('mace_jax.tools.gin_datasets.datasets.head_configs')
    assert 'pt_head' in head_cfgs
    assert head_cfgs['pt_head']['train_path'] == [str(pt_file)]
    assert 'pt_head' in gin.query_parameter('mace_jax.tools.gin_model.model.heads')
    gin.clear_config()


def test_cli_pt_head_respects_heads_list(tmp_path):
    gin.clear_config()
    pt_file = tmp_path / 'pt.xyz'
    pt_file.write_text('', encoding='utf-8')
    args, _ = train_cli.parse_args(
        [
            '--heads',
            'Default',
            '--pt_train_file',
            str(pt_file),
        ]
    )
    train_cli.apply_cli_overrides(args)
    heads = gin.query_parameter('mace_jax.tools.gin_model.model.heads')
    assert heads == ('pt_head', 'Default')
    gin.clear_config()


def test_cli_foundation_model_helper(tmp_path, monkeypatch):
    gin.clear_config()

    monkeypatch.setattr(
        train_cli,
        '_load_foundation_model',
        lambda *_, **__: _DummyTorchModel(),
    )
    monkeypatch.setattr(train_cli, '_is_named_mp_foundation', lambda name: True)
    args, _ = train_cli.parse_args(
        [
            '--foundation_model',
            'small',
            '--foundation_head',
            'Surface',
        ]
    )
    train_cli.apply_cli_overrides(args)

    checkpoint = gin.query_parameter('mace_jax.tools.gin_model.model.torch_checkpoint')
    assert checkpoint.endswith('.pt')
    assert Path(checkpoint).exists()
    assert gin.query_parameter('mace_jax.tools.gin_model.model.torch_head') == 'Surface'
    gin.clear_config()
    train_cli._cleanup_foundation_artifacts()


def test_cli_checkpoint_flags_with_foundation_model(tmp_path, monkeypatch):
    gin.clear_config()

    monkeypatch.setattr(
        train_cli,
        '_load_foundation_model',
        lambda *_, **__: _DummyTorchModel(),
    )
    monkeypatch.setattr(train_cli, '_is_named_mp_foundation', lambda name: True)

    checkpoint_dir = tmp_path / 'ckpts'
    resume_path = tmp_path / 'resume.ckpt'
    resume_path.write_bytes(b'dummy')

    args, _ = train_cli.parse_args(
        [
            '--foundation_model',
            'small',
            '--checkpoint-dir',
            str(checkpoint_dir),
            '--checkpoint-every',
            '3',
            '--checkpoint-keep',
            '5',
            '--resume-from',
            str(resume_path),
        ]
    )
    train_cli.apply_cli_overrides(args)

    assert gin.query_parameter(
        'mace_jax.tools.gin_functions.train.checkpoint_dir'
    ) == str(checkpoint_dir)
    assert (
        gin.query_parameter('mace_jax.tools.gin_functions.train.checkpoint_every') == 3
    )
    assert (
        gin.query_parameter('mace_jax.tools.gin_functions.train.checkpoint_keep') == 5
    )
    assert gin.query_parameter('mace_jax.tools.gin_functions.train.resume_from') == str(
        resume_path
    )
    gin.clear_config()
    train_cli._cleanup_foundation_artifacts()


def test_cli_foundation_conflict(tmp_path):
    gin.clear_config()
    dummy_ckpt = tmp_path / 'model.pt'
    dummy_ckpt.write_text('torch', encoding='utf-8')
    args, _ = train_cli.parse_args(
        [
            '--foundation_model',
            str(dummy_ckpt),
            '--torch-checkpoint',
            str(dummy_ckpt),
        ]
    )
    with pytest.raises(
        ValueError, match='either --torch-checkpoint or --foundation-model'
    ):
        train_cli.apply_cli_overrides(args)
    gin.clear_config()


def test_cli_schedulefree_binds_params():
    gin.clear_config()
    args, _ = train_cli.parse_args(
        [
            '--optimizer',
            'schedulefree',
            '--schedule-free-b1',
            '0.97',
            '--schedule-free-weight-lr-power',
            '1.5',
        ]
    )
    train_cli.apply_cli_overrides(args)
    assert gin.query_parameter('mace_jax.tools.gin_functions.optimizer.schedule_free')
    assert (
        gin.query_parameter('mace_jax.tools.gin_functions.optimizer.schedule_free_b1')
        == 0.97
    )
    assert (
        gin.query_parameter(
            'mace_jax.tools.gin_functions.optimizer.schedule_free_weight_lr_power'
        )
        == 1.5
    )
    gin.clear_config()


def test_cli_device_distributed_bindings():
    gin.clear_config()
    args, _ = train_cli.parse_args(
        [
            '--device',
            'cpu',
            '--distributed',
            '--process-count',
            '2',
            '--process-index',
            '1',
            '--coordinator-address',
            'localhost',
            '--coordinator-port',
            '12345',
        ]
    )
    train_cli.apply_cli_overrides(args)
    assert gin.query_parameter('mace_jax.tools.gin_functions.flags.device') == 'cpu'
    assert gin.query_parameter('mace_jax.tools.gin_functions.flags.distributed') is True
    assert gin.query_parameter('mace_jax.tools.gin_functions.flags.process_count') == 2
    assert gin.query_parameter('mace_jax.tools.gin_functions.flags.process_index') == 1
    assert (
        gin.query_parameter('mace_jax.tools.gin_functions.flags.coordinator_address')
        == 'localhost'
    )
    assert (
        gin.query_parameter('mace_jax.tools.gin_functions.flags.coordinator_port')
        == 12345
    )
    gin.clear_config()


def test_cli_multiheads_adjusts_defaults(monkeypatch):
    gin.clear_config()
    monkeypatch.setattr(
        train_cli,
        '_load_foundation_model',
        lambda *_, **__: _DummyTorchModel(),
    )
    monkeypatch.setattr(train_cli, '_is_named_mp_foundation', lambda name: True)
    args, _ = train_cli.parse_args(
        [
            '--foundation_model',
            'small',
            '--multiheads_finetuning',
        ]
    )
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
    monkeypatch.setattr(train_cli, '_is_named_mp_foundation', lambda name: True)
    args, _ = train_cli.parse_args(
        [
            '--foundation_model',
            'small',
            '--multiheads_finetuning',
            '--force_mh_ft_lr',
            '--lr',
            '0.005',
            '--ema-decay',
            '0.9',
        ]
    )
    train_cli.apply_cli_overrides(args)
    assert gin.query_parameter('mace_jax.tools.gin_functions.optimizer.lr') == 0.005
    assert gin.query_parameter('mace_jax.tools.gin_functions.train.ema_decay') == 0.9
    gin.clear_config()


def test_cli_multiheads_without_foundation():
    gin.clear_config()
    args, _ = train_cli.parse_args(
        [
            '--multiheads_finetuning',
        ]
    )
    train_cli.apply_cli_overrides(args)
    assert gin.query_parameter('mace_jax.tools.gin_functions.optimizer.lr') == 1e-4
    assert (
        gin.query_parameter('mace_jax.tools.gin_functions.train.ema_decay') == 0.99999
    )
    gin.clear_config()


def test_cli_swa_stage_two_options(tmp_path, simple_hdf5_path):
    gin.clear_config()
    train_hdf5 = simple_hdf5_path
    args, _ = train_cli.parse_args(
        [
            '--train-file',
            str(train_hdf5),
            '--valid-file',
            'None',
            '--swa',
            '--start_swa',
            '3',
            '--swa-loss',
            'huber',
            '--swa-energy-weight',
            '0.5',
            '--swa-lr',
            '1e-4',
        ]
    )
    train_cli.apply_cli_overrides(args)
    swa_cfg = gin.query_parameter('mace_jax.tools.gin_functions.train.swa_config')
    assert swa_cfg.stage_loss_factory is modules.WeightedHuberEnergyForcesStressLoss
    assert swa_cfg.stage_loss_kwargs == {'energy_weight': 0.5}
    assert (
        gin.query_parameter('mace_jax.tools.gin_functions.optimizer.stage_two_lr')
        == 1e-4
    )
    assert (
        gin.query_parameter('mace_jax.tools.gin_functions.optimizer.stage_two_interval')
        == 3
    )
    gin.clear_config()


def test_cli_eval_train_flag(tmp_path, simple_hdf5_path):
    gin.clear_config()
    train_hdf5 = simple_hdf5_path
    args, _ = train_cli.parse_args(
        [
            '--train-file',
            str(train_hdf5),
            '--valid-file',
            'None',
            '--eval-train',
            '--no-eval-test',
        ]
    )
    train_cli.apply_cli_overrides(args)
    assert gin.query_parameter('mace_jax.tools.gin_functions.train.eval_train') is True
    assert gin.query_parameter('mace_jax.tools.gin_functions.train.eval_test') is False
    gin.clear_config()


def test_cli_adamw_sets_decoupled(tmp_path):
    gin.clear_config()


def test_foundation_model_populates_heads(monkeypatch, tmp_path):
    gin.clear_config()

    class _DummyRMax:
        def __init__(self, value):
            self._value = value

        def item(self):
            return self._value

    class _DummyTorchModel:
        def __init__(self):
            self.heads = ['Default', 'Surface']
            self.r_max = _DummyRMax(4.0)

    def _fake_loader(name, default_dtype=None):
        assert name == 'dummy-foundation'
        assert default_dtype is None
        return _DummyTorchModel()

    monkeypatch.setattr(train_cli, '_load_foundation_model', _fake_loader)
    monkeypatch.setattr(train_cli, '_is_named_mp_foundation', lambda name: True)
    monkeypatch.setattr(train_cli.torch, 'save', lambda *_, **__: None)
    tmp_dir = tmp_path / 'tmp'
    tmp_dir.mkdir()
    monkeypatch.setattr(
        train_cli.tempfile,
        'mkdtemp',
        lambda suffix=None, prefix=None, dir=None: str(tmp_dir),
    )

    args, _ = train_cli.parse_args(
        [
            '--foundation-model',
            'dummy-foundation',
        ]
    )
    train_cli.apply_cli_overrides(args)
    assert gin.query_parameter('mace_jax.tools.gin_model.model.heads') == (
        'Default',
        'Surface',
    )
    assert gin.query_parameter('mace_jax.tools.gin_datasets.datasets.heads') == (
        'Default',
        'Surface',
    )
    train_cli._cleanup_foundation_artifacts()
    gin.clear_config()


def test_foundation_multihead_without_pt_disables(
    monkeypatch, tmp_path, caplog, simple_hdf5_path
):
    gin.clear_config()

    class _DummyRMax:
        def __init__(self, value):
            self._value = value

        def item(self):
            return self._value

    class _DummyTorchModel:
        def __init__(self):
            self.heads = ['Default']
            self.r_max = _DummyRMax(4.0)

    def _fake_loader(name, default_dtype=None):
        return _DummyTorchModel()

    monkeypatch.setattr(train_cli, '_load_foundation_model', _fake_loader)
    tmp_dir = tmp_path / 'tmp'
    tmp_dir.mkdir()
    monkeypatch.setattr(
        train_cli.tempfile,
        'mkdtemp',
        lambda suffix=None, prefix=None, dir=None: str(tmp_dir),
    )
    monkeypatch.setattr(train_cli.torch, 'save', lambda *_, **__: None)
    monkeypatch.setattr(train_cli, '_is_named_mp_foundation', lambda name: False)

    args, _ = train_cli.parse_args(
        [
            '--foundation-model',
            'custom-foundation',
            '--multiheads_finetuning',
        ]
    )
    with caplog.at_level(logging.WARNING):
        train_cli.apply_cli_overrides(args)
    with pytest.raises(ValueError):
        gin.query_parameter('mace_jax.tools.gin_functions.optimizer.lr')
    assert 'Multihead finetuning requires pretraining data' in caplog.text
    train_cli._cleanup_foundation_artifacts()
    gin.clear_config()
    train_hdf5 = simple_hdf5_path
    args, _ = train_cli.parse_args(
        [
            '--train-file',
            str(train_hdf5),
            '--valid-file',
            'None',
            '--optimizer',
            'adamw',
            '--scheduler',
            'plateau',
            '--lr-factor',
            '0.5',
            '--scheduler-patience',
            '4',
        ]
    )
    train_cli.apply_cli_overrides(args)
    assert (
        gin.query_parameter(
            'mace_jax.tools.gin_functions.optimizer.decoupled_weight_decay'
        )
        is True
    )
    assert (
        gin.query_parameter('mace_jax.tools.gin_functions.optimizer.scheduler')
        == gin_functions.reduce_on_plateau
    )
    gin.clear_config()


def test_cli_sets_runtime_and_training_controls(tmp_path, simple_hdf5_path):
    gin.clear_config()
    train_hdf5 = simple_hdf5_path
    args, _ = train_cli.parse_args(
        [
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
            str(train_hdf5),
            '--valid_file',
            'None',
            '--energy_key',
            'E',
            '--forces_key',
            'F',
            '--clip_grad',
            '0.5',
            '--ema',
            '--loss',
            'huber',
            '--energy_weight',
            '0.7',
            '--forces_weight',
            '2.5',
            '--stress_weight',
            '0.1',
            '--huber_delta',
            '0.05',
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
            '--eval-interval',
            '2',
            '--max-num-intervals',
            '7',
            '--patience',
            '4',
            '--eval-train-fraction',
            '0.5',
            '--eval-test',
            '--log-errors',
            'PerAtomMAE',
            '--beta',
            '0.7',
        ]
    )
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
    ) == str(train_hdf5)
    assert (
        gin.query_parameter('mace_jax.tools.gin_datasets.datasets.valid_path') is None
    )
    assert gin.query_parameter('mace_jax.tools.gin_datasets.datasets.energy_key') == 'E'
    assert gin.query_parameter('mace_jax.tools.gin_datasets.datasets.forces_key') == 'F'
    loss_fn = gin_functions.loss()
    assert isinstance(loss_fn, modules.WeightedHuberEnergyForcesStressLoss)
    assert loss_fn.energy_weight == pytest.approx(0.7)
    assert loss_fn.forces_weight == pytest.approx(2.5)
    assert loss_fn.stress_weight == pytest.approx(0.1)
    assert loss_fn.huber_delta == pytest.approx(0.05)
    assert (
        gin.query_parameter('mace_jax.tools.gin_functions.train.max_grad_norm') == 0.5
    )
    assert gin.query_parameter('mace_jax.tools.gin_functions.train.ema_decay') == 0.99
    assert gin.query_parameter('mace_jax.tools.gin_functions.optimizer.max_epochs') == 7
    assert gin.query_parameter('mace_jax.tools.gin_functions.train.patience') == 4
    assert gin.query_parameter(
        'mace_jax.tools.gin_functions.train.eval_train'
    ) == pytest.approx(0.5)
    assert gin.query_parameter('mace_jax.tools.gin_functions.train.eval_test') is True
    assert gin.query_parameter('mace_jax.tools.gin_functions.train.eval_interval') == 2
    assert (
        gin.query_parameter('mace_jax.tools.gin_functions.train.log_errors')
        == 'PerAtomMAE'
    )
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
    assert gin.query_parameter('amsgrad.b1') == 0.7
    assert (
        gin.query_parameter(
            'mace_jax.tools.gin_functions.optimizer.decoupled_weight_decay'
        )
        is False
    )
    gin.clear_config()
