import json
from pathlib import Path

import pytest

from mace_jax.cli import mace_jax_train as train_cli
from mace_jax.cli import mace_jax_train_plot as train_plot
import gin


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
        '\n'.join(
            [
                '{"interval": 1, "mode": "eval", "loss": 0.5, "mae_e": 0.2, "mae_f": 0.3}',
                '{"interval": 1, "mode": "eval_train", "loss": 0.4}',
                '{"interval": 2, "mode": "eval", "loss": 0.3, "mae_e": 0.1, "mae_f": 0.2}',
                '{"interval": 2, "mode": "eval_train", "loss": 0.35}',
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
