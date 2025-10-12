from pathlib import Path

import pytest

from mace_jax.cli import mace_jax_train as train_cli
from mace_jax.cli import mace_jax_train_plot as train_plot


@pytest.mark.slow
def test_run_train_cli_dry_run(tmp_path, monkeypatch):
    results_dir = tmp_path / 'results'
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
    ]

    train_cli.main(args)

    gin_files = list(Path(results_dir).glob('*.gin'))
    assert gin_files, 'Expected operative gin config to be written during dry run.'


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
