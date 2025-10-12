from __future__ import annotations

import argparse
import dataclasses
import glob
import json
import logging
from collections.abc import Iterable
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams.update({'font.size': 8})
plt.style.use('seaborn-v0_8-paper')

colors = [
    '#1f77b4',
    '#d62728',
    '#ff7f0e',
    '#2ca02c',
    '#9467bd',
    '#8c564b',
    '#e377c2',
    '#7f7f7f',
    '#bcbd22',
    '#17becf',
]


@dataclasses.dataclass(frozen=True)
class RunDescriptor:
    directory: Path
    stem: str


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Plot training curves from MACE-JAX metrics files.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--path',
        required=True,
        help='Path to a .metrics file or a directory containing such files.',
    )
    parser.add_argument(
        '--min-interval',
        type=int,
        default=0,
        help='Discard data recorded before this interval.',
    )
    parser.add_argument(
        '--keys',
        type=str,
        default='rmse_e_per_atom,rmse_f',
        help='Comma-separated list of metric keys to plot alongside the loss curves.',
    )
    parser.add_argument(
        '--linear',
        action='store_true',
        help='Use linear rather than log scales for plots.',
    )
    parser.add_argument(
        '--error-bars',
        action='store_true',
        help='Draw mean ± one standard deviation if multiple runs are present.',
    )
    parser.add_argument(
        '--start-interval',
        type=int,
        default=None,
        help='Optional interval at which a second training stage started; '
        'drawn as a vertical dashed line.',
    )
    parser.add_argument(
        '--output-format',
        type=str,
        default='png',
        help='Image format to use when saving plots.',
    )
    return parser.parse_args(argv)


def collect_metric_paths(path: str) -> list[Path]:
    candidate = Path(path)
    if candidate.is_file():
        return [candidate]

    matches = sorted(Path(p) for p in glob.glob(str(candidate / '*.metrics')))
    if not matches:
        raise FileNotFoundError(f"No '.metrics' files found under '{path}'.")
    return matches


def parse_metrics(path: Path) -> list[dict]:
    descriptor = RunDescriptor(directory=path.parent, stem=path.stem)
    records: list[dict] = []
    with path.open('rt', encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                logging.warning('Skipping malformed line in %s: %s', path, line)
                continue
            data['run_directory'] = str(descriptor.directory)
            data['run_name'] = descriptor.stem
            records.append(data)
    return records


def load_dataframe(paths: Iterable[Path]) -> pd.DataFrame:
    rows: list[dict] = []
    for metric_path in paths:
        rows.extend(parse_metrics(metric_path))
    if not rows:
        raise RuntimeError('No metrics data could be parsed.')
    frame = pd.DataFrame(rows)
    if 'interval' not in frame:
        raise KeyError("Metrics files do not contain an 'interval' field.")
    return frame


def _aggregate(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    columns = ['run_directory', 'run_name', 'mode', 'interval']
    available_keys = [k for k in keys if k in frame.columns]
    if available_keys:
        columns.extend(available_keys)
    if 'loss' in frame.columns:
        columns.append('loss')

    trimmed = frame[columns].copy()
    grouped = (
        trimmed.groupby(['run_directory', 'run_name', 'mode', 'interval'])
        .agg(['mean', 'std'])
        .reset_index()
    )
    grouped.columns = [
        '_'.join(filter(None, col)).strip('_') for col in grouped.columns
    ]
    return grouped


def plot_run(
    data: pd.DataFrame,
    keys: list[str],
    *,
    output_path: Path,
    linear: bool,
    error_bars: bool,
    start_interval: int | None,
) -> None:
    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(10, 3),
        constrained_layout=True,
    )

    def _plot_series(ax, series, label, color):
        ax.plot(series['interval'], series['loss_mean'], label=label, color=color)
        if error_bars and 'loss_std' in series:
            y = series['loss_mean']
            std = series['loss_std'].fillna(0.0)
            ax.fill_between(
                series['interval'],
                y - std,
                y + std,
                alpha=0.2,
                color=color,
            )

    loss_ax = axes[0]
    eval_data = data[data['mode'] == 'eval']
    train_data = data[data['mode'] == 'eval_train']

    if not eval_data.empty:
        _plot_series(loss_ax, eval_data, 'Validation', colors[0])
    if not train_data.empty:
        _plot_series(loss_ax, train_data, 'Train (eval interval)', colors[3])

    if start_interval is not None:
        loss_ax.axvline(start_interval, linestyle='--', color='black', alpha=0.5)

    loss_ax.set_xlabel('Interval')
    loss_ax.set_ylabel('Loss')
    if not linear:
        loss_ax.set_xscale('log')
        loss_ax.set_yscale('log')
    loss_ax.legend()

    metric_ax = axes[1]
    for idx, key in enumerate(keys):
        column_mean = f'{key}_mean'
        if column_mean not in data.columns:
            logging.debug("Metric '%s' not present in metrics file; skipping.", key)
            continue
        series = data[data['mode'] == 'eval']
        if series.empty:
            continue
        metric_ax.plot(
            series['interval'],
            series[column_mean],
            label=key,
            color=colors[(idx + 1) % len(colors)],
        )
        if error_bars:
            column_std = f'{key}_std'
            if column_std in series:
                std = series[column_std].fillna(0.0)
                metric_ax.fill_between(
                    series['interval'],
                    series[column_mean] - std,
                    series[column_mean] + std,
                    alpha=0.2,
                    color=colors[(idx + 1) % len(colors)],
                )

    metric_ax.set_xlabel('Interval')
    metric_ax.legend()
    if not linear:
        metric_ax.set_xscale('log')
        metric_ax.set_yscale('log')

    fig.savefig(output_path, format=output_path.suffix.lstrip('.'))
    plt.close(fig)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    metric_paths = collect_metric_paths(args.path)
    frame = load_dataframe(metric_paths)

    frame = frame[frame['interval'] >= args.min_interval]
    if frame.empty:
        raise RuntimeError('No records remain after applying the interval filter.')

    keys = [key.strip() for key in args.keys.split(',') if key.strip()]
    aggregated = _aggregate(frame, keys)

    for (directory, name), group in aggregated.groupby(['run_directory', 'run_name']):
        output_dir = Path(directory)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f'{name}.{args.output_format}'
        plot_run(
            group,
            keys,
            output_path=output_file,
            linear=args.linear,
            error_bars=args.error_bars,
            start_interval=args.start_interval,
        )
        logging.info('Saved plot to %s', output_file)


if __name__ == '__main__':
    main()
