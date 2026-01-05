#!/usr/bin/env python3
"""Benchmark sequential HDF5 read throughput for MACE datasets."""

from __future__ import annotations

import argparse
import random
import time

from mace_jax.data import HDF5Dataset


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Measure sequential HDF5 read throughput.',
    )
    parser.add_argument('path', help='Path to a MACE HDF5 file')
    parser.add_argument(
        '--count',
        type=int,
        default=None,
        help='Number of entries to read (default: all)',
    )
    parser.add_argument(
        '--repeat',
        type=int,
        default=1,
        help='Number of measurement repeats',
    )
    parser.add_argument(
        '--warmup',
        type=int,
        default=0,
        help='Number of warmup passes to ignore',
    )
    parser.add_argument(
        '--stride',
        type=int,
        default=1,
        help='Stride between entries (default: 1)',
    )
    parser.add_argument(
        '--shuffle',
        action='store_true',
        help='Also run a shuffled (random-order) read benchmark.',
    )
    parser.add_argument(
        '--shuffle-seed',
        type=int,
        default=0,
        help='Seed for shuffled index order.',
    )
    parser.add_argument(
        '--touch',
        action='store_true',
        help='Touch arrays on the atoms object to force materialization',
    )
    return parser.parse_args()


def _iter_indices(total: int, count: int | None, stride: int) -> range:
    if stride <= 0:
        raise ValueError('stride must be a positive integer.')
    limit = total if count is None else min(int(count), total)
    return range(0, limit * stride, stride)


def _read_pass(dataset: HDF5Dataset, indices, *, touch: bool):
    start = time.monotonic()
    seen = 0
    for idx in indices:
        atoms = dataset[int(idx)]
        if touch:
            _ = atoms.get_atomic_numbers()
            _ = atoms.positions
            _ = atoms.arrays.get('forces')
        seen += 1
    elapsed = time.monotonic() - start
    return seen, elapsed


def _run_mode(
    label: str,
    dataset: HDF5Dataset,
    *,
    make_indices,
    repeats: int,
    warmup: int,
    touch: bool,
) -> None:
    total_seen = 0
    total_time = 0.0
    for idx in range(max(warmup, 0)):
        _read_pass(dataset, make_indices(idx), touch=touch)
    for idx in range(max(repeats, 1)):
        seen, elapsed = _read_pass(dataset, make_indices(idx), touch=touch)
        total_seen += seen
        total_time += elapsed
        rate = seen / elapsed if elapsed > 0 else 0.0
        print(
            f'{label} pass: entries={seen} elapsed={elapsed:.3f}s '
            f'rate={rate:.2f} entries/s'
        )
    avg_rate = total_seen / total_time if total_time > 0 else 0.0
    print(
        f'{label} avg: entries={total_seen} elapsed={total_time:.3f}s '
        f'rate={avg_rate:.2f} entries/s'
    )


def main() -> None:
    args = _parse_args()
    with HDF5Dataset(args.path, mode='r') as dataset:

        def make_sequential(_):
            return _iter_indices(len(dataset), args.count, args.stride)

        def make_shuffled(idx):
            indices = list(_iter_indices(len(dataset), args.count, args.stride))
            rng = random.Random(int(args.shuffle_seed) + int(idx))
            rng.shuffle(indices)
            return indices

        _run_mode(
            'sequential',
            dataset,
            make_indices=make_sequential,
            repeats=args.repeat,
            warmup=args.warmup,
            touch=args.touch,
        )
        if args.shuffle:
            _run_mode(
                'shuffled',
                dataset,
                make_indices=make_shuffled,
                repeats=args.repeat,
                warmup=args.warmup,
                touch=args.touch,
            )


if __name__ == '__main__':
    main()
