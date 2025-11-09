"""Nox automation tasks for mace-jax."""

from __future__ import annotations

import nox


PYTHON = '3.10'
PACKAGE = 'mace_jax'


def _install_package(session: nox.Session, *extras: str) -> None:
    target = '.'
    if extras:
        target = f'.[{",".join(extras)}]'
    session.install('pip', 'setuptools', 'wheel')
    session.install('-e', target)


@nox.session(python=PYTHON)
def lint(session: nox.Session) -> None:
    """Run static analysis."""

    session.install('ruff')
    session.run('ruff', 'check', PACKAGE, 'tests')


@nox.session(python=PYTHON)
def tests(session: nox.Session) -> None:
    """Run the unit test suite."""

    _install_package(session, 'test')
    session.run('pytest')


@nox.session(python=PYTHON)
def fusion_benchmark(session: nox.Session) -> None:
    """Smoke-test the conv-fusion benchmark pipeline."""

    _install_package(session, 'test', 'conv_fusion', 'plot')
    session.run(
        'python',
        'playground/benchmark_inference.py',
        '--repeats',
        '1',
        '--warmup',
        '1',
        '--cue-conv-fusion',
    )
