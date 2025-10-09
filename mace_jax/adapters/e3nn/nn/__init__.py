"""Thin Flax-compatible wrappers for ``e3nn.nn`` building blocks."""

from ._activation import Activation
from ._extract import Extract
from ._fc import FullyConnectedNet
from ._gate import Gate

__all__ = ['Activation', 'Extract', 'FullyConnectedNet', 'Gate']
