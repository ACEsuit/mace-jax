"""Supported OpenEquivariance adapters.

Standalone and fully connected tensor products are not public because the
OpenEquivariance 0.6.8 backward path is unsafe for general multi-problem use.
"""

from .tensor_product import TensorProduct

__all__ = ['TensorProduct']
