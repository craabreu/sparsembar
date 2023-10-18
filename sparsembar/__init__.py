"""
SparseMBAR: A sparse-matrix variant of the multistate Bennett acceptance ratio (MBAR)
estimator
"""

# flake8: noqa
# pylint: disable=wrong-import-position
import jax

jax.config.update("jax_enable_x64", True)

from ._version import __version__
from .multigaussian import MultiGaussian
from .sparsembar import SparseMBAR
from .stategroup import StateGroup
