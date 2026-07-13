"""
Top-level package for SpectoPrep.
SpectroPrep: A comprehensive toolkit for spectroscopic data preprocessing and modeling.

This package provides tools for preprocessing spectroscopic data,
pipeline optimization, and modeling using Ridge regression.
"""

__author__ = """Habeeb Babatunde"""
__email__ = "babatundehabeeb2@gmail.com"
__version__ = "1.0.1-alpha"

from .logging import configure_logging, get_logger
from .modelling.ridge import OptimizedRidgeCV
from .pipeline.optimizer import PipelineOptimizer
from .visualization.plots import SpectroPrepPlotter

__all__ = [
    "PipelineOptimizer",
    "OptimizedRidgeCV",
    "SpectroPrepPlotter",
    "configure_logging",
    "get_logger",
    "__version__",
]
