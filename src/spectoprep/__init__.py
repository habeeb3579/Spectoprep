"""
Top-level package for SpectoPrep.
SpectroPrep: A comprehensive toolkit for spectroscopic data preprocessing and modeling.

This package provides tools for preprocessing spectroscopic data,
pipeline optimization, and modeling using Ridge regression.
"""
from .pipeline.optimizer import PipelineOptimizer
from .modelling.ridge import OptimizedRidgeCV
from .visualization.plots import SpectroPrepPlotter


__all__ = [
    'PipelineOptimizer',
    'OptimizedRidgeCV',
    'SpectroPrepPlotter'
]

__author__ = """Habeeb Babatunde"""
__email__ = 'babatundehabeeb2@gmail.com'
__version__ = '0.1.0'