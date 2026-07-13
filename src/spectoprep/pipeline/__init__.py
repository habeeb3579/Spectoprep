"""Pipeline construction and Bayesian optimization for SpectoPrep."""

from .builder import build_preprocessor_from_bayes
from .config import AVAILABLE_STEPS, DEFAULT_PARAM_BOUNDS, INCOMPATIBLE_SETS
from .optimizer import PipelineOptimizer
from .utils import (
    calculate_r2,
    calculate_rmse,
    choose_nearest,
    generate_pipeline_configurations,
    is_valid_pipeline,
)

__all__ = [
    "PipelineOptimizer",
    "build_preprocessor_from_bayes",
    "generate_pipeline_configurations",
    "is_valid_pipeline",
    "choose_nearest",
    "calculate_rmse",
    "calculate_r2",
    "AVAILABLE_STEPS",
    "INCOMPATIBLE_SETS",
    "DEFAULT_PARAM_BOUNDS",
]
