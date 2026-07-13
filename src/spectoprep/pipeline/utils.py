"""
Utility functions for pipeline optimization.
"""

import itertools

import numpy as np

from spectoprep.logging import get_logger

logger = get_logger("PipelineUtils")


def choose_nearest(value: float, allowed: list[float]) -> float:
    """
    Choose the nearest allowed value to the given value.

    Args:
        value: The input value
        allowed: List of allowed values

    Returns:
        The nearest allowed value
    """
    return min(allowed, key=lambda x: abs(x - value))


def is_valid_pipeline(pipeline: tuple[str, ...], incompatible_sets: list[list[str]]) -> bool:
    """
    Check if a pipeline configuration is valid based on incompatibility rules.

    Args:
        pipeline: Tuple of preprocessing step names
        incompatible_sets: List of sets, each containing mutually incompatible steps

    Returns:
        bool: True if the pipeline is valid, False otherwise
    """
    # Convert pipeline to set for easier intersection checks
    pipeline_set = set(pipeline)

    # Check against each incompatibility set
    for incompatible_set in incompatible_sets:
        # If the intersection of pipeline and incompatible set has more than 1 element,
        # it means we have incompatible steps in the pipeline
        if len(pipeline_set.intersection(incompatible_set)) > 1:
            return False

    return True


def generate_pipeline_configurations(
    preprocessing_steps: list[str],
    incompatible_sets: list[list[str]],
    max_length: int = 5,
    allowed_lengths: int | list[int] | tuple[int, ...] | None = None,
) -> list[tuple[str, ...]]:
    """
    Generate all valid pipeline configurations based on available steps and incompatibility rules.

    Args:
        preprocessing_steps: List of available preprocessing step names
        incompatible_sets: List of sets, each containing mutually incompatible steps
        max_length: Maximum number of steps in a pipeline
        allowed_lengths: Specific pipeline lengths to allow (e.g., [1, 2])

    Returns:
        List of tuples, each representing a valid pipeline configuration
    """
    # Normalize allowed lengths
    if allowed_lengths is None:
        allowed_lengths = list(range(1, max_length + 1))
    elif isinstance(allowed_lengths, int):
        allowed_lengths = [allowed_lengths]

    # Ensure allowed lengths are within bounds
    allowed_lengths = [length for length in allowed_lengths if 1 <= length <= max_length]

    # Generate all possible combinations for the allowed lengths
    all_valid_pipelines = []

    logger.info(
        "generating_pipeline_configurations",
        lengths=allowed_lengths,
        n_steps=len(preprocessing_steps),
    )

    for length in allowed_lengths:
        # Generate all combinations of the given length
        combinations = list(itertools.combinations(preprocessing_steps, length))

        # Filter out invalid combinations based on incompatibility rules
        valid_combinations = [
            comb for comb in combinations if is_valid_pipeline(comb, incompatible_sets)
        ]

        all_valid_pipelines.extend(valid_combinations)

    logger.info("pipeline_configurations_generated", n_configs=len(all_valid_pipelines))

    return all_valid_pipelines


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Root Mean Square Error (RMSE).

    Args:
        y_true: True target values
        y_pred: Predicted target values

    Returns:
        float: RMSE value
    """
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def calculate_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the R² score.

    Args:
        y_true: True target values
        y_pred: Predicted target values

    Returns:
        float: R² value
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot if ss_tot != 0 else 0.0
