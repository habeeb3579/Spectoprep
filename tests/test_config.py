"""Tests validating the configuration constants."""

from spectoprep.pipeline.config import (
    AVAILABLE_STEPS,
    DEFAULT_PARAM_BOUNDS,
    INCOMPATIBLE_SETS,
)


def test_available_steps_keys_unique_and_nonempty():
    assert len(AVAILABLE_STEPS) == len(set(AVAILABLE_STEPS))
    assert all(isinstance(k, str) and k for k in AVAILABLE_STEPS)


def test_incompatible_sets_subset_of_available_steps():
    known = set(AVAILABLE_STEPS)
    for group in INCOMPATIBLE_SETS:
        assert set(group).issubset(known), f"unknown steps in {group}"


def test_param_bounds_low_less_than_high():
    for key, (low, high) in DEFAULT_PARAM_BOUNDS.items():
        assert low < high, f"bound {key} has low >= high"


def test_ridge_alpha_bound_present():
    assert "ridge_alpha" in DEFAULT_PARAM_BOUNDS


def test_pipeline_config_not_in_default_bounds():
    # pipeline_config is injected at runtime by the optimizer, not a default.
    assert "pipeline_config" not in DEFAULT_PARAM_BOUNDS
