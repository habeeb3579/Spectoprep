"""Tests for spectoprep.pipeline.utils pure helper functions."""

import numpy as np
import pytest

from spectoprep.pipeline.config import INCOMPATIBLE_SETS
from spectoprep.pipeline.utils import (
    calculate_r2,
    calculate_rmse,
    choose_nearest,
    generate_pipeline_configurations,
    is_valid_pipeline,
)


class TestChooseNearest:
    def test_exact_match(self):
        assert choose_nearest(9, [7, 9, 11]) == 9

    def test_nearest(self):
        assert choose_nearest(10.4, [7, 9, 11, 13]) == 11

    def test_tie_returns_first(self):
        # 8 is equidistant from 7 and 9; min returns the first minimum.
        assert choose_nearest(8, [7, 9, 11, 13]) == 7

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            choose_nearest(1.0, [])


class TestIsValidPipeline:
    def test_incompatible_pair_rejected(self):
        assert is_valid_pipeline(("snv", "msc"), INCOMPATIBLE_SETS) is False

    def test_compatible_pair_accepted(self):
        assert is_valid_pipeline(("snv", "savgol"), INCOMPATIBLE_SETS) is True

    def test_empty_pipeline_valid(self):
        assert is_valid_pipeline((), INCOMPATIBLE_SETS) is True

    def test_single_step_valid(self):
        assert is_valid_pipeline(("snv",), INCOMPATIBLE_SETS) is True


class TestGeneratePipelineConfigurations:
    def test_length_one_count(self):
        steps = ["snv", "savgol", "detrend"]
        configs = generate_pipeline_configurations(steps, INCOMPATIBLE_SETS, allowed_lengths=1)
        assert len(configs) == 3
        assert all(len(c) == 1 for c in configs)

    def test_pairs_only_and_all_valid(self):
        steps = ["snv", "msc", "savgol", "detrend"]
        configs = generate_pipeline_configurations(steps, INCOMPATIBLE_SETS, allowed_lengths=2)
        assert all(len(c) == 2 for c in configs)
        # No returned pair violates the incompatibility rules.
        assert all(is_valid_pipeline(c, INCOMPATIBLE_SETS) for c in configs)
        # snv+msc are incompatible and must be excluded.
        assert ("snv", "msc") not in configs

    def test_out_of_range_lengths_dropped(self):
        steps = ["snv", "savgol"]
        configs = generate_pipeline_configurations(
            steps, INCOMPATIBLE_SETS, max_length=2, allowed_lengths=[5]
        )
        assert configs == []

    def test_default_lengths_span_one_to_max(self):
        steps = ["snv", "savgol"]
        configs = generate_pipeline_configurations(steps, INCOMPATIBLE_SETS, max_length=2)
        lengths = {len(c) for c in configs}
        assert lengths == {1, 2}


class TestMetrics:
    def test_rmse_zero_on_identical(self):
        y = np.array([1.0, 2.0, 3.0])
        assert calculate_rmse(y, y) == pytest.approx(0.0)

    def test_rmse_known_value(self):
        y_true = np.array([0.0, 0.0])
        y_pred = np.array([3.0, 4.0])
        assert calculate_rmse(y_true, y_pred) == pytest.approx(np.sqrt((9 + 16) / 2))

    def test_r2_perfect(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        assert calculate_r2(y, y) == pytest.approx(1.0)

    def test_r2_constant_true_returns_zero(self):
        y_true = np.array([5.0, 5.0, 5.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        assert calculate_r2(y_true, y_pred) == 0.0
