"""Unit tests for metrics module."""

import pytest

from themis_ml import metrics


def _get_point_est(md_tuple):
    """Get point estimate in mean_difference/normalized_mean_difference."""
    return md_tuple[0]


def test_mean_difference_full_discrimination():
    """Binary case: disadvantaged group is fully discriminated against."""
    y = [1, 1, 1, 1, 0, 0, 0, 0]
    s = [0, 0, 0, 0, 1, 1, 1, 1]
    assert _get_point_est(metrics.mean_difference(y, s)) == 1
    assert _get_point_est(metrics.normalized_mean_difference(y, s)) == 1


def test_mean_difference_partial_discrimination():
    """Binary case: disadvantaged group is partially discriminated against."""
    y = [1, 1, 1, 1, 0, 0, 1, 1]
    s = [0, 0, 0, 0, 1, 1, 1, 1]
    assert _get_point_est(metrics.mean_difference(y, s)) == 0.5
    assert _get_point_est(metrics.normalized_mean_difference(y, s)) == 1


def test_mean_difference_no_discrimination():
    """Binary case: proportion in each group with +ve outcome are equal."""
    y = [0, 0, 1, 1, 0, 0, 1, 1]
    s = [0, 0, 0, 0, 1, 1, 1, 1]
    assert _get_point_est(metrics.mean_difference(y, s)) == 0
    assert _get_point_est(metrics.normalized_mean_difference(y, s)) == 0


def test_mean_difference_partial_reverse_discrimination():
    """Binary case: advantaged group is partially discriminated against."""
    y = [1, 1, 0, 0, 1, 1, 1, 1]
    s = [0, 0, 0, 0, 1, 1, 1, 1]
    assert _get_point_est(metrics.mean_difference(y, s)) == -0.5
    assert _get_point_est(metrics.normalized_mean_difference(y, s)) == -1


def test_mean_difference_full_reverse_discrimination():
    """Binary case: advantaged group is fully discriminated against."""
    y = [0, 0, 0, 0, 1, 1, 1, 1]
    s = [0, 0, 0, 0, 1, 1, 1, 1]
    assert _get_point_est(metrics.mean_difference(y, s)) == -1
    assert _get_point_est(metrics.normalized_mean_difference(y, s)) == -1


def test_error_non_binary_protected_class():
    with pytest.raises(ValueError):
        metrics.mean_difference(
            [0, 0, 0, 0, 1, 1, 1, 1],
            [1, 0, 0, 1, 2, 3, 1, 0])
    with pytest.raises(ValueError):
        metrics.normalized_mean_difference(
            [0, 0, 0, 0, 1, 1, 1, 1],
            [1, 0, 0, 1, 2, 3, 1, 0])


def test_error_non_numeric_target_and_protected_class():
    # mean difference
    with pytest.raises(ValueError):
        metrics.mean_difference(
            ["a", "b", "c", "d"],
            ["a", "b", "c", "d"])
    with pytest.raises(ValueError):
        metrics.mean_difference(
            ["a", "b", "c", "d"],
            [1, 0, 0, 1])
    with pytest.raises(ValueError):
        metrics.mean_difference(
            [1, 0, 0, 1],
            ["a", "b", "c", "d"])
    # normalized mean difference
    with pytest.raises(ValueError):
        metrics.normalized_mean_difference(
            ["a", "b", "c", "d"],
            ["a", "b", "c", "d"])
    with pytest.raises(ValueError):
        metrics.normalized_mean_difference(
            ["a", "b", "c", "d"],
            [1, 0, 0, 1])
    with pytest.raises(ValueError):
        metrics.normalized_mean_difference(
            [1, 0, 0, 1],
            ["a", "b", "c", "d"])
