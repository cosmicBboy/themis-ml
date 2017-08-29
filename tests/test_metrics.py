"""Unit tests for metrics module."""

import pytest

from themis_ml import metrics


def test_mean_difference_full_discrimination():
    """Binary case: disadvantaged group is fully discriminated against."""
    assert metrics.mean_difference(
        [1, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 1]) == 1


def test_mean_difference_partial_discrimination():
    """Binary case: disadvantaged group is partially discriminated against."""
    assert metrics.mean_difference(
        [1, 1, 1, 1, 0, 0, 1, 1],
        [0, 0, 0, 0, 1, 1, 1, 1]) == 0.5


def test_mean_difference_no_discrimination():
    """Binary case: proportion in each group with +ve outcome are equal."""
    assert metrics.mean_difference(
        [0, 0, 1, 1, 0, 0, 1, 1],
        [0, 0, 0, 0, 1, 1, 1, 1]) == 0


def test_mean_difference_partial_reverse_discrimination():
    """Binary case: advantaged group is partially discriminated against."""
    assert metrics.mean_difference(
        [1, 1, 0, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, 1, 1, 1, 1]) == -0.5


def test_mean_difference_full_reverse_discrimination():
    """Binary case: advantaged group is fully discriminated against."""
    assert metrics.mean_difference(
        [0, 0, 0, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, 1, 1, 1, 1]) == -1


def test_error_non_binary_protected_class():
    with pytest.raises(ValueError):
        metrics.mean_difference(
            [0, 0, 0, 0, 1, 1, 1, 1],
            [1, 0, 0, 1, 2, 3, 1, 0])


def test_error_non_numeric_target_and_protected_class():
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
