"""Unit tests for stats_utils functions."""

import numpy as np
import pytest

from themis_ml import stats_utils


@pytest.fixture
def label_pred_data():
    return np.array([
        [1, 0.8],
        [1, 0.7],
        [1, 0.6],
        [1, 0.5],
        [0, 0.4],
        [0, 0.3],
        [0, 0.2],
        [0, 0.1],
    ])


def test_pearson_residuals(label_pred_data):
    y = label_pred_data[:, 0]
    pred = label_pred_data[:, 1]
    r = stats_utils.pearson_residuals(y, pred)
    assert r.dtype == np.dtype("float64")
    # for y = {0, 1} and 0 < pred 1:
    # 0 labels will be negative since both numerator and
    # denominator are negative:
    # - pearson_residuals = (y - pred) / np.sqrt(pred * (1 - pred))
    #   where y and pred are positive real numbers.
    assert (r[y == 1] > 0).all()
    assert (r[y == 0] < 0).all()


def test_deviance_residuals(label_pred_data):
    y = label_pred_data[:, 0]
    pred = label_pred_data[:, 1]
    d = stats_utils.pearson_residuals(y, pred)
    assert d.dtype == np.dtype("float64")
    # for y = {0, 1} and 0 < pred 1:
    # deviance for 0 labels evaluate to a negative number, and
    # deviance for 1 labels evaluate to a positive number.
    # see the `themis_ml.stats_utils.deviance_residuals` docstring for
    # mathematical details.
    assert (d[y == 1] > 0).all()
    assert (d[y == 0] < 0).all()

