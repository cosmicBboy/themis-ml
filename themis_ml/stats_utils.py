"""Utility functions for computing useful statistics."""

import numpy as np


def pearson_residuals(y, pred):
    """Compute Pearson residuals.

    Reference:
    https://web.as.uky.edu/statistics/users/pbreheny/760/S11/notes/4-12.pdf

    :param array-like[int] y: target labels. 1 is positive label, 0 is negative
        label
    :param array-like[float] pred: predicted labels.
    :returns: pearson residual.
    :rtype: array-like[float]
    """
    y, pred = np.array(y), np.array(pred)
    return (y - pred) / np.sqrt(pred * (1 - pred))


def deviance_residuals(y, pred):
    """Compute Deviance residuals.

    Reference:
    https://web.as.uky.edu/statistics/users/pbreheny/760/S11/notes/4-12.pdf

    Formula:
    d = sign * sqrt(-2 * {y * log(p) + (1 - y) * log(1 - p)})
    - where sign is -1 if y = 1 and 1 if y = 0
    - y is the true label
    - p is the predicted probability

    :param array-like[int] y: target labels. 1 is positive label, 0 is negative
        label
    :param array-like[float] pred: predicted labels.
    :returns: deviance residual.
    :rtype: array-like[float]
    """
    y, pred = np.array(y), np.array(pred)
    sign = np.array([1 if y_i else -1 for y_i in y])
    return sign * np.sqrt(-2 * (y * np.log(pred) + (1 - y) * np.log(1 - pred)))
