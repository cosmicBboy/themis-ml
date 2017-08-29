"""Module for Fairness-aware scoring metrics."""

import numpy as np

from .checks import check_binary


def mean_difference(y, s):
    """Compute the mean difference in y with respect to protected class s.

    In the binary target case, the mean difference metric measures the
    difference in the following conditional probabilities:

    mean_difference = p(y+ | s_0) - p(y+ | s_1)

    In the continuous target case, the mean difference metric measures the
    difference in the expected value of y conditioned on the protected class:

    mean_difference = E(y+ | s_0) - E(y+ | s_1)

    Where y+ is the desireable outcome, s_0 is the advantaged group, and
    s_1 is the disadvantaged group.

    :param array-like y: shape (n, ) containing binary target variable, where
        1 is the desireable outcome and 0 is the undesireable outcome.
    :param array-like s: shape (n, ) containing binary protected class
        variable where 0 is the advantaged groupd and 1 is the disadvantaged
        group.
    :returns: mean difference between advantaged group and disadvantaged group.
    :rtype: float
    """
    y, s = np.array(y).astype(float), np.array(s)
    check_binary(s)
    return np.mean(y[np.where(s == 0)]) - np.mean(y[np.where(s == 1)])
