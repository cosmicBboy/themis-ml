"""Module for Fairness-aware scoring metrics."""

import numpy as np

from .checks import check_binary


def _mean_difference(y, s):
    """Compute mean difference."""
    return np.mean(y[np.where(s == 0)]) - np.mean(y[np.where(s == 1)])


def mean_difference(y, s):
    """Compute the mean difference in y with respect to protected class s.

    In the binary target case, the mean difference metric measures the
    difference in the following conditional probabilities:

    mean_difference = p(y+ | s0) - p(y+ | s1)

    In the continuous target case, the mean difference metric measures the
    difference in the expected value of y conditioned on the protected class:

    mean_difference = E(y+ | s0) - E(y+ | s1)

    Where y+ is the desireable outcome, s0 is the advantaged group, and
    s1 is the disadvantaged group.

    Reference:
    Zliobaite, I. (2015). A survey on measuring indirect discrimination in
    machine learning. arXiv preprint arXiv:1511.00148.

    :param array-like y: shape (n, ) containing binary target variable, where
        1 is the desireable outcome and 0 is the undesireable outcome.
    :param array-like s: shape (n, ) containing binary protected class
        variable where 0 is the advantaged groupd and 1 is the disadvantaged
        group.
    :returns: mean difference between advantaged group and disadvantaged group.
    :rtype: float
    """
    y = check_binary(np.array(y).astype(int))
    s = check_binary(np.array(s).astype(int))
    return _mean_difference(y, s)


def normalized_mean_difference(y, s):
    """Compute normalized mean difference in y with respect to s.

    Same the mean difference score, except the score takes into account the
    maximum possible discrimination at a given positive outcome rate. Is only
    defined when y and s are both binary variables.

    normalized_mean_difference = mean_difference / d_max

    where d_max = min( (p(y+) / p(s0)), ((p(y-) / p(s1)) )

    The d_max normalization term denotes the smaller value of either the
    ratio of positive labels and advantaged observations or the ratio of
    negative labels and disadvantaged observations.

    Therefore the normalized mean difference will report a higher score than
    mean difference in two cases:
    - if there are fewer positive examples than there are advantaged
      observations
    - if there are fewer negative examples than there are disadvanted
      observations

    Reference:
    Zliobaite, I. (2015). A survey on measuring indirect discrimination in
    machine learning. arXiv preprint arXiv:1511.00148.
    """
    y = check_binary(np.array(y).astype(int))
    s = check_binary(np.array(s).astype(int))
    d_max = min(np.mean(y) / (1 - np.mean(s)), (1 - np.mean(y)) / np.mean(s))
    return _mean_difference(y, s) / float(d_max)
