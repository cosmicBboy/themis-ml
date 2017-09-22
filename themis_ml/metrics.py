"""Module for Fairness-aware scoring metrics."""

import numpy as np
import scipy

from .checks import check_binary
from math import sqrt
from scipy.stats import t

DEFAULT_CI = 0.975


def mean_confidence_interval(x, confidence=0.95):
    a = np.array(x) * 1.0
    mu, se = np.mean(a), scipy.stats.sem(a)
    me = se * t._ppf((1 + confidence) / 2., len(a) - 1)
    return mu, mu - me, mu + me


def mean_differences_ci(y, s, ci=DEFAULT_CI):
    """Calculate the mean difference and confidence interval.

    :param array-like y: shape (n, ) containing binary target variable, where
        1 is the desireable outcome and 0 is the undesireable outcome.
    :param array-like s: shape (n, ) containing binary protected class
        variable where 0 is the advantaged groupd and 1 is the disadvantaged
        group.
    :param float ci: % confidence interval to compute. Default: 97.5% to
        compute 95% two-sided t-statistic associated with degrees of freedom.
    :returns: mean difference between advantaged group and disadvantaged group
        with lower and upper bound confidence interval estimates.
    :rtype: tuple
    """
    n0 = (s == 0).sum().astype(float)
    n1 = (s == 1).sum().astype(float)
    df = n0 + n1 - 2
    std0 = y[s == 0].std()
    std1 = y[s == 1].std()
    std_n0n1 = sqrt(((n1 - 1)*(std1)**2 + (n0 - 1)*(std0)**2) / df)
    mean_diff = y[s == 0].mean() - y[s == 1].mean()
    margin_error = t.ppf(ci, df) * std_n0n1 * sqrt(1/n0 + 1 / float(n1))
    return mean_diff, mean_diff - margin_error, mean_diff + margin_error


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

    :param numpy.array y: shape (n, ) containing binary target variable, where
        1 is the desireable outcome and 0 is the undesireable outcome.
    :param numpy.array s: shape (n, ) containing binary protected class
        variable where 0 is the advantaged groupd and 1 is the disadvantaged
        group.
    :returns: mean difference between advantaged group and disadvantaged group.
    :rtype: float
    """
    y = check_binary(np.array(y).astype(int))
    s = check_binary(np.array(s).astype(int))
    return mean_differences_ci(y, s)


def normalized_mean_difference(y, s, norm_y=None, ci=DEFAULT_CI):
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
      observations.
    - if there are fewer negative examples than there are disadvantaged
      observations.

    Reference:
    Zliobaite, I. (2015). A survey on measuring indirect discrimination in
    machine learning. arXiv preprint arXiv:1511.00148.

    :param numpy.array y: shape (n, ) containing binary target variable, where
        1 is the desireable outcome and 0 is the undesireable outcome.
    :param numpy.array s: shape (n, ) containing binary protected class
        variable where 0 is the advantaged groupd and 1 is the disadvantaged
        group.
    :param numpy.array|None norm_y: shape (n, ) or None. If provided, this
        array is used to compute the normalization factor d_max.
    :returns: mean difference between advantaged group and disadvantaged group
        with lower and upper confidence interval bounds
    :rtype: tuple(float)
    """
    y = check_binary(np.array(y).astype(int))
    s = check_binary(np.array(s).astype(int))
    norm_y = y if norm_y is None else norm_y
    d_max = float(
        min(np.mean(norm_y) / (1 - np.mean(s)),
            (1 - np.mean(norm_y)) / np.mean(s)))
    md = mean_differences_ci(y, s)
    # TODO: Figure out if scaling the CI bounds by d_max makes sense here.
    if d_max == 0:
        return md
    lower_ci = md[1] / d_max
    lower_ci = lower_ci if lower_ci > -1 else -1
    upper_ci = md[2] / d_max
    upper_ci = upper_ci if upper_ci < 1 else 1
    return (md[0] / d_max, lower_ci, upper_ci)


def abs_mean_difference_delta(y, pred, s):
    """Compute lift in mean difference between y and pred.

    This measure represents the delta between absolute mean difference score
    in true y and predicted y. Values are in the range [0, 1] where the higher
    the value, the better. Note that this takes into account the reverse
    discrimintion case.

    :param numpy.array y: shape (n, ) containing binary target variable, where
        1 is the desireable outcome and 0 is the undesireable outcome.
    :param numpy.array pred: shape (n, ) containing binary predicted target,
        where 1 is the desireable outcome and 0 is the undesireable outcome.
    :param numpy.array s: shape (n, ) containing binary protected class
        variable where 0 is the advantaged groupd and 1 is the disadvantaged
        group.
    :returns: absolute difference in mean difference score between true y and
        predicted y
    :rtype: float
    """
    return abs(mean_difference(y, s)[0]) - abs(mean_difference(pred, s)[0])


def abs_normalized_mean_difference_delta(y, pred, s):
    """Compute lift in normalized mean difference between y and pred.

    This measure represents the delta between absolute normalized mean
    difference score in true y and predicted y. Values are in the range [0, 1]
    where the higher the value, the better. Note that this takes into account
    the reverse discrimintion case. Also note that the normalized mean
    difference score for predicted y's uses the true target for the
    normalization factor.

    :param numpy.array y: shape (n, ) containing binary target variable, where
        1 is the desireable outcome and 0 is the undesireable outcome.
    :param numpy.array pred: shape (n, ) containing binary predicted target,
        where 1 is the desireable outcome and 0 is the undesireable outcome.
    :param numpy.array s: shape (n, ) containing binary protected class
        variable where 0 is the advantaged groupd and 1 is the disadvantaged
        group.
    :returns: absolute difference in mean difference score between true y and
        predicted y
    :rtype: float
    """
    return (abs(normalized_mean_difference(y, s)[0]) -
            abs(normalized_mean_difference(pred, s)[0]))
