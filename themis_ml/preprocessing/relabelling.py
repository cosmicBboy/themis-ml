"""Relabel examples in a dataset for fairness-aware model training."""

import numpy as np
import math

from sklearn.base import BaseEstimator, TransformerMixin, MetaEstimatorMixin
from sklearn.utils.validation import check_array, check_X_y, check_is_fitted
from sklearn.linear_model import LogisticRegression

from ..checks import check_binary


def _n_relabels(y, s):
    """Compute the number of promotions/demotions that need to occur.

    :param np.array y: target labels
    :param np.array s: protected class labels
    :returns: number of promotions/demotions to occur.
    :rtype: int
    """
    total = float(len(s))
    s1 = s.sum()
    s0 = total - s1
    s1_positive = ((s == 1) & (y == 1)).sum()
    s0_positive = ((s == 0) & (y == 1)).sum()
    return int(math.ceil(((s1 * s0_positive) - (s0 * s1_positive)) / total))


def _relabel(y, s, r, promote_ranks, demote_ranks):
    if ((s and not y and r in promote_ranks) or
            (not s and y and r in demote_ranks)):
        return int(not y)
    return y


def _relabel_targets(y, s, ranks, n_relabels):
    """Compute relabelled targets based on predicted ranks."""
    demote_ranks = set(sorted(ranks[(s == 0) & (y == 1)])[:n_relabels])
    promote_ranks = set(sorted(ranks[(s == 1) & (y == 0)])[-n_relabels:])
    return np.array([
        _relabel(_y, _s, _r, promote_ranks, demote_ranks)
        for _y, _s, _r in zip(y, s, ranks)])


class Relabeller(BaseEstimator, TransformerMixin, MetaEstimatorMixin):

    def __init__(self, ranker=LogisticRegression()):
        """Create a Relabeller.

        This technique relabels target variables using a function that can
        compute a decision boundary in input data space using the following
        heuristic

        - The top `n` -ve labelled observations in the disadvantaged group `s1`
          that are closest to the decision boundary are "promoted" to the +ve
          label.
        - the top `n` +ve labelled observations in the advantaged group s0
          closest to the decision boundary are "demoted' to the -ve label.

        `n` is the number of promotions/demotions needed to make
        p(+|s0) = p(+|s1)

        :param BaseEstimator ranker: estimator to use as the ranker for
            relabelling observations close to the decision boundary. Default:
            LogisticRegression
        """
        self.ranker = ranker

    def fit(self, X, y=None, s=None):
        """Fit relabeller."""
        X, y = check_X_y(X, y)
        y = check_binary(y)
        s = check_binary(np.array(s).astype(int))
        if s.shape[0] != y.shape[0]:
            raise ValueError("`s` must be the same shape as `y`")
        self.n_relabels_ = _n_relabels(y, s)
        self.ranks_ = self.ranker.fit(X, y).predict_proba(X)[:, 1]
        self.X_ = X
        self.y_ = y
        self.s_ = s
        return self

    def transform(self, X):
        """Transform relabeller."""
        check_is_fitted(self, ["n_relabels_", "ranks_"])
        X = check_array(X)
        # Input X should be equal to the input to `fit`
        if not np.isclose(X, self.X_).all():
            raise ValueError(
                "`transform` input X must be equal to input X to `fit`")
        return _relabel_targets(
            self.y_, self.s_, self.ranks_, self.n_relabels_)
