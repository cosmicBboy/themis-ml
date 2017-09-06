"""Module for Fairness-aware base estimators."""

import numpy as np

from sklearn.base import (
    BaseEstimator, ClassifierMixin, MetaEstimatorMixin, clone)
from sklearn.utils.validation import check_array, check_X_y, check_is_fitted
from checks import check_binary, s_is_needed_on_fit, s_is_needed_on_predict


class FairnessAwareMetaEstimator(
        BaseEstimator, ClassifierMixin, MetaEstimatorMixin):

    def __init__(self, estimator, relabeller=None):
        """Initialize metaestimator for composing fairness-aware methods.

        :param Estimator estimator:
        :param Transformer|None relabeller:
        """
        self.relabeller = relabeller
        self.estimator = estimator

    def fit(self, X, y, s=None):
        X, y = check_X_y(X, y)
        y = check_binary(y)
        self.relabeller_ = None
        self.estimator_ = clone(self.estimator)
        # fit_transform y labels using estimator
        if self.relabeller is not None:
            self.relabeller_ = clone(self.relabeller)
            y = self.relabeller_.fit_transform(X, y, s=s)
        # fit estimator
        if s_is_needed_on_fit(self.estimator_, s):
            s = check_binary(np.array(s).astype(int))
            self.estimator_.fit(X, y, s)
        else:
            # since relabeller by definition needs s, this checks whether
            # relabeller is None and the `s` array is provided.
            if self.relabeller_ is None and s is not None:
                raise ValueError(
                    "`s` arg provided but %s fit doesn't accept `s`" %
                    self.estimator_)
            self.estimator_.fit(X, y)

    def predict(self, X, s=None):
        check_is_fitted(self, ["estimator_", "relabeller_"])
        X = check_array(X)
        if s_is_needed_on_predict(self.estimator_, s):
            s = check_binary(np.array(s).astype(int))
            return self.estimator_.predict(X, s)
        else:
            if s is not None:
                raise ValueError(
                    "`s` arg provided but %s predict doesn't accept `s`" %
                    self.estimator_)
            return self.estimator_.predict(X)

    def predict_proba(self, X, s=None):
        if not hasattr(self.estimator_, "predict_proba"):
            raise AttributeError(
                "%s has no method `predict_proba`" % self.estimator_)
        check_is_fitted(self, ["estimator_", "relabeller_"])
        X = check_array(X)
        if s_is_needed_on_predict(self.estimator_, s):
            s = check_binary(np.array(s).astype(int))
            return self.estimator_.predict_proba(X, s)
        else:
            if s is not None:
                raise ValueError(
                    "`s` arg provided but %s predict doesn't accept `s`" %
                    self.estimator_)
            return self.estimator_.predict_proba(X)
