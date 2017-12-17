"""Train counterfactually fair models.

This module contains an implementation of a linear counterfactually fair
model that uses the protected class variable to compute the residuals for each
input variable and uses those residuals to learn a function that maps from
inputs to the target variable.

Reference:
Kusner, M. J., Loftus, J. R., Russell, C., & Silva, R. (2017).
Counterfactual Fairness. Available at arXiv: https://arxiv.org/abs/1703.06856.
"""

import numpy as np

from enum import Enum
from functools import partial
from sklearn.base import (
    BaseEstimator, ClassifierMixin, MetaEstimatorMixin, clone)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.utils.validation import check_array, check_X_y, check_is_fitted

from ..checks import check_binary, is_binary, is_continuous
from ..stats_utils import pearson_residuals, deviance_residuals


def _get_binary_X_index(X):
    return np.where(np.apply_along_axis(is_binary, 0, X))[0]


def _get_continuous_X_index(X):
    return np.where(np.apply_along_axis(is_continuous, 0, X))[0]


def _compute_binary_residuals(estimator, s, true, residual_type):
    if residual_type == _BinaryResidualTypes.absolute:
        return _compute_absolute_residuals(
            estimator, s, true, predict_proba=True)
    elif residual_type == _BinaryResidualTypes.pearson:
        residual_func = pearson_residuals
    elif residual_type == _BinaryResidualTypes.deviance:
        residual_func = deviance_residuals
    else:
        raise ValueError("unsupported residual type: %s" % residual_type)
    return residual_func(true, estimator.predict_proba(s)[:, 1])


def _compute_absolute_residuals(estimator, s, true, predict_proba=False):
    if predict_proba:
        return true - estimator.predict_proba(s)[:, 1]
    return true - estimator.predict(s)


class _BinaryResidualTypes(Enum):
    absolute = 0
    pearson = 1
    deviance = 2


class LinearACFClassifier(BaseEstimator, ClassifierMixin, MetaEstimatorMixin):

    VALID_BINARY_RESIDUAL_TYPES = [e.name for e in _BinaryResidualTypes]
    S_ON_FIT = True
    S_ON_PREDICT = True

    def __init__(self, target_estimator=LogisticRegression(),
                 continuous_estimator=LinearRegression(),
                 binary_estimator=LogisticRegression(),
                 binary_residual_type="pearson"):
        """Instantiate a linear additive counterfactually-fair classifier.

        :param BaseEstimator target_estimator: A classifier for learning a
            function that maps to input residuals and target variable.
        :param BaseEstimator continuous_estimator: A regressor for computing
            the residuals for continuous X inputs.
        :param BaseEstimator binary_estimator: A classifier estimator for
            computing the residuals for binary X inputs.
        :param str binary_residual_type: The type of residual to use for binary
            residuals. Options: {"pearson", "deviance"}. Default: "pearson".
        """
        if binary_residual_type not in self.VALID_BINARY_RESIDUAL_TYPES:
            raise ValueError(
                "invalid binary residual type: %s. Must be one of %s"
                % (binary_residual_type, self.VALID_BINARY_RESIDUAL_TYPES))

        self.target_estimator = target_estimator
        self.continuous_estimator = continuous_estimator
        self.binary_estimator = binary_estimator
        self.binary_residual_type = binary_residual_type

    def fit(self, X, y, s):
        """Fit model."""
        X, y = check_X_y(X, y)
        y = check_binary(y)
        s = check_binary(np.array(s).astype(int))

        # save the indices on the X adxis
        self.binary_index_ = _get_binary_X_index(X)
        self.continuous_index_ = _get_continuous_X_index(X)
        self.n_input_variables_ = X.shape[1]

        self.residual_estimators_ = []
        self.compute_residual_funcs_ = []
        self.target_estimator_ = clone(self.target_estimator)

        # for quick lookup
        binary_index_set = set(self.binary_index_)
        continuous_index_set = set(self.continuous_index_)

        # store residuals
        self.fit_residuals_ = np.zeros_like(X, dtype="float64")
        residual_input = s.reshape(-1, 1)

        # fit residual estimators and compute residuals
        for i in range(self.n_input_variables_):
            if i in binary_index_set and len(set(X[:, i])) == 1:
                # if a binary variable only contains one of the classes
                # in the training set, then no residual can be computed.
                estimator, compute_residual_func = None, None
            elif i in continuous_index_set:
                estimator = clone(self.continuous_estimator)
                compute_residual_func = _compute_absolute_residuals
            elif i in binary_index_set:
                estimator = clone(self.binary_estimator)
                compute_residual_func = partial(
                    _compute_binary_residuals,
                    residual_type=self._binary_residual_type)
            else:
                raise ValueError(
                    "index %s is not in continuous_index_ or binary_index_")
            # fit residual estimator and compute residuals
            if estimator and compute_residual_func:
                estimator.fit(residual_input, X[:, i])
                self.fit_residuals_[:, i] = compute_residual_func(
                    estimator, residual_input, X[:, i])
            else:
                self.fit_residuals_[:, i] = 0
            self.compute_residual_funcs_.append(compute_residual_func)
            self.residual_estimators_.append(estimator)

        # fit target_estimator_
        self.target_estimator_.fit(self.fit_residuals_, y)
        return self

    def _compute_residuals_on_predict(self, X, s):
        predict_residuals = np.zeros_like(X, dtype="float64")
        residual_input = s.reshape(-1, 1)
        for i, (estimator, compute_residual_func) in enumerate(
                zip(self.residual_estimators_, self.compute_residual_funcs_)):
            if estimator and compute_residual_func:
                predict_residuals[:, i] = compute_residual_func(
                    estimator, residual_input, X[:, i])
            else:
                predict_residuals[:, i] = self.fit_residuals_[:, i]
        return predict_residuals

    def _check_fitted(self, X):
        X = check_array(X)
        if X.shape[1] != self.n_input_variables_:
            raise ValueError(
                "input `X` has %s variables but %s expected %s variables."
                % (X.shape[1], self.__name__, self.n_input_variables_))
        for i in self.binary_index_:
            check_binary
        # TODO: check that binary_index and continuous_index in X are indeed
        # binary and continuous.
        check_is_fitted(
            self,
            ["binary_index_",
             "continuous_index_",
             "n_input_variables_",
             "target_estimator_",
             "residual_estimators_",
             "compute_residual_funcs_",
             "fit_residuals_"
             ])
        return X

    def predict(self, X, s):
        """Generate predicted labels."""
        X = self._check_fitted(X)
        return self.target_estimator_.predict(
            self._compute_residuals_on_predict(X, s))

    def predict_proba(self, X, s):
        """Generate predicted probabilities."""
        self._check_fitted(X)
        predict_residuals = self._compute_residuals_on_predict(X, s)
        return self.target_estimator_.predict_proba(predict_residuals)

    @property
    def _binary_residual_type(self):
        return _BinaryResidualTypes[self.binary_residual_type]
