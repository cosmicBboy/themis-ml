"""Post-processing estimators to make fair predictions."""

import numpy as np

from sklearn.utils.validation import check_array, check_X_y, check_is_fitted
from sklearn.base import (
    BaseEstimator, ClassifierMixin, MetaEstimatorMixin, clone)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

from ..checks import check_binary

DECISION_THRESHOLD = 0.5
DEFAULT_ENSEMBLE_ESTIMATORS = [
    LogisticRegression(), DecisionTreeClassifier()]


class SingleROClassifier(
        BaseEstimator, ClassifierMixin, MetaEstimatorMixin):

    def __init__(self, estimator=LogisticRegression(), theta=0.1, demote=True):
        """Initialize Single Reject-Option Classifier.

        This fairness-aware technique produces fair predictions with the
        following heuristic:
        - an training an initial classifier on dataset D
        - generating predicted probabilities on the test set
        - computing the proximity of each prediction to the decision boundary
          learned by the classifier
        - within the critical region threshold theta around the decision
          boundary, where 0.5 < theta < 1, X_s1 (disadvantaged observations)
          are assigned as y+ and X_s0 (advantaged observations are assigned as
          y-.

        param BaseEstimator estimator: LogisticRegression by default
        param float theta: critical region threshold for demoting advantaged
            group and promoting advantaged group
        param bool demote: if True, demotes +ve labelled advantaged group
            observations at predict time. If False, only promote -ve labelled
            disadvantaged group observations at predict time.
        """
        # TODO: assert that estimator has a predict_proba method.
        self.estimator = estimator
        self.theta = theta
        self.demote = demote

    def fit(self, X, y):
        """Fit model."""
        X, y = check_X_y(X, y)
        y = check_binary(y)
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, y)
        return self

    def predict(self, X, s):
        """Generate predicted labels."""
        return (
            self.predict_proba(X, s)[:, 1] > DECISION_THRESHOLD).astype(int)

    def predict_proba(self, X, s):
        """Generate predicted probabilities."""
        pred_prob = self._raw_predict_proba(X, s)[:, 1]
        return self._flip_predictions(pred_prob, s)

    def _raw_predict_proba(self, X, s):
        X = check_array(X)
        s = check_binary(np.array(s).astype(int))
        check_is_fitted(self, ["estimator_"])
        return self.estimator_.predict_proba(X)

    def _flip_predictions(self, pred_prob, s):
        """Flip predictions based on protected class membership.

        :param np.array[float] pred_prob: predicted probabilities
        :param np.array[int] s: protected class membership, where
            1 = disadvantaged group, 0 = advantaged group.
        """
        flip_candidates = np.ones_like(pred_prob).astype(bool) \
            if self.demote else s == 1

        # find index where predictions are below theta threshold
        under_theta_index = np.where(
            (np.abs(pred_prob - 0.5) < self.theta) & flip_candidates)
        # flip the probability
        pred_prob[under_theta_index] = 1 - pred_prob[under_theta_index]
        pred_prob = pred_prob.reshape(-1, 1)
        return np.concatenate([1 - pred_prob, pred_prob], axis=1)


class MultipleROClassifier(SingleROClassifier):

    def __init__(
            self, estimators=DEFAULT_ENSEMBLE_ESTIMATORS,
            theta=0.1, demote=True, weighted_prediction=True):
        """Initialize Multiple Reject-Option Classifier.

        param list|tuple[BaseEstimator] estimators: A list or tuple of
            estimators to train multiple classifiers. By default, use
            LogisticRegression and DecisionTreeClassifier.
        param bool weighted_prediction: if True, uses the training accuracy
            as weights to compute ensembled prediction
        param bool demote: if True, demotes +ve labelled advantaged group
            observations at predict time. If False, only promote -ve labelled
            disadvantaged group observations at predict time.
        param bool weighted_prediction: if True, then uses accuracy score as
            weights to compute ensembled predicted probability. If False,
            ensembled probability is the mean of probabilities.
        """
        # TODO: assert that all estimators have a predict_proba method.
        # TODO: add support for customizing the performance function used
        # to compute the estimator weights used in the ensembled prediction.
        # Currently this class only supports accuracy.
        super(MultipleROClassifier, self).__init__()
        self.estimators = estimators
        self.demote = demote
        self.weighted_prediction = weighted_prediction

    def fit(self, X, y):
        """Fit model."""
        X, y = check_X_y(X, y)
        y = check_binary(y)
        self.estimators_ = []
        self.pred_weights_ = []
        for estimator in self.estimators:
            e = clone(estimator)
            self.estimators_.append(e.fit(X, y))
            # uniform weights if weighted_prediction is False
            self.pred_weights_.append(
                accuracy_score(y, e.predict(X)) if self.weighted_prediction
                else 1.0)
        self.pred_weights_ = np.array(self.pred_weights_)
        return self

    def _raw_predict_proba(self, X, s):
        X = check_array(X)
        s = check_binary(np.array(s).astype(int))
        check_is_fitted(self, ["estimators_", "pred_weights_"])
        # use uniform weights if pred_weights_ is False otherwise use
        # performance scores learned during
        pred_probs = np.concatenate([
            e.predict_proba(X)[:, 1].reshape(-1, 1) * w for e, w in
            zip(self.estimators_, self.pred_weights_)
        ], axis=1).sum(axis=1) / self.pred_weights_.sum()
        pred_probs = pred_probs.reshape(-1, 1)
        return np.concatenate([1 - pred_probs, pred_probs], axis=1)
