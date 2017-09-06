"""Unit tests for themis_ml meta estimators."""

import pytest

from sklearn.linear_model import LogisticRegression

from themis_ml.meta_estimators import FairnessAwareMetaEstimator
from themis_ml.linear_model import counterfactually_fair_models
from themis_ml.preprocessing import relabelling
from themis_ml.postprocessing import reject_option_classification

from conftest import create_linear_X, create_y, create_s


def test_fairness_aware_meta_estimator():
    X = create_linear_X()
    y = create_y()
    s = create_s()

    # use sklearn estimator
    lr = FairnessAwareMetaEstimator(LogisticRegression())
    lr.fit(X, y)
    lr.predict(X)
    lr.predict_proba(X)

    # use themis_ml LinearACFClassifier estimator
    linear_acf = FairnessAwareMetaEstimator(
        counterfactually_fair_models.LinearACFClassifier(
            binary_residual_type="absolute"))
    linear_acf.fit(X, y, s)
    linear_acf.predict(X, s)
    linear_acf.predict_proba(X, s)

    # when fit/predict methods need s, raise ValueError
    with pytest.raises(ValueError):
        linear_acf.fit(X, y, s=None)
    with pytest.raises(ValueError):
        linear_acf.predict(X, s=None)
    with pytest.raises(ValueError):
        linear_acf.predict_proba(X, s=None)

    # use themis_ml relabeller preprocessor
    relabel_clf = FairnessAwareMetaEstimator(
        estimator=LogisticRegression(), relabeller=relabelling.Relabeller())
    relabel_clf.fit(X, y, s)
    relabel_clf.predict(X)
    relabel_clf.predict_proba(X)

    # when estimator method does not need `s` on predict, raise ValueError if
    # it is provided
    with pytest.raises(ValueError):
        relabel_clf.predict(X, s)
        relabel_clf.predict_proba(X, s)

    # use themis_ml RejectOption Classifier
    single_reject_option_clf = FairnessAwareMetaEstimator(
        reject_option_classification.SingleROClassifier())
    single_reject_option_clf.fit(X, y)
    single_reject_option_clf.predict(X, s)
    single_reject_option_clf.predict_proba(X, s)

    with pytest.raises(ValueError):
        single_reject_option_clf.fit(X, y, s)
    with pytest.raises(ValueError):
        single_reject_option_clf.predict(X, s=None)
    with pytest.raises(ValueError):
        single_reject_option_clf.predict_proba(X, s=None)

    multi_reject_option_clf = FairnessAwareMetaEstimator(
        reject_option_classification.MultipleROClassifier())
    multi_reject_option_clf.fit(X, y)
    multi_reject_option_clf.predict(X, s)
    multi_reject_option_clf.predict_proba(X, s)

    with pytest.raises(ValueError):
        multi_reject_option_clf.fit(X, y, s)
    with pytest.raises(ValueError):
        multi_reject_option_clf.predict(X, s=None)
    with pytest.raises(ValueError):
        multi_reject_option_clf.predict_proba(X, s=None)
