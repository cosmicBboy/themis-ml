"""Unit tests for reject option classification."""

import math
import numpy as np
import pytest

from sklearn.exceptions import NotFittedError

from themis_ml.postprocessing.reject_option_classification import (
    SingleROClassifier, MultipleROClassifier, DECISION_THRESHOLD)
from conftest import create_linear_X, create_y, create_s


def test_single_ro_clf_fit_predict_demote_true():
    """Test fit and predict methods when demote is True.

    The create_linear_X, create_y, and create_Y functions (see conftest.py)
    are set up such a linear classifier can find a neat decision boundary to
    perfectly classify examples.
    """
    X = create_linear_X()
    y = create_y()
    s = create_s()
    roc_clf = SingleROClassifier(theta=0.2)
    roc_clf.fit(X, y)
    raw_pred_proba = roc_clf._raw_predict_proba(X, s)
    raw_pred = (raw_pred_proba[:, 1] > DECISION_THRESHOLD).astype(int)
    pred_proba = roc_clf.predict_proba(X, s)
    pred = roc_clf.predict(X, s)
    # all raw predictions should perfectly classify
    assert (raw_pred == y).all()
    assert (pred == (pred_proba[:, 1] > DECISION_THRESHOLD)).all()

    # with the theta critical region threshold of 0.2, the middle two
    # observations in create_linear_X should be flipped in label.
    midpoint = (X.shape[0] - 1) / 2.0
    promote_index = int(math.floor(midpoint))
    demote_index = int(math.ceil(midpoint))
    other_index = [i for i in range(X.shape[0])
                   if i not in [promote_index, demote_index]]
    # check correctly flipped predicted labels
    assert pred[promote_index] == (1 - raw_pred[promote_index])
    assert pred[demote_index] == (1 - raw_pred[demote_index])
    assert (pred[other_index] == raw_pred[other_index]).all()
    # check correctly flipped predicted probability
    assert pred_proba[promote_index, 1] == \
        (1 - raw_pred_proba[promote_index, 1])
    assert pred_proba[demote_index, 1] == (1 - raw_pred_proba[demote_index, 1])
    assert (pred[other_index] == raw_pred[other_index]).all()


def test_single_ro_clf_fit_predict_demote_false():
    """Test fit and predict methods when demote is False.

    When demote=False, only disadvantage group examples should be promoted
    (i.e. no advantaged group demotions).
    """
    X = create_linear_X()
    y = create_y()
    s = create_s()

    roc_clf = SingleROClassifier(theta=0.2, demote=False)
    roc_clf.fit(X, y)
    raw_pred_proba = roc_clf._raw_predict_proba(X, s)
    raw_pred = (raw_pred_proba[:, 1] > DECISION_THRESHOLD).astype(int)
    pred_proba = roc_clf.predict_proba(X, s)
    pred = roc_clf.predict(X, s)
    # all raw predictions should perfectly classify
    assert (raw_pred == y).all()
    assert (pred == (pred_proba[:, 1] > DECISION_THRESHOLD)).all()

    # with the theta critical region threshold of 0.2, and demote=False,
    # only the disadvantaged observation will be promoted.
    midpoint = (X.shape[0] - 1) / 2.0
    promote_index = int(math.floor(midpoint))
    other_index = [i for i in range(X.shape[0]) if i not in [promote_index]]
    # check correctly flipped predicted labels
    assert pred[promote_index] == (1 - raw_pred[promote_index])
    assert (pred[other_index] == raw_pred[other_index]).all()
    # check correctly flipped predicted probability
    assert pred_proba[promote_index, 1] == \
        (1 - raw_pred_proba[promote_index, 1])
    assert (pred[other_index] == raw_pred[other_index]).all()


def test_multiple_ro_clf_fit_predict_demote_true():
    """Test fit and predict methods when demote is True.

    The create_linear_X, create_y, and create_Y functions (see conftest.py)
    are set up such a linear classifier can find a neat decision boundary to
    perfectly classify examples.
    """
    X = create_linear_X()
    y = create_y()
    s = create_s()
    roc_clf = MultipleROClassifier(theta=0.2)
    roc_clf.fit(X, y)
    raw_pred_proba = roc_clf._raw_predict_proba(X, s)
    raw_pred = (raw_pred_proba[:, 1] > DECISION_THRESHOLD).astype(int)
    pred_proba = roc_clf.predict_proba(X, s)
    pred = roc_clf.predict(X, s)

    # all raw predictions should perfectly classify
    assert (raw_pred == y).all()
    assert (pred == (pred_proba[:, 1] > DECISION_THRESHOLD)).all()

    # probabilities should be the weighted mean of probabilities from the
    # specified estimators in the ensemble
    expected_probs = np.zeros_like(y, dtype="float64")
    for e, w in zip(roc_clf.estimators_, roc_clf.pred_weights_):
        expected_probs += e.predict_proba(X)[:, 1] * w
    expected_probs = expected_probs / roc_clf.pred_weights_.sum()
    assert (raw_pred_proba[:, 1] == expected_probs).all()


def test_not_fitted_error():
    """Test raises not fitted error if predict before fit."""
    with pytest.raises(NotFittedError):
        SingleROClassifier().predict(create_linear_X(), create_y())
