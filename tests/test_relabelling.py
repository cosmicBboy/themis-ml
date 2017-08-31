"""Unit tests for relabelling estimator."""

import numpy as np
import pytest


from themis_ml.preprocessing.relabelling import Relabeller


def X():
    return np.array([range(10), range(11, 21)]).T


def y():
    return np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])


def s():
    return np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])


def test_relabeller_fit():
    """Test that relabeller fitting """
    relabeller = Relabeller()
    X_input = X()
    targets = y()
    protected_class = s()
    # The formula to determine how many observations to promote/demote is
    # the number needed to make the proportion of positive labels equal
    # between the two groups. This proportion is rounded up.
    expected_n = 3
    # Given data specified in X function, the default LogisticRegression
    # estimator should be able to draw a perfect decision boundary to seperate
    # the y target.
    relabeller.fit(X_input, targets, protected_class)
    assert relabeller.n_relabels_ == expected_n
    assert (relabeller.X_ == X_input).all()
    assert (relabeller.y_ == targets).all()
    assert (relabeller.s_ == protected_class).all()


def test_relabeller_transform():
    """Test that relabeller correctly relabels targets."""
    expected = np.array([[0, 0, 1, 1, 1, 0, 0, 0, 1, 1]])
    assert (Relabeller().fit_transform(X(), y(), s=s()) == expected).all()


def test_fit_error():
    """Test fit method errors out."""
    # case: s array not the same length as y array
    with pytest.raises(ValueError):
        Relabeller().fit(X(), y(), np.array([1, 0, 0, 1]))
    # case: y targets are not a binary variable
    with pytest.raises(TypeError):
        targets = y()
        targets[0] = 100
        Relabeller().fit_transform(X(), targets, s())
    # case: s protected classes are not a binary variable
    with pytest.raises(TypeError):
        s_classes = y()
        s_classes[0] = 100
        Relabeller().fit_transform(X(), targets, s_classes)


def test_fit_transform_error():
    """Test fit_transform method errors out.

    ValueError should occur when X input to transform method is not the same
    as the X input to fit method.
    """
    X_input = X()
    with pytest.raises(ValueError):
        Relabeller().fit(X_input, y(), s()).transform(X_input.T)
