"""Unit tests for datasets."""

from themis_ml import datasets


def test_german_credit():
    """Test correct shape and content of german credit data."""
    data = datasets.german_credit()
    # correct number of rows and columns
    assert data.shape == (1000, 46)
    # correct proportion of target values
    assert (data["credit_risk"].value_counts().loc[[0, 1]] == [300, 700]).all()


def test_census_income():
    """Test correct shape and content of census income data."""
    data = datasets.census_income()
    # correct number of rows and columns
    assert data.shape == (299285, 396)
    # correct proportion of target values
    assert (
        data["income_gt_50k"].value_counts().loc[[0, 1]] ==
        [280717, 18568]).all()
    # correct proportion of training and test data (the original data contained)
    assert (
        data["dataset_partition"].value_counts().loc[
            ["training_set", "test_set"]]).all()
