"""Datasets for Fairness-aware Analysis or Modeling."""

import pandas as pd

from pathlib2 import Path
from os.path import dirname

from .german_credit_data_map import (
    german_credit_variable_map, preprocess_german_credit_data)
from .census_income_data_map import (
    preprocess_census_income_data, census_income_variable_map)
from .data_types import VariableType


def _data_path():
    return Path(dirname(__file__)) / "data"


def _map_transformer(series, variable_map):
    """Private function for making categorical variables human-readable.

    For raw datasets that use non-human-readable codes in categorical
    variables, this function is used to convert them to human-readable values.
    """
    variable = variable_map[series.name]
    if variable.transformer is None:
        return series
    try:
        return series.map(lambda x: variable.transformer[x])
    except TypeError:
        return series.map(variable.transformer)


def _apply_data_map(df, variable_map):
    return df.apply(_map_transformer, variable_map=variable_map)


def german_credit(raw=False):
    """Load German Credit Dataset.

    The target variable is "credit_risk", where 0 = bad and 1 = good

    :param bool raw: If True, return raw data, otherwise return model-ready
        data. The model-ready data has columns arranged in the order of:

        - numeric features.
        - ordered categorical features.
        - binary features.
        - non-ordered categorical features.
        - target.

        Note: Raw data does not have this ordering, nor does it have dummified
        categorical variables.
    :returns: DataFrame of raw or model-ready data.
    """
    out = _apply_data_map(
        pd.read_csv(str(_data_path() / "german_credit.csv")),
        german_credit_variable_map.variable_map)
    if raw:
        return out
    return preprocess_german_credit_data(out)


def census_income(raw=False):
    """Load Census Income Data from 1994 - 1995.

    The target variable is "income_gt_50k" (income above $50,000), where 0 is
    below and 1 is above.

    :param bool raw: if True, return raw data, otherwise return model-ready
        data. The model-ready data has columns arranged in the the order of:

        - numeric features.
        - ordered categorical features.
        - binary features.
        - non-ordered categorical features.
        - target.

    :returns: DataFrame of raw or model-ready data.
    """
    train = pd.read_csv(
        str(_data_path() / "census_income_1994_1995_train.csv"),
        names=census_income_variable_map.all_variables) \
        .pipe(_apply_data_map, census_income_variable_map.variable_map)
    test = pd.read_csv(
        str(_data_path() / "census_income_1994_1995_test.csv"),
        names=census_income_variable_map.all_variables) \
        .pipe(_apply_data_map, census_income_variable_map.variable_map)
    out = (
        pd.concat([
            train.assign(dataset_partition="training_set"),
            test.assign(dataset_partition="test_set")]))
    if raw:
        return out
    return preprocess_census_income_data(out)
