"""Datasets for Fairness-aware Analysis or Modeling."""

import pandas as pd

from pathlib2 import Path
from os.path import dirname

from .german_credit_data_map import (
    GERMAN_CREDIT_VARIABLE_MAP, preprocess_german_credit_data)
from .data_types import VariableType


def _map_categorical_variables(series, variable_map):
    variable = variable_map[series.name]
    if variable.variable_type == VariableType.NUMERIC:
        return series
    return series.map(lambda x: variable.categorical_dict[x])


def _apply_data_map_and_preprocessor(df, variable_map, preprocessor):
    return (
        df
        .apply(_map_categorical_variables, variable_map=variable_map)
        .pipe(preprocessor)
    )


def german_credit():
    """Load German Credit Dataset.

    The target variable is "credit_risk", where 0 = bad and 1 = good
    """
    data_path = Path(dirname(__file__)) / "data"
    # the data file is space-delimited.
    return _apply_data_map_and_preprocessor(
        pd.read_csv(str(data_path / "german_credit.csv")),
        GERMAN_CREDIT_VARIABLE_MAP.variable_map, preprocess_german_credit_data)
