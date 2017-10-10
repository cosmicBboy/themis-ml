"""Specify data types."""

import enum

from collections import OrderedDict


class VariableType(enum.Enum):
    BINARY = 0
    NON_ORDERED_CATEGORICAL = 1
    ORDERED_CATEGORICAL = 2
    NUMERIC = 3


class Variable(object):

    def __init__(
            self, name, variable_type, transformer=None, is_target=False,
            ignore=False):
        self.name = name
        self.variable_type = variable_type
        self.transformer = transformer
        self.is_target = is_target
        self.ignore = ignore


class VariableMap(object):

    def __init__(self, variables):
        self._variables = variables
        targets = [v.name for v in variables if v.is_target]
        self._targets = targets if len(targets) > 0 else None

    @property
    def all_variables(self):
        return [v.name for v in self._variables]

    @property
    def variable_map(self):
        return OrderedDict([(v.name, v) for v in self._variables])

    def _get_variables(self, variable_type):
        return [
            k for k, v in self.variable_map.items()
            if v.variable_type == variable_type
            and not v.is_target and not v.ignore]

    @property
    def binary_variables(self):
        return self._get_variables(VariableType.BINARY)

    @property
    def non_ordered_categorical_variables(self):
        return self._get_variables(VariableType.NON_ORDERED_CATEGORICAL)

    @property
    def ordered_categorical_variables(self):
        return self._get_variables(VariableType.ORDERED_CATEGORICAL)

    @property
    def numeric_variables(self):
        return self._get_variables(VariableType.NUMERIC)

    @property
    def targets(self):
        return self._targets


def string_cleaner(s):
    """Function for cleaning raw string values.

    :param str s: string to clean.
    :returns: string, lowercased with spaces replaced with "_".
    """
    return s.strip().lower().replace(" ", "_")
