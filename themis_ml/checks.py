"""Utility functions for doing checks."""

CONTINUOUS_DTYPES = [int, float]


def check_binary(x):
    if not is_binary(x):
        raise ValueError("%s must be a binary variable" % x)
    return x


def check_continuous(x):
    if not is_continuous(x):
        raise ValueError("%s must be a continuous variable" % x)
    return x


def is_binary(x):
    return set(x.ravel()).issubset({0, 1})


def is_continuous(x):
    return not is_binary(x) and x.dtype in CONTINUOUS_DTYPES
