"""Utility functions for doing checks."""


def check_binary(x):
    if set(x) != {0, 1}:
        raise ValueError("%s must be a binary variable" % x)
    return x
