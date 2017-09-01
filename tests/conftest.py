"""Fixtures for testing."""

import pytest
import numpy as np

SEED = 10


@pytest.fixture
def random_X_data():
    return {
        "bin": create_random_binary_X(),
        "cont": create_random_continuous_X()}


def create_random_X(data_fixture):
    return np.concatenate([
        data_fixture["bin"], data_fixture["cont"],
        data_fixture["bin"], data_fixture["cont"]], axis=1)


def create_random_binary_X():
    np.random.seed(SEED)
    return np.random.randint(0, 2, (10, 3))


def create_random_continuous_X():
    np.random.seed(SEED)
    return np.random.randint(0, 100, (10, 3))


def create_linear_X():
    return np.array([range(10), range(11, 21)]).T


def create_y():
    return np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])


def create_s():
    return np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
