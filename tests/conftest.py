import numpy as np
import pytest

from soccer_tracking.config import Config


@pytest.fixture
def cfg():
    return Config()


@pytest.fixture
def rng():
    return np.random.default_rng(0)
