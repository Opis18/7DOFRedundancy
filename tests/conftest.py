"""
conftest.py
===========
Shared fixtures for the 7DOFRedundancy test suite.

Adds src/ to sys.path so tests can `import kinematics.lie_algebra` etc.
without installing the package, and centralizes the two robot configs
(generic_7dof, iiwa7_r800) so every test file draws from the same source
of truth instead of re-deriving robot parameters.
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(ROOT, "..", "src"))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from kinematics.robot_param import get_robot  # noqa: E402


@pytest.fixture(scope="session")
def generic_robot():
    return get_robot("generic_7dof")


@pytest.fixture(scope="session")
def iiwa7(): 
    return get_robot("iiwa7_r800")


@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng(42)
