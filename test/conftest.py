
import os
import sys
from pathlib import Path

import numpy as np
import pinocchio as pin
import pytest

# Make project root importable (assumes tests live in manipulator_kinematics/test/ and
# the module file is at manipulator_kinematics/manipulator_kinematics.py)
THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent.parent  # up two levels
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from manipulator_kinematics.manipulator_kinematics import (
    ManipulatorKinematics,
    KinematicsSolverParameters,
    TCP_FRAME_NAME,
)

DEFAULT_URDF = "panda.urdf"


def pytest_addoption(parser):
    parser.addoption(
        "--urdf",
        action="store",
        default=os.environ.get("BH_URDF", DEFAULT_URDF),
        help="Path to the robot URDF to use for tests (env BH_URDF overrides default).",
    )


@pytest.fixture(scope="session")
def urdf_path(pytestconfig):
    p = Path(pytestconfig.getoption("--urdf")).expanduser()
    if not p.exists():
        pytest.skip(f"URDF not found at {p}. Set --urdf or BH_URDF to a valid file.")
    return str(p)


@pytest.fixture(scope="session")
def mk(urdf_path):
    # Use slightly smaller damping for tighter DIK tests, but still stable
    params = KinematicsSolverParameters(damping=1e-8, eps=1e-6, timeStep=0.2, maxIterationsCount=2000)
    return ManipulatorKinematics(urdf_path, parameters=params)


@pytest.fixture()
def data(mk):
    # Provide a fresh Data per test to ensure thread-safety/concurrency readiness
    return pin.Data(mk.getModel())


@pytest.fixture()
def random_q(mk):
    # robust random configuration consistent with limits
    return pin.randomConfiguration(mk.getModel())


@pytest.fixture()
def small_qdot(mk):
    # small random joint velocity to keep linearization good for comparisons
    nv = mk.getModel().nv
    return 0.05 * (2*np.random.rand(nv) - 1.0)

@pytest.fixture()
def zero_q(mk):
    nv = mk.getModel().nv
    return np.zeros(nv)


@pytest.fixture(scope="session")
def tol():
    return {
        "pos": 1e-6,
        "rot": 1e-6,
        "twist": 1e-6,
        "ik_err": 1e-4,
    }