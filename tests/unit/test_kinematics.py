"""Unit tests for kinematics/forward_kinematics.py and body_jacobian.py."""
import numpy as np
import pytest

from kinematics.forward_kinematics import forward_kinematics
from kinematics.body_jacobian import body_jacobian, body_jacobian_numerical


class TestForwardKinematics:
    def test_home_configuration(self, generic_robot):
        q0 = np.zeros(generic_robot["n_joints"])
        T = forward_kinematics(q0, generic_robot["S_list"], generic_robot["M"])
        assert np.allclose(T, generic_robot["M"], atol=1e-10)

    def test_output_is_valid_se3(self, generic_robot, rng):
        q = rng.uniform(-1, 1, size=generic_robot["n_joints"])
        T = forward_kinematics(q, generic_robot["S_list"], generic_robot["M"])
        R = T[:3, :3]
        assert np.allclose(R.T @ R, np.eye(3), atol=1e-8)
        assert np.isclose(np.linalg.det(R), 1.0, atol=1e-8)
        assert np.allclose(T[3, :], [0, 0, 0, 1])

    def test_iiwa7_home_configuration(self, iiwa7):
        q0 = np.zeros(iiwa7["n_joints"])
        T = forward_kinematics(q0, iiwa7["S_list"], iiwa7["M"])
        assert np.allclose(T, iiwa7["M"], atol=1e-10)


class TestBodyJacobian:
    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_matches_numerical_jacobian(self, generic_robot, seed):
        rng = np.random.default_rng(seed)
        q = rng.uniform(-1, 1, size=generic_robot["n_joints"])
        J_analytical = body_jacobian(q, generic_robot["S_list"], generic_robot["M"])
        J_numerical = body_jacobian_numerical(q, generic_robot["S_list"], generic_robot["M"])
        assert np.allclose(J_analytical, J_numerical, atol=1e-5)

    def test_shape(self, iiwa7, rng):
        q = rng.uniform(-1, 1, size=iiwa7["n_joints"])
        J = body_jacobian(q, iiwa7["S_list"], iiwa7["M"])
        assert J.shape == (6, iiwa7["n_joints"])

    def test_iiwa7_matches_numerical(self, iiwa7, rng):
        q = rng.uniform(-1, 1, size=iiwa7["n_joints"])
        J_analytical = body_jacobian(q, iiwa7["S_list"], iiwa7["M"])
        J_numerical = body_jacobian_numerical(q, iiwa7["S_list"], iiwa7["M"])
        assert np.allclose(J_analytical, J_numerical, atol=1e-5)
