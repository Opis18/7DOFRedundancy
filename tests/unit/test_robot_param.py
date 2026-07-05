"""Unit tests for kinematics/robot_param.py."""
import numpy as np
import pytest

from kinematics.robot_param import get_robot, list_robots, _REGISTRY


class TestRobotConfigs:
    @pytest.mark.parametrize("name", ["generic_7dof", "iiwa7_r800"])
    def test_config_structure(self, name):
        robot = get_robot(name)
        assert robot["n_joints"] == 7
        assert len(robot["S_list"]) == 7
        assert robot["M"].shape == (4, 4)
        assert robot["joint_limits"].shape == (7, 2)

    @pytest.mark.parametrize("name", ["generic_7dof", "iiwa7_r800"])
    def test_joint_limits_ordered(self, name):
        limits = get_robot(name)["joint_limits"]
        assert np.all(limits[:, 0] < limits[:, 1])

    def test_unknown_robot_raises(self):
        with pytest.raises(ValueError):
            get_robot("not_a_real_robot")

    def test_list_robots_includes_both(self, capsys):
        # list_robots() is documented as print-only (returns None); assert
        # against its printed output and the underlying registry instead
        # of a return value it was never designed to produce.
        list_robots()
        captured = capsys.readouterr()
        assert "generic_7dof" in captured.out
        assert "iiwa7_r800" in captured.out
        assert "generic_7dof" in _REGISTRY
        assert "iiwa7_r800" in _REGISTRY

    def test_iiwa7_screw_axes_unit_angular_part(self):
        # Each screw axis's angular part should be a unit vector (revolute joints).
        robot = get_robot("iiwa7_r800")
        for S in robot["S_list"]:
            S = np.asarray(S)
            assert np.isclose(np.linalg.norm(S[:3]), 1.0, atol=1e-8)
