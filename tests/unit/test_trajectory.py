"""Unit tests for simulation/trajectory.py — log-admissibility check and
the screw-motion trajectory generator."""
import numpy as np
import pytest

from kinematics.lie_algebra import exp_map
from simulation.trajectory import check_log_admissible, screw_trajectory


def pose_from_twist(xi):
    return exp_map(xi)


class TestLogAdmissible:
    def test_small_rotation_is_admissible(self):
        g_start = np.eye(4)
        g_end = pose_from_twist(np.array([0.2, 0.0, 0.0, 0.0, 0.0, 0.0]))
        ok, angle = check_log_admissible(g_start, g_end)
        assert ok
        assert np.isclose(angle, 0.2, atol=1e-8)

    def test_near_pi_rotation_is_not_admissible(self):
        g_start = np.eye(4)
        g_end = pose_from_twist(np.array([3.13, 0.0, 0.0, 0.0, 0.0, 0.0]))
        ok, angle = check_log_admissible(g_start, g_end)
        assert not ok

    def test_margin_parameter_tightens_the_bound(self):
        g_start = np.eye(4)
        # angle chosen to sit inside (pi - 0.05) but outside (pi - 0.5)
        g_end = pose_from_twist(np.array([np.pi - 0.2, 0.0, 0.0, 0.0, 0.0, 0.0]))
        ok_loose, _ = check_log_admissible(g_start, g_end, margin=0.05)
        ok_tight, _ = check_log_admissible(g_start, g_end, margin=0.5)
        assert ok_loose and not ok_tight


class TestScrewTrajectory:
    def test_endpoints_match(self):
        g_start = np.eye(4)
        g_end = pose_from_twist(np.array([0.3, 0.1, 0.0, 1.0, 0.0, 0.5]))
        traj = screw_trajectory(g_start, g_end, T=2.0)
        g0, v0 = traj(0.0)
        gT, vT = traj(2.0)
        assert np.allclose(g0, g_start, atol=1e-8)
        assert np.allclose(gT, g_end, atol=1e-6)

    def test_velocity_zero_at_endpoints(self):
        # Quintic time-scaling => zero velocity at tau=0 and tau=1.
        g_start = np.eye(4)
        g_end = pose_from_twist(np.array([0.3, 0.1, 0.0, 1.0, 0.0, 0.5]))
        traj = screw_trajectory(g_start, g_end, T=2.0)
        _, v0 = traj(0.0)
        _, vT = traj(2.0)
        assert np.allclose(v0, 0.0, atol=1e-8)
        assert np.allclose(vT, 0.0, atol=1e-8)

    def test_clamps_outside_domain(self):
        g_start = np.eye(4)
        g_end = pose_from_twist(np.array([0.3, 0.0, 0.0, 0.0, 0.0, 0.0]))
        traj = screw_trajectory(g_start, g_end, T=1.0)
        g_before, v_before = traj(-1.0)
        g_after, v_after = traj(5.0)
        assert np.allclose(g_before, g_start, atol=1e-8)
        assert np.allclose(v_before, 0.0, atol=1e-8)
        assert np.allclose(g_after, g_end, atol=1e-6)
        assert np.allclose(v_after, 0.0, atol=1e-8)

    def test_nonpositive_duration_raises(self):
        with pytest.raises(ValueError):
            screw_trajectory(np.eye(4), np.eye(4), T=0.0)

    def test_unsupported_time_scaling_raises(self):
        with pytest.raises(NotImplementedError):
            screw_trajectory(np.eye(4), np.eye(4), T=1.0, time_scaling="trapezoidal")
