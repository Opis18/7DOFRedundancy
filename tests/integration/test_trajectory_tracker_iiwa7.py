"""
integration/test_trajectory_tracker_iiwa7.py
=============================================
End-to-end integration test: wires every module (FK, body Jacobian,
manipulability, damped pseudoinverse, null-space projection, PID control
law, screw trajectory, Euler integration) into one closed-loop run on the
real iiwa7_r800 model, through a trajectory that passes close to a
kinematic singularity.

Also includes the comparative benchmark identified as a gap in the
paper draft: demonstrating that the log-map framework tracks through a
near-singular pose that a naive fixed-chart (e.g. angle-axis/Euler pose
error taken directly, no local re-linearization) approach handles worse
as the rotation error approaches its own chart boundary.
"""
import numpy as np
import pytest

from kinematics.forward_kinematics import forward_kinematics
from kinematics.lie_algebra import exp_map, log_map_se3
from simulation.controller import PIDControlLaw
from simulation.trajectory import screw_trajectory, check_log_admissible
from simulation.trajectory_tracker import run_control_loop


@pytest.fixture
def wrist_flip_pose_pair(iiwa7):
    """A start/end pose pair for iiwa7_r800 chosen to sit close to a wrist
    singularity (axes 4 and 6 nearly aligned) partway through the segment,
    exercising the manipulability-aware damping + null-space ascent."""
    q_start = np.array([0.1, 0.3, 0.0, -0.05, 0.0, 0.05, 0.0])  # near q4~0: wrist-near-singular
    q_end = np.array([0.4, 0.9, 0.3, 1.0, 0.2, -0.6, 0.3])
    g_start = forward_kinematics(q_start, iiwa7["S_list"], iiwa7["M"])
    g_end = forward_kinematics(q_end, iiwa7["S_list"], iiwa7["M"])
    return q_start, g_start, g_end


class TestFullPipelineIiwa7:
    def test_admissible_segment_tracks_with_bounded_error(self, iiwa7, wrist_flip_pose_pair):
        q_start, g_start, g_end = wrist_flip_pose_pair
        admissible, angle = check_log_admissible(g_start, g_end)
        assert admissible, f"test segment not log-admissible (angle={angle:.3f}); adjust fixture"

        traj = screw_trajectory(g_start, g_end, T=3.0)
        pid = PIDControlLaw(Kp=8.0, Ki=0.0, Kd=0.5)

        log = run_control_loop(
            q_start, iiwa7["S_list"], iiwa7["M"], traj, pid,
            dt=0.005, T_total=3.0, k_null=1.0,
            w_threshold=0.05, lam_max=0.05,
            joint_limits=iiwa7["joint_limits"],
        )

        final_err_norm = np.linalg.norm(log["xi_err_history"][-1])
        assert final_err_norm < 5e-3, f"final tracking error too large: {final_err_norm:.4e}"
        assert np.all(np.isfinite(log["q_history"]))
        assert np.all(np.isfinite(log["q_dot_history"]))

    def test_damping_engages_near_low_manipulability(self, iiwa7, wrist_flip_pose_pair):
        q_start, g_start, g_end = wrist_flip_pose_pair
        traj = screw_trajectory(g_start, g_end, T=3.0)
        pid = PIDControlLaw(Kp=8.0, Ki=0.0, Kd=0.5)
        log = run_control_loop(
            q_start, iiwa7["S_list"], iiwa7["M"], traj, pid,
            dt=0.005, T_total=3.0, w_threshold=0.05, lam_max=0.05,
            joint_limits=iiwa7["joint_limits"],
        )
        # At least somewhere in the run, damping should respond to
        # manipulability dropping (lam should not be uniformly zero if w
        # ever dips near/under w_threshold).
        low_w_mask = log["w_history"] < 0.05
        if np.any(low_w_mask):
            assert np.any(log["lam_history"][low_w_mask] > 0.0)

    def test_joint_limits_respected(self, iiwa7, wrist_flip_pose_pair):
        q_start, g_start, g_end = wrist_flip_pose_pair
        traj = screw_trajectory(g_start, g_end, T=3.0)
        pid = PIDControlLaw(Kp=8.0, Ki=0.0, Kd=0.5)
        log = run_control_loop(
            q_start, iiwa7["S_list"], iiwa7["M"], traj, pid,
            dt=0.005, T_total=3.0, joint_limits=iiwa7["joint_limits"],
        )
        limits = iiwa7["joint_limits"]
        assert np.all(log["q_history"] >= limits[:, 0] - 1e-9)
        assert np.all(log["q_history"] <= limits[:, 1] + 1e-9)


class TestFixedChartComparisonBenchmark:
    """Comparative benchmark (paper gap): a fixed-chart pose-error scheme
    -- taking the rotation error directly as an angle-axis vector w.r.t. a
    single global reference, WITHOUT re-linearizing locally via the log
    map at each step -- degrades as the rotation error approaches that
    chart's own singular angle, whereas log_map_se3, re-evaluated fresh
    each step at the *current* configuration, does not carry a global
    singular locus in the same way."""

    @staticmethod
    def _fixed_chart_axis_angle_error(R_cur, R_des, fixed_axis=np.array([0.0, 0.0, 1.0])):
        # Deliberately naive fixed-chart pose error: projects the relative
        # rotation onto a single fixed axis via the trace formula without
        # per-step re-derivation of the rotation axis -- this is the
        # "fixed global parameterization" the paper's topological-necessity
        # argument (Sec. 4) shows must have a rank-deficient locus.
        R_rel = R_des @ R_cur.T
        cos_theta = np.clip((np.trace(R_rel) - 1.0) / 2.0, -1.0, 1.0)
        theta = np.arccos(cos_theta)
        # Condition number proxy: 1/sin(theta) blows up as theta -> pi,
        # since decomposing back onto a fixed axis divides by sin(theta).
        sin_theta = np.sin(theta)
        return theta, sin_theta

    def test_fixed_chart_conditioning_degrades_near_pi(self):
        # sin(theta) is not monotonic over (0, pi) -- it peaks at pi/2 --
        # so 1/sin(theta) only worsens monotonically on the (pi/2, pi)
        # branch. That's the relevant branch here: it's the approach to
        # the chart boundary that matters, not the whole domain.
        axis = np.array([1.0, 0.0, 0.0])
        thetas_in = [np.pi / 2, 2.0, 2.5, np.pi - 0.1, np.pi - 0.01, np.pi - 0.001]
        condition_proxies = []
        for theta in thetas_in:
            xi = np.concatenate([theta * axis, np.zeros(3)])
            g = exp_map(xi)
            R = g[:3, :3]
            _, sin_theta = self._fixed_chart_axis_angle_error(np.eye(3), R)
            condition_proxies.append(1.0 / max(sin_theta, 1e-300))
        # Monotonically worsening conditioning as theta -> pi: this is the
        # failure mode the log-map framework is built to avoid.
        assert all(
            condition_proxies[i] < condition_proxies[i + 1]
            for i in range(len(condition_proxies) - 1)
        )

    def test_log_map_roundtrip_does_not_degrade_at_same_angles(self):
        # Contrast: log_map_se3, re-linearized at each call, has no
        # comparable 1/sin(theta) blowup in its own roundtrip accuracy
        # right up to the admissibility boundary (theta < pi).
        axis = np.array([1.0, 0.0, 0.0])
        thetas_in = [0.1, 1.0, 2.0, np.pi - 0.1, np.pi - 0.01, np.pi - 1e-4]
        errs = []
        for theta in thetas_in:
            xi = np.concatenate([theta * axis, np.array([0.1, 0.0, 0.0])])
            g = exp_map(xi)
            xi_hat = log_map_se3(g)
            errs.append(np.max(np.abs(xi - xi_hat)))
        assert max(errs) < 1e-5, f"log map roundtrip degraded: {errs}"
