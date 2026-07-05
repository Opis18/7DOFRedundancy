"""
trajectory.py
==============
Task-space reference trajectory generation. Produces a trajectory_func
matching the exact contract run_control_loop (trajectory_tracker.py)
expects:

    trajectory_func(t) -> (g_des, V_ff)
        g_des : (4,4) ndarray -- desired SE(3) end-effector pose at time t
        V_ff  : (6,)  ndarray -- desired feedforward body twist at time t

This file supplies ONE example generator, screw_trajectory: a single
constant-screw-axis segment between two SE(3) poses, with quintic time
scaling for a smooth start/stop. It is intentionally basic -- no
multi-waypoint chaining yet. Chaining several of these segments together
later (or replacing the interpolation scheme entirely) requires no change
to trajectory_tracker.py, only a different trajectory_func passed into
run_control_loop: this file's whole purpose is to be a drop-in replacement
for make_static_setpoint_trajectory, satisfying the same contract.

Why constant-screw-axis interpolation
---------------------------------------
For g(s) = g_start @ exp_map(s * xi_rel), where
xi_rel = log_map_se3(inv(g_start) @ g_end), the body-frame velocity of
the path is EXACTLY constant and equal to xi_rel at every s -- a standard
Lie-group fact (Ad_{exp(s*xi)}(xi) = xi, i.e. a twist commutes with its
own exponential). So the whole segment's feedforward twist is just
xi_rel, scaled by the time-scaling profile's ds/dt. This is the same
log/exp machinery as the paper's own local chart (Section 6), so the
trajectory generator and the controller's error signal share one
consistent piece of math.

Frame note (see trajectory_tracker.py's module docstring): the V_ff
returned here is the body-frame velocity of g_des(t) in g_des(t)'s OWN
frame, not the robot's current body frame T_sb. Those coincide once
tracking error is small -- the usual operating regime -- but not exactly
during large transients. This is the same approximation already flagged
in trajectory_tracker.py, not a new one introduced here.

Log-admissibility (Section 6)
-------------------------------
xi_rel is computed via log_map_se3, which requires the relative rotation
between g_start and g_end to stay under pi. check_log_admissible below is
a lightweight pre-flight check for exactly that, meant to be called once
per segment before handing it to run_control_loop, not per timestep.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from kinematics.lie_algebra import log_map_se3, exp_map


# ──────────────────────────────────────────────────────────────────────────────
# Log-admissibility pre-flight check
# ──────────────────────────────────────────────────────────────────────────────

def check_log_admissible(g_start, g_end, margin=0.1):
    """
    Check whether the relative rotation between g_start and g_end stays
    safely under pi -- the domain boundary of log_map_se3 (Section 6).

    Parameters
    ----------
    g_start, g_end : array_like, shape (4, 4)
        The two SE(3) poses defining a segment.
    margin : float, optional (default 0.1)
        Safety margin subtracted from pi, in radians. The check requires
        the relative rotation angle to stay below (pi - margin), not just
        below pi, since a segment sitting exactly at the boundary is
        numerically fragile even where it is technically admissible.

    Returns
    -------
    is_admissible : bool
    angle : float
        The relative rotation angle (radians), returned regardless of
        pass/fail so the caller can see how much margin remains.
    """
    g_start = np.asarray(g_start, dtype=float)
    g_end = np.asarray(g_end, dtype=float)
    xi_rel = log_map_se3(np.linalg.inv(g_start) @ g_end)
    angle = np.linalg.norm(xi_rel[:3])
    is_admissible = angle < (np.pi - margin)
    return is_admissible, angle


# ──────────────────────────────────────────────────────────────────────────────
# Time scaling
# ──────────────────────────────────────────────────────────────────────────────

def _quintic_time_scaling(tau):
    """
    Standard quintic time-scaling s(tau), tau in [0,1]: s(0)=0, s(1)=1,
    with zero velocity AND zero acceleration at both endpoints (a smooth
    start and stop, no jerk-inducing discontinuity at the boundary).

    Returns (s, ds_dtau).
    """
    s = 10 * tau ** 3 - 15 * tau ** 4 + 6 * tau ** 5
    ds_dtau = 30 * tau ** 2 - 60 * tau ** 3 + 30 * tau ** 4
    return s, ds_dtau


# ──────────────────────────────────────────────────────────────────────────────
# Trajectory generator
# ──────────────────────────────────────────────────────────────────────────────

def screw_trajectory(g_start, g_end, T, time_scaling='quintic'):
    """
    Single-segment constant-screw-axis trajectory from g_start to g_end
    over duration T, with quintic time scaling.

    Parameters
    ----------
    g_start, g_end : array_like, shape (4, 4)
        Start and end SE(3) poses.
    T : float
        Segment duration (seconds). Must be > 0.
    time_scaling : str, optional (default 'quintic')
        Only 'quintic' is implemented for now. Reserved for future
        options (trapezoidal, constant-velocity) without changing the
        returned function's contract.

    Returns
    -------
    trajectory_func : callable
        trajectory_func(t) -> (g_des, V_ff), matching the contract
        required by run_control_loop. For t outside [0, T], the pose and
        velocity are clamped to the segment's endpoints (holds g_end with
        zero feedforward velocity for t > T, holds g_start for t < 0).
    """
    if time_scaling != 'quintic':
        raise NotImplementedError(
            f"time_scaling='{time_scaling}' not implemented -- only "
            f"'quintic' is available for now.")
    if T <= 0:
        raise ValueError(f"T must be positive, got {T}")

    g_start = np.asarray(g_start, dtype=float)
    g_end = np.asarray(g_end, dtype=float)
    xi_rel = log_map_se3(np.linalg.inv(g_start) @ g_end)

    def trajectory_func(t):
        tau = np.clip(t / T, 0.0, 1.0)
        s, ds_dtau = _quintic_time_scaling(tau)
        g_des = g_start @ exp_map(s * xi_rel)
        s_dot = ds_dtau / T
        V_ff = s_dot * xi_rel
        return g_des, V_ff

    return trajectory_func


# ──────────────────────────────────────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────────────────────────────────────

def _run_validation():
    """
    Validation suite for trajectory.py.

    Tests
    -----
    1. Quintic time-scaling boundary conditions: s(0)=0, s(1)=1,
       ds/dtau(0)=0, ds/dtau(1)=0.
    2. screw_trajectory endpoints: g_des(0) == g_start, g_des(T) == g_end.
    3. Feedforward velocity matches a numerical (central-difference)
       estimate of the path's own body-frame velocity, confirming V_ff is
       computed correctly and not just dimensionally plausible.
    4. Feedforward velocity is exactly zero at t=0 and t=T (quintic's
       zero-endpoint-velocity property carrying through correctly).
    5. check_log_admissible passes for a small rotation and fails for a
       near-pi rotation.
    6. Out-of-range t (t<0, t>T) clamps to the segment's endpoints rather
       than extrapolating.
    """
    print("=" * 60)
    print(" trajectory.py — Validation Suite")
    print("=" * 60)

    def check(name, condition):
        print(f"  [{'PASS' if condition else 'FAIL'}] {name}")

    # ── Test 1: quintic boundary conditions ───────────────────────────────────
    s0, sdot0 = _quintic_time_scaling(0.0)
    s1, sdot1 = _quintic_time_scaling(1.0)
    check(f"Quintic time-scaling: s(0)=0, s(1)=1, ds/dtau(0)=0, ds/dtau(1)=0 "
          f"(got s0={s0}, s1={s1}, sdot0={sdot0}, sdot1={sdot1})",
          np.isclose(s0, 0) and np.isclose(s1, 1)
          and np.isclose(sdot0, 0) and np.isclose(sdot1, 0))

    # ── Test setup: an arbitrary segment ──────────────────────────────────────
    rng = np.random.default_rng(0)
    g_start = np.eye(4)
    g_start[:3, 3] = [0.3, -0.1, 0.5]

    xi_disp = np.array([0.4, -0.3, 0.2, 0.1, 0.2, -0.15])
    g_end = g_start @ exp_map(xi_disp)
    T_seg = 2.0
    traj = screw_trajectory(g_start, g_end, T_seg)

    # ── Test 2: endpoints ──────────────────────────────────────────────────────
    g_at_0, _ = traj(0.0)
    g_at_T, _ = traj(T_seg)
    check(f"g_des(0) == g_start (max diff={np.max(np.abs(g_at_0 - g_start)):.2e})",
          np.allclose(g_at_0, g_start, atol=1e-9))
    check(f"g_des(T) == g_end (max diff={np.max(np.abs(g_at_T - g_end)):.2e})",
          np.allclose(g_at_T, g_end, atol=1e-6))

    # ── Test 3: V_ff matches a numerical body-velocity estimate ───────────────
    h = 1e-5
    max_ff_err = 0.0
    for frac in [0.1, 0.3, 0.5, 0.7, 0.9]:
        t_mid = frac * T_seg
        g_minus, _ = traj(t_mid - h)
        g_plus, _ = traj(t_mid + h)
        V_numerical = log_map_se3(np.linalg.inv(g_minus) @ g_plus) / (2 * h)
        _, V_ff_analytical = traj(t_mid)
        err = np.max(np.abs(V_numerical - V_ff_analytical))
        max_ff_err = max(max_ff_err, err)
    check(f"V_ff matches central-difference body velocity along the path "
          f"(max_err={max_ff_err:.2e})", max_ff_err < 1e-4)

    # ── Test 4: V_ff is exactly zero at both endpoints ─────────────────────────
    _, V_ff_0 = traj(0.0)
    _, V_ff_T = traj(T_seg)
    check(f"V_ff(0) == 0 (quintic zero-velocity endpoint) "
          f"(norm={np.linalg.norm(V_ff_0):.2e})", np.allclose(V_ff_0, 0, atol=1e-9))
    check(f"V_ff(T) == 0 (quintic zero-velocity endpoint) "
          f"(norm={np.linalg.norm(V_ff_T):.2e})", np.allclose(V_ff_T, 0, atol=1e-9))

    # ── Test 5: check_log_admissible ──────────────────────────────────────────
    xi_small = np.array([0.2, 0.1, -0.1, 0.05, 0.05, 0.05])
    g_end_small = g_start @ exp_map(xi_small)
    admissible_small, angle_small = check_log_admissible(g_start, g_end_small)
    check(f"check_log_admissible passes for a small rotation "
          f"(angle={angle_small:.4f} rad)", admissible_small)

    xi_huge = np.array([np.pi - 0.02, 0.0, 0.0, 0.0, 0.0, 0.0])
    g_end_huge = g_start @ exp_map(xi_huge)
    admissible_huge, angle_huge = check_log_admissible(g_start, g_end_huge)
    check(f"check_log_admissible fails for a near-pi rotation "
          f"(angle={angle_huge:.4f} rad)", not admissible_huge)

    # ── Test 6: out-of-range t clamps to endpoints ─────────────────────────────
    g_before, V_before = traj(-1.0)
    g_after, V_after = traj(T_seg + 1.0)
    check("t < 0 clamps to g_start with zero feedforward velocity",
          np.allclose(g_before, g_start, atol=1e-9) and np.allclose(V_before, 0, atol=1e-9))
    check("t > T clamps to g_end with zero feedforward velocity",
          np.allclose(g_after, g_end, atol=1e-6) and np.allclose(V_after, 0, atol=1e-9))

    print("\n" + "=" * 60)


if __name__ == "__main__":
    _run_validation()