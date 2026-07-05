"""
trajectory_tracker.py
======================
Main control loop for task-space trajectory tracking on a redundant
manipulator. This module supplies no math of its own -- it wires together
every module already built on the "control" branch into one per-timestep
control law and integrates q(t) forward:

    body_jacobian                          -- kinematics branch, J_b(q)
    yoshikawa_measure, manipulability_gradient   -- manipulability.py
    adaptive_damping, pinv_dls                   -- pseudoinverse.py
    null_space_projector, redundancy_resolution  -- null_space.py

This matches the paper's Section 7 "Simulation and Testing Methodology":
7.1 hierarchical null-space control, 7.2 adaptive damped least squares,
7.3 manipulability-gradient secondary objective. This file is the
apparatus that runs them together; it proves nothing on its own.

Pluggable inputs
----------------
Both the reference trajectory and the feedback control law are supplied
by the caller rather than hard-coded here, following the same
dependency-injection pattern manipulability_gradient already uses for
jacobian_func:

    trajectory_func(t) -> (g_des, V_ff)
        g_des : (4,4) ndarray -- desired SE(3) end-effector pose at time t
        V_ff  : (6,)  ndarray -- desired feedforward body twist at time t

        make_static_setpoint_trajectory() below is a PLACEHOLDER
        satisfying this contract (a fixed pose, zero feedforward). It
        exists only to exercise run_control_loop() while the real
        waypoint / screw-interpolated trajectory generator is developed
        separately -- swapping it in later requires no change to this
        file, only a different trajectory_func argument.

        Note: V_ff is assumed to already be expressed in the same body
        frame that J_b uses -- i.e. it is added directly to the feedback
        term without any adjoint transform. This is exact once tracking
        error is small (the usual operating regime) but is technically
        an approximation while xi_err is large, since a moving reference
        frame's own body-frame velocity is not identical to the current
        robot body frame's until the two poses coincide. If a future
        trajectory generator defines V_ff in its own frame instead, the
        fix is to transform it via Ad(log-map argument) before adding it
        to V_fb -- flagged here rather than silently absorbed, since it's
        exactly the kind of representation subtlety Section 6 cares about.

    control_law_func(xi_err, dt) -> V_fb
        xi_err : (6,) ndarray -- pose error twist in the CURRENT body
                 frame, defined as log_map_se3(inv(T_sb) @ g_des). This
                 is exactly phi_t^{-1}(g_des) from Section 6: the local
                 tangent-space coordinates of the desired pose relative
                 to the current one, evaluated at the current timestep's
                 recentred chart.
        dt     : float -- loop timestep, needed by stateful laws (PID's
                 integral/derivative terms); ignored by memoryless ones.
        V_fb   : (6,) ndarray -- feedback contribution, added to V_ff to
                 form V_cmd.

        proportional_control_law() and PIDControlLaw now live in
        control_law.py, separated out so gains can be tuned and new
        control laws added without touching this file or re-running its
        (much heavier, full-robot) validation suite.

Integration scheme
-------------------
Explicit (forward) Euler: q_next = q + q_dot * dt. This is a first-order
kinematic loop -- q_dot is a velocity command, not an acceleration -- so
Euler's O(dt^2) local truncation error is standard and sufficient; RK4
would quadruple the FK/Jacobian/pseudoinverse work per step for accuracy
this loop doesn't need.

Log-admissibility (Section 6)
-------------------------------
The framework's algorithmic-singularity-free guarantee requires the
rotation angle of inv(T_sb) @ g_des to stay under pi at every timestep.
This holds automatically whenever tracking error is small relative to a
smooth trajectory. run_control_loop optionally warns (verbose=True) if
it is ever violated -- a violation signals excessive tracking error or an
overly aggressive trajectory jump, not a flaw in the control law itself.
"""

import numpy as np
import sys
import os

# ── Allow import from project root or control/ directly ──────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from kinematics.lie_algebra import log_map_se3, exp_map
from kinematics.forward_kinematics import forward_kinematics
from kinematics.body_jacobian import body_jacobian

try:
    from control.manipulability import yoshikawa_measure, manipulability_gradient
    from control.pseudoinverse import pinv_dls, adaptive_damping
    from control.null_space import null_space_projector, redundancy_resolution
    from simulation.controller import proportional_control_law, PIDControlLaw
except ModuleNotFoundError:
    from control.manipulability import yoshikawa_measure, manipulability_gradient
    from control.pseudoinverse import pinv_dls, adaptive_damping
    from control.null_space import null_space_projector, redundancy_resolution
    from simulation.controller import proportional_control_law, PIDControlLaw


# ──────────────────────────────────────────────────────────────────────────────
# Trajectory contract + placeholder (real generator to be supplied later)
# ──────────────────────────────────────────────────────────────────────────────

def make_static_setpoint_trajectory(g_des):
    """
    Placeholder trajectory generator: a single fixed SE(3) setpoint held
    for all time, with zero feedforward velocity.

    Exists ONLY to exercise run_control_loop() in isolation while the real
    task-space trajectory generator (waypoint / screw-interpolated,
    matching this same (t) -> (g_des, V_ff) contract) is developed
    separately. Swapping that generator in later requires no change to
    run_control_loop -- only the trajectory_func argument passed to it.

    Parameters
    ----------
    g_des : array_like, shape (4, 4)
        Fixed desired end-effector pose.

    Returns
    -------
    trajectory_func : callable
        trajectory_func(t) -> (g_des, V_ff), matching the contract
        required by run_control_loop. V_ff is always zeros(6): a static
        setpoint has no reference motion, so all commanded motion comes
        from the feedback term.
    """
    g_des = np.asarray(g_des, dtype=float)

    def trajectory_func(t):
        return g_des, np.zeros(6)

    return trajectory_func


# ──────────────────────────────────────────────────────────────────────────────
# Main control loop
# ──────────────────────────────────────────────────────────────────────────────

def run_control_loop(q0, S_list, M, trajectory_func, control_law_func,
                      dt, T_total, k_null=1.0, w_threshold=0.05, lam_max=0.05,
                      joint_limits=None, verbose=False):
    """
    Integrate joint configuration q(t) forward under closed-loop task-space
    control, wiring every control-branch module into one Euler step per
    timestep.

    Per-timestep pipeline
    ----------------------
    1. T_sb = forward_kinematics(q, S_list, M)
    2. g_des, V_ff = trajectory_func(t)
    3. xi_err = log_map_se3(inv(T_sb) @ g_des)        -- Section 6's phi_t^{-1}(g_des)
    4. V_fb = control_law_func(xi_err, dt)
       V_cmd = V_ff + V_fb
    5. J_b = body_jacobian(q, S_list, M, T_sb=T_sb)    -- T_sb reused, per
                                                            body_jacobian.py's
                                                            own docstring note
    6. w = yoshikawa_measure(J_b)
       lam = adaptive_damping(w, w_threshold, lam_max)
       J_pinv = pinv_dls(J_b, lam)
    7. grad_w = manipulability_gradient(q, S_list, M, body_jacobian)
       N = null_space_projector(J_b, J_pinv)
       q_dot = redundancy_resolution(J_b, V_cmd, k_null * grad_w,
                                      J_pinv=J_pinv, N=N)
    8. q = q + q_dot * dt                               -- explicit Euler
       (optionally clipped to joint_limits)

    Parameters
    ----------
    q0 : array_like, shape (n,)
        Initial joint configuration.
    S_list : list of array_like, each shape (6,)
        Space-frame screw axes (same convention as forward_kinematics /
        body_jacobian throughout the project).
    M : array_like, shape (4, 4)
        Home configuration SE(3) matrix.
    trajectory_func : callable
        trajectory_func(t) -> (g_des, V_ff). See module docstring for the
        full contract. make_static_setpoint_trajectory provides a
        placeholder satisfying it.
    control_law_func : callable
        control_law_func(xi_err, dt) -> V_fb. See module docstring for the
        full contract. proportional_control_law / PIDControlLaw provide
        ready-made implementations satisfying it.
    dt : float
        Integration timestep (seconds).
    T_total : float
        Total simulated duration (seconds). num_steps = ceil(T_total / dt).
    k_null : float, optional (default 1.0)
        Gain on the manipulability-gradient secondary objective. Must stay
        positive: manipulability.py's documented sign convention requires
        +grad(w) (gradient ASCENT) for the null-space term to never
        actively steer toward a singularity -- flipping the sign inverts
        that guarantee, so a negative value triggers a warning.
    w_threshold, lam_max : float, optional
        Passed straight through to adaptive_damping every step (the
        Nakamura-Hanafusa schedule from pseudoinverse.py).
    joint_limits : array_like, shape (n, 2), optional
        [q_min, q_max] per joint. If given, q is clipped after each Euler
        step. None (default) disables clipping.
    verbose : bool, optional
        Print a log-admissibility warning (Section 6) the first time the
        pose error's rotation angle approaches pi.

    Returns
    -------
    log : dict
        q_history       : (num_steps+1, n) ndarray -- joint trajectory, incl. q0
        t_history       : (num_steps+1,) ndarray
        xi_err_history  : (num_steps, 6) ndarray -- pose error twist each step
        q_dot_history   : (num_steps, n) ndarray -- commanded joint velocity
        w_history       : (num_steps,) ndarray -- manipulability each step
        lam_history     : (num_steps,) ndarray -- damping factor each step
        V_cmd_history   : (num_steps, 6) ndarray
        This is the raw material for Section 8's stability/accuracy
        analysis (tracking error over time, manipulability over time,
        damping behavior near kinematic singularities).
    """
    if k_null < 0:
        print("  [WARNING] k_null < 0 inverts manipulability.py's "
              "gradient-ascent sign convention and will actively steer "
              "toward singularities. This is very likely not what you want.")

    q = np.asarray(q0, dtype=float).flatten()
    n = len(q)
    num_steps = int(np.ceil(T_total / dt))

    q_history = np.zeros((num_steps + 1, n))
    t_history = np.zeros(num_steps + 1)
    xi_err_history = np.zeros((num_steps, 6))
    q_dot_history = np.zeros((num_steps, n))
    w_history = np.zeros(num_steps)
    lam_history = np.zeros(num_steps)
    V_cmd_history = np.zeros((num_steps, 6))

    q_history[0] = q
    t_history[0] = 0.0
    warned = False

    for k in range(num_steps):
        t = k * dt

        # ── 1-2. Current pose + reference ─────────────────────────────────
        T_sb = forward_kinematics(q, S_list, M)
        g_des, V_ff = trajectory_func(t)
        g_des = np.asarray(g_des, dtype=float)
        V_ff = np.asarray(V_ff, dtype=float).flatten()

        # ── 3. Pose error in the current body frame (Section 6: phi_t^{-1}) ──
        xi_err = log_map_se3(np.linalg.inv(T_sb) @ g_des)

        if verbose and not warned and np.linalg.norm(xi_err[:3]) >= np.pi - 1e-3:
            print(f"  [WARNING] t={t:.4f}: pose-error rotation angle "
                  f"{np.linalg.norm(xi_err[:3]):.4f} rad is approaching pi -- "
                  f"log-admissibility (Sec. 6) may be violated.")
            warned = True

        # ── 4. Feedback + feedforward ──────────────────────────────────────
        V_fb = np.asarray(control_law_func(xi_err, dt), dtype=float).flatten()
        V_cmd = V_ff + V_fb

        # ── 5. Jacobian (reuse T_sb) ─────────────────────────────────────────
        J_b = body_jacobian(q, S_list, M, T_sb=T_sb)

        # ── 6. Manipulability-aware damped pseudoinverse ─────────────────────
        w = yoshikawa_measure(J_b)
        lam = adaptive_damping(w, w_threshold, lam_max)
        J_pinv = pinv_dls(J_b, lam)

        # ── 7. Null-space secondary objective + redundancy resolution ───────
        grad_w = manipulability_gradient(q, S_list, M, body_jacobian)
        N = null_space_projector(J_b, J_pinv)
        q_dot = redundancy_resolution(J_b, V_cmd, k_null * grad_w,
                                       J_pinv=J_pinv, N=N)

        # ── 8. Explicit Euler integration ────────────────────────────────────
        q = q + q_dot * dt
        if joint_limits is not None:
            joint_limits = np.asarray(joint_limits, dtype=float)
            q = np.clip(q, joint_limits[:, 0], joint_limits[:, 1])

        # ── Log ───────────────────────────────────────────────────────────────
        q_history[k + 1] = q
        t_history[k + 1] = t + dt
        xi_err_history[k] = xi_err
        q_dot_history[k] = q_dot
        w_history[k] = w
        lam_history[k] = lam
        V_cmd_history[k] = V_cmd

    return {
        "q_history": q_history,
        "t_history": t_history,
        "xi_err_history": xi_err_history,
        "q_dot_history": q_dot_history,
        "w_history": w_history,
        "lam_history": lam_history,
        "V_cmd_history": V_cmd_history,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Validation helpers
# ──────────────────────────────────────────────────────────────────────────────

def _load_test_robot():
    """
    Try to load the real iiwa7_r800 model via robot_params/robot_param
    (module name is inconsistent across the project, so both spellings
    are tried, with and without the kinematics. prefix). Falls back to a
    self-contained axis-aligned 7-joint chain if unavailable, so this
    validation suite still runs standalone.

    Returns
    -------
    S_list, M, joint_limits (or None), using_real_robot (bool)
    """
    for module_name in ("kinematics.robot_param", "kinematics.robot_params",
                        "robot_param", "robot_params"):
        try:
            module = __import__(module_name, fromlist=["get_robot"])
            robot = module.get_robot("iiwa7_r800")
            return robot["S_list"], robot["M"], robot.get("joint_limits"), True
        except (ImportError, ModuleNotFoundError, AttributeError, KeyError):
            continue

    # ── Fallback: simple axis-aligned 7-joint chain (alternating z/y axes,
    #    same structural pattern as the real iiwa7_r800) ─────────────────────
    def _screw(w, r):
        w = np.array(w, dtype=float)
        r = np.array(r, dtype=float)
        return np.concatenate([w, -np.cross(w, r)])

    z, y = [0, 0, 1], [0, 1, 0]
    heights = [0.34, 0.34, 0.74, 0.74, 1.14, 1.14, 1.266]
    S_list = [_screw(z if i % 2 == 0 else y, [0, 0, heights[i]]) for i in range(7)]
    M = np.eye(4)
    M[2, 3] = 1.266
    return S_list, M, None, False


# ──────────────────────────────────────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────────────────────────────────────

def _run_validation():
    """
    Validation suite for trajectory_tracker.py.

    Tests
    -----
    1. Convergence: P-controlled static-setpoint tracking drives
       ||xi_err|| to near zero within the simulated duration.
    2. Task-space consistency: at sampled steps, J_b(q) @ q_dot matches
       the commanded V_cmd -- the redundancy_resolution guarantee,
       checked end-to-end through the full loop.
    3. Pluggability: run_control_loop calls trajectory_func and
       control_law_func exactly once per timestep.
    4. PIDControlLaw also converges, and .reset() clears its state.
    5. Joint-limit clipping keeps q within bounds throughout the run.
    6. Null-space ascent direction has non-negative instantaneous dw/dt
       contribution (algebraic guarantee from manipulability.py's
       documented sign convention), plus an informational empirical
       comparison of min(w) over a run with vs. without ascent.
    7. Log-admissibility check condition fires on a deliberately
       near-pi single-step rotation gap.
    8. (Informational) Notes whether the real iiwa7_r800 model or the
       fallback chain was used.
    """
    print("=" * 60)
    print(" trajectory_tracker.py — Validation Suite")
    print("=" * 60)

    def check(name, condition):
        print(f"  [{'PASS' if condition else 'FAIL'}] {name}")

    S_list, M, joint_limits_full, using_real_robot = _load_test_robot()
    n = len(S_list)

    # NOTE: q = 0 turns out to be an exactly singular configuration for the
    # fallback chain (w = 0, rank-deficient J_b) -- several joints share
    # the same effective screw axis at the all-zeros pose. That's a
    # legitimate, useful configuration for stress-testing the damping/
    # ascent machinery (Test 6 uses it on purpose), but it is the wrong
    # starting point for tests that check EXACT, undamped behavior. Those
    # use a well-conditioned configuration instead (w ~ 0.14, found by a
    # quick random search over the fallback chain).
    q0 = np.array([-0.0149, 1.1581, -0.2038, 1.1850, 0.2627, 0.9993, 0.1844])[:n]
    q0_singular = np.zeros(n)
    T_home = forward_kinematics(q0, S_list, M)

    # ── Test 1: convergence with a static setpoint + P control ───────────────
    xi_target = np.array([0.1, -0.05, 0.05, 0.05, 0.05, -0.05])
    target_pose = T_home @ exp_map(xi_target)
    trajectory_func = make_static_setpoint_trajectory(target_pose)
    control_law = proportional_control_law(Kp=2.0)

    log1 = run_control_loop(q0, S_list, M, trajectory_func, control_law,
                             dt=0.01, T_total=5.0, k_null=0.0)
    final_err_norm = np.linalg.norm(log1["xi_err_history"][-1])
    check(f"Convergence: P control on static setpoint drives ||xi_err|| -> ~0 "
          f"(final={final_err_norm:.2e})", final_err_norm < 0.01)

    # ── Test 2: task-space consistency (J_b @ q_dot == V_cmd) ────────────────
    max_consistency_err = 0.0
    for k in range(0, log1["q_dot_history"].shape[0], 20):
        J_b_k = body_jacobian(log1["q_history"][k], S_list, M)
        err_k = np.max(np.abs(J_b_k @ log1["q_dot_history"][k]
                               - log1["V_cmd_history"][k]))
        max_consistency_err = max(max_consistency_err, err_k)
    check(f"Task-space consistency: J_b @ q_dot == V_cmd along trajectory "
          f"(max_err={max_consistency_err:.2e})", max_consistency_err < 1e-6)

    # ── Test 3: pluggability -- both callables invoked exactly once/step ─────
    traj_calls = [0]
    ctrl_calls = [0]

    def counting_trajectory(t):
        traj_calls[0] += 1
        return trajectory_func(t)

    def counting_control_law(xi_err, dt):
        ctrl_calls[0] += 1
        return control_law(xi_err, dt)

    num_steps_expected = int(np.ceil(2.0 / 0.01))
    run_control_loop(q0, S_list, M, counting_trajectory, counting_control_law,
                      dt=0.01, T_total=2.0, k_null=0.0)
    check(f"Pluggability: trajectory_func called once per step "
          f"({traj_calls[0]} == {num_steps_expected})",
          traj_calls[0] == num_steps_expected)
    check(f"Pluggability: control_law_func called once per step "
          f"({ctrl_calls[0]} == {num_steps_expected})",
          ctrl_calls[0] == num_steps_expected)

    # ── Test 4: PIDControlLaw converges; .reset() clears state ───────────────
    pid = PIDControlLaw(Kp=2.0, Ki=0.5, Kd=0.05)
    log_pid = run_control_loop(q0, S_list, M, trajectory_func, pid,
                                dt=0.01, T_total=5.0, k_null=0.0)
    final_err_pid = np.linalg.norm(log_pid["xi_err_history"][-1])
    check(f"PIDControlLaw also converges (final ||xi_err||={final_err_pid:.2e})",
          final_err_pid < 0.01)

    pid.reset()
    check("PIDControlLaw.reset() clears integral and derivative state",
          np.allclose(pid._integral, 0.0) and pid._prev_error is None)

    # ── Test 5: joint-limit clipping keeps q within bounds ────────────────────
    tight_limits = np.stack([q0 - 0.02, q0 + 0.02], axis=1)
    log_clipped = run_control_loop(q0, S_list, M, trajectory_func, control_law,
                                    dt=0.01, T_total=2.0, k_null=0.0,
                                    joint_limits=tight_limits)
    within_bounds = (np.all(log_clipped["q_history"] >= tight_limits[:, 0] - 1e-9)
                      and np.all(log_clipped["q_history"] <= tight_limits[:, 1] + 1e-9))
    check("Joint-limit clipping keeps q within bounds throughout the run",
          within_bounds)

    # ── Test 6: null-space ascent -- algebraic guarantee + empirical check ───
    q_sample = np.array([0.3, -0.5, 0.8, -0.2, 1.1, -0.7, 0.4])[:n]
    J_sample = body_jacobian(q_sample, S_list, M)
    grad_w_sample = manipulability_gradient(q_sample, S_list, M, body_jacobian)
    N_sample = null_space_projector(J_sample, pinv_dls(J_sample, lam=0.0))
    dw_dt_contribution = grad_w_sample @ N_sample @ (1.0 * grad_w_sample)
    check(f"Null-space ascent direction's dw/dt contribution is non-negative "
          f"({dw_dt_contribution:.4e} >= 0)", dw_dt_contribution >= -1e-9)

    xi_target_2 = np.array([0.3, 0.2, -0.2, 0.2, 0.2, -0.2])
    T_home_singular = forward_kinematics(q0_singular, S_list, M)
    target_pose_2 = T_home_singular @ exp_map(xi_target_2)
    trajectory_func_2 = make_static_setpoint_trajectory(target_pose_2)

    log_no_ascent = run_control_loop(q0_singular, S_list, M, trajectory_func_2,
                                      proportional_control_law(Kp=2.0),
                                      dt=0.01, T_total=5.0, k_null=0.0)
    log_with_ascent = run_control_loop(q0_singular, S_list, M, trajectory_func_2,
                                        proportional_control_law(Kp=2.0),
                                        dt=0.01, T_total=5.0, k_null=1.0)
    w_min_no_ascent = np.min(log_no_ascent["w_history"])
    w_min_with_ascent = np.min(log_with_ascent["w_history"])
    # Both runs start at the same exactly-singular q0, so the trajectory-
    # wide minimum is trivially ~0 for both regardless of ascent -- compare
    # the second-half average instead, which actually reflects whether the
    # ascent term helped the trajectory move to (and stay in) a better-
    # conditioned region.
    halfway = len(log_no_ascent["w_history"]) // 2
    w_avg_2nd_half_no_ascent = np.mean(log_no_ascent["w_history"][halfway:])
    w_avg_2nd_half_with_ascent = np.mean(log_with_ascent["w_history"][halfway:])
    print(f"  [INFO] min(w) over full run (both start at the same singular "
          f"q0, so trivially ~equal): no-ascent={w_min_no_ascent:.4f}, "
          f"with-ascent={w_min_with_ascent:.4f}")
    print(f"  [INFO] mean(w) over 2nd half of run (reflects whether ascent "
          f"helped): no-ascent={w_avg_2nd_half_no_ascent:.4f}, "
          f"with-ascent={w_avg_2nd_half_with_ascent:.4f}")

    # ── Test 7: log-admissibility condition fires on a near-pi rotation gap ──
    xi_huge = np.array([np.pi - 5e-4, 0.0, 0.0, 0.0, 0.0, 0.0])
    T_sb_test = forward_kinematics(q0, S_list, M)
    g_huge = T_sb_test @ exp_map(xi_huge)
    xi_err_huge = log_map_se3(np.linalg.inv(T_sb_test) @ g_huge)
    triggers_warning = np.linalg.norm(xi_err_huge[:3]) >= np.pi - 1e-3
    check(f"Log-admissibility check condition fires on a near-pi rotation gap "
          f"(||omega||={np.linalg.norm(xi_err_huge[:3]):.4f})", triggers_warning)

    # ── Test 8 (informational): which robot model was used ───────────────────
    if using_real_robot:
        print("  [INFO] Validation ran on the real iiwa7_r800 model "
              "(robot_params/robot_param import succeeded).")
    else:
        print("  [INFO] robot_params/robot_param not found on path -- "
              "validation ran on a self-contained fallback 7-joint chain.")

    # ── Test 9 (optional): tracks a genuinely moving reference ────────────────
    # Everything above uses make_static_setpoint_trajectory, which never
    # exercises V_ff as anything but zero. If trajectory.py's real
    # screw_trajectory is available, use it here to confirm feedforward +
    # feedback combine correctly against an actually-moving reference.
    try:
        from simulation.trajectory import screw_trajectory
    except ModuleNotFoundError:
        try:
            from trajectory import screw_trajectory
        except ModuleNotFoundError:
            screw_trajectory = None

    if screw_trajectory is not None:
        xi_move = np.array([0.2, 0.1, -0.1, 0.15, -0.1, 0.1])
        g_move_end = T_home @ exp_map(xi_move)
        moving_traj = screw_trajectory(T_home, g_move_end, T=3.0)
        log_moving = run_control_loop(q0, S_list, M, moving_traj,
                                       proportional_control_law(Kp=2.0),
                                       dt=0.005, T_total=3.0, k_null=0.0)
        final_err_moving = np.linalg.norm(log_moving["xi_err_history"][-1])
        check(f"Tracks a genuinely moving (screw-interpolated) reference, "
              f"not just a static setpoint "
              f"(final ||xi_err||={final_err_moving:.2e})",
              final_err_moving < 0.05)
    else:
        print("  [INFO] trajectory.py not found on path -- skipping the "
              "moving-reference tracking test.")

    print("\n" + "=" * 60)


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    _run_validation()