"""
controller.py
================
Pluggable feedback control laws for task-space trajectory tracking.
Separated out of trajectory_tracker.py so gains can be tuned, and new
control laws added, without touching the control loop itself or
re-running its (much heavier, full-robot) validation suite.

Contract
--------
    control_law_func(xi_err, dt) -> V_fb

    xi_err : (6,) ndarray -- pose error twist in the current body frame,
             as computed by trajectory_tracker.py: xi_err =
             log_map_se3(inv(T_sb) @ g_des). This is exactly phi_t^{-1}
             (g_des) from Section 6: the local tangent-space coordinates
             of the desired pose relative to the current one.
    dt     : float -- control loop timestep, needed by stateful laws
             (PID's integral/derivative terms). Ignored by memoryless
             ones (e.g. plain P).
    V_fb   : (6,) ndarray -- feedback contribution. trajectory_tracker.py
             adds this to the trajectory's feedforward twist V_ff to form
             V_cmd.

Two implementations are provided here: proportional_control_law (a
stateless factory) and PIDControlLaw (a stateful class -- one instance
must be reused across every timestep of a single run_control_loop call,
since it carries the integral accumulator and previous error between
calls). Anything else satisfying the same (xi_err, dt) -> V_fb contract
can be swapped in without changing trajectory_tracker.py at all.
"""

import numpy as np


def _to_gain_matrix(K):
    """Broadcast a scalar or (6,) vector gain onto a diagonal (6,6) matrix;
    pass a full (6,6) matrix through unchanged (allows decoupled
    angular/linear gains -- e.g. different gains on the omega block vs.
    the v block, since these have very different natural units/scales)."""
    K = np.asarray(K, dtype=float)
    if K.ndim == 0:
        return K * np.eye(6)
    elif K.ndim == 1:
        return np.diag(K)
    return K


def proportional_control_law(Kp):
    """
    Stateless proportional feedback: V_fb = Kp @ xi_err.

    Parameters
    ----------
    Kp : array_like, shape (6, 6) or (6,) or float
        Proportional gain (see _to_gain_matrix for broadcasting rules).

    Returns
    -------
    control_law_func : callable
        control_law_func(xi_err, dt) -> V_fb. dt is accepted for
        interface consistency but unused by a memoryless P law.
    """
    Kp = _to_gain_matrix(Kp)

    def control_law_func(xi_err, dt):
        return Kp @ np.asarray(xi_err, dtype=float)

    return control_law_func


class PIDControlLaw:
    """
    Stateful PID feedback on the body-frame pose-error twist:

        V_fb = Kp @ xi_err + Ki @ integral(xi_err dt) + Kd @ d(xi_err)/dt

    Carries the integral accumulator and previous error between calls, so
    -- unlike proportional_control_law -- one instance (not just the
    class) must be reused across every timestep of a single
    run_control_loop call.

    Parameters
    ----------
    Kp, Ki, Kd : array_like, shape (6, 6) or (6,) or float
        Gains (see _to_gain_matrix for broadcasting rules).
    integral_limit : float, optional
        Element-wise clamp on the integral accumulator (anti-windup).
        None (default) disables clamping.
    """

    def __init__(self, Kp, Ki, Kd, integral_limit=None):
        self.Kp = _to_gain_matrix(Kp)
        self.Ki = _to_gain_matrix(Ki)
        self.Kd = _to_gain_matrix(Kd)
        self.integral_limit = integral_limit
        self._integral = np.zeros(6)
        self._prev_error = None

    def reset(self):
        """Clear the accumulated integral and derivative history."""
        self._integral = np.zeros(6)
        self._prev_error = None

    def __call__(self, xi_err, dt):
        xi_err = np.asarray(xi_err, dtype=float)

        self._integral = self._integral + xi_err * dt
        if self.integral_limit is not None:
            self._integral = np.clip(self._integral, -self.integral_limit,
                                      self.integral_limit)

        if self._prev_error is None:
            derivative = np.zeros(6)
        else:
            derivative = (xi_err - self._prev_error) / dt
        self._prev_error = xi_err

        return self.Kp @ xi_err + self.Ki @ self._integral + self.Kd @ derivative


# ──────────────────────────────────────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────────────────────────────────────

def _run_validation():
    """
    Validation suite for control_law.py. Fully self-contained -- no robot,
    Jacobian, or FK needed, since these are pure functions/classes of
    (xi_err, dt).

    Tests
    -----
    1. proportional_control_law with a scalar gain broadcasts correctly.
    2. proportional_control_law with a (6,) vector gain broadcasts onto a
       diagonal matrix (decoupled per-axis gains).
    3. proportional_control_law with a full (6,6) matrix passes through
       unchanged.
    4. PIDControlLaw's first call has zero derivative (no previous error
       to difference against yet).
    5. PIDControlLaw's integral accumulator matches a manually-computed
       running sum across several calls.
    6. PIDControlLaw's derivative term matches (current-previous)/dt on
       the second call.
    7. PIDControlLaw's integral_limit clamps the accumulator correctly
       (anti-windup).
    8. PIDControlLaw.reset() clears both integral and previous-error state.
    9. PIDControlLaw with Ki=0, Kd=0 reduces exactly to
       proportional_control_law with the same Kp -- confirms the two
       implementations are consistent with each other, not just each
       individually plausible.
    """
    print("=" * 60)
    print(" control_law.py — Validation Suite")
    print("=" * 60)

    def check(name, condition):
        print(f"  [{'PASS' if condition else 'FAIL'}] {name}")

    xi_err = np.array([0.1, -0.2, 0.05, 0.3, -0.1, 0.2])
    dt = 0.01

    # ── Test 1: scalar gain ───────────────────────────────────────────────────
    ctrl = proportional_control_law(2.0)
    V_fb = ctrl(xi_err, dt)
    check(f"proportional_control_law: scalar Kp broadcasts correctly "
          f"(max diff={np.max(np.abs(V_fb - 2.0 * xi_err)):.2e})",
          np.allclose(V_fb, 2.0 * xi_err))

    # ── Test 2: vector gain ───────────────────────────────────────────────────
    Kp_vec = np.array([1.0, 2.0, 3.0, 0.5, 0.5, 0.5])
    ctrl_vec = proportional_control_law(Kp_vec)
    V_fb_vec = ctrl_vec(xi_err, dt)
    expected_vec = Kp_vec * xi_err
    check(f"proportional_control_law: (6,) vector Kp broadcasts onto a "
          f"diagonal matrix (max diff={np.max(np.abs(V_fb_vec - expected_vec)):.2e})",
          np.allclose(V_fb_vec, expected_vec))

    # ── Test 3: full matrix gain passes through unchanged ─────────────────────
    Kp_mat = np.diag([1, 2, 3, 4, 5, 6]).astype(float)
    Kp_mat[0, 1] = 0.3  # off-diagonal coupling term
    ctrl_mat = proportional_control_law(Kp_mat)
    V_fb_mat = ctrl_mat(xi_err, dt)
    check(f"proportional_control_law: full (6,6) matrix Kp passes through "
          f"unchanged, including off-diagonal terms "
          f"(max diff={np.max(np.abs(V_fb_mat - Kp_mat @ xi_err)):.2e})",
          np.allclose(V_fb_mat, Kp_mat @ xi_err))

    # ── Test 4: PID first call has zero derivative ────────────────────────────
    pid = PIDControlLaw(Kp=1.0, Ki=0.5, Kd=0.1)
    V_fb_1 = pid(xi_err, dt)
    expected_1 = 1.0 * xi_err + 0.5 * (xi_err * dt)  # integral after 1 step, deriv=0
    check(f"PIDControlLaw: first call has zero derivative contribution "
          f"(max diff={np.max(np.abs(V_fb_1 - expected_1)):.2e})",
          np.allclose(V_fb_1, expected_1))

    # ── Test 5: integral accumulates as a running sum ─────────────────────────
    pid2 = PIDControlLaw(Kp=0.0, Ki=1.0, Kd=0.0)
    errors = [np.array([0.1] * 6), np.array([0.2] * 6), np.array([0.15] * 6)]
    running_integral = np.zeros(6)
    max_integral_err = 0.0
    for e in errors:
        V_fb_step = pid2(e, dt)
        running_integral = running_integral + e * dt
        max_integral_err = max(max_integral_err,
                                np.max(np.abs(V_fb_step - running_integral)))
    check(f"PIDControlLaw: integral term matches a manually-computed "
          f"running sum across calls (max_err={max_integral_err:.2e})",
          max_integral_err < 1e-12)

    # ── Test 6: derivative term on the second call ────────────────────────────
    pid3 = PIDControlLaw(Kp=0.0, Ki=0.0, Kd=1.0)
    e1 = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
    e2 = np.array([0.13, 0.0, 0.0, 0.0, 0.0, 0.0])
    pid3(e1, dt)  # first call, derivative=0, just sets _prev_error
    V_fb_deriv = pid3(e2, dt)
    expected_deriv = (e2 - e1) / dt
    check(f"PIDControlLaw: derivative term == (current-previous)/dt on the "
          f"second call (max diff={np.max(np.abs(V_fb_deriv - expected_deriv)):.2e})",
          np.allclose(V_fb_deriv, expected_deriv))

    # ── Test 7: integral_limit clamps (anti-windup) ────────────────────────────
    pid4 = PIDControlLaw(Kp=0.0, Ki=1.0, Kd=0.0, integral_limit=0.05)
    for _ in range(20):  # would accumulate to 0.1*20*0.01=huge without clamping
        pid4(np.array([1.0] * 6), dt)
    check(f"PIDControlLaw: integral_limit clamps the accumulator "
          f"(max |integral|={np.max(np.abs(pid4._integral)):.4f} <= 0.05)",
          np.all(np.abs(pid4._integral) <= 0.05 + 1e-12))

    # ── Test 8: reset() clears state ───────────────────────────────────────────
    pid4.reset()
    check("PIDControlLaw.reset() clears integral and derivative state",
          np.allclose(pid4._integral, 0.0) and pid4._prev_error is None)

    # ── Test 9: PID with Ki=Kd=0 matches proportional_control_law exactly ────
    pid5 = PIDControlLaw(Kp=2.0, Ki=0.0, Kd=0.0)
    ctrl_p_equiv = proportional_control_law(2.0)
    max_consistency_err = 0.0
    for e in [np.array([0.1] * 6), np.array([-0.2, 0.3, 0.1, 0.0, 0.05, -0.1])]:
        v_pid = pid5(e, dt)
        v_p = ctrl_p_equiv(e, dt)
        max_consistency_err = max(max_consistency_err, np.max(np.abs(v_pid - v_p)))
    check(f"PIDControlLaw(Ki=0, Kd=0) matches proportional_control_law "
          f"with the same Kp exactly (max_err={max_consistency_err:.2e})",
          max_consistency_err < 1e-12)

    print("\n" + "=" * 60)


if __name__ == "__main__":
    _run_validation()