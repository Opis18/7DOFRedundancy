"""
null_space.py
=============
Null-space projection and redundancy resolution for the 7-DOF manipulator,
via the standard gradient-projection method (Liegeois 1977; Nakamura 1991;
Siciliano & Chiaverini's redundancy-resolution formulation).

    xi_dot (task twist) ──► J^+ ──► J^+ @ xi_dot
                                          │
    q_dot_secondary ──► N = (I - J^+ J) ──► N @ q_dot_secondary
                                          │
                                          ▼
                    q_dot = J^+ @ xi_dot + N @ q_dot_secondary

For a redundant manipulator (n=7 joints, m=6 task-space DOF), null(J) is
generically 1-dimensional. Any joint velocity projected through N produces
zero end-effector twist, so a secondary objective can be pursued "for
free" alongside the primary task. In this project the secondary
objective is manipulability ascent (see manipulability.py):

    q_dot_secondary = +k * manipulability_gradient(q, ...),  k > 0

which — per manipulability.py's documented sign convention — makes the
null-space term's contribution to dw/dt provably non-negative when N is
the exact (undamped) projector.

Pseudoinverse convention (resolved — standard/simple option)
--------------------------------------------------------------
This module uses a single J_pinv for both the primary task term and the
null-space projector, rather than maintaining separate damped/undamped
pseudoinverses. This is the standard formulation found throughout the
redundancy-resolution literature. The one thing worth knowing: N is an
*exact* orthogonal projector (idempotent, J @ N = 0 exactly) only when
J_pinv is the undamped Moore-Penrose pseudoinverse. N stays symmetric
regardless of damping (N = I - J^T A^-1 J is symmetric whenever
A = J J^T + lam^2 I is, which it always is), but idempotency and the
exact null-space property degrade as lam grows: N @ N != N and
J @ N != 0, both by an amount that scales with lam. If a damped J_pinv
(pseudoinverse.pinv_dls with lam > 0) is passed in near a singularity,
some task-space leakage from the secondary term is expected in that
regime. This is a standard, accepted tradeoff (the same regime where
damping is needed for stability in the first place) and is validated
explicitly below rather than engineered around.

Scope note
----------
A joint-limit-avoidance secondary objective (gradient of a joint-limit
potential) was considered as a second candidate for q_dot_secondary, but
is not implemented here — resolved as out of scope for the current
paper, which uses manipulability ascent as its sole secondary objective.

Depends on
----------
- numpy
- No hard dependency on pseudoinverse.py or manipulability.py: both
  functions below accept J_pinv as a plain argument (defaulting to
  np.linalg.pinv(J) if omitted), keeping this module decoupled and
  testable in isolation, matching the pattern used throughout the
  project.

Functions
---------
null_space_projector   : N = I - J^+ J
redundancy_resolution  : q_dot = J^+ @ xi_dot + N @ q_dot_secondary
"""

import numpy as np
import sys
import os


# ──────────────────────────────────────────────────────────────────────────────
# 1. null_space_projector — N = I - J^+ J
# ──────────────────────────────────────────────────────────────────────────────

def null_space_projector(J, J_pinv=None):
    """
    Compute the null-space projection matrix N = I - J^+ J.

    Any vector v satisfies J @ (N @ v) == 0 (exactly, for the undamped
    Moore-Penrose J^+; approximately for a damped J^+, which also makes
    N only approximately idempotent — see module docstring), i.e. N @ v
    lies in the null space of J and produces no task-space motion when
    used as a joint-velocity command.

    Parameters
    ----------
    J : array_like, shape (m, n)
        Jacobian (typically J_b, the 6x7 body Jacobian).
    J_pinv : array_like, shape (n, m), optional
        Pseudoinverse of J. If None, computed via np.linalg.pinv(J)
        (the standard undamped Moore-Penrose pseudoinverse). Callers
        using an adaptively-damped pseudoinverse (pseudoinverse.pinv_dls
        with lam > 0) should pass it in directly.

    Returns
    -------
    N : np.ndarray, shape (n, n)
        Null-space projector, N = I_n - J_pinv @ J.
    """
    J = np.asarray(J, dtype=float)
    n = J.shape[1]

    if J_pinv is None:
        J_pinv = np.linalg.pinv(J)
    else:
        J_pinv = np.asarray(J_pinv, dtype=float)

    return np.eye(n) - J_pinv @ J


# ──────────────────────────────────────────────────────────────────────────────
# 2. redundancy_resolution — standard gradient-projection formula
# ──────────────────────────────────────────────────────────────────────────────

def redundancy_resolution(J, xi_dot, q_dot_secondary, J_pinv=None, N=None):
    """
    Combine a primary task-space objective with a secondary null-space
    objective into a single joint-velocity command, via the standard
    gradient-projection method:

        q_dot = J^+ @ xi_dot + N @ q_dot_secondary

    Parameters
    ----------
    J : array_like, shape (6, n)
        Body Jacobian at the current configuration.
    xi_dot : array_like, shape (6,)
        Desired body-frame task twist (primary objective).
    q_dot_secondary : array_like, shape (n,)
        Candidate joint-velocity vector encoding the secondary objective.
        In this project: +k * manipulability_gradient(q, ...), k > 0
        (gradient ascent on manipulability — see manipulability.py's
        sign-convention note). Only its projection onto null(J) survives.
    J_pinv : array_like, shape (n, 6), optional
        Precomputed pseudoinverse of J. Computed via np.linalg.pinv(J)
        if not provided. The same J_pinv is used for both the primary
        term and (if N is not separately provided) the null-space
        projector — see the module docstring's pseudoinverse convention.
    N : array_like, shape (n, n), optional
        Precomputed null-space projector. Computed internally via
        null_space_projector(J, J_pinv) if not provided.

    Returns
    -------
    q_dot : np.ndarray, shape (n,)
        Combined joint-velocity command.
    """
    J = np.asarray(J, dtype=float)
    xi_dot = np.asarray(xi_dot, dtype=float).flatten()
    q_dot_secondary = np.asarray(q_dot_secondary, dtype=float).flatten()

    if J_pinv is None:
        J_pinv = np.linalg.pinv(J)
    else:
        J_pinv = np.asarray(J_pinv, dtype=float)

    if N is None:
        N = null_space_projector(J, J_pinv)
    else:
        N = np.asarray(N, dtype=float)

    return J_pinv @ xi_dot + N @ q_dot_secondary


# ──────────────────────────────────────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────────────────────────────────────

def _planar_3r_jacobian(q, l1=1.0, l2=1.0, l3=1.0):
    """
    Closed-form Jacobian of a 3-link planar RRR arm reaching a 2D target
    (validation helper only) — the standard textbook example of a
    redundant manipulator: n=3 joints, m=2 task dims, null(J) generically
    1-dimensional, exactly mirroring this project's n=7, m=6 structure at
    minimal scale.

        x = l1*cos(t1) + l2*cos(t1+t2) + l3*cos(t1+t2+t3)
        y = l1*sin(t1) + l2*sin(t1+t2) + l3*sin(t1+t2+t3)
    """
    q = np.asarray(q, dtype=float).flatten()
    t1, t2, t3 = q
    s1, c1 = np.sin(t1), np.cos(t1)
    s12, c12 = np.sin(t1 + t2), np.cos(t1 + t2)
    s123, c123 = np.sin(t1 + t2 + t3), np.cos(t1 + t2 + t3)

    return np.array([
        [-l1 * s1 - l2 * s12 - l3 * s123, -l2 * s12 - l3 * s123, -l3 * s123],
        [ l1 * c1 + l2 * c12 + l3 * c123,  l2 * c12 + l3 * c123,  l3 * c123],
    ])


def _run_validation():
    """
    Validation suite for null_space.py.

    Tests
    -----
    1. J @ N ≈ 0 (defining property) for a random well-conditioned 6x7 J.
    2. N is idempotent: N @ N ≈ N.
    3. N is symmetric: N ≈ N.T (holds for the undamped/Moore-Penrose case).
    4. Rank check: trace(N) ≈ 1 for a 6x7 full-row-rank J (redundancy = 1).
    5. redundancy_resolution reduces to J_pinv @ xi_dot when
       q_dot_secondary = 0.
    6. Task-space consistency: J @ redundancy_resolution(...) ≈ xi_dot
       regardless of q_dot_secondary (secondary term contributes zero
       task-space motion by construction).
    7. Hand-verifiable sanity check on a 3-link planar arm (n=3, m=2,
       null-space dimension 1) — same structural checks as 1-4 at a
       scale simple enough to reason about directly.
    8. Cross-check against pseudoinverse.pinv_dls (this same "control"
       branch): null_space_projector using pinv_dls(J, lam=0) matches
       using the default np.linalg.pinv(J); and using a damped J_pinv
       (lam > 0) is shown to only approximately satisfy J @ N ≈ 0,
       confirming the documented pseudoinverse-convention caveat.
    9. Full local pipeline: manipulability.manipulability_gradient as
       the secondary objective, verifying task-space consistency still
       holds exactly (undamped case).
    10. (Optional, skipped if unavailable) Full end-to-end cross-check
        on the real iiwa7_r800 model via robot_params + body_jacobian.
    """
    print("=" * 60)
    print(" null_space.py — Validation Suite")
    print("=" * 60)

    tol = 1e-8

    def check(name, result, expected, tol=tol):
        err = np.max(np.abs(np.asarray(result, dtype=float) - np.asarray(expected, dtype=float)))
        status = "PASS" if err < tol else "FAIL"
        print(f" [{status}] {name} (max_err={err:.2e})")

    # ── Test 1-4: defining properties on a random 6x7 J ───────────────────
    np.random.seed(0)
    J = np.random.randn(6, 7)
    N = null_space_projector(J)

    check("J @ N ≈ 0 (defining property)", J @ N, np.zeros((6, 7)), tol=1e-6)
    check("N idempotent: N @ N ≈ N", N @ N, N, tol=1e-6)
    check("N symmetric: N ≈ N.T", N, N.T, tol=1e-6)
    check("trace(N) ≈ 1 (redundancy = n - m = 1)", np.trace(N), 1.0, tol=1e-6)

    # ── Test 5-6: redundancy_resolution properties ────────────────────────
    xi_dot = np.array([0.1, -0.2, 0.3, 0.05, -0.1, 0.2])
    J_pinv = np.linalg.pinv(J)

    q_dot_no_secondary = redundancy_resolution(J, xi_dot, np.zeros(7), J_pinv=J_pinv, N=N)
    check("redundancy_resolution(secondary=0) == J_pinv @ xi_dot",
          q_dot_no_secondary, J_pinv @ xi_dot, tol=1e-6)

    q_dot_secondary = np.random.randn(7)
    q_dot_combined = redundancy_resolution(J, xi_dot, q_dot_secondary, J_pinv=J_pinv, N=N)
    check("Task-space consistency: J @ q_dot ≈ xi_dot regardless of secondary term",
          J @ q_dot_combined, xi_dot, tol=1e-6)

    # ── Test 7: hand-verifiable 3-link planar arm sanity check ───────────
    q_3r = np.array([0.4, 0.9, -0.6])
    J_3r = _planar_3r_jacobian(q_3r)
    N_3r = null_space_projector(J_3r)

    check("[3R arm] J @ N ≈ 0", J_3r @ N_3r, np.zeros((2, 3)), tol=1e-6)
    check("[3R arm] N idempotent", N_3r @ N_3r, N_3r, tol=1e-6)
    check("[3R arm] trace(N) ≈ 1 (n=3, m=2)", np.trace(N_3r), 1.0, tol=1e-6)

    # ── Test 8: cross-check against pseudoinverse.py (sibling module) ────
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    try:
        try:
            from control.pseudoinverse import pinv_dls
        except ModuleNotFoundError:
            from control.pseudoinverse import pinv_dls

        J_pinv_dls0 = pinv_dls(J, lam=0.0)
        N_dls0 = null_space_projector(J, J_pinv_dls0)
        check("null_space_projector(pinv_dls(lam=0)) matches np.linalg.pinv version",
              N_dls0, N, tol=1e-6)

        J_pinv_damped = pinv_dls(J, lam=0.3)
        N_damped = null_space_projector(J, J_pinv_damped)
        leak = np.max(np.abs(J @ N_damped))
        idem_err = np.max(np.abs(N_damped @ N_damped - N_damped))
        sym_err = np.max(np.abs(N_damped - N_damped.T))
        print(f"     [INFO] damped J_pinv (lam=0.3): J @ N leakage max={leak:.4f}, "
              f"idempotency error max={idem_err:.4f} "
              f"(expected non-zero — see documented caveat, not a failure)")
        print(f"     [INFO] N stays symmetric regardless of damping "
              f"(symmetry error max={sym_err:.2e}, as expected from N = I - J^T A^-1 J)")
    except ModuleNotFoundError:
        print(" [SKIP] pseudoinverse.py not found on path.")

    # ── Test 9: manipulability ascent as the real secondary objective ────
    try:
        try:
            from control.manipulability import manipulability_gradient
        except ModuleNotFoundError:
            from manipulability import manipulability_gradient

        def _jacobian_func_3r(q, S_list=None, M=None):
            return _planar_3r_jacobian(q)

        grad_w = manipulability_gradient(q_3r, None, None, _jacobian_func_3r)
        xi_dot_3r = np.array([0.2, -0.1])
        q_dot_pipeline = redundancy_resolution(J_3r, xi_dot_3r, grad_w, J_pinv=None, N=N_3r)
        check("[pipeline] J @ q_dot ≈ xi_dot with manipulability-gradient secondary term",
              J_3r @ q_dot_pipeline, xi_dot_3r, tol=1e-6)
    except ModuleNotFoundError:
        print(" [SKIP] manipulability.py not found on path.")

    # ── Test 10 (optional): full pipeline on the real iiwa7_r800 model ───
    try:
        try:
            from kinematics.robot_param import get_robot
            from kinematics.body_jacobian import body_jacobian
            from control.manipulability import manipulability_gradient as grad_fn
            from control.pseudoinverse import pinv_dls as pinv_fn
        except ModuleNotFoundError:
            from kinematics.robot_param import get_robot
            from kinematics.body_jacobian import body_jacobian
            from manipulability import manipulability_gradient as grad_fn
            from control.pseudoinverse import pinv_dls as pinv_fn

        robot = get_robot("iiwa7_r800")
        q_iiwa = np.array([0.3, -0.5, 0.8, -0.2, 1.1, -0.7, 0.4])
        J_iiwa = body_jacobian(q_iiwa, robot["S_list"], robot["M"])
        J_pinv_iiwa = pinv_fn(J_iiwa, lam=0.0)
        grad_w_iiwa = grad_fn(q_iiwa, robot["S_list"], robot["M"], body_jacobian)

        xi_dot_iiwa = np.array([0.1, 0.0, -0.1, 0.0, 0.05, 0.0])
        q_dot_iiwa = redundancy_resolution(J_iiwa, xi_dot_iiwa, grad_w_iiwa, J_pinv=J_pinv_iiwa)
        err = np.max(np.abs(J_iiwa @ q_dot_iiwa - xi_dot_iiwa))
        status = "PASS" if err < 1e-6 else "FAIL"
        print(f" [{status}] Full pipeline (iiwa7_r800): task-space consistency "
              f"(max_err={err:.2e})")
    except ModuleNotFoundError:
        print(" [SKIP] robot_params.py / body_jacobian.py not found on path — "
              "run from within the project structure for this cross-check.")

    print("\n" + "=" * 60)


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    _run_validation()