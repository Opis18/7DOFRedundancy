"""
pseudoinverse.py
================
Damped least-squares (DLS) pseudoinverse of the body Jacobian J_b(q), with
a Nakamura–Hanafusa adaptive damping schedule driven by manipulability.

The plain Moore–Penrose pseudoinverse J^+ = J^T (J J^T)^-1 blows up as
det(J J^T) -> 0 near a kinematic singularity. Damped least squares trades
a small amount of tracking accuracy for numerical stability:

    J^+_damped = J^T (J J^T + lam^2 * I)^-1

lam is chosen adaptively from the manipulability measure w(q) (see
manipulability.yoshikawa_measure), via the standard Nakamura–Hanafusa
quadratic ramp:

    lam(w) = 0                              if w >= w_threshold
    lam(w) = lam_max * (1 - w/w_threshold)^2 if w <  w_threshold

i.e. damping is off entirely away from singularities and ramps up
smoothly (quadratically, so with zero slope at w = w_threshold — no
kink) as w collapses toward 0.

Role in the "control" branch
-----------------------------
    q ──► body_jacobian(q) ──► J_b
                  │
                  ├──► manipulability.yoshikawa_measure(J_b) ──► w
                  │                     │
                  │                     ▼
                  │            adaptive_damping(w, ...) ──► lam
                  │                     │
                  │                     ▼
                  └──────────► pinv_dls(J_b, lam) ──► J_b^+
                                        │
                                        ▼
                          (consumed by null_space.py to build the
                           null-space projector and resolve redundancy)

This module takes w and lam as plain float arguments rather than
importing manipulability.py directly — keeps it decoupled and testable
in isolation, matching the pattern used throughout the project (each
control module is generic over its inputs rather than reaching into
its neighbors).

Functions
---------
pinv_dls          : damped least-squares pseudoinverse of a Jacobian
adaptive_damping  : Nakamura–Hanafusa damping schedule, w -> lam
"""

import numpy as np
import sys
import os


# ──────────────────────────────────────────────────────────────────────────────
# 1. pinv_dls — damped least-squares pseudoinverse
# ──────────────────────────────────────────────────────────────────────────────

def pinv_dls(J, lam=0.0):
    """
    Damped least-squares pseudoinverse of a Jacobian.

        J^+ = J^T (J J^T + lam^2 * I)^-1

    Reduces to the exact Moore–Penrose pseudoinverse when lam = 0 (for
    a full-row-rank J, this matches np.linalg.pinv(J) to numerical
    precision).

    Parameters
    ----------
    J : array_like, shape (m, n)
        Jacobian to invert (typically the 6xn body Jacobian J_b). This
        project's use case has m=6, n=7 (redundant manipulator), i.e. J
        is wide and generically full row rank away from singularities.
    lam : float, optional (default 0.0)
        Damping factor. lam = 0 recovers the undamped pseudoinverse.
        Larger lam trades tracking accuracy for conditioning near
        singularities. Typically supplied by adaptive_damping(w, ...)
        rather than hand-set.

    Returns
    -------
    J_pinv : np.ndarray, shape (n, m)
        Damped least-squares pseudoinverse of J.

    Notes
    -----
    This is the "right" damped pseudoinverse (via J J^T, an m x m
    matrix), appropriate for the wide/redundant case (m < n) used
    throughout this project. It is not the general Moore–Penrose
    pseudoinverse for arbitrary-shaped or rank-deficient J.
    """
    J = np.asarray(J, dtype=float)
    m = J.shape[0]

    JJt = J @ J.T
    damped = JJt + (lam ** 2) * np.eye(m)

    return J.T @ np.linalg.inv(damped)


# ──────────────────────────────────────────────────────────────────────────────
# 2. adaptive_damping — Nakamura–Hanafusa quadratic damping ramp
# ──────────────────────────────────────────────────────────────────────────────

def adaptive_damping(w, w_threshold, lam_max):
    """
    Nakamura–Hanafusa adaptive damping schedule.

        lam(w) = 0                              if w >= w_threshold
        lam(w) = lam_max * (1 - w/w_threshold)^2 if w <  w_threshold

    Damping is off entirely away from singularities and ramps up
    quadratically (zero slope at w = w_threshold, so the transition is
    smooth — no kink in lam as a function of w) as manipulability
    collapses toward a kinematic singularity.

    Parameters
    ----------
    w : float
        Manipulability measure at the current configuration, from
        manipulability.yoshikawa_measure(J_b).
    w_threshold : float
        Manipulability value below which damping begins to engage.
    lam_max : float
        Damping factor applied in the limit w -> 0.

    Returns
    -------
    lam : float
        Damping factor to pass into pinv_dls.
    """
    if w >= w_threshold:
        return 0.0

    ratio = w / w_threshold
    return lam_max * (1.0 - ratio) ** 2


# ──────────────────────────────────────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────────────────────────────────────

def _near_singular_jacobian(m, n, sigma_min, seed=0):
    """
    Build a synthetic (m, n) matrix with a controlled smallest singular
    value, via a random SVD construction (validation helper only).

    Singular values are sigma_min followed by (min(m,n)-1) values drawn
    uniformly from [0.5, 1.5], so the matrix is well-conditioned except
    for one small singular value that mimics an approaching kinematic
    singularity.
    """
    rng = np.random.default_rng(seed)
    U, _ = np.linalg.qr(rng.standard_normal((m, m)))
    V, _ = np.linalg.qr(rng.standard_normal((n, n)))

    k = min(m, n)
    sigmas = np.concatenate([[sigma_min], rng.uniform(0.5, 1.5, size=k - 1)])

    S = np.zeros((m, n))
    S[:k, :k] = np.diag(sigmas)

    return U @ S @ V.T


def _run_validation():
    """
    Validation suite for pseudoinverse.py.

    Tests
    -----
    1. pinv_dls(J, lam=0) matches np.linalg.pinv(J) for a well-conditioned
       random 6x7 J.
    2. Roundtrip: J @ pinv_dls(J, 0) @ J ≈ J (Moore–Penrose identity), and
       J @ pinv_dls(J, 0) ≈ I_6 (right-inverse property for full-row-rank J).
    3. Damped solution norm ||pinv_dls(J, lam) @ xi_dot|| is monotonically
       non-increasing in lam, for a fixed target twist xi_dot (standard
       Tikhonov/ridge property).
    4. At a synthetic near-singular J (smallest singular value 1e-8),
       lam=0 blows up (huge solution norm) while lam>0 stays bounded.
    5. adaptive_damping returns 0 at w=w_threshold and lam_max in the
       limit w -> 0, with the expected quadratic value at the midpoint.
    6. (Optional, skipped if unavailable) Cross-check the full
       w -> adaptive_damping -> pinv_dls pipeline on the real
       iiwa7_r800 body Jacobian, via robot_params + body_jacobian +
       manipulability.
    """
    print("=" * 60)
    print(" pseudoinverse.py — Validation Suite")
    print("=" * 60)

    tol = 1e-8

    def check(name, result, expected, tol=tol):
        err = np.max(np.abs(np.asarray(result, dtype=float) - np.asarray(expected, dtype=float)))
        status = "PASS" if err < tol else "FAIL"
        print(f" [{status}] {name} (max_err={err:.2e})")

    # ── Test 1: undamped pinv_dls matches np.linalg.pinv ──────────────────
    np.random.seed(0)
    J = np.random.randn(6, 7)
    J_pinv_dls = pinv_dls(J, lam=0.0)
    J_pinv_np = np.linalg.pinv(J)
    check("pinv_dls(J, lam=0) matches np.linalg.pinv(J)", J_pinv_dls, J_pinv_np)

    # ── Test 2: roundtrip / right-inverse property ────────────────────────
    roundtrip = J @ J_pinv_dls @ J
    check("Roundtrip: J @ J^+ @ J ≈ J", roundtrip, J, tol=1e-6)
    right_inverse = J @ J_pinv_dls
    check("Right-inverse: J @ J^+ ≈ I_6 (full row rank)", right_inverse, np.eye(6), tol=1e-6)

    # ── Test 3: damped solution norm is non-increasing in lam ─────────────
    xi_dot = np.array([0.1, -0.2, 0.3, 0.05, -0.1, 0.2])
    lam_values = [0.0, 0.01, 0.1, 0.5, 1.0]
    norms = [float(np.linalg.norm(pinv_dls(J, lam) @ xi_dot)) for lam in lam_values]
    monotonic = all(norms[i] >= norms[i + 1] - 1e-12 for i in range(len(norms) - 1))
    print(f" [{'PASS' if monotonic else 'FAIL'}] ||q_dot|| non-increasing in lam "
          f"({[round(n, 4) for n in norms]})")

    # ── Test 4: near-singular robustness ───────────────────────────────────
    J_sing = _near_singular_jacobian(6, 7, sigma_min=1e-8, seed=1)
    xi_dot_sing = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    norm_undamped = np.linalg.norm(pinv_dls(J_sing, lam=0.0) @ xi_dot_sing)
    norm_damped = np.linalg.norm(pinv_dls(J_sing, lam=0.1) @ xi_dot_sing)
    blows_up = norm_undamped > 1e4
    stays_bounded = norm_damped < 100
    print(f" [{'PASS' if blows_up else 'FAIL'}] undamped solution blows up near singularity "
          f"(||q_dot||={norm_undamped:.2e})")
    print(f" [{'PASS' if stays_bounded else 'FAIL'}] damped solution stays bounded "
          f"(||q_dot||={norm_damped:.2e})")

    # ── Test 5: adaptive_damping schedule values ──────────────────────────
    w_threshold, lam_max = 0.1, 0.05
    lam_at_threshold = adaptive_damping(w_threshold, w_threshold, lam_max)
    lam_at_zero = adaptive_damping(0.0, w_threshold, lam_max)
    lam_at_mid = adaptive_damping(w_threshold / 2, w_threshold, lam_max)
    expected_mid = lam_max * (1 - 0.5) ** 2  # = lam_max / 4
    check("adaptive_damping(w=w_threshold) == 0", lam_at_threshold, 0.0)
    check("adaptive_damping(w=0) == lam_max", lam_at_zero, lam_max)
    check("adaptive_damping(w=w_threshold/2) == lam_max*(1-0.5)^2", lam_at_mid, expected_mid)
    # Above threshold should also clamp to 0 (not just at the boundary)
    lam_above = adaptive_damping(2 * w_threshold, w_threshold, lam_max)
    check("adaptive_damping(w > w_threshold) == 0", lam_above, 0.0)

    # ── Test 6 (optional): full pipeline on the real iiwa7_r800 model ────
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    try:
        try:
            from kinematics.robot_param import get_robot
            from kinematics.body_jacobian import body_jacobian
            from control.manipulability import yoshikawa_measure
        except ModuleNotFoundError:
            from kinematics.robot_param import get_robot
            from kinematics.body_jacobian import body_jacobian
            from manipulability import yoshikawa_measure

        robot = get_robot("iiwa7_r800")
        q_iiwa = np.array([0.3, -0.5, 0.8, -0.2, 1.1, -0.7, 0.4])
        J_iiwa = body_jacobian(q_iiwa, robot["S_list"], robot["M"])
        w_iiwa = yoshikawa_measure(J_iiwa)
        lam_iiwa = adaptive_damping(w_iiwa, w_threshold=0.05, lam_max=0.05)
        J_pinv_iiwa = pinv_dls(J_iiwa, lam_iiwa)

        shape_ok = J_pinv_iiwa.shape == (7, 6)
        print(f" [{'PASS' if shape_ok else 'FAIL'}] Full pipeline runs on iiwa7_r800 "
              f"(w={w_iiwa:.4f}, lam={lam_iiwa:.4f}, J^+ shape={J_pinv_iiwa.shape})")
    except ModuleNotFoundError:
        print(" [SKIP] robot_params.py / body_jacobian.py / manipulability.py not found "
              "on path — run from within the project structure for this cross-check.")

    print("\n" + "=" * 60)


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    _run_validation()