"""
manipulability.py
==================
Yoshikawa manipulability measure and its gradient — used for algorithmic
singularity avoidance in the "control" branch.

The Yoshikawa manipulability measure

    w(q) = sqrt(det(J(q) J(q)^T))

is proportional to the volume of the manipulability ellipsoid and vanishes
exactly at kinematic singularities (rank(J) < 6). It supplies:

  1. A scalar "distance from singularity" fed into pseudoinverse.py's
     Nakamura-style adaptive damping schedule.
  2. A gradient direction, ∇w(q), used as the secondary objective in
     null_space.py's redundancy resolution: pushing +∇w through the
     null-space projector lets the robot climb away from singular
     configurations without disturbing the primary task-space motion.

Convention (Lynch & Park, consistent with lie_algebra.py / body_jacobian.py):
    Twist ξ = [ω, v]. w(q) itself is invariant to this row ordering — it
    depends only on det(J J^T), which is unchanged under any fixed row
    permutation of J — so no convention-specific handling is needed here.

Sign convention (resolved)
---------------------------
Downstream callers (null_space.redundancy_resolution) should use
q_dot_secondary = +k * manipulability_gradient(...), k > 0, i.e. gradient
ASCENT on w. For the undamped null-space projector N = I - J^+ J (which
is symmetric idempotent), this guarantees the null-space term's
contribution to dw/dt is non-negative:

    grad(w)^T @ N @ (k * grad(w)) = k * ||N @ grad(w)||^2 >= 0

Flipping the sign would actively steer toward singularities, so this is
a hard convention, not a tunable — it's set here in the producer of
grad(w) so every consumer inherits it.

Functions
---------
yoshikawa_measure       : w(q) = sqrt(det(J J^T))
manipulability_gradient : numerical (central-difference) gradient dw/dq
"""

import numpy as np
import sys
import os


# ──────────────────────────────────────────────────────────────────────────────
# 1. yoshikawa_measure — w(q) = sqrt(det(J J^T))
# ──────────────────────────────────────────────────────────────────────────────

def yoshikawa_measure(J):
    """
    Compute the Yoshikawa manipulability measure of a Jacobian.

        w = sqrt(det(J @ J.T))

    Equivalently, the product of the singular values of J. w > 0 away
    from kinematic singularities and w -> 0 continuously as the
    configuration approaches one (rank(J) < 6).

    Parameters
    ----------
    J : array_like, shape (m, n)
        Jacobian (typically the 6xn body Jacobian J_b). Requires m <= n
        (task-space dimension no greater than joint count) — true for
        this project's 6x7 body Jacobian.

    Returns
    -------
    w : float
        Manipulability measure, w >= 0.

    Notes
    -----
    J @ J.T is symmetric positive semi-definite, so det(J @ J.T) is
    mathematically non-negative; it is clipped at 0 before the sqrt to
    absorb floating-point noise that can otherwise produce a tiny
    negative determinant right at a singularity.
    """
    J = np.asarray(J, dtype=float)
    if J.ndim != 2:
        raise ValueError(f"yoshikawa_measure expects a 2D matrix, got shape {J.shape}")

    JJt = J @ J.T
    det = np.linalg.det(JJt)
    det = max(det, 0.0)  # guard against tiny negative floating-point noise near w=0

    return float(np.sqrt(det))


# ──────────────────────────────────────────────────────────────────────────────
# 2. manipulability_gradient — numerical gradient of w(q)
# ──────────────────────────────────────────────────────────────────────────────

def manipulability_gradient(q, S_list, M, jacobian_func, eps=1e-6):
    """
    Central-difference gradient of w(q) with respect to joint angles.

        grad(w)_i = [ w(q + eps*e_i) - w(q - eps*e_i) ] / (2*eps)

    Used as the secondary null-space objective for singularity avoidance:
    downstream callers should pass +grad(w) (gradient ASCENT — see the
    module docstring's "Sign convention" section) into
    null_space.redundancy_resolution.

    Parameters
    ----------
    q : array_like, shape (n,)
        Current joint configuration.
    S_list : list of array_like, each shape (6,)
        Screw axes defining the manipulator, passed straight through to
        jacobian_func (same S_list used throughout the project — see
        forward_kinematics.py / body_jacobian.py).
    M : array_like, shape (4, 4)
        Home configuration of the end-effector, passed straight through
        to jacobian_func.
    jacobian_func : callable
        Function with signature jacobian_func(q, S_list, M) -> J,
        i.e. body_jacobian.body_jacobian in the full pipeline. Passed in
        rather than imported directly so this module stays decoupled
        from body_jacobian.py, matching the pattern used elsewhere in
        the project (forward_kinematics.py and body_jacobian.py are
        each generic over S_list/M rather than hard-coding a robot).
    eps : float, optional (default 1e-6)
        Finite-difference step size.

    Returns
    -------
    grad_w : np.ndarray, shape (n,)
        Central-difference approximation of d(w)/dq.
    """
    q = np.asarray(q, dtype=float).flatten()
    n = len(q)
    grad_w = np.zeros(n)

    for i in range(n):
        q_fwd = q.copy()
        q_fwd[i] += eps
        q_bwd = q.copy()
        q_bwd[i] -= eps

        w_fwd = yoshikawa_measure(jacobian_func(q_fwd, S_list, M))
        w_bwd = yoshikawa_measure(jacobian_func(q_bwd, S_list, M))

        grad_w[i] = (w_fwd - w_bwd) / (2 * eps)

    return grad_w


# ──────────────────────────────────────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────────────────────────────────────

def _planar_2r_jacobian(q, S_list=None, M=None, l1=1.0, l2=1.0):
    """
    Closed-form Jacobian of a 2-link planar RR arm (validation helper only).

    Classic textbook result (e.g. Lynch & Park):
        J(theta) = [[-l1*sin(t1) - l2*sin(t1+t2), -l2*sin(t1+t2)],
                    [ l1*cos(t1) + l2*cos(t1+t2),  l2*cos(t1+t2)]]
        det(J) = l1*l2*sin(t2)

    This gives a hand-verifiable, dependency-free ground truth for
    yoshikawa_measure and manipulability_gradient: w(q) = l1*l2*|sin(t2)|
    is singular (w=0) exactly at the arm's fully-extended / fully-folded
    configurations (t2 = 0, pi), and independent of t1.

    Matches the (q, S_list, M) -> J calling signature expected by
    manipulability_gradient; S_list and M are accepted and ignored.
    """
    q = np.asarray(q, dtype=float).flatten()
    t1, t2 = q[0], q[1]
    s1, c1 = np.sin(t1), np.cos(t1)
    s12, c12 = np.sin(t1 + t2), np.cos(t1 + t2)
    return np.array([
        [-l1 * s1 - l2 * s12, -l2 * s12],
        [ l1 * c1 + l2 * c12,  l2 * c12],
    ])


def _run_validation():
    """
    Validation suite for manipulability.py.

    Tests
    -----
    1. yoshikawa_measure matches the SVD-based equivalent prod(singular
       values) on a random 6x7 matrix.
    2. w > 0 at a known non-singular 2R-arm configuration.
    3. w ≈ 0 at a synthetic rank-deficient 6x7 matrix (duplicated row).
    4. w -> 0 continuously as the 2R arm approaches its singularity
       (t2 -> 0), matching the closed form w = l1*l2*|sin(t2)|.
    5. manipulability_gradient matches the closed-form analytical
       gradient of the 2R arm's w(q) (dw/dt1 = 0, dw/dt2 = l1*l2*cos(t2)).
    6. Directional check: a small step along +grad(w) increases w(q)
       (confirms the gradient-ascent sign convention documented above).
    7. (Optional, skipped if unavailable) Cross-check manipulability_gradient
       against robot_params + body_jacobian on the real iiwa7_r800 model,
       if those modules are importable from the project structure.
    """
    print("=" * 60)
    print(" manipulability.py — Validation Suite")
    print("=" * 60)

    tol = 1e-8

    def check(name, result, expected, tol=tol):
        err = np.max(np.abs(np.asarray(result, dtype=float) - np.asarray(expected, dtype=float)))
        status = "PASS" if err < tol else "FAIL"
        print(f" [{status}] {name} (max_err={err:.2e})")

    # ── Test 1: SVD equivalence ───────────────────────────────────────────
    np.random.seed(0)
    J_rand = np.random.randn(6, 7)
    w_det = yoshikawa_measure(J_rand)
    w_svd = float(np.prod(np.linalg.svd(J_rand, compute_uv=False)))
    check("yoshikawa_measure matches SVD product", w_det, w_svd)

    # ── Test 2: w > 0 at a non-singular 2R-arm configuration ─────────────
    q_ok = np.array([0.3, np.pi / 2])  # t2 = 90°, well away from singularity
    J_ok = _planar_2r_jacobian(q_ok)
    w_ok = yoshikawa_measure(J_ok)
    expected_ok = 1.0 * 1.0 * abs(np.sin(np.pi / 2))  # l1=l2=1
    check("w > 0 at non-singular 2R config (t2=90°)", w_ok, expected_ok)

    # ── Test 3: w ≈ 0 at a synthetic rank-deficient matrix ────────────────
    # Note: det(J @ J.T) for an exactly rank-deficient 6x6 matrix is
    # mathematically 0, but numerical LU-based determinants of a
    # near-degenerate matrix with O(1-10) entries carry floating-point
    # error well above 1e-8 in absolute terms — use a looser tolerance
    # here than the rest of the suite, scaled to that matrix's entries.
    J_deg = np.random.randn(6, 7)
    J_deg[5, :] = J_deg[0, :]  # duplicate a row -> rank <= 5
    w_deg = yoshikawa_measure(J_deg)
    check("w ≈ 0 at rank-deficient J (duplicated row)", w_deg, 0.0, tol=1e-4)

    # ── Test 4: w -> 0 continuously as 2R arm approaches singularity ─────
    t2_values = [0.5, 0.1, 0.01, 0.001]
    w_values = [yoshikawa_measure(_planar_2r_jacobian([0.2, t2])) for t2 in t2_values]
    monotonic = all(w_values[i] > w_values[i + 1] for i in range(len(w_values) - 1))
    expected_w = [abs(np.sin(t2)) for t2 in t2_values]
    check("w matches closed form l1*l2*|sin(t2)| across approach to singularity",
          w_values, expected_w)
    print(f"     {'PASS' if monotonic else 'FAIL'}: w decreases monotonically as t2 -> 0 "
          f"({[round(w, 4) for w in w_values]})")

    # ── Test 5: gradient matches closed-form analytical gradient ─────────
    q_grad = np.array([0.3, 0.8])  # sin(t2) > 0, so w = l1*l2*sin(t2) locally
    grad_num = manipulability_gradient(q_grad, None, None, _planar_2r_jacobian)
    grad_analytical = np.array([0.0, np.cos(q_grad[1])])  # dw/dt1=0, dw/dt2=cos(t2)
    check("manipulability_gradient matches closed-form (dw/dt1=0, dw/dt2=cos t2)",
          grad_num, grad_analytical, tol=1e-6)

    # ── Test 6: directional ascent check ──────────────────────────────────
    q_near_sing = np.array([0.2, 0.05])  # close to the t2=0 singularity
    w_before = yoshikawa_measure(_planar_2r_jacobian(q_near_sing))
    grad = manipulability_gradient(q_near_sing, None, None, _planar_2r_jacobian)
    alpha = 0.05
    q_after = q_near_sing + alpha * grad  # gradient ASCENT (+grad, per sign convention)
    w_after = yoshikawa_measure(_planar_2r_jacobian(q_after))
    ascended = w_after > w_before
    print(f" [{'PASS' if ascended else 'FAIL'}] step along +grad(w) increases w "
          f"({w_before:.4f} -> {w_after:.4f})")

    # ── Test 7 (optional): cross-check on the real iiwa7_r800 model ──────
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    try:
        try:
            from kinematics.robot_param import get_robot
            from kinematics.body_jacobian import body_jacobian
        except ModuleNotFoundError:
            from kinematics.robot_param import get_robot
            from kinematics.body_jacobian import body_jacobian

        robot = get_robot("iiwa7_r800")
        q_iiwa = np.array([0.3, -0.5, 0.8, -0.2, 1.1, -0.7, 0.4])
        grad_iiwa = manipulability_gradient(
            q_iiwa, robot["S_list"], robot["M"], body_jacobian
        )
        # Self-consistency: gradient computed at two different eps values
        # should agree closely if the finite-difference step is well-scaled.
        grad_iiwa_alt = manipulability_gradient(
            q_iiwa, robot["S_list"], robot["M"], body_jacobian, eps=1e-5
        )
        err = np.max(np.abs(grad_iiwa - grad_iiwa_alt))
        status = "PASS" if err < 1e-4 else "FAIL"
        print(f" [{status}] iiwa7_r800 gradient stable across eps=1e-6 vs 1e-5 "
              f"(max_err={err:.2e})")
        w_iiwa = yoshikawa_measure(body_jacobian(q_iiwa, robot["S_list"], robot["M"]))
        print(f"     w(q) = {w_iiwa:.4f} at test configuration on iiwa7_r800")
    except ModuleNotFoundError:
        print(" [SKIP] robot_params.py / body_jacobian.py not found on path — "
              "run from within the project structure for this cross-check.")

    print("\n" + "=" * 60)


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    _run_validation()