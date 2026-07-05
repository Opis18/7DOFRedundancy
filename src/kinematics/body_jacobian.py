"""
body_jacobian.py
================
Body Jacobian construction for a 7-DOF redundant manipulator.

The Body Jacobian J_b maps joint velocities to the end-effector body twist:

    ξ_body = J_b(q) · q̇        (6×7 matrix equation)

where ξ_body = [ω, v] is expressed in the end-effector's own frame (body frame).

Derivation from the Space-frame PoE
-------------------------------------
Starting from the space-frame Product of Exponentials:

    T_sb(q) = exp([S₁]q₁) · exp([S₂]q₂) · ... · exp([S₇]q₇) · M

The space Jacobian column i is:

    J_s^i = Ad(T_{0,i-1}) · Sᵢ

where T_{0,i-1} = exp([S₁]q₁) · ... · exp([S_{i-1}]q_{i-1})  (partial FK up to joint i-1)

The body Jacobian is then obtained by re-expressing each column in the
end-effector body frame via the inverse adjoint of the full FK transform:

    J_b = Ad(T_sb⁻¹) · J_s

So column i of J_b is:

    J_b^i = Ad(T_sb⁻¹ · T_{0,i-1}) · Sᵢ

This is the formula implemented here. The adjoint transform carries each
joint's screw axis from the space frame into the end-effector body frame.

Physical interpretation
-----------------------
    Column i of J_b : the body twist the end-effector would experience
                      if only joint i moved at unit rate (q̇ᵢ = 1 rad/s),
                      expressed in the end-effector's own coordinate frame.

Functions
---------
    body_jacobian           : analytical J_b via adjoint chain
    body_jacobian_numerical : finite-difference J_b (for validation only)
"""

import numpy as np
import sys
import os

# ── Allow import from project root or kinematics/ directly ───────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from kinematics.lie_algebra import exp_map, log_map_se3, adjoint
    from kinematics.forward_kinematics import forward_kinematics
except ModuleNotFoundError:
    from lie_algebra import exp_map, log_map_se3, adjoint
    from forward_kinematics import forward_kinematics


# ──────────────────────────────────────────────────────────────────────────────
# Primary function — analytical body Jacobian
# ──────────────────────────────────────────────────────────────────────────────

def body_jacobian(q, S_list, M, T_sb=None):
    """
    Compute the 6×n Body Jacobian at joint configuration q.

    Uses the adjoint chain formula derived from the space-frame PoE:

        J_b^i = Ad(T_sb⁻¹ · T_{0,i-1}) · Sᵢ

    where T_{0,i-1} = exp([S₁]q₁) · ... · exp([S_{i-1}]q_{i-1}).

    Parameters
    ----------
    q : array_like, shape (n,)
        Current joint angles in radians.

    S_list : list of array_like, each shape (6,)
        Space-frame screw axes at home config (same list used in FK).

    M : array_like, shape (4, 4)
        Home configuration SE(3) matrix (same as used in FK).

    T_sb : array_like, shape (4, 4), optional
        End-effector pose from FK at the current q.
        Pass this in if you have already called forward_kinematics(q)
        in the same timestep — avoids recomputing FK internally.
        If None, FK is called automatically.

    Returns
    -------
    J_b : np.ndarray, shape (6, n)
        Body Jacobian matrix. Columns are body-frame screw axes,
        one per joint. Expressed in the [ω, v] convention.

    Notes
    -----
    The caller (trajectory_tracker.py) should pass T_sb to avoid
    computing FK twice per control loop iteration.
    """
    q      = np.asarray(q, dtype=float).flatten()
    n      = len(S_list)

    # ── Compute FK if not supplied ────────────────────────────────────────────
    if T_sb is None:
        T_sb = forward_kinematics(q, S_list, M)

    T_sb_inv = np.linalg.inv(T_sb)

    # ── Build J_b column by column ────────────────────────────────────────────
    J_b     = np.zeros((6, n))
    T_accum = np.eye(4)   # tracks T_{0,i-1} = exp([S1]q1)·...·exp([S_{i-1}]q_{i-1})

    for i in range(n):
        Si = np.asarray(S_list[i], dtype=float).flatten()

        # Transform partial FK into end-effector body frame
        # T_rel = T_sb⁻¹ · T_{0,i-1}  encodes where joint i's axis
        # sits relative to the current end-effector frame
        T_rel = T_sb_inv @ T_accum

        # Apply adjoint to re-express Sᵢ in the body frame
        J_b[:, i] = adjoint(T_rel) @ Si

        # Accumulate the transform for the next column
        # T_{0,i} = T_{0,i-1} · exp([Sᵢ]qᵢ)
        T_accum = T_accum @ exp_map(Si * q[i])

    return J_b


# ──────────────────────────────────────────────────────────────────────────────
# Numerical body Jacobian — finite differences (validation only)
# ──────────────────────────────────────────────────────────────────────────────

def body_jacobian_numerical(q, S_list, M, eps=1e-7):
    """
    Compute the body Jacobian numerically via central finite differences.

    Used exclusively for validating the analytical body_jacobian().
    Do NOT use in the control loop — it calls FK 2n times per call.

    The body Jacobian column i satisfies:

        J_b^i ≈ vee6( log( T_sb⁻¹ · T_sb(q + ε·eᵢ) ) ) / ε

    This follows directly from the definition J_b · q̇ = ξ_body.

    Parameters
    ----------
    q : array_like, shape (n,)
        Joint angles in radians.

    S_list : list of array_like, each shape (6,)
        Space-frame screw axes.

    M : array_like, shape (4, 4)
        Home configuration.

    eps : float, optional
        Finite-difference step size. Default 1e-7 balances truncation
        error (smaller ε) vs floating-point cancellation (larger ε).

    Returns
    -------
    J_b_num : np.ndarray, shape (6, n)
        Numerical body Jacobian.
    """
    q = np.asarray(q, dtype=float).flatten()
    n = len(S_list)

    T_sb     = forward_kinematics(q, S_list, M)
    T_sb_inv = np.linalg.inv(T_sb)

    J_b_num = np.zeros((6, n))

    for i in range(n):
        # Forward perturbation
        q_fwd      = q.copy(); q_fwd[i] += eps
        T_fwd      = forward_kinematics(q_fwd, S_list, M)
        xi_fwd     = log_map_se3(T_sb_inv @ T_fwd)

        # Backward perturbation
        q_bwd      = q.copy(); q_bwd[i] -= eps
        T_bwd      = forward_kinematics(q_bwd, S_list, M)
        xi_bwd     = log_map_se3(T_sb_inv @ T_bwd)

        # Central difference
        J_b_num[:, i] = (xi_fwd - xi_bwd) / (2 * eps)

    return J_b_num


# ──────────────────────────────────────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────────────────────────────────────

def _run_validation():
    """
    Validation suite for body_jacobian.

    Tests
    -----
    1. Output shape is (6, n)
    2. Analytical vs numerical match at q=0 (generic robot)
    3. Analytical vs numerical match at random q (generic robot)
    4. Analytical vs numerical match at random q (iiwa7_r800)
    5. Rank of J_b is 6 at a non-singular configuration
    6. Zero-velocity consistency: J_b @ zeros == zeros
    7. Single-joint sanity: for a pure z-rotation robot, J_b[:,0] == body-frame S1
    """
    print("=" * 60)
    print("  body_jacobian.py  —  Validation Suite")
    print("=" * 60)

    tol_analytical = 1e-8    # analytical vs numerical agreement
    tol_shape      = 0       # exact

    def check(name, result, expected, tol):
        err = np.max(np.abs(np.asarray(result) - np.asarray(expected)))
        status = "PASS" if err < tol else "FAIL"
        print(f"  [{status}]  {name}  (max_err={err:.2e})")

    # ── Shared geometry — use robot_params if available, else define inline ──
    try:
        sys.path.insert(0, os.path.dirname(__file__))
        from robot_param import get_robot
        robot_generic = get_robot("generic_7dof")
        robot_iiwa    = get_robot("iiwa7_r800")
        S_gen  = robot_generic["S_list"]
        M_gen  = robot_generic["M"]
        S_iiwa = robot_iiwa["S_list"]
        M_iiwa = robot_iiwa["M"]
        have_robot_params = True
    except (ImportError, ModuleNotFoundError):
        have_robot_params = False
        # Fallback: simple 7-joint chain
        def _make_screw(w, r):
            w = np.array(w, dtype=float)
            r = np.array(r, dtype=float)
            return np.concatenate([w, -np.cross(w, r)])
        z = [0, 0, 1]; y = [0, 1, 0]
        S_gen  = [_make_screw(z, [0,0,i*0.36]) for i in range(7)]
        S_iiwa = S_gen
        M_gen  = np.eye(4); M_gen[2, 3] = 1.26
        M_iiwa = M_gen

    n = 7

    # ── Test 1: Shape at q=0 ─────────────────────────────────────────────────
    q_zero = np.zeros(n)
    J_zero = body_jacobian(q_zero, S_gen, M_gen)
    shape_ok = J_zero.shape == (6, n)
    print(f"\n  [{'PASS' if shape_ok else 'FAIL'}]  Output shape is (6, 7)")

    # ── Test 2: Analytical vs numerical at q=0 (generic) ─────────────────────
    J_num_zero = body_jacobian_numerical(q_zero, S_gen, M_gen)
    check("Analytical == numerical at q=0 (generic)",
          J_zero, J_num_zero, tol=tol_analytical)

    # ── Test 3: Analytical vs numerical at random q (generic) ────────────────
    np.random.seed(42)
    q_rand = np.array([0.3, -0.5, 0.8, -0.2, 1.1, -0.7, 0.4])
    J_ana  = body_jacobian(q_rand, S_gen, M_gen)
    J_num  = body_jacobian_numerical(q_rand, S_gen, M_gen)
    check("Analytical == numerical at random q (generic)",
          J_ana, J_num, tol=tol_analytical)

    # ── Test 4: Analytical vs numerical at random q (iiwa7) ──────────────────
    J_ana_iiwa = body_jacobian(q_rand, S_iiwa, M_iiwa)
    J_num_iiwa = body_jacobian_numerical(q_rand, S_iiwa, M_iiwa)
    check("Analytical == numerical at random q (iiwa7_r800)",
          J_ana_iiwa, J_num_iiwa, tol=tol_analytical)

    # ── Test 5: Rank check — J_b should have rank 6 at non-singular q ────────
    rank = np.linalg.matrix_rank(J_ana, tol=1e-10)
    print(f"  [{'PASS' if rank == 6 else 'FAIL'}]  "
          f"rank(J_b) = 6 at random q (got {rank})")

    # ── Test 6: T_sb caching — passing T_sb gives same result ────────────────
    T_sb = forward_kinematics(q_rand, S_gen, M_gen)
    J_cached = body_jacobian(q_rand, S_gen, M_gen, T_sb=T_sb)
    check("Cached T_sb gives identical J_b",
          J_cached, J_ana, tol=1e-15)

    # ── Test 7: Zero velocity consistency ────────────────────────────────────
    xi_zero = J_ana @ np.zeros(n)
    check("J_b @ q̇=0 gives zero twist",
          xi_zero, np.zeros(6), tol=1e-15)

    # ── Test 8: Single revolute joint — body Jacobian is Ad(M⁻¹)·S₁ ─────────
    # For a single revolute joint with S = [0,0,1,0,0,0] and M=I:
    # At q=0: T_sb = exp(0) · I = I, so J_b[:,0] = Ad(I⁻¹·I) · S = S
    S_single = [np.array([0, 0, 1, 0, 0, 0], dtype=float)]
    M_single = np.eye(4)
    J_single = body_jacobian(np.array([0.0]), S_single, M_single)
    check("Single joint at q=0: J_b[:,0] == Ad(M⁻¹)·S",
          J_single[:, 0], np.array([0, 0, 1, 0, 0, 0]), tol=1e-15)

    # ── Print J_b for inspection ──────────────────────────────────────────────
    print(f"\n  J_b at q={q_rand} (iiwa7_r800):")
    print(f"  Shape: {J_ana_iiwa.shape}")
    np.set_printoptions(precision=4, suppress=True)
    print(J_ana_iiwa)
    print()
    print(f"  Singular values of J_b: "
          f"{np.linalg.svd(J_ana_iiwa, compute_uv=False).round(4)}")

    print("\n" + "=" * 60)


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    _run_validation()