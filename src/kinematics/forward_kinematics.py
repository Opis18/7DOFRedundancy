"""
forward_kinematics.py
=====================
Forward kinematics for a 7-DOF redundant manipulator via the
Product of Exponentials (PoE) formula.

Given joint angles q and the robot's screw axes S_list (defined in the
space frame at the zero/home configuration), the end-effector pose T in
SE(3) is computed as:

    T(q) = exp([S₁]q₁) · exp([S₂]q₂) · ... · exp([S₇]q₇) · M

where M is the home configuration (end-effector pose when all qᵢ = 0).

Convention (Lynch & Park, Modern Robotics):
    Screw axis  Sᵢ = [ω̂ᵢ, vᵢ]   where  vᵢ = -ω̂ᵢ × rᵢ
    ω̂ᵢ  : unit rotation axis of joint i in space frame
    rᵢ   : any point on joint i's axis in space frame
    M    : SE(3) home configuration (4×4)

Functions
---------
    forward_kinematics   : compute SE(3) end-effector pose from joint angles
    fk_positions         : extract (x, y, z) of each joint for visualization
"""

import numpy as np
import sys
import os

# ── Allow import whether running from project root or kinematics/ directly ───
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from kinematics.lie_algebra import exp_map
except ModuleNotFoundError:
    from lie_algebra import exp_map


# ──────────────────────────────────────────────────────────────────────────────
# Primary function
# ──────────────────────────────────────────────────────────────────────────────

def forward_kinematics(q, S_list, M):
    """
    Compute the end-effector pose using the Space-frame Product of Exponentials.

    Formula:
        T(q) = exp([S₁]q₁) · exp([S₂]q₂) · ... · exp([Sₙ]qₙ) · M

    Each term exp([Sᵢ]qᵢ) is the SE(3) rigid-body transform produced by
    rotating/translating joint i by angle qᵢ about/along its screw axis Sᵢ.
    Multiplying them left-to-right chains the transforms from base to
    end-effector. M is right-multiplied last to account for the home pose.

    Parameters
    ----------
    q : array_like, shape (n,)
        Joint angles in radians. n must equal len(S_list).

    S_list : list of array_like, each shape (6,)
        Screw axes [ω̂ᵢ, vᵢ] expressed in the space frame at home config.
        For a revolute joint:
            ωᵢ = unit rotation axis
            vᵢ = -ωᵢ × rᵢ  (rᵢ is any point on the joint axis)
        For a prismatic joint:
            ωᵢ = [0, 0, 0]
            vᵢ = unit translation direction

    M : array_like, shape (4, 4)
        Home configuration — end-effector SE(3) pose when all qᵢ = 0.

    Returns
    -------
    T : np.ndarray, shape (4, 4)
        End-effector pose in SE(3).

    Raises
    ------
    ValueError
        If len(q) != len(S_list) or M is not (4, 4).

    Example
    -------
    >>> # Single revolute joint rotating about z, home at identity
    >>> S = [np.array([0, 0, 1, 0, 0, 0])]
    >>> M = np.eye(4)
    >>> T = forward_kinematics([np.pi/2], S, M)
    >>> # T[:3,:3] should be a 90° rotation about z
    """
    q      = np.asarray(q, dtype=float).flatten()
    M      = np.asarray(M, dtype=float)
    n      = len(S_list)

    # ── Input validation ─────────────────────────────────────────────────────
    if len(q) != n:
        raise ValueError(
            f"len(q)={len(q)} does not match len(S_list)={n}"
        )
    if M.shape != (4, 4):
        raise ValueError(
            f"M must be a (4,4) SE(3) matrix, got shape {M.shape}"
        )
    for i, S in enumerate(S_list):
        S = np.asarray(S, dtype=float).flatten()
        if S.shape != (6,):
            raise ValueError(
                f"S_list[{i}] must be a 6-vector, got shape {S.shape}"
            )

    # ── Product of Exponentials ───────────────────────────────────────────────
    T = np.eye(4)

    for i in range(n):
        Si    = np.asarray(S_list[i], dtype=float).flatten()
        xi_i  = Si * q[i]          # scale screw axis by joint angle
        T     = T @ exp_map(xi_i)  # left-accumulate each transform

    T = T @ M                      # right-multiply home configuration

    return T


# ──────────────────────────────────────────────────────────────────────────────
# Helper: joint positions along the kinematic chain (for visualization)
# ──────────────────────────────────────────────────────────────────────────────

def fk_positions(q, S_list, joint_origins_home):
    """
    Return the 3D position of each joint frame for visualization.

    This is NOT part of the control loop — it is used for plotting the robot
    configuration (stick-figure kinematic chain) during simulation.

    Parameters
    ----------
    q : array_like, shape (n,)
        Joint angles in radians.

    S_list : list of array_like, each shape (6,)
        Screw axes in the space frame (same as forward_kinematics).

    joint_origins_home : array_like, shape (n+1, 3)
        Position of each joint origin (plus end-effector) at home config q=0.
        Row 0    = base frame origin
        Row 1..n = joint i origin
        Row n    = end-effector origin

    Returns
    -------
    positions : np.ndarray, shape (n+1, 3)
        Cartesian positions of each joint frame after applying q.
    """
    q                  = np.asarray(q, dtype=float).flatten()
    joint_origins_home = np.asarray(joint_origins_home, dtype=float)
    n                  = len(S_list)

    positions = np.zeros((n + 1, 3))

    # Base frame origin is fixed
    positions[0] = joint_origins_home[0]

    # Accumulate transforms joint by joint
    T_accum = np.eye(4)
    for i in range(n):
        Si       = np.asarray(S_list[i], dtype=float).flatten()
        xi_i     = Si * q[i]
        T_accum  = T_accum @ exp_map(xi_i)

        # Transform the home position of joint i+1 into the current frame
        p_home   = np.append(joint_origins_home[i + 1], 1.0)  # homogeneous
        positions[i + 1] = (T_accum @ p_home)[:3]

    return positions


# ──────────────────────────────────────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────────────────────────────────────

def _run_validation():
    """Roundtrip and known-answer tests for forward_kinematics."""

    print("=" * 55)
    print("  forward_kinematics.py  —  Validation Suite")
    print("=" * 55)

    tol = 1e-10

    def check(name, result, expected, tol=tol):
        err = np.max(np.abs(result - expected))
        status = "PASS" if err < tol else "FAIL"
        print(f"  [{status}]  {name}  (max_err={err:.2e})")


    # ── Test 1: At q=0, result should equal M exactly ────────────────────────
    # Any M, any screw axes — the product collapses to identity * M = M
    M_test = np.array([
        [1, 0, 0, 0.5],
        [0, 1, 0, 0.2],
        [0, 0, 1, 1.0],
        [0, 0, 0, 1.0],
    ], dtype=float)

    # Arbitrary screw axes (won't matter at q=0)
    S_list_test = [
        np.array([0, 0, 1,  0,  0, 0]),   # joint 1: z-rotation at origin
        np.array([0, 1, 0,  0,  0, 1]),   # joint 2: y-rotation, axis at x=1
        np.array([0, 0, 1, -2,  0, 0]),   # joint 3: z-rotation, axis at y=2
        np.array([0, 1, 0,  0,  0, 2]),   # joint 4: y-rotation, axis at x=2
        np.array([0, 0, 1, -3,  0, 0]),   # joint 5: z-rotation, axis at y=3
        np.array([0, 1, 0,  0,  0, 3]),   # joint 6: y-rotation, axis at x=3
        np.array([0, 0, 1, -4,  0, 0]),   # joint 7: z-rotation, axis at y=4
    ]

    q_zero = np.zeros(7)
    T_zero = forward_kinematics(q_zero, S_list_test, M_test)
    check("FK at q=0 equals M", T_zero, M_test)


    # ── Test 2: Single joint — 90° rotation about z, M = identity ────────────
    # exp([0,0,1, 0,0,0] * π/2) should give a 90° z-rotation with no translation
    S_single = [np.array([0, 0, 1, 0, 0, 0])]
    M_id     = np.eye(4)
    T_90     = forward_kinematics([np.pi / 2], S_single, M_id)

    R_expected = np.array([
        [ 0, -1, 0],
        [ 1,  0, 0],
        [ 0,  0, 1],
    ], dtype=float)
    check("Single joint 90° z-rotation (R block)", T_90[:3, :3], R_expected, tol=1e-10)
    check("Single joint 90° z-rotation (p=0)",     T_90[:3, 3],  np.zeros(3), tol=1e-10)


    # ── Test 3: Single prismatic joint — pure translation along x ─────────────
    S_prismatic = [np.array([0, 0, 0, 1, 0, 0])]   # linear along x, no rotation
    T_trans     = forward_kinematics([2.5], S_prismatic, M_id)

    T_expected = np.eye(4)
    T_expected[0, 3] = 2.5
    check("Single prismatic joint, d=2.5 along x", T_trans, T_expected)


    # ── Test 4: Two joints, known geometry ────────────────────────────────────
    # Joint 1: rotate about z at origin
    # Joint 2: rotate about z, axis at x=1
    # Home config: end-effector at [2, 0, 0]
    # At q1=π/2, q2=0: end-effector should be at [0, 1, 0]... wait, let me think.
    #
    # S1 = [0,0,1, 0,0,0]   → rotation about z at origin
    # S2 = [0,0,1, 0,-1,0]  → rotation about z, passing through x=1
    #                          v = -ω × r = -[0,0,1]×[1,0,0] = -[0,1,0] = [0,-1,0]
    # M  = [[1,0,0,2],[0,1,0,0],[0,0,1,0],[0,0,0,1]]  (end-effector at x=2)
    #
    # At q1=π/2, q2=0:
    #   T = exp([S1]*π/2) · I · M
    #   exp([S1]*π/2) rotates everything by 90° about z:
    #   p_ee = R_90 · [2,0,0] = [0,2,0]  → end-effector at y=2

    S1 = np.array([0, 0, 1,  0,  0, 0])
    S2 = np.array([0, 0, 1,  0, -1, 0])
    M2 = np.eye(4); M2[0, 3] = 2.0

    T_2joint = forward_kinematics([np.pi/2, 0.0], [S1, S2], M2)
    p_expected_2joint = np.array([0.0, 2.0, 0.0])
    check("Two joints, q=[π/2, 0]: end-effector position", T_2joint[:3, 3], p_expected_2joint, tol=1e-10)


    # ── Test 5: Full 7-joint chain — FK then verify SE(3) structure ───────────
    # T must be a valid SE(3) element:
    #   R should be orthogonal: Rᵀ R = I
    #   det(R) = +1
    #   Bottom row = [0, 0, 0, 1]

    q_rand = np.array([0.3, -0.5, 0.8, -0.2, 1.1, -0.7, 0.4])
    T_rand = forward_kinematics(q_rand, S_list_test, M_test)

    R_rand = T_rand[:3, :3]
    check("FK result: Rᵀ R = I (orthogonality)", R_rand.T @ R_rand, np.eye(3), tol=1e-12)

    det_R = np.linalg.det(R_rand)
    check("FK result: det(R) = 1", np.array([[det_R]]), np.array([[1.0]]), tol=1e-12)

    bottom_row_expected = np.array([[0.0, 0.0, 0.0, 1.0]])
    check("FK result: bottom row = [0,0,0,1]", T_rand[3:, :], bottom_row_expected, tol=1e-15)

    print("=" * 55)


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    _run_validation()
