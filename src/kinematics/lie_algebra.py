"""
lie_algebra.py
==============
Lie algebra utilities for SE(3) / se(3) operations.

Implements the core mathematical building blocks for the local tangent-space
mapping framework for algorithmic singularity avoidance in redundant manipulators.

Functions (in dependency order):
    hat3         : R^3  -> so(3)   skew-symmetric matrix
    vee3         : so(3) -> R^3    inverse of hat3
    hat6         : R^6  -> se(3)   4x4 twist matrix
    vee6         : se(3) -> R^6    inverse of hat6
    exp_map      : se(3) -> SE(3)  matrix exponential (Rodrigues)
    log_map_se3  : SE(3) -> se(3)  matrix logarithm  ← core of the framework
    adjoint      : SE(3) -> R^6x6  adjoint representation

Convention (Lynch & Park):
    Twist ξ = [ω, v]  where ω ∈ R^3 is angular, v ∈ R^3 is linear
    Body Jacobian columns live in se(3), re-expressed in end-effector frame
"""

import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Numerical thresholds
# ──────────────────────────────────────────────────────────────────────────────

_EPS       = 1e-10   # near-zero threshold for angle checks
_PI_NEAR   = 1e-7    # threshold for θ ≈ π (log map special case)


# ──────────────────────────────────────────────────────────────────────────────
# 1. hat3 — R^3 → so(3)
# ──────────────────────────────────────────────────────────────────────────────

def hat3(omega):
    """
    Convert a 3D angular velocity vector into its 3x3 skew-symmetric matrix.

    For any vector v, the cross product ω × v equals hat3(ω) @ v.
    This is the Lie algebra map for SO(3).

    Parameters
    ----------
    omega : array_like, shape (3,)
        Angular velocity vector [ω₁, ω₂, ω₃].

    Returns
    -------
    Omega : np.ndarray, shape (3, 3)
        Skew-symmetric matrix representation.

    Example
    -------
    >>> hat3([1, 2, 3])
    array([[ 0, -3,  2],
           [ 3,  0, -1],
           [-2,  1,  0]])
    """
    omega = np.asarray(omega, dtype=float).flatten()
    if omega.shape != (3,):
        raise ValueError(f"hat3 expects a 3-vector, got shape {omega.shape}")

    w1, w2, w3 = omega

    return np.array([
        [ 0.0, -w3,  w2],
        [ w3,  0.0, -w1],
        [-w2,   w1,  0.0]
    ])


# ──────────────────────────────────────────────────────────────────────────────
# 2. vee3 — so(3) → R^3  (inverse of hat3)
# ──────────────────────────────────────────────────────────────────────────────

def vee3(Omega):
    """
    Extract the 3D angular velocity vector from a 3x3 skew-symmetric matrix.

    Inverse of hat3: vee3(hat3(ω)) == ω.

    Parameters
    ----------
    Omega : array_like, shape (3, 3)
        Skew-symmetric matrix.

    Returns
    -------
    omega : np.ndarray, shape (3,)
        Angular velocity vector [ω₁, ω₂, ω₃].
    """
    Omega = np.asarray(Omega, dtype=float)
    if Omega.shape != (3, 3):
        raise ValueError(f"vee3 expects a (3,3) matrix, got shape {Omega.shape}")

    return np.array([
        Omega[2, 1],   # ω₁
        Omega[0, 2],   # ω₂
        Omega[1, 0],   # ω₃
    ])


# ──────────────────────────────────────────────────────────────────────────────
# 3. hat6 — R^6 → se(3)
# ──────────────────────────────────────────────────────────────────────────────

def hat6(xi):
    """
    Convert a 6D body twist into its 4x4 se(3) matrix representation.

    Parameters
    ----------
    xi : array_like, shape (6,)
        Body twist [ω₁, ω₂, ω₃, v₁, v₂, v₃].
        First 3 entries: angular velocity ω.
        Last  3 entries: linear velocity  v.

    Returns
    -------
    X : np.ndarray, shape (4, 4)
        se(3) matrix:
            [ hat3(ω)  v ]
            [  0  0  0  0 ]

    Notes
    -----
    This is a Lie algebra element. Its matrix exponential gives an SE(3) pose.
    """
    xi = np.asarray(xi, dtype=float).flatten()
    if xi.shape != (6,):
        raise ValueError(f"hat6 expects a 6-vector, got shape {xi.shape}")

    omega = xi[:3]
    v     = xi[3:]

    X = np.zeros((4, 4))
    X[:3, :3] = hat3(omega)
    X[:3,  3] = v

    return X


# ──────────────────────────────────────────────────────────────────────────────
# 4. vee6 — se(3) → R^6  (inverse of hat6)
# ──────────────────────────────────────────────────────────────────────────────

def vee6(X):
    """
    Extract the 6D body twist from a 4x4 se(3) matrix.

    Inverse of hat6: vee6(hat6(ξ)) == ξ.

    Parameters
    ----------
    X : array_like, shape (4, 4)
        se(3) Lie algebra element.

    Returns
    -------
    xi : np.ndarray, shape (6,)
        Body twist [ω₁, ω₂, ω₃, v₁, v₂, v₃].
    """
    X = np.asarray(X, dtype=float)
    if X.shape != (4, 4):
        raise ValueError(f"vee6 expects a (4,4) matrix, got shape {X.shape}")

    omega = vee3(X[:3, :3])   # extract angular part from skew block
    v     = X[:3, 3]          # extract linear part from top-right column

    return np.concatenate([omega, v])


# ──────────────────────────────────────────────────────────────────────────────
# 5. exp_map — se(3) → SE(3)
# ──────────────────────────────────────────────────────────────────────────────

def exp_map(xi):
    """
    Compute the matrix exponential of a 6D twist, mapping se(3) → SE(3).

    Uses Rodrigues' formula for the rotation and the corresponding
    closed-form G matrix for the translation.

    Parameters
    ----------
    xi : array_like, shape (6,)
        Body twist [ω, v]. Does NOT need to be unit-normalized;
        the rotation angle is encoded in norm(ω).

    Returns
    -------
    T : np.ndarray, shape (4, 4)
        SE(3) homogeneous transformation matrix.

    Notes
    -----
    G matrix (Chirikjian / Lynch & Park notation):
        G(θ) = I·θ + (1 - cosθ)·Ω̂ + (θ - sinθ)·Ω̂²
    such that p = G(θ) · v_unit recovers the translation vector.
    """
    xi = np.asarray(xi, dtype=float).flatten()
    if xi.shape != (6,):
        raise ValueError(f"exp_map expects a 6-vector, got shape {xi.shape}")

    omega = xi[:3]
    v     = xi[3:]

    theta = np.linalg.norm(omega)

    T = np.eye(4)

    # ── Case 1: Pure translation (no rotation) ──────────────────────────────
    if theta < _EPS:
        T[:3, 3] = v
        return T

    # ── Case 2: Has rotation ─────────────────────────────────────────────────
    omega_unit = omega / theta
    v_unit     = v     / theta

    Omega = hat3(omega_unit)   # unit skew matrix
    Omega2 = Omega @ Omega     # Ω² used in both R and G

    # Rodrigues' rotation formula
    R = (np.eye(3)
         + np.sin(theta)       * Omega
         + (1 - np.cos(theta)) * Omega2)

    # G matrix: maps unit linear velocity to displacement
    G = (np.eye(3) * theta
         + (1 - np.cos(theta)) * Omega
         + (theta - np.sin(theta)) * Omega2)

    p = G @ v_unit

    T[:3, :3] = R
    T[:3,  3] = p

    return T


# ──────────────────────────────────────────────────────────────────────────────
# 6. log_map_se3 — SE(3) → se(3)   ← core of the framework
# ──────────────────────────────────────────────────────────────────────────────

def log_map_se3(T):
    """
    Compute the matrix logarithm of an SE(3) pose, mapping SE(3) → se(3).

    This is the central operation of the local tangent-space framework.
    Given the pose error T_err = T_cur⁻¹ @ T_des, this function returns
    the 6D body twist ξ_err that represents the error in the local tangent
    plane at the current configuration.

    Parameters
    ----------
    T : array_like, shape (4, 4)
        SE(3) homogeneous transformation matrix.
        Typically T_err = T_cur_inv @ T_des.

    Returns
    -------
    xi : np.ndarray, shape (6,)
        Body twist [ω, v] such that exp_map(xi) ≈ T.
        This is fed directly into the pseudoinverse solver as the
        pose error in the control loop.

    Notes
    -----
    Three cases handled:

        Case 1 — θ ≈ 0   (near-identity, no rotation):
            ω = 0,  v = p directly.

        Case 2 — θ ≈ π   (180° rotation, sin(θ) → 0):
            Standard formula is numerically unstable.
            Use diagonal of (R + I) to recover the rotation axis.

        Case 3 — General (0 < θ < π):
            Recover Ω̂_unit from (R - Rᵀ) / (2 sinθ).
            Invert the G matrix analytically to recover v.

    G_inv formula (Chirikjian):
        G⁻¹ = (1/θ)·I  -  (1/2)·Ω̂  +  (1/θ - (1/2)·cot(θ/2))·Ω̂²
    """
    T = np.asarray(T, dtype=float)
    if T.shape != (4, 4):
        raise ValueError(f"log_map_se3 expects a (4,4) matrix, got shape {T.shape}")

    R = T[:3, :3]
    p = T[:3,  3]

    # Rotation angle from the trace formula
    # Clamp to [-1, 1] to guard against floating-point errors outside arccos domain
    cos_theta = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(cos_theta)

    # ── Case 1: Near-zero rotation ───────────────────────────────────────────
    if theta < _EPS:
        omega = np.zeros(3)
        v     = p.copy()
        return np.concatenate([omega, v])

    # ── Case 2: Near-180° rotation (sin(θ) → 0, standard formula unstable) ──
    if np.pi - theta < _PI_NEAR:
        # Recover rotation axis from symmetric part of R
        # (R + I)/2 has columns proportional to the eigenvector for eigenvalue +1
        S = (R + np.eye(3)) / 2.0

        # Find the column with the largest diagonal entry — most numerically stable
        idx = np.argmax(np.diag(S))
        axis = S[:, idx]

        # Normalize to get unit axis, then scale by θ
        axis_norm = np.linalg.norm(axis)
        if axis_norm < _EPS:
            # Degenerate: R = -I, any axis works (pick x-axis)
            omega_unit = np.array([1.0, 0.0, 0.0])
        else:
            omega_unit = axis / axis_norm

        omega = theta * omega_unit
        Omega_unit = hat3(omega_unit)

        # G maps unit linear velocity (v/θ) to displacement p.
        # So G_inv @ p recovers v/θ — multiply by θ to get v.
        G_inv = _g_inv_matrix(theta, Omega_unit)
        v = theta * (G_inv @ p)

        return np.concatenate([omega, v])

    # ── Case 3: General rotation ─────────────────────────────────────────────

    # Recover unit skew matrix from the antisymmetric part of R
    Omega_unit = (R - R.T) / (2.0 * np.sin(theta))

    # Extract the angular velocity vector
    omega_unit = vee3(Omega_unit)
    omega      = theta * omega_unit

    # G maps unit linear velocity (v/θ) to displacement p.
    # So G_inv @ p recovers v/θ — multiply by θ to get v.
    G_inv = _g_inv_matrix(theta, Omega_unit)
    v     = theta * (G_inv @ p)

    return np.concatenate([omega, v])


def _g_inv_matrix(theta, Omega_unit):
    """
    Compute the inverse of the G matrix analytically.

    G⁻¹(θ) = (1/θ)·I  −  (1/2)·Ω̂  +  (1/θ − (1/2)·cot(θ/2))·Ω̂²

    This maps the position vector p back to the linear velocity v
    such that G(θ)·v = p  →  v = G_inv·p.

    Parameters
    ----------
    theta      : float     rotation angle (radians)
    Omega_unit : np.ndarray, shape (3,3)  unit skew-symmetric matrix

    Returns
    -------
    G_inv : np.ndarray, shape (3, 3)
    """
    Omega2 = Omega_unit @ Omega_unit

    # cot(θ/2) = cos(θ/2) / sin(θ/2)
    cot_half = np.cos(theta / 2.0) / np.sin(theta / 2.0)

    G_inv = ((1.0 / theta)         * np.eye(3)
             - 0.5                 * Omega_unit
             + (1.0/theta - 0.5 * cot_half) * Omega2)

    return G_inv


# ──────────────────────────────────────────────────────────────────────────────
# 7. adjoint — SE(3) → R^{6×6}
# ──────────────────────────────────────────────────────────────────────────────

def adjoint(T):
    """
    Compute the 6x6 Adjoint representation of an SE(3) transformation.

    The Adjoint is used in body_jacobian.py to re-express each joint's
    screw axis (defined in its own local frame) into the end-effector
    body frame, which is the common frame for the body Jacobian.

    For a twist ξ in frame A, the same twist in frame B is:
        ξ_B = Ad(T_{B←A}) · ξ_A

    Parameters
    ----------
    T : array_like, shape (4, 4)
        SE(3) homogeneous transformation matrix.

    Returns
    -------
    Ad_T : np.ndarray, shape (6, 6)
        6x6 Adjoint matrix:
            [ R        0 ]
            [ p̂ · R   R ]
        where p̂ = hat3(p) is the skew matrix of the position vector.

    Notes
    -----
    The block structure places angular components in the top half and
    linear components in the bottom half, consistent with the
    [ω, v] twist convention used throughout this module.
    """
    T = np.asarray(T, dtype=float)
    if T.shape != (4, 4):
        raise ValueError(f"adjoint expects a (4,4) matrix, got shape {T.shape}")

    R = T[:3, :3]
    p = T[:3,  3]

    p_hat = hat3(p)

    Ad_T = np.zeros((6, 6))
    Ad_T[:3, :3] = R
    Ad_T[3:, :3] = p_hat @ R
    Ad_T[3:, 3:] = R

    return Ad_T


# ──────────────────────────────────────────────────────────────────────────────
# Validation — roundtrip tests
# ──────────────────────────────────────────────────────────────────────────────

def _run_validation():
    """
    Run roundtrip consistency checks on all functions.
    Prints PASS / FAIL for each test.
    """
    print("=" * 55)
    print("  lie_algebra.py  —  Validation Suite")
    print("=" * 55)

    tol = 1e-10

    def check(name, result, expected, tol=tol):
        err = np.max(np.abs(result - expected))
        status = "PASS" if err < tol else "FAIL"
        print(f"  [{status}]  {name}  (max_err={err:.2e})")


    # ── Test 1: hat3 / vee3 roundtrip ───────────────────────────────────────
    omega = np.array([1.0, 2.0, 3.0])
    check("hat3 / vee3 roundtrip", vee3(hat3(omega)), omega)

    # ── Test 2: hat6 / vee6 roundtrip ───────────────────────────────────────
    xi = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    check("hat6 / vee6 roundtrip", vee6(hat6(xi)), xi)

    # ── Test 3: exp_map → pure translation (no rotation) ────────────────────
    xi_trans = np.array([0.0, 0.0, 0.0, 1.0, 2.0, 3.0])
    T_trans  = exp_map(xi_trans)
    expected_trans = np.eye(4)
    expected_trans[:3, 3] = [1.0, 2.0, 3.0]
    check("exp_map pure translation", T_trans, expected_trans)

    # ── Test 4: exp_map → pure rotation about z (90°) ───────────────────────
    angle = np.pi / 2
    xi_rot = np.array([0.0, 0.0, angle, 0.0, 0.0, 0.0])
    T_rot  = exp_map(xi_rot)
    R_expected = np.array([
        [ 0.0, -1.0,  0.0],
        [ 1.0,  0.0,  0.0],
        [ 0.0,  0.0,  1.0]
    ])
    check("exp_map pure rotation (90° about z)",
          T_rot[:3, :3], R_expected, tol=1e-10)

    # ── Test 5: exp_map / log_map_se3 roundtrip (general twist) ─────────────
    xi_gen = np.array([0.1, -0.2, 0.3, 0.5, -0.4, 0.2])
    T_gen  = exp_map(xi_gen)
    xi_rec = log_map_se3(T_gen)
    check("exp / log roundtrip (general twist)", xi_rec, xi_gen)

    # ── Test 6: log_map_se3 → identity gives zero twist ─────────────────────
    xi_ident = log_map_se3(np.eye(4))
    check("log_map_se3 of identity = zero", xi_ident, np.zeros(6))

    # ── Test 7: exp / log roundtrip near θ = π ──────────────────────────────
    xi_pi = np.array([np.pi - 0.01, 0.0, 0.0, 0.1, 0.0, 0.0])
    T_pi  = exp_map(xi_pi)
    xi_pi_rec = log_map_se3(T_pi)
    check("exp / log roundtrip (θ near π)", xi_pi_rec, xi_pi, tol=1e-7)

    # ── Test 8: adjoint invertibility ────────────────────────────────────────
    T_adj = exp_map(np.array([0.3, -0.1, 0.2, 1.0, -0.5, 0.8]))
    Ad    = adjoint(T_adj)
    Ad_inv = adjoint(np.linalg.inv(T_adj))
    product = Ad @ Ad_inv
    check("adjoint · adjoint_inv = I (6x6)", product, np.eye(6))

    # ── Test 9: adjoint twist transformation ─────────────────────────────────
    # A twist in one frame should have the same physical meaning after
    # re-expression in another frame via the adjoint
    xi_in  = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])  # pure z-rotation
    T_id   = np.eye(4)
    Ad_id  = adjoint(T_id)
    xi_out = Ad_id @ xi_in
    check("adjoint of identity is identity map", xi_out, xi_in)

    print("=" * 55)


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    _run_validation()