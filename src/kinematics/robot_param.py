"""
robot_params.py
===============
Robot configuration definitions for the 7-DOF manipulator framework.

This file is where all physical robot measurements live — screw axes,
home configurations, joint limits, and link geometry. The rest of the
framework (FK, Jacobian, control) imports from here and never hard-codes
robot-specific numbers.

Two configurations are provided:

    GENERIC_7DOF
        A clean, symmetric 7-DOF chain with round-number link lengths.
        Good for initial testing, debugging, and paper illustrations
        where you want a tractable geometry to reason about.

    IIWA7_R800
        KUKA LBR iiwa 7 R800 parameters from the published datasheet and
        verified DH chain (d_BS=0.340, d_SE=0.400, d_EW=0.400, d_FL=0.126).
        Joint pair origins (1,2), (3,4), (5,6) are collocated — no lateral
        offsets. Joint limits from the official KUKA axis data table.

Usage
-----
    from robot_params import get_robot

    robot = get_robot("generic_7dof")   # or "iiwa7_r800"

    T = forward_kinematics(q, robot["S_list"], robot["M"])

Structure of each config dict
------------------------------
    name              str         human-readable label
    n_joints          int         number of joints (7)
    S_list            list[ndarray(6,)]   space-frame screw axes at home
    M                 ndarray(4,4)        home configuration SE(3)
    joint_limits      ndarray(7,2)        [q_min, q_max] per joint (radians)
    joint_origins     ndarray(8,3)        base + 7 joint origins + EE (for viz)
    link_lengths      dict                named physical dimensions (metres)
"""

import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Helper: build a screw axis from geometry
# ──────────────────────────────────────────────────────────────────────────────

def make_screw(w, r):
    """
    Build a 6D space-frame screw axis from a rotation axis and a point on it.

    For a revolute joint:
        S = [ω̂, v]   where   v = -ω̂ × r

    Parameters
    ----------
    w : array_like, shape (3,)
        Unit rotation axis direction in the space frame.
        For prismatic joints pass [0, 0, 0] and set r to the
        translation direction (place it in v directly instead).

    r : array_like, shape (3,)
        Any point on the joint's axis, expressed in the space frame
        at the home configuration (all joints at q=0).

    Returns
    -------
    S : np.ndarray, shape (6,)
        Space-frame screw axis [ω̂, v].
    """
    w = np.asarray(w, dtype=float)
    r = np.asarray(r, dtype=float)
    v = -np.cross(w, r)
    return np.concatenate([w, v])


def make_rotation_matrix(axis, angle_deg):
    """
    Rodrigues rotation: rotate `angle_deg` about `axis` (unit vector).
    Used to build home configurations with non-trivial orientations.
    """
    axis  = np.asarray(axis, dtype=float)
    axis  = axis / np.linalg.norm(axis)
    theta = np.deg2rad(angle_deg)
    K     = np.array([
        [0,        -axis[2],  axis[1]],
        [axis[2],   0,       -axis[0]],
        [-axis[1],  axis[0],  0      ]
    ])
    return np.eye(3) + np.sin(theta)*K + (1 - np.cos(theta))*(K @ K)


def make_T(R, p):
    """Assemble a 4×4 SE(3) matrix from R (3×3) and p (3,)."""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3,  3] = p
    return T


# ──────────────────────────────────────────────────────────────────────────────
# Configuration 1 — GENERIC_7DOF
# ──────────────────────────────────────────────────────────────────────────────
#
# Geometry (all in metres, space frame at home config q = 0):
#
#   Joint   Axis    Origin          Notes
#   ─────   ────    ──────          ─────
#     1      +z     [0, 0, 0]       base, shoulder yaw
#     2      +y     [0, 0, 0.40]    shoulder pitch
#     3      +z     [0, 0, 0.72]    upper-arm roll
#     4      +y     [0, 0, 0.72]    elbow pitch
#     5      +z     [0, 0, 1.08]    forearm roll
#     6      +y     [0, 0, 1.08]    wrist pitch
#     7      +z     [0, 0, 1.08]    wrist roll
#
#   End-effector at home:  [0, 0, 1.26]  pointing up (+z)
#
# Link lengths (all 0.36m except the first segment at 0.40m):
#   d1 = 0.40   base  → joint 2
#   d2 = 0.32   joint 2 → joint 3
#   d3 = 0.36   joint 3 → joint 5
#   d4 = 0.18   joint 5 → EE
#
# This is intentionally symmetric and axis-aligned — easy to verify by
# hand and useful for debugging the Jacobian and control loop.

def _build_generic_7dof():
    # ── Joint origins in the space frame at q = 0 ───────────────────────────
    base = np.array([0.0, 0.0, 0.0])
    r1   = np.array([0.0, 0.0, 0.0])
    r2   = np.array([0.0, 0.0, 0.40])
    r3   = np.array([0.0, 0.0, 0.72])
    r4   = np.array([0.0, 0.0, 0.72])
    r5   = np.array([0.0, 0.0, 1.08])
    r6   = np.array([0.0, 0.0, 1.08])
    r7   = np.array([0.0, 0.0, 1.08])
    p_ee = np.array([0.0, 0.0, 1.26])

    # ── Rotation axes ────────────────────────────────────────────────────────
    z =  np.array([0.0, 0.0,  1.0])
    y =  np.array([0.0, 1.0,  0.0])

    # ── Screw axes  S = [ω̂, -ω̂ × r] ────────────────────────────────────────
    S_list = [
        make_screw(z, r1),   # joint 1 — shoulder yaw
        make_screw(y, r2),   # joint 2 — shoulder pitch
        make_screw(z, r3),   # joint 3 — upper-arm roll
        make_screw(y, r4),   # joint 4 — elbow pitch
        make_screw(z, r5),   # joint 5 — forearm roll
        make_screw(y, r6),   # joint 6 — wrist pitch
        make_screw(z, r7),   # joint 7 — wrist roll
    ]

    # ── Home configuration M — end-effector at p_ee, pointing up (+z) ───────
    M = make_T(np.eye(3), p_ee)

    # ── Joint limits (radians) ───────────────────────────────────────────────
    # Symmetric ±2π/3 (~120°) for all joints — conservative generic limits
    limit = 2 * np.pi / 3
    joint_limits = np.array([[-limit, limit]] * 7)

    # ── Joint origins for visualization (base + 7 joints + EE) ──────────────
    joint_origins = np.array([base, r1, r2, r3, r4, r5, r6, r7, p_ee])

    return {
        "name"          : "generic_7dof",
        "n_joints"      : 7,
        "S_list"        : S_list,
        "M"             : M,
        "joint_limits"  : joint_limits,
        "joint_origins" : joint_origins,
        "link_lengths"  : {
            "d1" : 0.40,   # base → joint 2
            "d2" : 0.32,   # joint 2 → joint 3
            "d3" : 0.36,   # joint 3 → joint 5
            "d4" : 0.18,   # joint 7 → end-effector
        },
    }


# ──────────────────────────────────────────────────────────────────────────────
# Configuration 2 — IIWA7_R800
# ──────────────────────────────────────────────────────────────────────────────
#
# KUKA LBR iiwa 7 R800 parameters.
#
# Source: KUKA LBR iiwa datasheet + DH chain verified against SmartPad
# cartesian coordinates (robot-forum.com, confirmed by s_visual_kinematics
# and kuka-isir/iiwa_description URDF).
#
# Physical dimensions (metres):
#   d_BS = 0.340   base mounting plane → shoulder joint pair (1,2)
#   d_SE = 0.400   shoulder → elbow joint pair (3,4)   ["R800": 400+400=800mm]
#   d_EW = 0.400   elbow → wrist joint pair (5,6)
#   d_FL = 0.126   wrist → tool flange (joint 7)
#   All lateral offsets = 0  (all joint axes pass through the robot centerline)
#
# Key structural difference from Panda:
#   Joint PAIRS share the same spatial origin — no lateral offsets anywhere.
#   (j1,j2) collocated at z=0.340, (j3,j4) at z=0.740, (j5,j6) at z=1.140
#   This gives the iiwa a clean spherical-pair structure at shoulder/elbow/wrist.
#
# Joint axes at home (pure alternating z/y, all on the centerline):
#   Joint  Axis   Origin          Description
#   ─────  ────   ──────          ───────────
#     1     +z    [0, 0, 0.340]   shoulder yaw
#     2     +y    [0, 0, 0.340]   shoulder pitch  (same origin as j1)
#     3     +z    [0, 0, 0.740]   upper-arm roll
#     4     +y    [0, 0, 0.740]   elbow pitch     (same origin as j3)
#     5     +z    [0, 0, 1.140]   forearm roll
#     6     +y    [0, 0, 1.140]   wrist pitch     (same origin as j5)
#     7     +z    [0, 0, 1.266]   wrist roll
#
# Joint limits (radians) — from KUKA iiwa 7 R800 axis data table:
#   A1: ±170°,  A2: ±120°,  A3: ±170°,  A4: ±120°
#   A5: ±170°,  A6: ±120°,  A7: ±175°
#   All limits are symmetric (unlike Panda which has asymmetric limits).

def _build_iiwa7_r800():
    # ── Physical dimensions (metres) ─────────────────────────────────────────
    d_BS = 0.340   # base → shoulder pair
    d_SE = 0.400   # shoulder → elbow pair
    d_EW = 0.400   # elbow → wrist pair
    d_FL = 0.126   # wrist pair → tool flange

    # ── Cumulative z-heights in the space frame at q = 0 ────────────────────
    z_shoulder = d_BS                       # 0.340  joints 1 & 2
    z_elbow    = d_BS + d_SE               # 0.740  joints 3 & 4
    z_wrist    = d_BS + d_SE + d_EW        # 1.140  joints 5 & 6
    z_flange   = d_BS + d_SE + d_EW + d_FL # 1.266  joint 7 / EE

    # ── Joint origins (all on the centerline — no x or y offsets) ───────────
    base = np.array([0.0, 0.0, 0.0])
    r1   = np.array([0.0, 0.0, z_shoulder])   # shoulder yaw
    r2   = np.array([0.0, 0.0, z_shoulder])   # shoulder pitch  — same as r1
    r3   = np.array([0.0, 0.0, z_elbow   ])   # upper-arm roll
    r4   = np.array([0.0, 0.0, z_elbow   ])   # elbow pitch     — same as r3
    r5   = np.array([0.0, 0.0, z_wrist   ])   # forearm roll
    r6   = np.array([0.0, 0.0, z_wrist   ])   # wrist pitch     — same as r5
    r7   = np.array([0.0, 0.0, z_flange  ])   # wrist roll / flange
    p_ee = np.array([0.0, 0.0, z_flange  ])   # end-effector at flange

    # ── Rotation axes ────────────────────────────────────────────────────────
    z = np.array([0.0, 0.0, 1.0])
    y = np.array([0.0, 1.0, 0.0])

    # ── Screw axes  S = [ω̂, v]  where  v = -ω̂ × r ──────────────────────────
    #
    # Note: for any joint whose axis lies on the z-axis (r has no x,y component),
    # the z-rotation screws always produce v = 0 since -z × [0,0,h] = 0.
    # The y-rotation screws produce v = [-h, 0, 0] since -y × [0,0,h] = [-h,0,0].
    # This gives a particularly clean screw axis set.
    S_list = [
        make_screw(z, r1),   # joint 1 — shoulder yaw    → [0,0,1,  0,    0, 0]
        make_screw(y, r2),   # joint 2 — shoulder pitch  → [0,1,0, -0.340, 0, 0]
        make_screw(z, r3),   # joint 3 — upper-arm roll  → [0,0,1,  0,    0, 0]
        make_screw(y, r4),   # joint 4 — elbow pitch     → [0,1,0, -0.740, 0, 0]
        make_screw(z, r5),   # joint 5 — forearm roll    → [0,0,1,  0,    0, 0]
        make_screw(y, r6),   # joint 6 — wrist pitch     → [0,1,0, -1.140, 0, 0]
        make_screw(z, r7),   # joint 7 — wrist roll      → [0,0,1,  0,    0, 0]
    ]

    # ── Home configuration M — EE at flange, pointing up (+z) ────────────────
    # At q=0 the iiwa stands fully upright; EE is directly above the base.
    # For hardware use, verify this orientation against the SmartPad reading.
    M = make_T(np.eye(3), p_ee)

    # ── Joint limits (radians) — from KUKA iiwa 7 R800 axis data table ───────
    deg = np.deg2rad
    joint_limits = np.array([
        [-deg(170),  deg(170)],   # A1 — shoulder yaw
        [-deg(120),  deg(120)],   # A2 — shoulder pitch
        [-deg(170),  deg(170)],   # A3 — upper-arm roll
        [-deg(120),  deg(120)],   # A4 — elbow pitch
        [-deg(170),  deg(170)],   # A5 — forearm roll
        [-deg(120),  deg(120)],   # A6 — wrist pitch
        [-deg(175),  deg(175)],   # A7 — wrist roll
    ])

    # ── Joint origins for visualization (base + 7 joints + EE = 9 rows) ─────
    joint_origins = np.array([base, r1, r2, r3, r4, r5, r6, r7, p_ee])

    return {
        "name"          : "iiwa7_r800",
        "n_joints"      : 7,
        "S_list"        : S_list,
        "M"             : M,
        "joint_limits"  : joint_limits,
        "joint_origins" : joint_origins,
        "link_lengths"  : {
            "d_BS" : d_BS,   # base → shoulder
            "d_SE" : d_SE,   # shoulder → elbow
            "d_EW" : d_EW,   # elbow → wrist
            "d_FL" : d_FL,   # wrist → flange
        },
    }


# ──────────────────────────────────────────────────────────────────────────────
# Registry and factory
# ──────────────────────────────────────────────────────────────────────────────

# Build configs once at import time
GENERIC_7DOF  = _build_generic_7dof()
IIWA7_R800    = _build_iiwa7_r800()

_REGISTRY = {
    "generic_7dof" : GENERIC_7DOF,
    "iiwa7_r800"   : IIWA7_R800,
}


def get_robot(name="generic_7dof"):
    """
    Retrieve a robot configuration dict by name.

    Parameters
    ----------
    name : str
        "generic_7dof"  — clean symmetric chain (default, good for debugging)
        "iiwa7_r800"    — KUKA LBR iiwa 7 R800

    Returns
    -------
    config : dict with keys:
        name, n_joints, S_list, M, joint_limits, joint_origins, link_lengths

    Example
    -------
    >>> robot = get_robot("iiwa7_r800")
    >>> T = forward_kinematics(q, robot["S_list"], robot["M"])
    """
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown robot '{name}'. Available: {list(_REGISTRY.keys())}"
        )
    return _REGISTRY[name]


def list_robots():
    """Print all available robot configurations."""
    print("Available robot configurations:")
    for name, cfg in _REGISTRY.items():
        ll = cfg["link_lengths"]
        print(f"  {name:20s}  —  {cfg['n_joints']} joints, "
              f"link_lengths={ll}")


# ──────────────────────────────────────────────────────────────────────────────
# Optional: stub for loading from a real URDF file
# ──────────────────────────────────────────────────────────────────────────────

def load_from_urdf(urdf_path):
    """
    [STUB] Load robot parameters from a URDF file.

    This function is a placeholder. To implement it, install the
    `urdfpy` or `yourdfpy` library and parse joint origins/axes from
    the URDF XML, then pass them through make_screw() to build S_list.

        pip install yourdfpy

    Parameters
    ----------
    urdf_path : str
        Path to the .urdf file.

    Returns
    -------
    config : dict  (same structure as get_robot())
    """
    raise NotImplementedError(
        "load_from_urdf() is not yet implemented.\n"
        "Install 'yourdfpy' and parse <joint> origin/axis tags,\n"
        "then call make_screw(axis, origin) for each revolute joint."
    )


# ──────────────────────────────────────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────────────────────────────────────

def _run_validation():
    print("=" * 55)
    print("  robot_params.py  —  Validation Suite")
    print("=" * 55)

    tol = 1e-10

    def check(name, result, expected, tol=tol):
        err = np.max(np.abs(np.asarray(result) - np.asarray(expected)))
        status = "PASS" if err < tol else "FAIL"
        print(f"  [{status}]  {name}  (max_err={err:.2e})")

    for robot_name in ["generic_7dof", "iiwa7_r800"]:
        print(f"\n  Robot: {robot_name}")
        robot = get_robot(robot_name)

        # ── Each S_list entry must be a 6-vector ─────────────────────────────
        all_shape_ok = all(
            np.asarray(S).shape == (6,) for S in robot["S_list"]
        )
        print(f"  [{'PASS' if all_shape_ok else 'FAIL'}]  All screw axes are 6-vectors")

        # ── Each rotation axis must be unit length (or zero for prismatic) ───
        all_unit = all(
            abs(np.linalg.norm(np.asarray(S)[:3]) - 1.0) < 1e-10
            for S in robot["S_list"]
        )
        print(f"  [{'PASS' if all_unit else 'FAIL'}]  All rotation axes are unit vectors")

        # ── M must be a valid SE(3) element ──────────────────────────────────
        M = robot["M"]
        R = M[:3, :3]
        RtR_err  = np.max(np.abs(R.T @ R - np.eye(3)))
        det_err  = abs(np.linalg.det(R) - 1.0)
        row_err  = np.max(np.abs(M[3] - [0, 0, 0, 1]))
        check("M: Rᵀ R = I",          RtR_err,  0.0, tol=1e-12)
        check("M: det(R) = 1",         det_err,  0.0, tol=1e-12)
        check("M: bottom row [0,0,0,1]", row_err, 0.0, tol=1e-15)

        # ── Joint limits must be (7, 2) with min < max ────────────────────────
        lim = robot["joint_limits"]
        shape_ok  = lim.shape == (7, 2)
        order_ok  = bool(np.all(lim[:, 0] < lim[:, 1]))
        print(f"  [{'PASS' if shape_ok else 'FAIL'}]  joint_limits shape (7,2)")
        print(f"  [{'PASS' if order_ok else 'FAIL'}]  joint_limits: min < max for all joints")

        # ── n_joints consistency ──────────────────────────────────────────────
        n_ok = (robot["n_joints"] == 7 == len(robot["S_list"]))
        print(f"  [{'PASS' if n_ok else 'FAIL'}]  n_joints == len(S_list) == 7")

    # ── get_robot raises on unknown name ─────────────────────────────────────
    try:
        get_robot("nonexistent_robot")
        print("  [FAIL]  get_robot raises ValueError for unknown name")
    except ValueError:
        print("\n  [PASS]  get_robot raises ValueError for unknown name")

    print("\n" + "=" * 55)


if __name__ == "__main__":
    list_robots()
    print()
    _run_validation()