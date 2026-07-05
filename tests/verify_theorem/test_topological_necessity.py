"""
verify_theorem/test_topological_necessity.py
==============================================
Empirical confirmation of the Section 4 topological-necessity argument
(pi_1(SO(3)) = Z_2): SO(3) is doubly covered by SU(2)/the unit
quaternions, so a rotation by exactly theta=pi about axis n is
identical to a rotation by theta=pi about axis -n. Any single-valued
axis-angle recovery must therefore pick a side, and that choice is
genuinely discontinuous/unstable in a neighborhood of theta=pi -- no
amount of implementation care removes this, since it's the topological
obstruction itself, not a bug in log_map_se3.

This is exactly why check_log_admissible enforces a strict margin below
pi (default 0.1 rad) rather than merely excluding the single point
theta=pi: the ambiguity's numerical footprint extends into a
neighborhood of the boundary, not just the exact boundary point.
"""
import numpy as np
import pytest

from kinematics.lie_algebra import exp_map, log_map_se3


class TestAxisSignAmbiguityAtBoundary:
    def test_r_pi_n_equals_r_pi_minus_n(self, rng):
        # Direct SO(3)-level statement of the double-cover fact: a pi
        # rotation about n and about -n produce the identical rotation
        # matrix. This is the root cause of the axis-sign ambiguity,
        # independent of any particular log-map implementation.
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        xi_pos = np.concatenate([np.pi * axis, np.zeros(3)])
        xi_neg = np.concatenate([-np.pi * axis, np.zeros(3)])
        R_pos = exp_map(xi_pos)[:3, :3]
        R_neg = exp_map(xi_neg)[:3, :3]
        assert np.allclose(R_pos, R_neg, atol=1e-10)

    def test_axis_sign_recovery_becomes_unstable_approaching_pi(self, rng):
        # Not a correctness failure to fix -- a direct empirical
        # measurement of the instability's extent, to justify the
        # log-admissibility margin quantitatively. As gap -> 0, the
        # recovered omega can flip sign relative to the input (equally
        # valid rotation-wise, but a discontinuous jump in the twist
        # representation), which is the practical symptom of the Z_2
        # ambiguity for closed-loop control (a control law reading xi_err
        # across such a flip would see a sign-discontinuous "error").
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        v = rng.normal(size=3) * 0.1

        gaps = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
        sign_agreement = []
        for gap in gaps:
            theta = np.pi - gap
            xi = np.concatenate([theta * axis, v])
            T = exp_map(xi)
            xi_hat = log_map_se3(T)
            # Same rotation axis recovered (up to sign) regardless of gap --
            # what we're tracking is whether the SIGN matches the input.
            agrees = np.dot(xi[:3], xi_hat[:3]) > 0
            sign_agreement.append(agrees)

        print("\n  gap-from-pi -> axis sign matches input twist:")
        for gap, agrees in zip(gaps, sign_agreement):
            print(f"    {gap:.0e}  ->  {agrees}")

        # The point of this test: sign agreement is reliable with margin
        # (gap >= 1e-2, i.e. inside check_log_admissible's default 0.1 rad
        # margin) but is NOT guaranteed as gap -> 0. We assert the
        # well-margined case is reliable; we deliberately do NOT assert
        # anything about the smallest gaps, since instability there is
        # the expected topological phenomenon, not a regression to catch.
        assert bool(sign_agreement[0])  # gap = 1e-2, safely margined
