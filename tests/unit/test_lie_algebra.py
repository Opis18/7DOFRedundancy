"""Unit tests for kinematics/lie_algebra.py — hat/vee, exp/log maps, adjoint."""
import numpy as np
import pytest

from kinematics.lie_algebra import hat3, vee3, hat6, vee6, exp_map, log_map_se3, adjoint


def random_twist(rng, scale=1.0):
    return rng.normal(size=6) * scale


class TestHatVee3:
    def test_vee_inverts_hat(self, rng):
        w = rng.normal(size=3)
        assert np.allclose(vee3(hat3(w)), w)

    def test_hat_is_skew_symmetric(self, rng):
        w = rng.normal(size=3)
        Omega = hat3(w)
        assert np.allclose(Omega, -Omega.T)

    def test_hat_matches_cross_product(self, rng):
        w = rng.normal(size=3)
        v = rng.normal(size=3)
        assert np.allclose(hat3(w) @ v, np.cross(w, v))

    def test_bad_shape_raises(self):
        with pytest.raises(ValueError):
            hat3([1, 2])


class TestHatVee6:
    def test_vee_inverts_hat(self, rng):
        xi = random_twist(rng)
        assert np.allclose(vee6(hat6(xi)), xi)

    def test_hat6_bottom_row_zero(self, rng):
        xi = random_twist(rng)
        X = hat6(xi)
        assert np.allclose(X[3, :], 0.0)


class TestExpLogRoundtrip:
    # NOTE: exp_map/log_map_se3 roundtrip is only well-defined for rotation
    # angle < pi (log map's principal domain -- see check_log_admissible).
    # A twist with ||omega|| > pi maps to the SAME rotation as a shorter
    # twist on the opposite axis, so log_map_se3 legitimately returns a
    # different (but equivalent) representative. scale is chosen here to
    # keep ||omega|| comfortably under pi.
    @pytest.mark.parametrize("scale", [0.01, 0.3, 0.6, 0.9])
    def test_roundtrip_general(self, rng, scale):
        xi = random_twist(rng, scale=scale)
        assert np.linalg.norm(xi[:3]) < np.pi, "test twist escaped the log-admissible domain"
        T = exp_map(xi)
        xi_recovered = log_map_se3(T)
        assert np.allclose(xi, xi_recovered, atol=1e-8)

    def test_identity_logs_to_zero(self):
        assert np.allclose(log_map_se3(np.eye(4)), np.zeros(6))

    def test_near_pi_rotation_stable(self, rng):
        # Inside the theta~pi special-case branch (_PI_NEAR = 1e-7), but
        # far enough from the literal theta=pi point that the axis-sign
        # recovery heuristic (argmax diagonal of (R+I)/2) stays
        # well-conditioned. See verify_theorem/test_topological_necessity.py
        # for what happens as gap -> 0: the axis sign genuinely becomes
        # ambiguous there (the pi_1(SO(3))=Z_2 double cover), which is
        # not a bug -- it's the boundary phenomenon Section 4 is about.
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        theta = np.pi - 1e-4
        xi = np.concatenate([theta * axis, rng.normal(size=3) * 0.1])
        T = exp_map(xi)
        xi_recovered = log_map_se3(T)
        assert np.allclose(xi, xi_recovered, atol=1e-6)

    def test_exp_map_output_is_se3(self, rng):
        xi = random_twist(rng)
        T = exp_map(xi)
        R = T[:3, :3]
        assert np.allclose(R.T @ R, np.eye(3), atol=1e-10)
        assert np.isclose(np.linalg.det(R), 1.0, atol=1e-10)
        assert np.allclose(T[3, :], [0, 0, 0, 1])


class TestAdjoint:
    def test_adjoint_of_identity_is_identity(self):
        assert np.allclose(adjoint(np.eye(4)), np.eye(6))

    def test_adjoint_inverse_matches_adjoint_of_inverse(self, rng):
        xi = random_twist(rng)
        T = exp_map(xi)
        Ad_T = adjoint(T)
        Ad_Tinv = adjoint(np.linalg.inv(T))
        assert np.allclose(Ad_T @ Ad_Tinv, np.eye(6), atol=1e-10)

    def test_determinant_is_one(self, rng):
        # Algebraic lemma used in Section 5 of the paper: det(Ad_g) = 1 for
        # all g in SE(3). Sampled broadly here; verify_theorem/ has the
        # dedicated, paper-facing version of this check.
        for _ in range(20):
            xi = random_twist(rng, scale=rng.uniform(0.01, 3.0))
            T = exp_map(xi)
            assert np.isclose(np.linalg.det(adjoint(T)), 1.0, atol=1e-8)
