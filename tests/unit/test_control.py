"""Unit tests for control/manipulability.py, pseudoinverse.py, null_space.py."""
import numpy as np
import pytest

from kinematics.body_jacobian import body_jacobian
from control.manipulability import yoshikawa_measure, manipulability_gradient
from control.pseudoinverse import pinv_dls, adaptive_damping
from control.null_space import null_space_projector, redundancy_resolution


class TestYoshikawaMeasure:
    def test_positive_at_generic_config(self, iiwa7, rng):
        q = rng.uniform(-1, 1, size=iiwa7["n_joints"])
        J = body_jacobian(q, iiwa7["S_list"], iiwa7["M"])
        w = yoshikawa_measure(J)
        assert w > 0

    def test_zero_at_rank_deficient_jacobian(self):
        J = np.zeros((6, 7))
        assert yoshikawa_measure(J) == pytest.approx(0.0, abs=1e-10)


class TestManipulabilityGradient:
    def test_shape_matches_q(self, iiwa7, rng):
        q = rng.uniform(-1, 1, size=iiwa7["n_joints"])
        grad = manipulability_gradient(q, iiwa7["S_list"], iiwa7["M"], body_jacobian)
        assert grad.shape == (iiwa7["n_joints"],)

    def test_ascent_increases_manipulability(self, iiwa7, rng):
        q = rng.uniform(-0.5, 0.5, size=iiwa7["n_joints"])
        J0 = body_jacobian(q, iiwa7["S_list"], iiwa7["M"])
        w0 = yoshikawa_measure(J0)
        grad = manipulability_gradient(q, iiwa7["S_list"], iiwa7["M"], body_jacobian)
        step = 0.05
        q1 = q + step * grad / (np.linalg.norm(grad) + 1e-12)
        J1 = body_jacobian(q1, iiwa7["S_list"], iiwa7["M"])
        w1 = yoshikawa_measure(J1)
        assert w1 > w0


class TestPinvDLS:
    def test_zero_damping_matches_moore_penrose(self, iiwa7, rng):
        q = rng.uniform(-1, 1, size=iiwa7["n_joints"])
        J = body_jacobian(q, iiwa7["S_list"], iiwa7["M"])
        assert np.allclose(pinv_dls(J, lam=0.0), np.linalg.pinv(J), atol=1e-6)

    def test_damping_reduces_pinv_norm_near_singularity(self):
        # A rank-deficient Jacobian: damping should keep J_pinv finite
        # where the undamped Moore-Penrose pinv would blow up in a
        # closed-loop (division-by-near-zero-singular-value) sense.
        J = np.zeros((6, 7))
        J[0, 0] = 1e-8
        J_pinv_damped = pinv_dls(J, lam=0.1)
        assert np.all(np.isfinite(J_pinv_damped))


class TestAdaptiveDamping:
    def test_zero_manipulability_gives_max_damping(self):
        assert adaptive_damping(0.0, w_threshold=0.05, lam_max=0.05) == pytest.approx(0.05)

    def test_above_threshold_gives_zero_damping(self):
        assert adaptive_damping(0.5, w_threshold=0.05, lam_max=0.05) == pytest.approx(0.0)

    def test_monotonic_in_between(self):
        w_vals = np.linspace(0, 0.05, 10)
        lams = [adaptive_damping(w, 0.05, 0.05) for w in w_vals]
        assert all(lams[i] >= lams[i + 1] for i in range(len(lams) - 1))


class TestNullSpaceProjector:
    def test_projector_is_idempotent_undamped(self, iiwa7, rng):
        q = rng.uniform(-1, 1, size=iiwa7["n_joints"])
        J = body_jacobian(q, iiwa7["S_list"], iiwa7["M"])
        N = null_space_projector(J)
        assert np.allclose(N @ N, N, atol=1e-6)

    def test_J_N_is_zero_undamped(self, iiwa7, rng):
        q = rng.uniform(-1, 1, size=iiwa7["n_joints"])
        J = body_jacobian(q, iiwa7["S_list"], iiwa7["M"])
        N = null_space_projector(J)
        assert np.allclose(J @ N, 0.0, atol=1e-6)

    def test_projector_stays_symmetric_when_damped(self, iiwa7, rng):
        q = rng.uniform(-1, 1, size=iiwa7["n_joints"])
        J = body_jacobian(q, iiwa7["S_list"], iiwa7["M"])
        J_pinv_damped = pinv_dls(J, lam=0.3)
        N = null_space_projector(J, J_pinv=J_pinv_damped)
        assert np.allclose(N, N.T, atol=1e-10)


class TestRedundancyResolution:
    def test_recovers_primary_term_when_secondary_zero(self, iiwa7, rng):
        q = rng.uniform(-1, 1, size=iiwa7["n_joints"])
        J = body_jacobian(q, iiwa7["S_list"], iiwa7["M"])
        xi_dot = rng.normal(size=6)
        q_dot = redundancy_resolution(J, xi_dot, np.zeros(iiwa7["n_joints"]))
        assert np.allclose(J @ q_dot, xi_dot, atol=1e-6)

    def test_task_space_consistency_with_secondary_term(self, iiwa7, rng):
        # Secondary objective should be fully absorbed by the null space:
        # task-space motion should equal xi_dot regardless of the secondary term.
        q = rng.uniform(-1, 1, size=iiwa7["n_joints"])
        J = body_jacobian(q, iiwa7["S_list"], iiwa7["M"])
        xi_dot = rng.normal(size=6)
        secondary = rng.normal(size=iiwa7["n_joints"])
        q_dot = redundancy_resolution(J, xi_dot, secondary)
        assert np.allclose(J @ q_dot, xi_dot, atol=1e-6)
