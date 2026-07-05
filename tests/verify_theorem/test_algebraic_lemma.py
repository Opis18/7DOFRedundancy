"""
verify_theorem/test_algebraic_lemma.py
=======================================
Direct, paper-facing numerical verification of the algebraic building
blocks behind the central theorem (see project_4 draft, Sections 5-6):

  1. det(Ad_g) = 1 for all g in SE(3)           (Section 5 algebraic lemma)
  2. dL_{g(t)} invertibility (via exp/log being mutually inverse on the
     log-admissible domain, i.e. away from the theta = pi chart boundary)
     (Section 6 central-theorem building block)

These are intentionally separate from tests/unit/test_lie_algebra.py:
that file checks implementation correctness (does the code do what its
docstring says); this file checks that the *mathematical claims made in
the paper* hold numerically, sampled broadly across SE(3). If a paper
claim and an implementation detail ever diverge, these should be the
tests that catch it.
"""
import numpy as np
import pytest

from kinematics.lie_algebra import exp_map, log_map_se3, adjoint


N_SAMPLES = 500


@pytest.fixture(scope="module")
def sampled_twists():
    rng = np.random.default_rng(7)
    # Angular magnitude sampled up to pi - margin (log-admissible domain);
    # linear part unconstrained (log map has no linear-part restriction).
    twists = []
    for _ in range(N_SAMPLES):
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        theta = rng.uniform(0, np.pi - 0.05)
        v = rng.normal(size=3) * rng.uniform(0, 2.0)
        twists.append(np.concatenate([theta * axis, v]))
    return twists


class TestDetAdjointIsOne:
    """Section 5: det(Ad_g) = 1 for all g in SE(3) -- SE(3) is unimodular,
    so the adjoint representation is volume/orientation preserving. This
    is what guarantees the body-Jacobian re-expression in body_jacobian.py
    never introduces a spurious scaling/orientation-flip factor."""

    def test_det_is_one_over_sampled_group_elements(self, sampled_twists):
        max_err = 0.0
        for xi in sampled_twists:
            g = exp_map(xi)
            det = np.linalg.det(adjoint(g))
            max_err = max(max_err, abs(det - 1.0))
        assert max_err < 1e-7, f"max |det(Ad_g) - 1| = {max_err:.3e}"

    def test_det_is_one_at_identity(self):
        assert np.isclose(np.linalg.det(adjoint(np.eye(4))), 1.0, atol=1e-12)

    def test_adjoint_is_a_group_homomorphism(self, sampled_twists, rng=np.random.default_rng(11)):
        # Ad_{g1 g2} = Ad_{g1} Ad_{g2} -- needed for the composition step
        # of the Section 6 synthesis argument (chaining local charts).
        max_err = 0.0
        for _ in range(100):
            xi1 = sampled_twists[rng.integers(0, len(sampled_twists))]
            xi2 = sampled_twists[rng.integers(0, len(sampled_twists))]
            g1, g2 = exp_map(xi1), exp_map(xi2)
            lhs = adjoint(g1 @ g2)
            rhs = adjoint(g1) @ adjoint(g2)
            max_err = max(max_err, np.max(np.abs(lhs - rhs)))
        assert max_err < 1e-6, f"max homomorphism residual = {max_err:.3e}"


class TestLocalInvertibilityAwayFromChartBoundary:
    """Section 6: dL_{g(t)} (equivalently, exp/log being mutually inverse)
    holds on the log-admissible domain theta < pi. This is the constructive
    half of the central theorem -- log_map_se3 recovers exactly the twist
    that generated g, so the local chart is non-degenerate there."""

    def test_exp_log_roundtrip_holds_broadly(self, sampled_twists):
        max_err = 0.0
        for xi in sampled_twists:
            g = exp_map(xi)
            xi_hat = log_map_se3(g)
            max_err = max(max_err, np.max(np.abs(xi - xi_hat)))
        assert max_err < 1e-6, f"max exp/log roundtrip residual = {max_err:.3e}"

    def test_invertibility_holds_throughout_log_admissible_domain(self):
        # check_log_admissible's default margin is 0.1 rad -- i.e. the
        # paper's actual operating claim is correctness for theta up to
        # pi - 0.1, not all the way to the literal boundary. Verify the
        # roundtrip holds cleanly everywhere inside that claimed domain.
        axis = np.array([1.0, 0.0, 0.0])
        margin = 0.1
        for theta in np.linspace(0.01, np.pi - margin, 25):
            xi = np.concatenate([theta * axis, np.array([0.1, 0.2, 0.3])])
            g = exp_map(xi)
            xi_hat = log_map_se3(g)
            assert np.allclose(xi, xi_hat, atol=1e-8), (
                f"roundtrip degraded inside the claimed log-admissible domain at theta={theta}"
            )

    def test_precision_characterization_near_chart_boundary(self):
        # Not a pass/fail correctness claim -- a numerical characterization
        # of exactly where roundtrip precision starts to degrade as
        # theta -> pi, to justify the 0.1 rad default margin quantitatively
        # in the paper.
        axis = np.array([1.0, 0.0, 0.0])
        gaps = [0.1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
        errors = []
        for gap in gaps:
            theta = np.pi - gap
            xi = np.concatenate([theta * axis, np.array([0.1, 0.2, 0.3])])
            g = exp_map(xi)
            xi_hat = log_map_se3(g)
            errors.append(np.max(np.abs(xi - xi_hat)))
        print("\n  gap-from-pi -> roundtrip error:")
        for gap, err in zip(gaps, errors):
            print(f"    {gap:.0e}  ->  {err:.3e}")
        # Sanity bound only: error should never exceed the rotation angle
        # itself (i.e. never catastrophically wrong, just imprecise).
        assert all(e < 1.0 for e in errors)
