"""Unit tests for simulation/controller.py — control-law contract."""
import numpy as np
import pytest

from simulation.controller import proportional_control_law, PIDControlLaw


class TestProportionalControlLaw:
    def test_scalar_gain(self, rng):
        law = proportional_control_law(2.0)
        xi_err = rng.normal(size=6)
        assert np.allclose(law(xi_err, dt=0.01), 2.0 * xi_err)

    def test_diagonal_gain(self, rng):
        Kp = rng.uniform(0.5, 2.0, size=6)
        law = proportional_control_law(Kp)
        xi_err = rng.normal(size=6)
        assert np.allclose(law(xi_err, dt=0.01), Kp * xi_err)


class TestPIDControlLaw:
    def test_matches_proportional_when_ki_kd_zero(self, rng):
        pid = PIDControlLaw(Kp=1.5, Ki=0.0, Kd=0.0)
        p_law = proportional_control_law(1.5)
        for _ in range(5):
            xi_err = rng.normal(size=6)
            assert np.allclose(pid(xi_err, dt=0.01), p_law(xi_err, dt=0.01))

    def test_derivative_term_on_second_call(self):
        pid = PIDControlLaw(Kp=0.0, Ki=0.0, Kd=1.0)
        xi1 = np.ones(6)
        xi2 = np.ones(6) * 2.0
        dt = 0.1
        pid(xi1, dt)
        out = pid(xi2, dt)
        expected = (xi2 - xi1) / dt
        assert np.allclose(out, expected)

    def test_integral_limit_clamps(self):
        pid = PIDControlLaw(Kp=0.0, Ki=1.0, Kd=0.0, integral_limit=0.05)
        for _ in range(50):
            pid(np.ones(6) * 10.0, dt=0.1)
        assert np.max(np.abs(pid._integral)) <= 0.05 + 1e-12

    def test_reset_clears_state(self):
        pid = PIDControlLaw(Kp=0.0, Ki=1.0, Kd=1.0)
        pid(np.ones(6), dt=0.1)
        pid.reset()
        assert np.allclose(pid._integral, 0.0)
        assert pid._prev_error is None
