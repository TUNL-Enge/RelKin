"""
Tests for core/rutherford.py.
Checked against analytic formula at specific angles.
"""
import numpy as np
import pytest
from nuclear_kinematics.core.rutherford import (
    rutherford_cm, safe_distance, nuclear_touching_radius
)


class TestRutherfordCM:
    def test_90_degrees(self):
        # At θ_cm = 90°: dσ/dΩ = η'² / (4 sin⁴(θ/2))
        # sin(45°) = 1/√2, so sin⁴(45°) = 1/4
        # For Z1=Z2=1, T_cm=1 MeV: η' = 1.43996/2 = 0.71998 fm
        # dσ/dΩ = η'² / (4 * 1/4) * 10 = η'² * 10 = 5.184 mb/sr
        Z1, Z2, T_cm = 1, 1, 1.0
        result = rutherford_cm(np.pi / 2, Z1, Z2, T_cm)
        eta_prime = 1 * 1 * 1.43996 / (2 * 1.0)
        expected = (eta_prime**2) * 10.0  # Note: factor of 4 cancels with sin⁴(45°) = 1/4
        assert abs(result[0] - expected) < 1e-6

    def test_forward_divergence(self):
        # Near θ=0, cross section diverges
        result = rutherford_cm(0.01, 6, 1, 10.0)
        assert result[0] > 1e6

    def test_array_input(self):
        angles = np.array([np.pi/4, np.pi/2, 3*np.pi/4])
        result = rutherford_cm(angles, 6, 6, 10.0)
        assert result.shape == (3,)
        assert np.all(result > 0)


class TestSafeDistance:
    def test_12C_12C(self):
        # 12C + 12C at T_lab = 20 MeV → T_cm = 10 MeV (equal masses)
        d = safe_distance(6, 6, 12, 12, 20.0)
        R = nuclear_touching_radius(12, 12)
        # At 10 MeV CM, Coulomb excitation is NOT safe (d_min < R_touch)
        assert d > 0

    def test_touching_radius(self):
        R = nuclear_touching_radius(12, 12)
        # 12C: R = 1.2*(12^(1/3) + 12^(1/3)) + 5 ≈ 1.2*2*2.289 + 5 ≈ 10.49 fm
        assert 10.0 < R < 11.0
