"""
Tests for core/kinematics.py.

Benchmarks:
  - Elastic scattering: Q=0, check energy conservation
  - Threshold energy for 13C(p,n)13N (Q ≈ -3.003 MeV)
  - Inverse kinematics max lab angle: d(12C,p)13C at 50 MeV/u
"""
import numpy as np
import pytest
from nuclear_kinematics.core.masses import get_mass_MeV, parse_nuclide
from nuclear_kinematics.core.kinematics import (
    cm_energy, threshold_energy, lab_angle_energy, max_lab_angle,
    kinematic_locus, recoil_max_lab_angle, kinematic_compression,
    lab_to_cm_solutions,
)


# Convenience: masses for common nuclides
mP  = get_mass_MeV(1, 1)    # proton (atomic H-1)
mN  = get_mass_MeV(0, 1)    # neutron
m12C = get_mass_MeV(6, 12)
m13C = get_mass_MeV(6, 13)
m13N = get_mass_MeV(7, 13)
mD   = get_mass_MeV(1, 2)   # deuteron


class TestCMEnergy:
    def test_at_rest(self):
        # T_A = 0: sqrt(s) = mA + mB
        sqrts = cm_energy(mP, m12C, 0.0)
        assert abs(sqrts - (mP + m12C)) < 1e-6

    def test_nonzero(self):
        # 12C beam at 50 MeV/u → T_A = 600 MeV
        sqrts = cm_energy(m12C, mD, 600.0)
        assert sqrts > m12C + mD  # must be above threshold


class TestThresholdEnergy:
    def test_exothermic_zero(self):
        # 12C(d,p)13C: Q > 0, threshold = 0
        T = threshold_energy(mD, m12C, mP, m13C)
        assert T == 0.0

    def test_endothermic_13C_pn(self):
        # 13C(p,n)13N: Q ≈ -3.003 MeV
        # T_thresh = [(mN + m13N)^2 - (mP + m13C)^2] / (2*m13C)
        T = threshold_energy(mP, m13C, mN, m13N)
        # Non-relativistic approx: T ≈ 3.003 * (mP + m13C) / m13C ≈ 3.24 MeV
        assert 3.0 < T < 3.5, f"Threshold {T:.3f} MeV outside expected range"


class TestLabAngleEnergy:
    def test_forward_angle_positive_energy(self):
        theta_lab, T_C = lab_angle_energy(
            np.array([0.5, 1.0, 1.5]),  # CM angles in radians
            m12C, mD, mP, m13C, 600.0
        )
        assert np.all(T_C > 0), "Ejectile energies must be positive"
        assert np.all(theta_lab >= 0), "Lab angles must be non-negative"

    def test_elastic_energy_at_0_deg(self):
        # Elastic p+p at 100 MeV: ejectile at theta_cm=0 gets maximum energy
        theta_lab, T_C = lab_angle_energy(
            np.array([0.01]),
            mP, mP, mP, mP, 100.0
        )
        assert T_C[0] > 90.0, "Forward ejectile in elastic p+p should get ~100 MeV"


class TestMaxLabAngle:
    def test_inverse_kinematics_has_max(self):
        # d(12C,12C)d elastic at 50 MeV/u: heavy beam → inverse kinematics → max angle exists
        # In inverse kinematics, the heavy ejectile (12C) has a maximum lab angle
        T_A = 50.0 * 12  # 50 MeV/u for 12C beam = 600 MeV total
        ang = max_lab_angle(m12C, mD, m12C, mD, T_A)
        assert ang is not None, "Inverse kinematics should give a max lab angle"
        # Heavy ejectile max angle should be much less than 90° (strong forward focusing)
        assert np.degrees(ang) < 45.0

    def test_normal_kinematics_no_max(self):
        # p + 12C → p + 12C elastic at 100 MeV: light beam, no max angle
        ang = max_lab_angle(mP, m12C, mP, m12C, 100.0)
        assert ang is None


class TestKinematicLocus:
    def test_returns_correct_shape(self):
        theta_deg, T_MeV = kinematic_locus(m12C, mD, mP, m13C, 600.0)
        assert len(theta_deg) == 500
        assert len(T_MeV) == 500

    def test_energies_positive(self):
        _, T_MeV = kinematic_locus(m12C, mD, mP, m13C, 600.0)
        assert np.all(T_MeV > 0)


class TestRecoilMaxLabAngle:
    def test_inverse_kinematics_cone(self):
        # 12C beam on d: mA >> mB, recoil cone should be small
        ang = recoil_max_lab_angle(m12C, mD, 600.0)
        assert np.degrees(ang) < 15.0  # heavy recoil strongly forward-focused

    def test_normal_kinematics_hemisphere(self):
        # p beam on 12C: mA < mB, recoil covers full forward hemisphere
        ang = recoil_max_lab_angle(mP, m12C, 100.0)
        assert abs(ang - np.pi / 2) < 1e-6


class TestKinematicCompression:
    """Tests for kinematic_compression (kp = (1/rho) drho/dtheta_lab)."""

    def test_kp_zero_at_theta_cm_zero(self):
        """kp = 0 at theta_cm = 0 (forward CM direction)."""
        kp = kinematic_compression(0.0, mP, m12C, mP, m12C, 100.0)
        assert abs(kp[0]) < 1e-10

    def test_kp_zero_at_theta_cm_pi(self):
        """kp = 0 at theta_cm = pi (backward CM direction)."""
        kp = kinematic_compression(np.pi, mP, m12C, mP, m12C, 100.0)
        assert abs(kp[0]) < 1e-10

    def test_kp_finite_normal_kinematics(self):
        """kp is finite everywhere in (0, pi) for normal kinematics."""
        theta = np.linspace(0.01, np.pi - 0.01, 200)
        kp = kinematic_compression(theta, mP, m12C, mP, m12C, 100.0)
        assert np.all(np.isfinite(kp))

    def test_kp_returns_array(self):
        """Scalar input still returns an ndarray."""
        kp = kinematic_compression(np.pi / 2, mP, m12C, mP, m12C, 100.0)
        assert isinstance(kp, np.ndarray)
        assert kp.shape == (1,)

    def test_kp_negative_midrange_normal_kinematics(self):
        """kp < 0 at theta_cm ~ pi/2 for p+12C elastic (rho decreases with angle)."""
        kp = kinematic_compression(np.pi / 2, mP, m12C, mP, m12C, 100.0)
        assert kp[0] < 0


class TestLabToCMSolutionsThetaBLab:
    """Tests for theta_B_lab in lab_to_cm_solutions() output."""

    def test_theta_B_lab_key_present(self):
        """Every solution dict must contain 'theta_B_lab'."""
        solutions = lab_to_cm_solutions(
            np.radians(30.0), mP, m12C, mP, m12C, 100.0
        )
        assert len(solutions) >= 1
        for sol in solutions:
            assert 'theta_B_lab' in sol

    def test_theta_B_lab_forward_hemisphere(self):
        """B is always in the forward hemisphere (theta_B_lab < pi/2) for
        normal kinematics with a light beam on a heavy target."""
        solutions = lab_to_cm_solutions(
            np.radians(30.0), mP, m12C, mP, m12C, 100.0
        )
        for sol in solutions:
            assert sol['theta_B_lab'] >= 0.0
            assert sol['theta_B_lab'] < np.pi / 2

    def test_theta_B_lab_nonnegative(self):
        """theta_B_lab is always >= 0 (arctan2 with positive pt_B_lab)."""
        for theta_deg in [10.0, 30.0, 60.0]:
            solutions = lab_to_cm_solutions(
                np.radians(theta_deg), mP, m12C, mP, m12C, 100.0
            )
            for sol in solutions:
                assert sol['theta_B_lab'] >= 0.0


class TestExcitedResidual:
    """Residual B in an excited state is handled via effective mass m_B + Ex."""

    def test_ex_zero_unchanged(self):
        sol0 = lab_to_cm_solutions(np.radians(30.0), mD, m12C, mP, m13C, 10.0)
        solx = lab_to_cm_solutions(np.radians(30.0), mD, m12C, mP, m13C + 0.0, 10.0)
        assert sol0 == solx

    def test_ejectile_energy_drops_and_energy_conserved(self):
        Ex = 3.854
        T_a = 10.0
        Q_gs = mD + m12C - mP - m13C
        gs = lab_to_cm_solutions(np.radians(30.0), mD, m12C, mP, m13C, T_a)
        ex = lab_to_cm_solutions(np.radians(30.0), mD, m12C, mP, m13C + Ex, T_a)
        assert len(gs) == 1 and len(ex) == 1
        drop = gs[0]['T_b_lab'] - ex[0]['T_b_lab']
        # At fixed lab angle the drop is close to, but not exactly, Ex
        assert 0.8 * Ex < drop < 1.2 * Ex
        assert abs(ex[0]['T_b_lab'] + ex[0]['T_B_lab'] - (T_a + Q_gs - Ex)) < 1e-9

    def test_inelastic_threshold(self):
        # 12C(p,p')12C*(4.439): threshold ≈ Ex * (m_p + m_12C) / m_12C ≈ 4.81 MeV
        T = threshold_energy(mP, m12C, mP, m12C + 4.439)
        assert abs(T - 4.439 * (mP + m12C) / m12C) < 0.01
