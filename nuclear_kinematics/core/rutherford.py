"""
Rutherford scattering and Coulomb excitation utilities.

Units: angles in radians, energies in MeV, cross sections in mb/sr, distances in fm.
"""
import numpy as np
from scipy.optimize import brentq

from .constants import E2_MEVFM, R0


def rutherford_cm(
    theta_cm: float | np.ndarray,
    Z1: int,
    Z2: int,
    T_cm: float,
) -> np.ndarray:
    """
    Rutherford differential cross section in the CM frame.

    dσ/dΩ_cm = η'² / (4 sin⁴(θ_cm/2))   [mb/sr]

    where η' = Z1*Z2*e² / (2*T_cm)

    Parameters
    ----------
    theta_cm : CM scattering angle(s) in radians
    Z1, Z2   : charge numbers of beam and target
    T_cm     : CM kinetic energy in MeV
    """
    theta_cm = np.atleast_1d(np.asarray(theta_cm, dtype=float))
    eta_prime = Z1 * Z2 * E2_MEVFM / (2.0 * T_cm)
    dsdo_fm2 = (eta_prime**2) / (4.0 * np.sin(theta_cm / 2.0)**4)
    return dsdo_fm2 * 10.0   # fm² → mb (1 fm² = 10 mb)


def rutherford_lab(
    theta_lab: float | np.ndarray,
    Z1: int,
    Z2: int,
    mA_amu: float,
    mB_amu: float,
    T_lab: float,
) -> np.ndarray:
    """
    Rutherford differential cross section in the lab frame.

    Transforms via the Jacobian:
      dσ/dΩ_lab = dσ/dΩ_cm * (1 + τ² + 2τ cosθ_cm)^(3/2) / (1 + τ cosθ_cm)
    where τ = mA / mB and
      tan(θ_lab) = sin(θ_cm) / (τ + cos(θ_cm))

    Parameters
    ----------
    theta_lab        : lab scattering angle(s) in radians
    Z1, Z2           : charge numbers
    mA_amu, mB_amu   : beam and target masses in atomic mass units
    T_lab            : lab kinetic energy of beam in MeV
    """
    theta_lab = np.atleast_1d(np.asarray(theta_lab, dtype=float))
    tau = mA_amu / mB_amu
    T_cm = (mB_amu / (mA_amu + mB_amu)) * T_lab

    # Solve for theta_cm from theta_lab numerically
    # tan(θ_lab) = sin(θ_cm) / (τ + cos(θ_cm))
    # For τ < 1 (normal kinematics) this is single-valued
    # Rearranged: sin(θ_cm) - tan(θ_lab)*(τ + cos(θ_cm)) = 0
    # Use analytic solution: θ_cm = θ_lab + arcsin(τ * sin(θ_lab))  [for τ≤1]
    # General: solve quadratic in cos(θ_cm)
    # cos(θ_cm) = [-τ*sin²(θ_lab) ± cos(θ_lab)*sqrt(1 - τ²*sin²(θ_lab))]
    #             / (1 - τ²*sin²(θ_lab) + ... ) — use atan2 form below

    tl = theta_lab
    # Compute cos(θ_cm) from tan(θ_lab) = sin(θ_cm)/(τ + cos(θ_cm))
    # Let x = cos(θ_cm). Then: (1+τ²-τ²)... use direct formula:
    # cos(θ_cm) = (cos(θ_lab) - τ*sin²(θ_lab)/cos(θ_lab)) ... numeric approach:

    def _theta_cm_scalar(t_lab_val):
        def eq(t_cm):
            return np.arctan2(np.sin(t_cm), tau + np.cos(t_cm)) - t_lab_val
        return brentq(eq, 1e-9, np.pi - 1e-9)

    theta_cm = np.array([_theta_cm_scalar(t) for t in tl])
    dsdo_cm = rutherford_cm(theta_cm, Z1, Z2, T_cm)

    factor = (1 + tau**2 + 2 * tau * np.cos(theta_cm))**1.5
    denom  = np.abs(1 + tau * np.cos(theta_cm))
    return dsdo_cm * factor / denom


def safe_distance(
    Z1: int,
    Z2: int,
    A1: int,
    A2: int,
    T_lab: float,
) -> float:
    """
    Minimum distance of closest approach d_min at θ_cm = 180° (head-on).

    Converts lab kinetic energy to CM energy internally:
      T_cm = A2 / (A1 + A2) * T_lab   [non-relativistic approximation]

    d_min = Z1*Z2*e² / T_cm   [fm]   (at 180° the factor (1 + 1/sin(θ/2)) = 2)

    Also computes the nuclear touching radius R_touch = 1.2*(A1^(1/3) + A2^(1/3)) + 5 fm.

    Returns d_min in fm. Caller should compare to R_touch to assess safety.
    """
    T_cm = (A2 / (A1 + A2)) * T_lab  # non-rel. CM energy from lab energy
    d_min = Z1 * Z2 * E2_MEVFM / T_cm  # factor of 2 absorbed: (1 + 1/sin(π/2)) = 2
    return float(d_min)


def nuclear_touching_radius(A1: int, A2: int) -> float:
    """Nuclear touching radius in fm: R_touch = 1.2*(A1^(1/3) + A2^(1/3)) + 5 fm."""
    return R0 * (A1**(1/3) + A2**(1/3)) + 5.0
