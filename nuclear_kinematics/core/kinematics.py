"""
Two-body relativistic kinematics: a + A → b + B.

Convention:
    a  – projectile (beam)
    A  – target (at rest in lab)
    b  – ejectile
    B  – residual

All masses in MeV/c², energies in MeV, angles in radians.
"""
import numpy as np


def cm_energy(m_a: float, m_A: float, T_a: float) -> float:
    """
    Invariant CM energy sqrt(s) in MeV.

    s = m_a² + m_A² + 2*m_A*(T_a + m_a)
    """
    E_a = T_a + m_a
    s = m_a**2 + m_A**2 + 2.0 * m_A * E_a
    return float(np.sqrt(s))


def threshold_energy(m_a: float, m_A: float, m_b: float, m_B: float) -> float:
    """
    Lab-frame threshold kinetic energy for an endothermic reaction (Q < 0).

    T_thresh = [(m_b + m_B)² - (m_a + m_A)²] / (2 * m_A)

    Returns 0.0 if the reaction is exothermic (no threshold).
    """
    num = (m_b + m_B)**2 - (m_a + m_A)**2
    if num <= 0.0:
        return 0.0
    return float(num / (2.0 * m_A))


def cm_kinetic_energies(m_b: float, m_B: float, sqrts: float) -> tuple[float, float]:
    """
    CM-frame kinetic energies of ejectile b and residual B.

    These are fixed for a given reaction (independent of scattering angle).

    Returns
    -------
    T_b_cm : CM kinetic energy of ejectile b (MeV)
    T_B_cm : CM kinetic energy of residual B (MeV)
    """
    s = sqrts**2
    E_star_b = (s + m_b**2 - m_B**2) / (2.0 * sqrts)
    E_star_B = sqrts - E_star_b
    return float(E_star_b - m_b), float(E_star_B - m_B)


def lab_angle_energy(
    theta_b_cm: "float | np.ndarray",
    m_a: float,
    m_A: float,
    m_b: float,
    m_B: float,
    T_a: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Transform ejectile b from CM frame to lab frame.

    Parameters
    ----------
    theta_b_cm : float or array, CM angle of ejectile b (radians)
    m_a, m_A   : projectile and target masses (MeV/c²)
    m_b, m_B   : ejectile and residual masses (MeV/c²)
    T_a        : beam lab kinetic energy (MeV)

    Returns
    -------
    theta_b_lab : array, lab angle of ejectile b (radians)
    T_b_lab     : array, lab kinetic energy of ejectile b (MeV)
    """
    theta_b_cm = np.atleast_1d(np.asarray(theta_b_cm, dtype=float))
    E_a = T_a + m_a
    p_a = np.sqrt(max(E_a**2 - m_a**2, 0.0))
    s = m_a**2 + m_A**2 + 2.0 * m_A * E_a
    sqrts = np.sqrt(s)
    beta  = p_a / (E_a + m_A)
    gamma = (E_a + m_A) / sqrts

    num = (s - (m_b + m_B)**2) * (s - (m_b - m_B)**2)
    p_star   = np.sqrt(max(num, 0.0)) / (2.0 * sqrts)
    E_star_b = (s + m_b**2 - m_B**2) / (2.0 * sqrts)

    cos_cm = np.cos(theta_b_cm)
    sin_cm = np.sin(theta_b_cm)

    pz_lab      = gamma * (p_star * cos_cm + beta * E_star_b)
    pt_lab      = p_star * sin_cm
    theta_b_lab = np.arctan2(pt_lab, pz_lab)
    E_b_lab     = gamma * (E_star_b + beta * p_star * cos_cm)
    T_b_lab     = E_b_lab - m_b

    return theta_b_lab, T_b_lab


def lab_to_cm_solutions(
    theta_lab: float,
    m_a: float,
    m_A: float,
    m_b: float,
    m_B: float,
    T_a: float,
) -> list[dict]:
    """
    Given the lab angle of ejectile b, return all physical CM-frame solutions.

    Uses the analytic quadratic inversion of the Lorentz boost. For normal
    kinematics one solution is returned; for inverse kinematics up to two
    solutions may exist (or zero if the angle exceeds the kinematic limit).

    Parameters
    ----------
    theta_lab : float, lab angle of ejectile b (radians)
    m_a, m_A  : projectile and target masses (MeV/c²)
    m_b, m_B  : ejectile and residual masses (MeV/c²)
    T_a       : beam lab kinetic energy (MeV)

    Returns
    -------
    List of dicts (0, 1, or 2 entries), each containing:
        theta_b_cm : CM angle of ejectile b (radians)
        T_b_lab    : lab kinetic energy of ejectile b (MeV)
        T_B_lab    : lab kinetic energy of residual B (MeV)
    """
    E_a = T_a + m_a
    p_a = np.sqrt(max(E_a**2 - m_a**2, 0.0))
    s = m_a**2 + m_A**2 + 2.0 * m_A * E_a
    sqrts = np.sqrt(s)
    beta  = p_a / (E_a + m_A)
    gamma = (E_a + m_A) / sqrts

    # Below threshold: reaction is closed, no solutions
    if s <= (m_b + m_B)**2:
        return []

    num = (s - (m_b + m_B)**2) * (s - (m_b - m_B)**2)
    p_star   = np.sqrt(max(num, 0.0)) / (2.0 * sqrts)
    E_star_b = (s + m_b**2 - m_B**2) / (2.0 * sqrts)
    E_star_B = sqrts - E_star_b

    Q = m_a + m_A - m_b - m_B
    cos_lab = np.cos(theta_lab)

    def _make_solution(cos_cm: float) -> dict:
        cos_cm = float(np.clip(cos_cm, -1.0, 1.0))
        theta_cm = float(np.arccos(cos_cm))
        sin_cm = float(np.sqrt(max(1.0 - cos_cm**2, 0.0)))
        E_b_lab = gamma * (E_star_b + beta * p_star * cos_cm)
        T_b_lab = float(E_b_lab - m_b)
        T_B_lab = float(T_a + Q - T_b_lab)
        # Lorentz boost for residual B (travels opposite to b in CM frame)
        pz_B_lab = gamma * (-p_star * cos_cm + beta * E_star_B)
        pt_B_lab = p_star * sin_cm
        theta_B_lab = float(np.arctan2(pt_B_lab, pz_B_lab))
        return {
            'theta_b_cm': theta_cm,
            'T_b_lab': T_b_lab,
            'T_B_lab': T_B_lab,
            'theta_B_lab': theta_B_lab,
        }

    # Special case: theta_lab = 90° (tan undefined, cos = 0)
    if abs(cos_lab) < 1e-12:
        if p_star < 1e-15:
            return []
        cos_cm = -(beta * E_star_b) / p_star
        if abs(cos_cm) > 1.0:
            return []
        return [_make_solution(cos_cm)]

    T_tan = np.tan(theta_lab)
    Tg = T_tan * gamma  # tan(theta_lab) * gamma_cm

    # Quadratic in c = cos(theta_b_cm):
    #   A_q*c² + B_q*c + C_q = 0
    # derived from: tan(theta_lab) = p* sin_cm / [gamma*(p* cos_cm + beta*E*_b)]
    A_q = p_star**2 * (Tg**2 + 1.0)
    B_q = 2.0 * Tg**2 * beta * E_star_b * p_star
    C_q = (Tg * beta * E_star_b)**2 - p_star**2

    disc = B_q**2 - 4.0 * A_q * C_q
    if disc < 0.0:
        return []

    sqrt_disc = np.sqrt(disc)
    solutions = []

    for sign in (+1.0, -1.0):
        cos_cm = (-B_q + sign * sqrt_disc) / (2.0 * A_q)
        if abs(cos_cm) > 1.0 + 1e-9:
            continue
        cos_cm_clipped = float(np.clip(cos_cm, -1.0, 1.0))

        # Reject spurious roots introduced by squaring: verify the candidate
        # reproduces theta_lab via the forward transform.
        sin_cm = float(np.sqrt(max(1.0 - cos_cm_clipped**2, 0.0)))
        pz = gamma * (p_star * cos_cm_clipped + beta * E_star_b)
        pt = p_star * sin_cm
        theta_lab_check = float(np.arctan2(pt, pz))
        if abs(theta_lab_check - theta_lab) > 1e-6:
            continue

        solutions.append(_make_solution(cos_cm_clipped))

    return solutions


def max_lab_angle(
    m_a: float, m_A: float, m_b: float, m_B: float, T_a: float
) -> "float | None":
    """
    Maximum lab angle for ejectile b when beta_cm >= beta*_b (inverse kinematics).

    Returns the maximum lab angle in radians, or None if the ejectile covers
    the full 0–180° lab range (normal kinematics).
    """
    E_a = T_a + m_a
    p_a = np.sqrt(max(E_a**2 - m_a**2, 0.0))
    s = m_a**2 + m_A**2 + 2.0 * m_A * E_a
    sqrts = np.sqrt(s)
    beta = p_a / (E_a + m_A)

    num = (s - (m_b + m_B)**2) * (s - (m_b - m_B)**2)
    p_star   = np.sqrt(max(num, 0.0)) / (2.0 * sqrts)
    E_star_b = (s + m_b**2 - m_B**2) / (2.0 * sqrts)
    beta_star_b = p_star / E_star_b

    if beta >= beta_star_b:
        return float(np.arcsin(beta_star_b / beta))
    return None


def kinematic_locus(
    m_a: float,
    m_A: float,
    m_b: float,
    m_B: float,
    T_a: float,
    n_points: int = 500,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Kinematic locus: lab angle vs. lab kinetic energy of ejectile b.

    Sweeps theta_b_cm from 0 to pi in n_points steps.

    Returns
    -------
    theta_b_lab_deg : array of lab angles in degrees
    T_b_lab         : array of lab kinetic energies in MeV
    """
    theta_cm = np.linspace(0.01, np.pi - 0.01, n_points)
    theta_b_lab, T_b_lab = lab_angle_energy(theta_cm, m_a, m_A, m_b, m_B, T_a)
    return np.degrees(theta_b_lab), T_b_lab


def kinematic_compression(
    theta_b_cm: "float | np.ndarray",
    m_a: float,
    m_A: float,
    m_b: float,
    m_B: float,
    T_a: float,
) -> np.ndarray:
    """
    Kinematic compression factor kp = (1/rho) drho/dtheta_lab.

    For an Enge split-pole spectrograph, kp quantifies how the magnetic
    rigidity rho = p_b / (qB) changes with lab angle. Since rho prop p_b,
    this is equivalent to (1/p_b) dp_b/dtheta_lab.

    Derived analytically via the chain rule through theta_cm:
        kp = (-gamma*pz*sin_cm + pt*cos_cm) / (pz*cos_cm + gamma*pt*sin_cm)

    Parameters
    ----------
    theta_b_cm : float or array, CM angle of ejectile b (radians)
    m_a, m_A   : projectile and target masses (MeV/c^2)
    m_b, m_B   : ejectile and residual masses (MeV/c^2)
    T_a        : beam lab kinetic energy (MeV)

    Returns
    -------
    kp : np.ndarray, kinematic compression factor (rad^-1)
         Positive: rho increases with theta_lab.
         kp = 0 at theta_cm = 0 and pi.
         Diverges at the kinematic limit (dtheta_lab/dtheta_cm = 0).
    """
    theta_b_cm = np.atleast_1d(np.asarray(theta_b_cm, dtype=float))

    E_a = T_a + m_a
    p_a = np.sqrt(max(E_a**2 - m_a**2, 0.0))
    s = m_a**2 + m_A**2 + 2.0 * m_A * E_a
    sqrts = np.sqrt(s)
    beta = p_a / (E_a + m_A)
    gamma = (E_a + m_A) / sqrts

    num = (s - (m_b + m_B)**2) * (s - (m_b - m_B)**2)
    p_star = np.sqrt(max(num, 0.0)) / (2.0 * sqrts)
    E_star_b = (s + m_b**2 - m_B**2) / (2.0 * sqrts)

    cos_cm = np.cos(theta_b_cm)
    sin_cm = np.sin(theta_b_cm)

    pz = gamma * (p_star * cos_cm + beta * E_star_b)
    pt = p_star * sin_cm

    numerator   = -gamma * pz * sin_cm + pt * cos_cm
    denominator =  pz * cos_cm + gamma * pt * sin_cm

    return numerator / denominator


def recoil_max_lab_angle(m_a: float, m_A: float, T_a: float) -> float:
    """
    Half-angle of the forward cone into which the residual B is focused
    in inverse kinematics.

    Non-relativistic approximation:
      theta_max(B) ≈ arcsin(m_A / m_a)

    Returns the half-angle in radians. Only physically meaningful when
    m_a > m_A (inverse kinematics); for normal kinematics (m_a <= m_A)
    the residual covers the full forward hemisphere.
    """
    ratio = m_A / m_a
    if ratio >= 1.0:
        return float(np.pi / 2)
    return float(np.arcsin(ratio))
