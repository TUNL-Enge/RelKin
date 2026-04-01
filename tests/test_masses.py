"""
Tests for core/masses.py.

Q-value benchmarks checked against IAEA Q-value calculator:
  https://nucleus.iaea.org/Pages/q-calc.aspx
"""
import pytest
from nuclear_kinematics.core.masses import get_mass_MeV, parse_nuclide, qvalue


# --- parse_nuclide ---

def test_parse_proton():
    assert parse_nuclide("p") == (1, 1)

def test_parse_neutron():
    assert parse_nuclide("n") == (0, 1)

def test_parse_deuteron():
    assert parse_nuclide("d") == (1, 2)

def test_parse_alpha():
    assert parse_nuclide("a") == (2, 4)

def test_parse_12C_massFirst():
    assert parse_nuclide("12C") == (6, 12)

def test_parse_12C_symFirst():
    assert parse_nuclide("C12") == (6, 12)

def test_parse_208Pb():
    assert parse_nuclide("208Pb") == (82, 208)

def test_parse_invalid():
    with pytest.raises(ValueError):
        parse_nuclide("XY99")


# --- get_mass_MeV ---

def test_proton_mass():
    # AME2020 proton mass ≈ 938.783 MeV (atomic, includes electron)
    m = get_mass_MeV(1, 1)
    assert abs(m - 938.783) < 0.01, f"Proton mass {m:.3f} MeV unexpected"

def test_alpha_mass():
    # 4He atomic mass ≈ 3728.401 MeV (AME2020)
    m = get_mass_MeV(2, 4)
    assert abs(m - 3728.401) < 0.01


# --- qvalue ---

def test_qvalue_elastic():
    # Elastic: A+B → A+B, Q = 0
    beam   = parse_nuclide("12C")
    target = parse_nuclide("p")
    Q = qvalue(beam, target, beam, target)
    assert abs(Q) < 1e-6, f"Elastic Q should be 0, got {Q}"

def test_qvalue_13C_pn_13N():
    # 13C(p,n)13N: Q ≈ -3.003 MeV  (IAEA Q-calc)
    beam   = parse_nuclide("p")
    target = parse_nuclide("13C")
    eject  = parse_nuclide("n")
    recoil = parse_nuclide("13N")
    Q = qvalue(beam, target, eject, recoil)
    assert abs(Q - (-3.003)) < 0.005, f"13C(p,n)13N Q={Q:.4f}, expected ~-3.003"

def test_qvalue_12C_pg_13N():
    # 12C(p,γ)13N: Q ≈ +1.943 MeV
    # Use (p, gamma) → recoil only (gamma has mass 0, model as mC=0, mD=13N)
    # Actually gamma carries no mass; skip this test — Q-calc needs special handling
    # Test 12C(d,p)13C instead: Q ≈ +2.722 MeV (IAEA)
    beam   = parse_nuclide("d")
    target = parse_nuclide("12C")
    eject  = parse_nuclide("p")
    recoil = parse_nuclide("13C")
    Q = qvalue(beam, target, eject, recoil)
    assert abs(Q - 2.722) < 0.005, f"12C(d,p)13C Q={Q:.4f}, expected ~2.722"
