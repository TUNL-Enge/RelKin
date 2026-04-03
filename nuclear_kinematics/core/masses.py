"""
AME2020 nuclear mass utilities.

All masses are *atomic* masses (includes electron masses).
For reactions conserving Z, electron masses cancel exactly.
For reactions changing Z (e.g. beta decay), a correction of
(delta_Z * m_electron) applies but is not applied automatically here.
"""
import functools
import re

import numpy as np
import pandas as pd

from .constants import U_TO_MEV

# Common nuclide aliases → (Z, A)
_ALIASES: dict[str, tuple[int, int]] = {
    "p":  (1, 1),
    "n":  (0, 1),
    "d":  (1, 2),
    "t":  (1, 3),
    "a":  (2, 4),
    "4he": (2, 4),
}

# Element symbol → Z mapping (Z=1..118)
_SYMBOLS = [
    "H",  "He", "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Ne",
    "Na", "Mg", "Al", "Si", "P",  "S",  "Cl", "Ar", "K",  "Ca",
    "Sc", "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y",  "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I",  "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W",  "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U",  "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
    "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
    "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
]
_SYMBOL_TO_Z = {s.upper(): i + 1 for i, s in enumerate(_SYMBOLS)}


@functools.lru_cache(maxsize=1)
def load_ame2020() -> pd.DataFrame:
    """Load AME2020 mass table via nuclearmasses package. Cached after first call."""
    from nuclearmasses.mass_table import MassTable
    return MassTable().full_data.query("TableYear == 2020")


def get_mass_MeV(Z: int, A: int) -> float:
    """
    Return the AME2020 atomic mass of nuclide (Z, A) in MeV/c².

    Atomic mass M = A * u_to_MeV + delta_keV / 1000
    where delta_keV is the AME2020 mass excess in keV.

    Raises KeyError if the nuclide is not in AME2020.
    """
    df = load_ame2020()
    row = df[(df["Z"] == Z) & (df["A"] == A)]
    if row.empty:
        raise KeyError(f"Nuclide Z={Z}, A={A} not found in AME2020")
    delta_keV = float(row["AMEMassExcess"].iloc[0])
    # print(f"A = {A}; delta_keV = {delta_keV}")
    return A * U_TO_MEV + delta_keV / 1000.0


def parse_nuclide(s: str) -> tuple[int, int]:
    """
    Parse a nuclide string into (Z, A).

    Accepted formats:
      "p", "n", "d", "t", "a"         — common aliases
      "12C", "4He", "208Pb"            — mass number + symbol
      "C12", "He4", "Pb208"            — symbol + mass number (alternative)

    Returns (Z, A) tuple.
    Raises ValueError if the string cannot be parsed.
    """
    key = s.strip().lower()
    if key in _ALIASES:
        return _ALIASES[key]

    # Try "12C" or "208Pb" format (digits first)
    m = re.fullmatch(r"(\d+)([A-Za-z]+)", s.strip())
    if m:
        A = int(m.group(1))
        sym = m.group(2).upper()
        if sym not in _SYMBOL_TO_Z:
            raise ValueError(f"Unknown element symbol: {m.group(2)!r}")
        return _SYMBOL_TO_Z[sym], A

    # Try "C12" or "Pb208" format (letters first)
    m = re.fullmatch(r"([A-Za-z]+)(\d+)", s.strip())
    if m:
        sym = m.group(1).upper()
        A = int(m.group(2))
        if sym not in _SYMBOL_TO_Z:
            raise ValueError(f"Unknown element symbol: {m.group(1)!r}")
        return _SYMBOL_TO_Z[sym], A

    raise ValueError(f"Cannot parse nuclide string: {s!r}")


def format_nuclide(Z: int, A: int) -> str:
    """Return canonical nuclide string, e.g. format_nuclide(6, 12) → '12C'."""
    if Z == 0 and A == 1:
        return "n"
    if Z < 1 or Z > len(_SYMBOLS):
        raise ValueError(f"Z={Z} out of range")
    return f"{A}{_SYMBOLS[Z - 1]}"


def qvalue(
    ZA: tuple[int, int],
    ZB: tuple[int, int],
    ZC: tuple[int, int],
    ZD: tuple[int, int],
) -> float:
    """
    Q-value for reaction A + B → C + D in MeV.

    Q = (m_A + m_B) - (m_C + m_D)

    Positive Q: exothermic. Negative Q: endothermic.
    Uses AME2020 atomic masses.
    """
    mA = get_mass_MeV(*ZA)
    mB = get_mass_MeV(*ZB)
    mC = get_mass_MeV(*ZC)
    mD = get_mass_MeV(*ZD)
    return (mA + mB) - (mC + mD)
