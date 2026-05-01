"""
ASME B31.3 Process Piping Code stress computations.

Computes:
  - Expansion stress SE from flexibility analysis moments
  - Sustained stress SL from pressure, weight, and sustained loads
  - Allowable stress SA = f * (1.25*Sc + 0.25*Sh)
  - Stress intensification factors (SIF) for bends and tees
  - Section modulus Z for circular pipe cross-sections

All formulas from ASME B31.3-2020 Chapter II.

References:
    Caponetto / Giudice et al. (2022), ANN-Based Optimization of Pressure Piping
    (Designs 6(6), 103, MDPI)
"""

import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


# ─────────────────────────────────────────────────────────────────────────────
# Pipe geometry and section properties
# ─────────────────────────────────────────────────────────────────────────────

# Standard pipe schedule lookup: (nominal_inch, schedule) → (OD_mm, t_mm)
PIPE_SCHEDULE = {
    (4,  "Sch40"): (114.3, 6.02),
    (4,  "Sch80"): (114.3, 8.56),
    (6,  "Sch40"): (168.3, 7.11),
    (6,  "Sch80"): (168.3, 10.97),
    (8,  "Sch40"): (219.1, 8.18),
    (8,  "Sch80"): (219.1, 12.70),
    (10, "Sch40"): (273.1, 9.27),
    (10, "Sch80"): (273.1, 15.09),
    (12, "Sch40"): (323.9, 9.53),
    (12, "Sch80"): (323.9, 17.48),
}


def section_modulus(D: float, t: float) -> float:
    """
    Pipe section modulus Z (m³).

        Z = π(D⁴ - d⁴) / (32D)

    where d = D - 2t is the inner diameter.

    Args:
        D : outer diameter (m)
        t : wall thickness (m)

    Returns:
        Z : section modulus (m³)
    """
    d = D - 2.0 * t
    return np.pi * (D**4 - d**4) / (32.0 * D)


def cross_section_area(D: float, t: float) -> float:
    """Pipe metal cross-section area (m²)."""
    d = D - 2.0 * t
    return np.pi / 4.0 * (D**2 - d**2)


def second_moment_of_area(D: float, t: float) -> float:
    """Second moment of area I (m⁴) for hollow circular section."""
    d = D - 2.0 * t
    return np.pi / 64.0 * (D**4 - d**4)


# ─────────────────────────────────────────────────────────────────────────────
# Stress intensification factors (SIF)
# ─────────────────────────────────────────────────────────────────────────────

def sif_bend(r: float, D: float, t: float) -> Tuple[float, float]:
    """
    In-plane (ii) and out-of-plane (io) SIF for a pipe bend (elbow).

    From ASME B31.3 Appendix D:
        h  = t * R / r²           (pipe bend characteristic)
        ii = 0.9 / h^(2/3)        (in-plane)
        io = 0.75 / h^(2/3)       (out-of-plane)
        (minimum SIF = 1.0)

    Args:
        r : bend radius to pipe centreline (m)
        D : pipe outer diameter (m)
        t : wall thickness (m)

    Returns:
        ii, io : in-plane and out-of-plane SIFs
    """
    R = D / 2.0
    h = t * r / R**2
    ii = max(0.9 / h**(2.0/3.0), 1.0)
    io = max(0.75 / h**(2.0/3.0), 1.0)
    return ii, io


def sif_tee_welding(D: float, t: float, Db: float, tb: float) -> Tuple[float, float]:
    """
    SIF for welding tee fitting (run and branch).

    Simplified formula from ASME B31.3 Appendix D, Table D300.

    Args:
        D, t   : run pipe OD and wall thickness (m)
        Db, tb : branch pipe OD and wall thickness (m)

    Returns:
        i_run, i_branch : SIFs for run and branch
    """
    r2 = D / 2.0
    T  = t    # header wall thickness
    # Flexibility characteristic
    h  = 4.4 * T / r2
    i_run    = max(0.9 / h**(2.0/3.0), 1.0)
    i_branch = max(0.9 / h**(2.0/3.0), 1.0) * (D / Db) ** 0.5
    return i_run, i_branch


# ─────────────────────────────────────────────────────────────────────────────
# Stress calculations
# ─────────────────────────────────────────────────────────────────────────────

def expansion_stress(
    Mi: float,
    Mo: float,
    Mt: float,
    ii: float,
    io: float,
    Z: float,
) -> float:
    """
    ASME B31.3 Expansion stress SE at a point.

        Sb = sqrt((ii*Mi)² + (io*Mo)²) / Z    [resultant bending stress]
        St = Mt / (2*Z)                        [torsional stress]
        SE = sqrt(Sb² + 4*St²)

    Args:
        Mi, Mo : in-plane and out-of-plane bending moments (N·m)
        Mt     : torsional moment (N·m)
        ii, io : in-plane and out-of-plane SIFs
        Z      : section modulus (m³)

    Returns:
        SE : expansion stress (Pa)
    """
    Sb = np.sqrt((ii * Mi) ** 2 + (io * Mo) ** 2) / Z
    St = Mt / (2.0 * Z)
    SE = np.sqrt(Sb ** 2 + 4.0 * St ** 2)
    return SE


def sustained_stress(
    P: float,
    D: float,
    t: float,
    i_factor: float,
    MA: float,
    Z: float,
) -> float:
    """
    ASME B31.3 Sustained stress SL.

        SL = P*D/(4t) + 0.75*i*MA/Z   ≤ Sh

    Args:
        P        : internal pressure (Pa)
        D        : outer diameter (m)
        t        : wall thickness (m)
        i_factor : SIF at the point (max of ii, io)
        MA       : resultant moment from weight + sustained loads (N·m)
        Z        : section modulus (m³)

    Returns:
        SL : sustained stress (Pa)
    """
    pressure_term  = P * D / (4.0 * t)
    bending_term   = 0.75 * i_factor * MA / Z
    return pressure_term + bending_term


def allowable_expansion_stress(
    Sc: float,
    Sh: float,
    SL: float,
    f: float = 1.0,
) -> float:
    """
    Allowable displacement stress range SA (ASME B31.3 Eq. 302.3.5).

        SA = f * (1.25*Sc + 0.25*Sh)
    If SL < Sh:
        SA = f * [1.25*(Sc + Sh) - SL]   (enhanced allowable)

    Args:
        Sc  : basic allowable stress at cold condition (Pa)
        Sh  : basic allowable stress at hot condition (Pa)
        SL  : sustained stress (Pa, to check enhanced allowable applicability)
        f   : stress range reduction factor (1.0 for ≤7000 cycles)

    Returns:
        SA : allowable displacement stress range (Pa)
    """
    base_SA = f * (1.25 * Sc + 0.25 * Sh)
    if SL < Sh:
        enhanced_SA = f * (1.25 * (Sc + Sh) - SL)
        return max(base_SA, enhanced_SA)
    return base_SA


def hoop_stress_b313(P: float, D: float, t: float) -> float:
    """
    Hoop stress from internal pressure (pressure design check).

        σ_h = P * D / (2 * t)

    Returns:
        sigma_h : hoop stress (Pa)
    """
    return P * D / (2.0 * t)


def utilisation_ratio(stress: float, allowable: float) -> float:
    """
    Code utilisation ratio = stress / allowable. ≤ 1.0 for compliant design.
    """
    return stress / (allowable + 1e-12)


# ─────────────────────────────────────────────────────────────────────────────
# Material allowable stresses lookup (Carbon steel A106 Gr.B / A53)
# ─────────────────────────────────────────────────────────────────────────────

def basic_allowable_stress_carbon_steel(T_C: float) -> Tuple[float, float]:
    """
    Approximate basic allowable stress for ASTM A106 Gr.B carbon steel pipe.
    Interpolated from ASME B31.3 Table A-1.

    Args:
        T_C : design temperature (°C)

    Returns:
        Sc : allowable at cold (ambient = 20°C) (Pa)
        Sh : allowable at hot temperature T_C (Pa)
    """
    # Table values (temperature °C, allowable stress MPa)
    T_table = np.array([20,  50, 100, 150, 200, 250, 300, 350, 400])
    S_table = np.array([138, 138, 138, 138, 131, 117, 110, 103, 97]) * 1e6

    Sc = float(np.interp(20.0,  T_table, S_table))
    Sh = float(np.interp(T_C,   T_table, S_table))
    return Sc, Sh


# ─────────────────────────────────────────────────────────────────────────────
# Complete code check for a single critical node
# ─────────────────────────────────────────────────────────────────────────────

def asme_b313_code_check(
    P: float,
    D: float,
    t: float,
    T_install: float,
    T_operate: float,
    Mi_exp: float,
    Mo_exp: float,
    Mt_exp: float,
    MA_sust: float,
    r_bend: Optional[float] = None,
    f_factor: float = 1.0,
) -> Dict:
    """
    Full ASME B31.3 code check at a single critical node (e.g., bend elbow).

    Args:
        P                : design pressure (Pa)
        D                : outer diameter (m)
        t                : wall thickness (m)
        T_install        : installation temperature (°C)
        T_operate        : operating temperature (°C)
        Mi_exp, Mo_exp   : in-plane and out-of-plane moments from thermal expansion (N·m)
        Mt_exp           : torsional moment from thermal expansion (N·m)
        MA_sust          : resultant sustained bending moment (N·m)
        r_bend           : bend centreline radius (m); None = straight pipe (SIF=1)
        f_factor         : stress range reduction factor

    Returns:
        results dict with SE, SL, SA, Sh, utilisation ratios, and PASS/FAIL flags
    """
    Z  = section_modulus(D, t)
    Sc, Sh = basic_allowable_stress_carbon_steel(T_install)
    _,  Sh_hot = basic_allowable_stress_carbon_steel(T_operate)

    if r_bend is not None:
        ii, io = sif_bend(r_bend, D, t)
    else:
        ii, io = 1.0, 1.0

    SE = expansion_stress(Mi_exp, Mo_exp, Mt_exp, ii, io, Z)
    SL = sustained_stress(P, D, t, max(ii, io), MA_sust, Z)
    SA = allowable_expansion_stress(Sc, Sh_hot, SL, f_factor)

    return {
        "SE":             SE,
        "SL":             SL,
        "SA":             SA,
        "Sh":             Sh_hot,
        "Sc":             Sc,
        "Z":              Z,
        "ii":             ii,
        "io":             io,
        "ur_expansion":   utilisation_ratio(SE, SA),
        "ur_sustained":   utilisation_ratio(SL, Sh_hot),
        "pass_expansion": SE <= SA,
        "pass_sustained": SL <= Sh_hot,
        "pass_all":       (SE <= SA) and (SL <= Sh_hot),
    }
