"""
pinnflow/physics/pde_reducer.py
────────────────────────────────
Phase 3-B — Reducer Transition PDE Residuals

Physics equations for concentric/eccentric reducer pressure drop and
stress amplification, for use in the MoEPINN expert loss.

References:
    ASME B31.3 § 304.7.4 — Reducers (included angle limit ≤ 30°)
    Idelchik (2008) Handbook §5 — Gradual contraction
    Crane TP-410 — contraction/expansion loss coefficients

Column layout of X_raw (standard 10-column schema):
    col 0: diameter_mm      (outlet / smaller end, D2)
    col 1: thickness_mm     (outlet wall)
    col 2: length_m         (reducer length)
    col 3: pressure_MPa
    col 4: soil_disp_mm
    col 5: delta_T_degC
    col 6: velocity_m_s     (velocity at outlet, larger)
    col 7: soil_stiffness
    col 8: shape_id         (3 = reducer)
    col 9: shape_param      (= D2 / D1, diameter ratio ∈ (0, 1.0))
                             where D1 = inlet (larger) diameter, D2 = outlet
"""
from __future__ import annotations

import numpy as np

_GAS_DENSITY_KG_M3: float = 0.8   # natural gas at ~50 bar, 289 K


def reducer_included_half_angle_deg(X_raw: np.ndarray) -> np.ndarray:
    """
    Compute the included half-angle (α) of the concentric reducer.

    Geometry:   α = arctan( (D1 - D2) / (2 × L) )
                D1 = D2 / shape_param  (inlet diameter derived from outlet + ratio)
                D2 = diameter_mm (col 0)
                L  = length_m (col 2) in mm for consistent units

    ASME B31.3 § 304.7.4 limit:  α ≤ 30°

    Returns:
        ndarray shape (N,)  — half-angle in degrees.
    """
    D2_mm = np.maximum(X_raw[:, 0], 1.0)
    L_m   = np.maximum(X_raw[:, 2], 0.1)
    d_ratio = np.clip(
        X_raw[:, 9] if X_raw.shape[1] > 9 else np.full(len(X_raw), 0.8),
        0.1, 0.99,
    )
    # D1 = D2 / ratio → (D1 - D2) = D2 × (1/ratio - 1)
    delta_D_mm = D2_mm * (1.0 / d_ratio - 1.0)
    L_mm = L_m * 1000.0
    alpha_rad = np.arctan(delta_D_mm / np.maximum(2.0 * L_mm, 1e-3))
    return np.degrees(alpha_rad)


def reducer_pressure_drop(X_raw: np.ndarray) -> np.ndarray:
    """
    Compute pressure drop across a concentric reducer (gradual contraction).

    For included half-angle α < 45°:
        K_c = 0.04 × tan(α)   (Idelchik gradual contraction)
    For α ≥ 45° (abrupt):
        K_c = 0.5 × (1 - (D2/D1)²)

    ΔP = K_c × ρ × v2² / 2   [Pa] → kPa

    Args:
        X_raw: (N, ≥10) raw feature matrix.

    Returns:
        ndarray shape (N,)  — ΔP in kPa.
    """
    v2     = np.maximum(X_raw[:, 6], 0.1)   # outlet velocity m/s
    d_ratio = np.clip(
        X_raw[:, 9] if X_raw.shape[1] > 9 else np.full(len(X_raw), 0.8),
        0.1, 0.99,
    )
    alpha_deg = reducer_included_half_angle_deg(X_raw)
    alpha_rad = np.radians(alpha_deg)

    K_gradual = 0.04 * np.tan(alpha_rad)
    K_abrupt  = 0.5 * (1.0 - d_ratio ** 2)
    K_c = np.where(alpha_deg < 45.0, K_gradual, K_abrupt)
    K_c = np.clip(K_c, 0.01, 0.5)

    dP_pa = K_c * _GAS_DENSITY_KG_M3 * v2 ** 2 / 2.0
    return dP_pa / 1000.0   # Pa → kPa


def reducer_stress_factor(X_raw: np.ndarray) -> np.ndarray:
    """
    Compute stress amplification factor for the reducer transition zone.

    For gradual reducers (α ≤ 15°): factor ≈ 1.0 (no amplification)
    Linear interpolation up to α = 30° (ASME limit): factor up to 1.2
    Beyond 30° (non-code geometry): factor extrapolated to 2.0 max.

    σ_reducer = factor × σ_hoop

    Returns:
        ndarray shape (N,)  — dimensionless amplification factor.
    """
    alpha_deg = reducer_included_half_angle_deg(X_raw)
    # Piecewise: 1.0 at α=0, 1.2 at α=30°, capped at 2.0
    factor = 1.0 + np.clip(alpha_deg / 30.0, 0.0, 1.0) * 0.2
    # Extra penalty beyond code limit
    factor += np.where(alpha_deg > 30.0, (alpha_deg - 30.0) / 30.0 * 0.5, 0.0)
    return np.clip(factor, 1.0, 2.0)


def reducer_amplified_stress(X_raw: np.ndarray) -> np.ndarray:
    """
    Compute the amplified hoop stress at the reducer transition.

    σ_reducer = factor × P × D2 / (2 × t)   [MPa]

    Returns:
        ndarray shape (N,)  — amplified stress in MPa.
    """
    D2_mm  = np.maximum(X_raw[:, 0], 1.0)
    t_mm   = np.maximum(X_raw[:, 1], 1.0)
    P_mpa  = np.maximum(X_raw[:, 3], 0.0)

    sigma_base = P_mpa * D2_mm / (2.0 * t_mm)
    factor     = reducer_stress_factor(X_raw)
    return np.clip(sigma_base * factor, 1.0, 600.0)


def reducer_residuals(
    X_raw: np.ndarray,
    sigma_pred: np.ndarray,
    dp_pred: np.ndarray,
) -> dict:
    """
    Compute PDE residual losses for the reducer MoE expert.

    Args:
        X_raw:      (N, ≥10) raw feature array.
        sigma_pred: (N,) or (N,1) predicted von Mises stress [MPa].
        dp_pred:    (N,) or (N,1) predicted pressure drop [kPa].

    Returns:
        dict with keys: "stress_residual", "flow_residual",
                        "alpha_deg", "stress_factors", "angle_violation"
    """
    sp = sigma_pred.ravel()
    fp = dp_pred.ravel()

    sigma_target = reducer_amplified_stress(X_raw)
    dp_target    = reducer_pressure_drop(X_raw)
    alpha_deg    = reducer_included_half_angle_deg(X_raw)
    factors      = reducer_stress_factor(X_raw)

    stress_res  = float(np.mean((sp - sigma_target) ** 2))
    flow_res    = float(np.mean((fp - dp_target) ** 2))
    # Flag segments exceeding ASME B31.3 § 304.7.4 half-angle limit of 30°
    angle_violation = bool(np.any(alpha_deg > 30.0))

    return {
        "stress_residual":  stress_res,
        "flow_residual":    flow_res,
        "alpha_deg":        alpha_deg,
        "stress_factors":   factors,
        "angle_violation":  angle_violation,
    }
