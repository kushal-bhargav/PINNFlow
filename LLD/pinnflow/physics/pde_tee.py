"""
pinnflow/physics/pde_tee.py
────────────────────────────
Phase 3-A — Tee Junction PDE Residuals

Physics equations for tee (branch) junction pressure drop and
stress concentration, for use in the MoEPINN expert loss.

References:
    ASME B31.3 Appendix D — Stress Intensification Factors
    Idelchik (2008) Handbook of Hydraulic Resistance — §5, branching flows
    Crane TP-410 — K-factors for tee fittings

All functions operate on raw (unscaled) feature arrays X_raw with the
standard 10-column layout:
    col 0: diameter_mm
    col 1: thickness_mm
    col 2: length_m
    col 3: pressure_MPa
    col 4: soil_disp_mm
    col 5: delta_T_degC
    col 6: velocity_m_s
    col 7: soil_stiffness
    col 8: shape_id          (2 = tee)
    col 9: shape_param       (= d_branch / d_run  for tees, ∈ [0.25, 1.0])
"""
from __future__ import annotations

import numpy as np

# ── Loss coefficients for branch-flow fittings ────────────────────────────
# From Crane TP-410 and Idelchik §5
_K_BRANCH: float = 1.0    # run → branch turn pressure loss coefficient
_K_RUN:    float = 0.6    # run → run (through) loss coefficient
_DENSITY_GAS_KG_M3: float = 1000.0   # Pa·s reference (water density used in normalisation; gas handled below)
_GAS_DENSITY_KG_M3: float = 0.8      # approximate natural gas density at 50 bar, 289 K


def tee_branch_pressure_drop(X_raw: np.ndarray) -> np.ndarray:
    """
    Compute branch-side pressure drop for a tee junction using
    the K-factor (resistance coefficient) method.

    ΔP_branch = K_branch × ρ × v_run² / 2   [Pa]
    ΔP_run    = K_run    × ρ × v_run² / 2   [Pa]

    We return the branch pressure drop (conservative, larger value)
    converted to kPa.

    Args:
        X_raw: ndarray shape (N, ≥10)  — raw feature matrix.

    Returns:
        ndarray shape (N,)  — branch-side ΔP in kPa.
    """
    v = np.maximum(X_raw[:, 6], 0.1)   # velocity m/s
    dP_pa = _K_BRANCH * _GAS_DENSITY_KG_M3 * v ** 2 / 2.0
    return dP_pa / 1000.0   # Pa → kPa


def tee_run_pressure_drop(X_raw: np.ndarray) -> np.ndarray:
    """
    Compute run-through pressure drop for a tee junction.

    Returns:
        ndarray shape (N,)  — run-through ΔP in kPa.
    """
    v = np.maximum(X_raw[:, 6], 0.1)
    dP_pa = _K_RUN * _GAS_DENSITY_KG_M3 * v ** 2 / 2.0
    return dP_pa / 1000.0


def tee_stress_concentration(X_raw: np.ndarray) -> np.ndarray:
    """
    Compute the tee stress intensification factor (SIF) per
    ASME B31.3 Appendix D form analogous to elbow SIF.

    h   = t × R_run / (D_run / 2)²     (characteristic parameter)
              where R_run = D_run / 2   (run pipe radius)
    i   = 0.9 / h^(2/3)                (SIF, same form as elbow)

    For tees the branch-to-run ratio (shape_param, col 9) also scales i:
        i_tee = i × (1 + 0.5 × d_ratio)   where d_ratio = d_branch/d_run ∈ [0.25, 1.0]

    Returns:
        ndarray shape (N,)  — SIF values (dimensionless).
    """
    D_mm = np.maximum(X_raw[:, 0], 1.0)
    t_mm = np.maximum(X_raw[:, 1], 1.0)
    d_ratio = np.clip(X_raw[:, 9] if X_raw.shape[1] > 9 else np.ones(len(X_raw)), 0.25, 1.0)

    R_m   = (D_mm / 2.0) / 1000.0   # metres
    t_m   = t_mm / 1000.0

    h = np.maximum(t_m * R_m / np.maximum(R_m ** 2, 1e-8), 1e-4)
    sif_base = 0.9 / h ** (2.0 / 3.0)
    sif_tee  = sif_base * (1.0 + 0.5 * d_ratio)
    return np.clip(sif_tee, 1.0, 15.0)


def tee_hoop_stress_sif(X_raw: np.ndarray) -> np.ndarray:
    """
    Compute the SIF-amplified hoop stress at the tee crotch.

    σ_tee = i_tee × P × D / (2 × t)    [MPa]

    Returns:
        ndarray shape (N,)  — amplified stress in MPa.
    """
    D_mm = np.maximum(X_raw[:, 0], 1.0)
    t_mm = np.maximum(X_raw[:, 1], 1.0)
    P_mpa = np.maximum(X_raw[:, 3], 0.0)

    sigma_base = P_mpa * D_mm / (2.0 * t_mm)
    sif        = tee_stress_concentration(X_raw)
    return np.clip(sigma_base * sif, 1.0, 600.0)


def tee_residuals(
    X_raw: np.ndarray,
    sigma_pred: np.ndarray,
    dp_pred: np.ndarray,
) -> dict:
    """
    Compute PDE residual losses for the tee junction MoE expert.

    Residuals:
        stress_res : MSE between predicted stress and SIF-amplified hoop stress target
        flow_res   : MSE between predicted ΔP and K-factor branch ΔP target

    Args:
        X_raw:      (N, ≥10) raw feature array
        sigma_pred: (N,) or (N,1) predicted von Mises stress [MPa]
        dp_pred:    (N,) or (N,1) predicted pressure drop [kPa]

    Returns:
        dict with keys: "stress_residual", "flow_residual", "sif_values"
    """
    sp = sigma_pred.ravel()
    fp = dp_pred.ravel()

    sigma_target = tee_hoop_stress_sif(X_raw)
    dp_target    = tee_branch_pressure_drop(X_raw)

    stress_res = float(np.mean((sp - sigma_target) ** 2))
    flow_res   = float(np.mean((fp - dp_target) ** 2))
    sif_vals   = tee_stress_concentration(X_raw)

    return {
        "stress_residual": stress_res,
        "flow_residual":   flow_res,
        "sif_values":      sif_vals,
    }
