"""
Synthetic dataset generation for ASME B31.3 piping stress prediction.

Performs a systematic parameter sweep over expansion loop geometry and operating
conditions, computes ASME B31.3 code stresses analytically via simplified
flexibility analysis, and stores the resulting training dataset.

Parameter ranges:
  - Loop height H     : [1.0, 5.0] m
  - Loop width W      : [0.5, 3.0] m
  - Pipe diameter D   : {4", 6", 8", 10", 12"} nominal
  - Wall schedule     : Sch40 or Sch80
  - Temperature diff  : [50, 250] °C
  - Pressure          : [0.5, 5.0] MPa

References:
    Caponetto / Giudice et al. (2022), ANN-Based Optimization of Pressure Piping
"""

import numpy as np
import pandas as pd
from itertools import product
from typing import Dict, Tuple, Optional
import warnings

from .asme_b313.stress_formulas import (
    section_modulus, asme_b313_code_check, PIPE_SCHEDULE,
    basic_allowable_stress_carbon_steel
)


# ─────────────────────────────────────────────────────────────────────────────
# Simplified flexibility analysis for an L-shaped expansion loop
# ─────────────────────────────────────────────────────────────────────────────

def simplified_thermal_moments(
    H: float,
    W: float,
    D: float,
    t: float,
    E: float,
    alpha: float,
    delta_T: float,
    bend_radius: float = None,
) -> Dict[str, float]:
    """
    Simplified analytical flexibility analysis for an L-shaped (2-leg) loop.

    Assumes:
      - Two legs of lengths H (vertical) and W (horizontal)
      - Fixed at both anchor points
      - Thermal expansion drives the loading

    Approximate resultant moments at the bend (critical location):

    This uses the guided-cantilever approximation commonly used in
    piping stress pre-screening (ASME B31.3 Appendix P, non-mandatory).

        δ_axial ≈ α * ΔT * W          (horizontal leg elongation)
        M_bend  ≈ 3EI * δ_axial / H²  (approximate fixed-end moment)

    Args:
        H, W       : loop leg lengths (m)
        D, t       : pipe OD and wall thickness (m)
        E          : Young's modulus (Pa)
        alpha      : thermal expansion coefficient (1/°C)
        delta_T    : temperature difference from ambient to operating (°C)
        bend_radius: elbow bend radius (m); default = 1.5 * D / 2

    Returns:
        dict with Mi_exp, Mo_exp, Mt_exp, MA_sust (N·m)
    """
    from .asme_b313.stress_formulas import second_moment_of_area, cross_section_area

    if bend_radius is None:
        bend_radius = 1.5 * D / 2.0

    I   = second_moment_of_area(D, t)
    A   = cross_section_area(D, t)
    EI  = E * I

    # Thermal displacement at free end of each leg
    delta_H = alpha * delta_T * H    # vertical leg expansion (m)
    delta_W = alpha * delta_T * W    # horizontal leg expansion (m)

    # Guided-cantilever moment: M ≈ 3EI * δ / L²
    M_from_W = 3.0 * EI * delta_W / (H ** 2 + 1e-8)
    M_from_H = 3.0 * EI * delta_H / (W ** 2 + 1e-8)

    Mi_exp = M_from_W                    # in-plane bending at elbow
    Mo_exp = M_from_H                    # out-of-plane bending
    Mt_exp = 0.0                         # torsion (zero for planar L-loop)

    # Sustained moment (weight): approximate self-weight bending at mid-span
    rho_steel = 7850.0
    g         = 9.81
    w_pipe    = rho_steel * A * g        # distributed weight (N/m)
    MA_sust   = w_pipe * max(H, W) ** 2 / 8.0  # simply-supported approximation

    return {
        "Mi_exp":    Mi_exp,
        "Mo_exp":    Mo_exp,
        "Mt_exp":    Mt_exp,
        "MA_sust":   MA_sust,
        "bend_radius": bend_radius,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Full parameter sweep and dataset assembly
# ─────────────────────────────────────────────────────────────────────────────

def generate_dataset(
    H_range:    Tuple[float, float, int]   = (1.0, 5.0, 20),
    W_range:    Tuple[float, float, int]   = (0.5, 3.0, 15),
    nominal_sizes: list = None,
    schedules:  list   = None,
    dT_range:   Tuple[float, float, int]   = (50.0, 250.0, 10),
    P_range:    Tuple[float, float, int]   = (0.5e6, 5.0e6, 5),
    E:          float  = 207e9,
    alpha:      float  = 1.2e-5,
    T_install:  float  = 20.0,
    random_seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Generate training dataset by sweeping over geometry and operating parameters.

    For each combination, computes:
      - SE (expansion stress, Pa)
      - SL (sustained stress, Pa)
      - SA (allowable expansion stress, Pa)
      - Sh (basic hot allowable, Pa)
      - ur_expansion = SE/SA
      - ur_sustained = SL/Sh
      - pass_all : Boolean

    Args:
        H_range     : (min, max, n_levels) for loop height
        W_range     : (min, max, n_levels) for loop width
        nominal_sizes : list of nominal pipe sizes in inches (default [4,6,8,10,12])
        schedules   : list of schedule strings (default ['Sch40','Sch80'])
        dT_range    : temperature difference range
        P_range     : pressure range (Pa)
        E, alpha    : material properties
        T_install   : installation temperature (°C)
        random_seed : if set, sub-sample dataset to reduce size

    Returns:
        DataFrame with columns: H, W, D_m, t_m, nominal_in, schedule,
                                 delta_T, P_Pa, SE, SL, SA, Sh, ur_exp, ur_sust, pass_all
    """
    if nominal_sizes is None:
        nominal_sizes = [4, 6, 8, 10, 12]
    if schedules is None:
        schedules = ["Sch40", "Sch80"]

    H_vals  = np.linspace(*H_range)
    W_vals  = np.linspace(*W_range)
    dT_vals = np.linspace(*dT_range)
    P_vals  = np.linspace(*P_range)

    records = []

    for nom, sch in product(nominal_sizes, schedules):
        key = (nom, sch)
        if key not in PIPE_SCHEDULE:
            continue
        OD_mm, t_mm = PIPE_SCHEDULE[key]
        D_m  = OD_mm * 1e-3
        t_m  = t_mm  * 1e-3

        for H, W, dT, P in product(H_vals, W_vals, dT_vals, P_vals):
            T_op = T_install + dT

            # Flexibility analysis
            moms = simplified_thermal_moments(H, W, D_m, t_m, E, alpha, dT)

            # Code check
            try:
                result = asme_b313_code_check(
                    P=P, D=D_m, t=t_m,
                    T_install=T_install, T_operate=T_op,
                    Mi_exp=moms["Mi_exp"],
                    Mo_exp=moms["Mo_exp"],
                    Mt_exp=moms["Mt_exp"],
                    MA_sust=moms["MA_sust"],
                    r_bend=moms["bend_radius"],
                )
            except Exception:
                continue

            records.append({
                "H":           H,
                "W":           W,
                "D_m":         D_m,
                "t_m":         t_m,
                "nominal_in":  nom,
                "schedule":    sch,
                "delta_T":     dT,
                "P_Pa":        P,
                "SE":          result["SE"],
                "SL":          result["SL"],
                "SA":          result["SA"],
                "Sh":          result["Sh"],
                "ur_exp":      result["ur_expansion"],
                "ur_sust":     result["ur_sustained"],
                "pass_all":    int(result["pass_all"]),
            })

    df = pd.DataFrame(records)
    print(f"Dataset generated: {len(df):,} samples, "
          f"{df['pass_all'].sum():,} passing ({df['pass_all'].mean()*100:.1f}%)")
    return df


def train_val_test_split(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac:   float = 0.15,
    seed:       int   = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split dataset into train / validation / test partitions.

    Returns:
        df_train, df_val, df_test
    """
    df_shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n = len(df_shuffled)
    n_train = int(n * train_frac)
    n_val   = int(n * val_frac)
    return (
        df_shuffled.iloc[:n_train],
        df_shuffled.iloc[n_train : n_train + n_val],
        df_shuffled.iloc[n_train + n_val :],
    )


def df_to_tensors(
    df: pd.DataFrame,
    input_cols: list,
    target_col: str,
    device: str = "cpu",
) -> Tuple:
    """
    Convert DataFrame splits to normalised PyTorch tensors.

    Returns:
        X_tensor (N, n_features), y_tensor (N,), x_mean, x_std (for denormalisation)
    """
    import torch

    X_np = df[input_cols].values.astype(np.float32)
    y_np = df[target_col].values.astype(np.float32)

    x_mean = X_np.mean(axis=0)
    x_std  = X_np.std(axis=0) + 1e-8

    X_norm = (X_np - x_mean) / x_std

    X_t = torch.tensor(X_norm, device=device)
    y_t = torch.tensor(y_np,   device=device)

    return X_t, y_t, x_mean, x_std
