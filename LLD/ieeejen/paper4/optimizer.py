"""
Constrained optimisation of pipe loop geometry using the trained ANN surrogate.

Objective: minimise pipe material volume (cost) subject to ASME B31.3 constraints.

Strategy:
  - ANN provides fast SE(H,W) and SL(H,W) predictions (microseconds per eval)
  - ANN gradients w.r.t. inputs computed via backpropagation (differentiable surrogate)
  - SLSQP constrained optimiser finds minimum geometry satisfying SE ≤ SA, SL ≤ Sh

Revamping scenario:
  - Existing design violates SE > SA
  - Optimizer finds minimum H, W increase to restore compliance

References:
    Caponetto / Giudice et al. (2022), ANN-Based Optimization of Pressure Piping
"""

import numpy as np
import torch
from scipy.optimize import minimize, differential_evolution
from typing import Dict, Tuple, Optional, Callable

from .ffnn import DualStressANN, StressPredictorANN
from .asme_b313.stress_formulas import allowable_expansion_stress, basic_allowable_stress_carbon_steel


# ─────────────────────────────────────────────────────────────────────────────
# ANN-based objective and constraint functions
# ─────────────────────────────────────────────────────────────────────────────

class ANNSurrogateOptimizer:
    """
    Constrained geometry optimiser using trained ANN surrogate for stress prediction.

    Minimises:  pipe_volume(H, W, D, L_system)   [material cost proxy]
    Subject to:
        SE(H, W; D, t, dT, P) / SA  ≤ 1.0   [expansion code check]
        SL(H, W; D, t, dT, P) / Sh  ≤ 1.0   [sustained code check]
        H_min ≤ H ≤ H_max
        W_min ≤ W ≤ W_max

    Args:
        dual_ann      : trained DualStressANN
        fixed_params  : dict {D, t, delta_T, P, T_install, T_operate}
        H_bounds      : (H_min, H_max) in metres
        W_bounds      : (W_min, W_max) in metres
    """

    def __init__(
        self,
        dual_ann: DualStressANN,
        fixed_params: Dict,
        H_bounds: Tuple[float, float] = (0.5, 8.0),
        W_bounds: Tuple[float, float] = (0.3, 5.0),
    ):
        self.ann    = dual_ann
        self.fixed  = fixed_params
        self.H_bnd  = H_bounds
        self.W_bnd  = W_bounds

        # Pre-compute allowable stresses
        D, t   = fixed_params["D"], fixed_params["t"]
        dT     = fixed_params["delta_T"]
        P      = fixed_params["P"]
        T_in   = fixed_params.get("T_install", 20.0)
        T_op   = T_in + dT

        Sc, _   = basic_allowable_stress_carbon_steel(T_in)
        _, Sh   = basic_allowable_stress_carbon_steel(T_op)
        # Approximate SL for SA calculation (will be updated during optimisation)
        SL_approx = P * D / (4.0 * t)
        self.SA  = allowable_expansion_stress(Sc, Sh, SL_approx)
        self.Sh  = Sh
        self.Sc  = Sc

    def _build_X_raw(self, H: float, W: float) -> np.ndarray:
        """Build single-sample raw input array [H, W, D, t, dT, P]."""
        fp = self.fixed
        return np.array([[H, W, fp["D"], fp["t"],
                          fp["delta_T"], fp["P"]]], dtype=np.float32)

    def predict_stresses(self, H: float, W: float) -> Tuple[float, float]:
        """Return (SE, SL) predicted by the ANN surrogate."""
        X_raw = self._build_X_raw(H, W)
        SE, SL = self.ann.predict(X_raw)
        return float(SE[0]), float(SL[0])

    def objective(self, x: np.ndarray) -> float:
        """
        Objective: total pipe volume of the expansion loop (proportional to cost).
        Loop modelled as two legs: vertical H and horizontal W.

            V = A_pipe * (H + W)   [cross-section area × total leg length]
        """
        H, W = x
        D, t = self.fixed["D"], self.fixed["t"]
        A_pipe = np.pi / 4.0 * (D**2 - (D - 2*t)**2)
        return A_pipe * (H + W)    # m³

    def constraints(self, x: np.ndarray) -> list:
        """
        SLSQP constraint functions (each must be ≥ 0 for feasibility).

            g1 = SA - SE  ≥ 0   (expansion constraint)
            g2 = Sh - SL  ≥ 0   (sustained constraint)
        """
        H, W   = float(x[0]), float(x[1])
        SE, SL = self.predict_stresses(H, W)
        # Re-compute SA using current SL (enhanced allowable)
        SA_cur = allowable_expansion_stress(self.Sc, self.Sh, SL)
        return [SA_cur - SE, self.Sh - SL]

    def optimise(
        self,
        H0: float = 2.0,
        W0: float = 1.0,
        method: str = "SLSQP",
        tol: float = 1e-6,
        max_iter: int = 500,
        verbose: bool = True,
    ) -> Dict:
        """
        Run constrained optimisation.

        Args:
            H0, W0   : initial loop geometry guess
            method   : 'SLSQP' (gradient-based) or 'DE' (differential evolution)
            tol      : convergence tolerance
            max_iter : maximum iterations
            verbose  : print progress

        Returns:
            result dict with optimal H, W, SE, SL, SA, ur values, and solver info
        """
        if verbose:
            SE0, SL0 = self.predict_stresses(H0, W0)
            print(f"\nInitial design:  H={H0:.2f}m  W={W0:.2f}m")
            print(f"  SE={SE0/1e6:.2f} MPa  (SA={self.SA/1e6:.2f} MPa, "
                  f"ur={SE0/self.SA:.3f})")
            print(f"  SL={SL0/1e6:.2f} MPa  (Sh={self.Sh/1e6:.2f} MPa, "
                  f"ur={SL0/self.Sh:.3f})")
            print("─" * 50)

        x0     = np.array([H0, W0])
        bounds = [self.H_bnd, self.W_bnd]
        cons   = [{"type": "ineq", "fun": lambda x: self.constraints(x)[0]},
                  {"type": "ineq", "fun": lambda x: self.constraints(x)[1]}]

        if method == "SLSQP":
            res = minimize(
                self.objective, x0,
                method="SLSQP",
                bounds=bounds,
                constraints=cons,
                tol=tol,
                options={"maxiter": max_iter, "disp": verbose},
            )
        elif method == "DE":
            # Global search with penalty for constraint violation
            def penalised_obj(x):
                obj   = self.objective(x)
                g     = self.constraints(x)
                penalty = 1e6 * (max(0, -g[0]) + max(0, -g[1]))
                return obj + penalty

            res = differential_evolution(
                penalised_obj, bounds, seed=42,
                maxiter=max_iter, tol=tol,
                popsize=15, mutation=(0.5, 1.0), recombination=0.7,
                disp=verbose,
            )
        else:
            raise ValueError(f"Unknown method '{method}'")

        H_opt, W_opt = float(res.x[0]), float(res.x[1])
        SE_opt, SL_opt = self.predict_stresses(H_opt, W_opt)
        SA_opt = allowable_expansion_stress(self.Sc, self.Sh, SL_opt)

        if verbose:
            print(f"\nOptimal design:  H={H_opt:.3f}m  W={W_opt:.3f}m")
            print(f"  SE={SE_opt/1e6:.2f} MPa  "
                  f"(SA={SA_opt/1e6:.2f} MPa, ur={SE_opt/SA_opt:.3f}) "
                  f"{'✓' if SE_opt <= SA_opt else '✗'}")
            print(f"  SL={SL_opt/1e6:.2f} MPa  "
                  f"(Sh={self.Sh/1e6:.2f} MPa, ur={SL_opt/self.Sh:.3f}) "
                  f"{'✓' if SL_opt <= self.Sh else '✗'}")
            print(f"  Volume: {self.objective(res.x)*1e6:.2f} cm³  "
                  f"(Converged: {res.success})")

        return {
            "H_opt": H_opt, "W_opt": W_opt,
            "SE_opt": SE_opt, "SL_opt": SL_opt,
            "SA_opt": SA_opt, "Sh": self.Sh,
            "ur_exp": SE_opt / SA_opt,
            "ur_sust": SL_opt / self.Sh,
            "volume_m3": self.objective(res.x),
            "success": res.success,
            "message": res.message,
            "n_iter": res.nit if hasattr(res, "nit") else None,
        }

    def revamp_scenario(
        self,
        H_existing: float,
        W_existing: float,
        verbose: bool = True,
    ) -> Dict:
        """
        Revamping scenario: find minimum loop geometry adjustment to restore
        code compliance for an existing non-compliant design.

        Starts the optimisation from the existing geometry (H_existing, W_existing).
        """
        if verbose:
            SE_ex, SL_ex = self.predict_stresses(H_existing, W_existing)
            violations = []
            if SE_ex > self.SA: violations.append(f"SE={SE_ex/1e6:.1f} > SA={self.SA/1e6:.1f} MPa")
            if SL_ex > self.Sh: violations.append(f"SL={SL_ex/1e6:.1f} > Sh={self.Sh/1e6:.1f} MPa")
            print(f"\nRevamping scenario: existing H={H_existing:.2f}m, W={W_existing:.2f}m")
            if violations:
                print(f"  NON-COMPLIANT: {'; '.join(violations)}")

        return self.optimise(H0=H_existing, W0=W_existing, verbose=verbose)
