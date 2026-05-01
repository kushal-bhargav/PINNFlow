"""
Monte Carlo Simulation (MCS) for pipeline failure probability estimation.

Procedure:
  1. Train the PINN surrogate once offline (done in trainer.py)
  2. Generate N_MCS samples from the joint distribution of uncertain variables ξ
  3. For each sample, evaluate max strain via PINN forward pass (microseconds)
  4. Count failure events g(ξ) ≤ 0
  5. Estimate P_f = N_f / N_MCS

Key advantage: PINN evaluation replaces expensive FEA runs, enabling
10⁵–10⁶ samples in seconds instead of years.

References:
    Haghighat et al. (2025), PINN-RA for Buried Pipeline Reliability Analysis
"""

import torch
import numpy as np
import time
from typing import Dict, Optional, Tuple
import torch.nn as nn

from .distributions import UncertaintyModel, UncertaintySampler
from .limit_state import (
    evaluate_max_strain_pinn,
    limit_state_combined,
    DEFAULT_ALLOWABLES,
)


# ─────────────────────────────────────────────────────────────────────────────
# MCS Engine
# ─────────────────────────────────────────────────────────────────────────────

class MonteCarloSimulation:
    """
    Monte Carlo reliability analysis using a trained PINN surrogate.

    Args:
        network                 : trained PINNNetwork (inference mode)
        uncertainty_model       : UncertaintyModel with distribution parameters
        cross_section_integrator: CrossSectionIntegrator for strain recovery
        pipe_params             : dict {D, t, P, delta_T, E, nu, alpha_T, L, beta_deg}
        n_x                     : number of spatial evaluation points along pipe
        eps_allow_t             : tensile strain allowable
        eps_allow_c             : compressive strain allowable
        device                  : 'cpu' or 'cuda'
    """

    def __init__(
        self,
        network: nn.Module,
        uncertainty_model: UncertaintyModel,
        cross_section_integrator,
        pipe_params: dict,
        n_x: int = 200,
        eps_allow_t: float = DEFAULT_ALLOWABLES["tensile_strain_limit"],
        eps_allow_c: float = DEFAULT_ALLOWABLES["compressive_strain_limit"],
        device: str = "cpu",
    ):
        self.network   = network
        self.sampler   = UncertaintySampler(uncertainty_model)
        self.csi       = cross_section_integrator
        self.pipe_params = pipe_params
        self.n_x         = n_x
        self.eps_allow_t = eps_allow_t
        self.eps_allow_c = eps_allow_c
        self.device      = device

        # Spatial evaluation grid (uniform along pipe)
        L = pipe_params.get("L", 300.0)
        self.x_grid = torch.linspace(0.0, L, n_x, device=device)

    def run(
        self,
        n_mcs: int = 100_000,
        batch_size: int = 10_000,
        seed: Optional[int] = 42,
        verbose: bool = True,
    ) -> Dict:
        """
        Run Monte Carlo simulation.

        Args:
            n_mcs      : total number of MC samples
            batch_size : samples per batch (memory management)
            seed       : random seed for reproducibility
            verbose    : print progress

        Returns:
            results dict with keys:
                P_f         : estimated failure probability
                cov_Pf      : coefficient of variation of P_f estimate
                n_failures  : number of failure events
                n_mcs       : total samples
                beta_HL     : Hasofer-Lind reliability index β = -Φ⁻¹(P_f)
                g_values    : (n_mcs,) full limit state values
                elapsed_s   : wall-clock time (seconds)
        """
        t0 = time.time()

        if verbose:
            print(f"\nMonte Carlo Simulation  |  N_MCS={n_mcs:,}  |  device={self.device}")
            print(f"  Strain allowables: tensile={self.eps_allow_t*100:.1f}%  "
                  f"compressive={self.eps_allow_c*100:.2f}%")

        # ── Sample uncertain variables ────────────────────────────────────────
        xi_samples = self.sampler.sample(n_mcs, seed=seed,
                                         as_tensor=True, device=self.device)
        if verbose:
            print(f"  Sampling complete. Evaluating PINN surrogate ...")

        # ── Evaluate PINN for max strains ─────────────────────────────────────
        eps_t, eps_c = evaluate_max_strain_pinn(
            self.network, xi_samples, self.x_grid,
            self.csi, self.pipe_params, batch_size=batch_size
        )

        # ── Compute limit state ───────────────────────────────────────────────
        g = limit_state_combined(eps_t, eps_c, self.eps_allow_t, self.eps_allow_c)

        # ── Count failures ────────────────────────────────────────────────────
        failures   = (g <= 0).sum().item()
        P_f        = failures / n_mcs

        # COV of P_f estimate: COV_Pf ≈ sqrt((1-P_f) / (P_f * N_MCS))
        if P_f > 0:
            cov_Pf = np.sqrt((1.0 - P_f) / (P_f * n_mcs))
        else:
            cov_Pf = float("inf")

        # Hasofer-Lind reliability index
        from scipy.stats import norm
        if 0.0 < P_f < 1.0:
            beta_HL = -norm.ppf(P_f)
        else:
            beta_HL = float("inf") if P_f == 0.0 else float("-inf")

        elapsed = time.time() - t0

        if verbose:
            print(f"\n  {'─'*50}")
            print(f"  P_f        = {P_f:.4e}")
            print(f"  β_HL       = {beta_HL:.3f}")
            print(f"  COV(P_f)   = {cov_Pf:.3f}")
            print(f"  N_failures = {failures:,} / {n_mcs:,}")
            print(f"  Elapsed    = {elapsed:.2f} s")
            print(f"  {'─'*50}")

        return {
            "P_f":        P_f,
            "cov_Pf":     cov_Pf,
            "n_failures": failures,
            "n_mcs":      n_mcs,
            "beta_HL":    beta_HL,
            "g_values":   g.cpu().numpy(),
            "eps_tension":  eps_t.cpu().numpy(),
            "eps_compression": eps_c.cpu().numpy(),
            "elapsed_s":  elapsed,
        }

    def sensitivity_analysis(
        self,
        n_mcs: int = 50_000,
        seed: int = 0,
    ) -> Dict[str, float]:
        """
        Compute Spearman rank-correlation coefficients between each uncertain
        variable and the limit state value g(ξ).

        High |ρ| → variable is important driver of failure probability.

        Returns:
            correlations : dict mapping variable name → Spearman correlation with g
        """
        from scipy.stats import spearmanr

        xi = self.sampler.sample(n_mcs, seed=seed, as_tensor=True, device=self.device)
        eps_t, eps_c = evaluate_max_strain_pinn(
            self.network, xi, self.x_grid,
            self.csi, self.pipe_params
        )
        g = limit_state_combined(eps_t, eps_c, self.eps_allow_t, self.eps_allow_c)
        g_np = g.cpu().numpy()
        xi_np = xi.cpu().numpy()

        names = ["c (Pa)", "phi (deg)", "gamma (N/m3)", "delta (m)", "H (m)"]
        correlations = {}
        for i, name in enumerate(names):
            rho, _ = spearmanr(xi_np[:, i], g_np)
            correlations[name] = float(rho)

        return correlations

    def convergence_study(
        self,
        n_levels: list = None,
        seed: int = 0,
    ) -> Dict[str, list]:
        """
        Study P_f convergence as function of N_MCS.

        Returns:
            dict with 'n_mcs', 'P_f', 'cov_Pf' lists
        """
        if n_levels is None:
            n_levels = [1_000, 5_000, 10_000, 50_000, 100_000]

        results = {"n_mcs": [], "P_f": [], "cov_Pf": []}
        for n in n_levels:
            r = self.run(n_mcs=n, seed=seed, verbose=False)
            results["n_mcs"].append(n)
            results["P_f"].append(r["P_f"])
            results["cov_Pf"].append(r["cov_Pf"])
            print(f"  N={n:>8,}  P_f={r['P_f']:.4e}  COV={r['cov_Pf']:.3f}")
        return results
