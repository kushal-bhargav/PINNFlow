"""
Probability distributions for uncertain variables in the MCS reliability analysis.

Uncertain variables ξ = {c, φ, γ, δ, H}:
  - Soil cohesion c        : Lognormal(μ_c, σ_c)
  - Friction angle φ       : Normal(μ_φ, σ_φ)
  - Unit weight γ          : Normal(μ_γ, σ_γ)
  - Ground displacement δ  : Lognormal(μ_δ, σ_δ)
  - Burial depth H         : Deterministic or Normal

References:
    Haghighat et al. (2025), PINN-RA for Buried Pipeline Reliability Analysis
"""

import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Distribution parameter containers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class UncertaintyModel:
    """
    Holds mean and coefficient-of-variation (COV) for all uncertain variables.

    Cohesion c and displacement δ are Lognormal.
    Friction angle φ and unit weight γ are Normal.
    Burial depth H is treated as deterministic by default (COV=0).
    """
    # Soil cohesion (Pa)
    c_mean:   float = 10e3
    c_cov:    float = 0.30

    # Friction angle (degrees)
    phi_mean: float = 35.0
    phi_cov:  float = 0.10

    # Unit weight (N/m³)
    gamma_mean: float = 18e3
    gamma_cov:  float = 0.05

    # Ground displacement magnitude (m)
    delta_mean: float = 0.10
    delta_cov:  float = 0.40

    # Burial depth (m)
    H_mean:   float = 1.5
    H_cov:    float = 0.0    # 0 = deterministic


def lognormal_params(mean: float, cov: float) -> Tuple[float, float]:
    """
    Convert mean and COV of a Lognormal distribution to
    underlying Normal distribution parameters (mu_ln, sigma_ln).

        sigma_ln = sqrt(ln(1 + COV²))
        mu_ln    = ln(mean) - sigma_ln² / 2

    Args:
        mean : mean of the Lognormal variable
        cov  : coefficient of variation

    Returns:
        mu_ln, sigma_ln : parameters of the underlying Normal distribution
    """
    sigma_ln = np.sqrt(np.log(1.0 + cov ** 2))
    mu_ln    = np.log(mean) - 0.5 * sigma_ln ** 2
    return mu_ln, sigma_ln


def normal_params(mean: float, cov: float) -> Tuple[float, float]:
    """
    Return (mean, std) for a Normal distribution given mean and COV.

    Returns:
        mu, sigma : Normal distribution parameters
    """
    return mean, mean * cov


# ─────────────────────────────────────────────────────────────────────────────
# Sampler
# ─────────────────────────────────────────────────────────────────────────────

class UncertaintySampler:
    """
    Generates Monte Carlo samples from the joint distribution of
    uncertain variables ξ = [c, φ, γ, δ, H].

    Usage:
        model   = UncertaintyModel(c_mean=10e3, c_cov=0.3, ...)
        sampler = UncertaintySampler(model)
        samples = sampler.sample(n=100_000)   # (100000, 5) tensor
    """

    def __init__(self, model: UncertaintyModel):
        self.model = model

    def sample(
        self,
        n: int,
        seed: Optional[int] = None,
        as_tensor: bool = True,
        device: str = "cpu",
    ) -> torch.Tensor:
        """
        Draw n independent samples from the joint distribution.

        Column order: [c, φ, γ, δ, H]

        Args:
            n         : number of samples
            seed      : random seed
            as_tensor : if True, return torch.Tensor; else numpy array
            device    : torch device for tensor output

        Returns:
            samples : (n, 5) array/tensor of realizations
                      columns: [c (Pa), φ (deg), γ (N/m³), δ (m), H (m)]
        """
        rng = np.random.default_rng(seed)
        m   = self.model

        # Cohesion — Lognormal
        mu_c, sig_c = lognormal_params(m.c_mean, m.c_cov)
        c_samples   = rng.lognormal(mu_c, sig_c, size=n)
        c_samples   = np.clip(c_samples, 0.0, None)      # cohesion ≥ 0

        # Friction angle — Normal
        mu_phi, sig_phi = normal_params(m.phi_mean, m.phi_cov)
        phi_samples = rng.normal(mu_phi, sig_phi, size=n)
        phi_samples = np.clip(phi_samples, 15.0, 50.0)  # physical bounds

        # Unit weight — Normal
        mu_g, sig_g    = normal_params(m.gamma_mean, m.gamma_cov)
        gamma_samples  = rng.normal(mu_g, sig_g, size=n)
        gamma_samples  = np.clip(gamma_samples, 12e3, 25e3)

        # Ground displacement — Lognormal
        mu_d, sig_d   = lognormal_params(m.delta_mean, m.delta_cov)
        delta_samples = rng.lognormal(mu_d, sig_d, size=n)
        delta_samples = np.clip(delta_samples, 0.001, 2.0)

        # Burial depth — Normal or deterministic
        if m.H_cov > 0:
            mu_H, sig_H = normal_params(m.H_mean, m.H_cov)
            H_samples   = rng.normal(mu_H, sig_H, size=n)
            H_samples   = np.clip(H_samples, 0.5, 5.0)
        else:
            H_samples = np.full(n, m.H_mean)

        samples_np = np.column_stack([
            c_samples.astype(np.float32),
            phi_samples.astype(np.float32),
            gamma_samples.astype(np.float32),
            delta_samples.astype(np.float32),
            H_samples.astype(np.float32),
        ])

        if as_tensor:
            return torch.tensor(samples_np, device=device)
        return samples_np

    def summary_stats(self, n: int = 50_000, seed: int = 0) -> Dict[str, Dict[str, float]]:
        """
        Compute empirical mean and std of sampled distributions.
        Useful for verifying sampler correctness.
        """
        s = self.sample(n, seed=seed, as_tensor=False)
        names = ["c (Pa)", "phi (deg)", "gamma (N/m3)", "delta (m)", "H (m)"]
        stats = {}
        for i, name in enumerate(names):
            stats[name] = {
                "mean":   float(s[:, i].mean()),
                "std":    float(s[:, i].std()),
                "cov":    float(s[:, i].std() / (s[:, i].mean() + 1e-12)),
                "p05":    float(np.percentile(s[:, i], 5)),
                "p95":    float(np.percentile(s[:, i], 95)),
            }
        return stats


# ─────────────────────────────────────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model   = UncertaintyModel()
    sampler = UncertaintySampler(model)
    stats   = sampler.summary_stats(n=100_000, seed=42)
    for var, s in stats.items():
        print(f"{var:20s}  mean={s['mean']:.3e}  std={s['std']:.3e}  "
              f"cov={s['cov']:.3f}  [{s['p05']:.3e}, {s['p95']:.3e}]")
