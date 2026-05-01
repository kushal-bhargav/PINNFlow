"""
pinnflow/ablation.py
─────────────────────
MODULE 9 — Ablation Study  [P4]

Five ablation configurations to isolate each novelty contribution:
  A — Vanilla MLP          (no PDE, no log-target)
  B — Single-task PINN     (stress only)
  C — Multi-task PINN      [N1]
  D — Multi-task PINN + CA-VAE [N1+N2]   (measures E2E CSR)
  E — Full framework       [N1+N2+N3]

Runs n_seeds seeds per configuration and reports mean ± std.

[P1 FIX] Data is shuffled before train/test split in every seed so augmented
         samples from PhysicsSimulator reach the training set.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

from pinnflow.simulator  import PhysicsSimulator
from pinnflow.pinn       import MultiTaskPINN
from pinnflow.vae        import CAVAE
from pinnflow.environment import PipelineEnv
from pinnflow.agent      import PPOAgent
from pinnflow.baselines  import SimpleNN

FEAT = [
    "diameter", "thickness", "length", "pressure",
    "soil_disp", "delta_T", "velocity", "soil_stiffness",
]


def run_ablation(
    sim: PhysicsSimulator,
    n_samples: int = 800,
    n_seeds: int = 3,
) -> pd.DataFrame:
    """
    [P4] Run the 5-configuration ablation and return aggregated DataFrame.
    Prints a formatted ASCII table to stdout.
    """
    print("\n[ABLATION] Running ablation study...")
    results = []

    for seed in range(n_seeds):
        np.random.seed(seed * 17 + 42)
        df = sim.generate(n_samples)
        X  = df[FEAT].values
        Y  = df[["von_mises_stress", "pressure_drop_kPa"]].values

        # [P1 FIX] Shuffle before split so augmented regime reaches training set
        idx = np.random.permutation(len(X))
        X, Y = X[idx], Y[idx]
        sp  = int(0.8 * len(X))
        Xtr, Xte = X[:sp], X[sp:]
        Ytr, Yte = Y[:sp], Y[sp:]

        # ── A: Vanilla MLP ────────────────────────────────────────────────────
        nn = SimpleNN(n_in=8, hidden=128, lr=1e-3)
        nn.fit(Xtr, Ytr[:, 0], epochs=200, batch=64)
        pred_a = nn.predict(Xte)
        mae_a  = mean_absolute_error(Yte[:, 0], pred_a)
        rng_a  = Yte[:, 0].max() - Yte[:, 0].min()
        results.append({
            "seed": seed, "config": "A: Vanilla MLP",
            "stress_mae_pct": round(mae_a / rng_a * 100, 2),
            "stress_r2":      round(r2_score(Yte[:, 0], pred_a), 4),
            "pressure_mae_pct": float("nan"), "e2e_csr": float("nan"),
        })

        # ── B: Single-task PINN (stress only, log target) ─────────────────────
        Yz = Ytr.copy(); Yz[:, 1] = 0.1   # fluid target ignored (lam_f=0)
        pinn_b = MultiTaskPINN(
            n_in=8, hidden=(64, 128, 128, 64),
            lr=3e-3, lam_s=0.01, lam_f=0.0, mu_bc=0.01, nu_mono=0.0,
        )
        pinn_b.fit(Xtr, Yz, epochs=150, batch=64, verbose=False)
        pred_b = pinn_b.predict(Xte)
        mae_bs = mean_absolute_error(Yte[:, 0], pred_b[:, 0])
        rng_bs = Yte[:, 0].max() - Yte[:, 0].min()
        results.append({
            "seed": seed, "config": "B: Single-task PINN",
            "stress_mae_pct": round(mae_bs / rng_bs * 100, 2),
            "stress_r2":      round(r2_score(Yte[:, 0], pred_b[:, 0]), 4),
            "pressure_mae_pct": float("nan"), "e2e_csr": float("nan"),
        })

        # ── C: Multi-task PINN [N1] ───────────────────────────────────────────
        pinn_c = MultiTaskPINN(
            n_in=8, hidden=(64, 128, 128, 64),
            lr=3e-3, lam_s=0.005, lam_f=0.01, mu_bc=0.01, nu_mono=0.001,
        )
        pinn_c.fit(Xtr, Ytr, epochs=150, batch=64, verbose=False)
        pred_c = pinn_c.predict(Xte)
        mae_cs = mean_absolute_error(Yte[:, 0], pred_c[:, 0])
        mae_cf = mean_absolute_error(Yte[:, 1], pred_c[:, 1])
        rng_cs = Yte[:, 0].max() - Yte[:, 0].min()
        rng_cf = Yte[:, 1].max() - Yte[:, 1].min()
        results.append({
            "seed": seed, "config": "C: Multi-task PINN [N1]",
            "stress_mae_pct":   round(mae_cs / rng_cs * 100, 2),
            "stress_r2":        round(r2_score(Yte[:, 0], pred_c[:, 0]), 4),
            "pressure_mae_pct": round(mae_cf / rng_cf * 100, 2),
            "e2e_csr": float("nan"),
        })

        # ── D: [N1] + CA-VAE [N2] — measure generated CSR ────────────────────
        vae_d = CAVAE(x_dim=8, z_dim=12, hidden=(64, 32), lr=3e-3, gamma_phys=2.0)
        vae_d.fit(X, epochs=100, batch=64, verbose=False)
        csr_d, _ = vae_d.csr(200, pinn_c)
        results.append({
            "seed": seed, "config": "D: [N1]+CA-VAE [N2]",
            "stress_mae_pct":   round(mae_cs / rng_cs * 100, 2),
            "stress_r2":        round(r2_score(Yte[:, 0], pred_c[:, 0]), 4),
            "pressure_mae_pct": round(mae_cf / rng_cf * 100, 2),
            "e2e_csr": round(csr_d, 4),
        })

        # ── E: Full framework [N1+N2+N3] ─────────────────────────────────────
        env_ab = PipelineEnv(pinn_c, curriculum=True)
        ag_ab  = PPOAgent(sdim=8, adim=8, hidden=64, lr=5e-3)
        ag_ab.train(env_ab, n_ep=150, steps=20, verbose=False)
        csr_vals = []
        for _ in range(20):
            layout = vae_d.generate(1)[0]
            layout = np.clip(layout, env_ab.BOUNDS[:, 0], env_ab.BOUNDS[:, 1])
            env_ab.state = layout
            for _ in range(20):
                a, _, _ = ag_ab.select_action(env_ab.state)
                env_ab.step(a)
            pred_e = pinn_c.predict(env_ab.state.reshape(1, -1))[0]
            csr_vals.append(float(pred_e[0] < 200))
        results.append({
            "seed": seed, "config": "E: Full [N1+N2+N3]",
            "stress_mae_pct":   round(mae_cs / rng_cs * 100, 2),
            "stress_r2":        round(r2_score(Yte[:, 0], pred_c[:, 0]), 4),
            "pressure_mae_pct": round(mae_cf / rng_cf * 100, 2),
            "e2e_csr": round(float(np.mean(csr_vals)), 4),
        })

    df_ab = pd.DataFrame(results)
    agg   = df_ab.groupby("config").agg(
        stress_mae_mean  =("stress_mae_pct", "mean"),
        stress_mae_std   =("stress_mae_pct", "std"),
        stress_r2_mean   =("stress_r2",      "mean"),
        pressure_mae_mean=("pressure_mae_pct", "mean"),
        e2e_csr_mean     =("e2e_csr", "mean"),
        e2e_csr_std      =("e2e_csr", "std"),
    ).round(4)

    print("\n  ABLATION TABLE (mean ± std over 3 seeds)")
    print(f"  {'Config':<28} {'StressMAE%':>12} {'StressR²':>10} "
          f"{'PressMAE%':>11} {'E2E-CSR':>10}")
    print("  " + "-" * 74)
    for cfg, row in agg.iterrows():
        sm  = f"{row.stress_mae_mean:.2f}±{row.stress_mae_std:.2f}"
        pm  = f"{row.pressure_mae_mean:.2f}" if not np.isnan(row.pressure_mae_mean) else "—"
        csr = (f"{row.e2e_csr_mean:.3f}±{row.e2e_csr_std:.3f}"
               if not np.isnan(row.e2e_csr_mean) else "—")
        print(f"  {cfg:<28} {sm:>12} {row.stress_r2_mean:>10.4f} {pm:>11} {csr:>10}")

    return agg
