"""
pinnflow/visualization.py
──────────────────────────
MODULE 12 — Results figure (6 rows × 3 cols, 24 subplots).

plot_all() assembles the complete results figure and saves it to RESULTS_DIR.
All styling uses the colour palette from pinnflow.config.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

from pinnflow.config import (
    RESULTS_DIR,
    BLUE, LBLUE, GREEN, RED, GOLD, PURPLE, TEAL, ORANGE,
)


def _ax_style(ax, title: str, xl: str = "", yl: str = "", fs: float = 9.5) -> None:
    ax.set_facecolor("#FAFBFC")
    ax.set_title(title, fontsize=fs, fontweight="bold", color=BLUE, pad=7)
    ax.set_xlabel(xl, fontsize=8, color="#444")
    ax.set_ylabel(yl, fontsize=8, color="#444")
    ax.tick_params(labelsize=7.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.22, linestyle="--")


def plot_all(
    pinn, vae, agent,
    rand_rewards, Y_te, Y_pred,
    pinn_res, div_res, rl_res,
    e2e_b, e2e_rec,
    fem_b, fem_rec,
    ablation_df, v1_results,
) -> str:
    """
    Build and save the 6-row results figure.
    Returns the saved file path.
    """
    fig = plt.figure(figsize=(24, 32))
    fig.patch.set_facecolor("#F4F6F9")
    gs  = gridspec.GridSpec(6, 3, figure=fig, hspace=0.52, wspace=0.35)
    ep  = range(1, len(pinn.history["total"]) + 1)

    # ── Row 0: PINN ──────────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    ax.semilogy(ep, pinn.history["total"],  BLUE,  lw=2,   label="Total")
    ax.semilogy(ep, pinn.history["data"],   LBLUE, lw=1.5, label="L_data")
    ax.semilogy(ep, pinn.history["pde"],    GREEN, lw=1.5, label="λ·L_PDE")
    ax.semilogy(ep, pinn.history["bc"],     GOLD,  lw=1.5, label="μ·L_BC")
    ax.legend(fontsize=7, framealpha=0.7)
    _ax_style(ax, "[N1] Multi-task PINN loss\n(log scale + cosine LR)", "Epoch", "Loss")

    ax = fig.add_subplot(gs[0, 1])
    ax.scatter(Y_te[:, 0], Y_pred[:, 0], alpha=0.4, s=14, c=LBLUE, edgecolors="none")
    mn, mx = Y_te[:, 0].min(), Y_te[:, 0].max()
    ax.plot([mn, mx], [mn, mx], "r--", lw=1.5)
    r2s = pinn_res["von_mises_stress"]["R2"]
    mp  = pinn_res["von_mises_stress"]["MAE_pct"]
    ax.text(0.05, 0.90, f"R²={r2s:.4f}\nMAE={mp:.2f}%\n[P1] log(σ) target",
            transform=ax.transAxes, fontsize=8, color=BLUE, fontweight="bold")
    ax.text(0.05, 0.68, "v1: R²=0.88 / MAE=7.55%",
            transform=ax.transAxes, fontsize=7, color=RED, alpha=0.8)
    _ax_style(ax, "Phase 1: Stress accuracy\nvon Mises (MPa)", "Actual", "Predicted")

    ax = fig.add_subplot(gs[0, 2])
    ax.scatter(Y_te[:, 1], Y_pred[:, 1], alpha=0.4, s=14, c=GREEN, edgecolors="none")
    mn, mx = Y_te[:, 1].min(), Y_te[:, 1].max()
    ax.plot([mn, mx], [mn, mx], "r--", lw=1.5)
    r2f = pinn_res["pressure_drop_kPa"]["R2"]
    mpf = pinn_res["pressure_drop_kPa"]["MAE_pct"]
    ax.text(0.05, 0.90, f"R²={r2f:.4f}\nMAE={mpf:.2f}%\n[N1] fluid head",
            transform=ax.transAxes, fontsize=8, color=GREEN, fontweight="bold")
    ax.text(0.05, 0.68, "v1: R²=0.11 / MAE=12.52%",
            transform=ax.transAxes, fontsize=7, color=RED, alpha=0.8)
    _ax_style(ax, "Phase 1: Pressure drop accuracy\n(kPa)", "Actual", "Predicted")

    # ── Row 1: VAE ───────────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    ev = range(1, len(vae.history["elbo"]) + 1)
    ax.semilogy(ev, vae.history["recon"], BLUE,  lw=2,   label="Recon")
    ax.semilogy(ev, vae.history["kl"],   GREEN, lw=1.5, label="KL (free bits)")
    ax.semilogy(ev, vae.history["phys"], RED,   lw=1.5, label="Physics")
    ax.legend(fontsize=7)
    _ax_style(ax, "[N2] CA-VAE training\n[P2] KL annealing + free bits", "Epoch", "Loss")

    ax  = fig.add_subplot(gs[1, 1])
    gen = vae.generate(300)
    ax.hist(gen[:, 0],      bins=20, color=LBLUE,  alpha=0.75, label="Diameter (mm)")
    ax.hist(gen[:, 3] * 10, bins=20, color=GREEN,  alpha=0.75, label="Pressure ×10 (MPa)")
    ax.legend(fontsize=7)
    _ax_style(ax, "Phase 2: CA-VAE generated layouts\n(diversity)", "Value", "Count")

    ax  = fig.add_subplot(gs[1, 2])
    d_g = gen[:, 0]; t_g = gen[:, 1]; P_g = gen[:, 3]
    sig_g = P_g * d_g / (2 * t_g); ok_g = sig_g < 200
    ax.scatter(d_g[ok_g],  sig_g[ok_g],  c=GREEN, alpha=0.5, s=12, label="Feasible")
    ax.scatter(d_g[~ok_g], sig_g[~ok_g], c=RED,   alpha=0.5, s=12, label="Infeasible")
    ax.axhline(200, c=RED, lw=1.5, ls="--", label="ASME 200 MPa")
    ax.legend(fontsize=7)
    ax.text(0.05, 0.90, f"CSR={ok_g.mean():.1%}\n(v1: 1.8%)",
            transform=ax.transAxes, fontsize=8, color=BLUE, fontweight="bold")
    _ax_style(ax, "Phase 2: Constraint satisfaction\nASME B31.3", "Diameter (mm)", "σ (MPa)")

    # ── Row 2: RL ────────────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[2, 0])
    pp = pd.Series(agent.reward_hist).rolling(25).mean()
    rr = pd.Series(rand_rewards[:len(agent.reward_hist)]).rolling(25).mean()
    ax.plot(pp, c=BLUE, lw=2, label="PPO [N3]")
    ax.plot(rr, c=RED,  lw=2, ls="--", label="Random")
    ax.axvline(100, c=GOLD,  lw=1, ls=":", label="Phase 2 start")
    ax.axvline(200, c=GREEN, lw=1, ls=":", label="Phase 3 start")
    ax.legend(fontsize=7)
    _ax_style(ax, "Phase 3: Curriculum PPO\nvs random baseline [P3] linear reward",
              "Episode", "Reward")

    ax = fig.add_subplot(gs[2, 1])
    cs = pd.Series(agent.csr_hist).rolling(25).mean()
    ax.plot(cs, c=GREEN, lw=2, label="PPO CSR")
    ax.axhline(0.75, c=RED, lw=1.5, ls="--", label="v1 CSR=0.75")
    ax.set_ylim(0, 1.05); ax.legend(fontsize=7)
    _ax_style(ax, "Phase 3: Episode constraint\nsatisfaction (ASME B31.3)", "Episode", "CSR")

    ax = fig.add_subplot(gs[2, 2])
    es = pd.Series(agent.ent_hist).rolling(25).mean()
    ax.plot(es, c=PURPLE, lw=2)
    ax.axvline(100, c=GOLD,  lw=1, ls=":")
    ax.axvline(200, c=GREEN, lw=1, ls=":")
    _ax_style(ax, "Phase 3: Policy entropy\nexploration → exploitation", "Episode", "Entropy")

    # ── Row 3: Benchmark + ablation bar chart ─────────────────────────────────
    ax = fig.add_subplot(gs[3, 0])
    rl_sig  = [r["sigma"]            for r in e2e_rec]
    rl_dP   = [r["dP"]               for r in e2e_rec]
    fem_sig = [r["von_mises_stress"]  for r in fem_rec]
    fem_dP  = [r["pressure_drop_kPa"] for r in fem_rec]
    ax.scatter(fem_sig, fem_dP, c=RED,  alpha=0.5, s=15, label="FEM (same n)")
    ax.scatter(rl_sig,  rl_dP,  c=BLUE, alpha=0.8, s=40, zorder=5,
               label="PINN-RL-Gen [Ours]", edgecolors="white", lw=0.5)
    ax.axvline(200, c=RED, lw=1.5, ls="--", alpha=0.5, label="ASME limit")
    ax.legend(fontsize=7)
    _ax_style(ax, "Phase 4: Pareto frontier\n[P5] fair comparison (same n_designs)",
              "σ_max (MPa)", "ΔP (kPa)")

    ax  = fig.add_subplot(gs[3, 1])
    met = ["Stress\nMAE %", "Press.\nMAE %", "Stress\nR²×100", "Gen\nCSR %", "RL\nImpr %"]
    v1v = [v1_results["stress_mae_pct"], v1_results["pressure_mae_pct"],
           v1_results["stress_r2"] * 100, v1_results["gen_csr"] * 100, v1_results["ppo_impr"]]
    v3v = [pinn_res["von_mises_stress"]["MAE_pct"], pinn_res["pressure_drop_kPa"]["MAE_pct"],
           pinn_res["von_mises_stress"]["R2"] * 100, div_res["csr"] * 100, rl_res["improvement_pct"]]
    xb  = np.arange(len(met)); w = 0.35
    ax.bar(xb - w / 2, v1v, w, color=RED,  alpha=0.8, label="v1 baseline")
    ax.bar(xb + w / 2, v3v, w, color=BLUE, alpha=0.8, label="v3 improved")
    ax.set_xticks(xb); ax.set_xticklabels(met, fontsize=7.5)
    ax.legend(fontsize=7)
    _ax_style(ax, "v1 → v3 improvement\n(key metrics)", "", "Value")

    ax  = fig.add_subplot(gs[3, 2])
    m2  = ["Avg σ\n(MPa)", "Avg ΔP\n(kPa)", "CSR\n(%)"]
    e2v = [e2e_b["avg_sigma"], e2e_b["avg_dP"], e2e_b["csr"] * 100]
    fmv = [fem_b["avg_sigma"], fem_b["avg_dP"], fem_b["csr"] * 100]
    xb  = np.arange(len(m2)); wb = 0.35
    ax.bar(xb - wb / 2, e2v, wb, color=BLUE, alpha=0.85, label="PINN-RL-Gen")
    ax.bar(xb + wb / 2, fmv, wb, color=RED,  alpha=0.85, label="FEM (same n)")
    ax.set_xticks(xb); ax.set_xticklabels(m2, fontsize=7.5)
    ax.legend(fontsize=7)
    _ax_style(ax, "Phase 4: End-to-end benchmark\n[P5] n_designs=30 each")

    # ── Row 4: Ablation chart ─────────────────────────────────────────────────
    ax = fig.add_subplot(gs[4, :])
    ax.set_facecolor("#FAFBFC")
    _ax_style(ax, "[P4] Ablation study — each novelty contribution isolated (3-seed mean ± std)")
    configs     = ablation_df.index.tolist()
    xpos        = np.arange(len(configs))
    stress_means = ablation_df["stress_mae_mean"].values
    stress_stds  = ablation_df["stress_mae_std"].fillna(0).values
    csr_means    = ablation_df["e2e_csr_mean"].values
    csr_stds     = ablation_df["e2e_csr_std"].fillna(0).values
    ax2 = ax.twinx()
    ax.bar(xpos - 0.2, stress_means, 0.35, color=LBLUE, alpha=0.85, label="Stress MAE % (↓ better)")
    ax.errorbar(xpos - 0.2, stress_means, yerr=stress_stds,
                fmt="none", ecolor="black", capsize=4, lw=1)
    csr_plot     = np.where(np.isnan(csr_means), 0, csr_means)
    csr_std_plot = np.where(np.isnan(csr_stds),  0, csr_stds)
    ax2.bar(xpos + 0.2, csr_plot, 0.35, color=GREEN, alpha=0.85, label="E2E CSR (↑ better)")
    ax2.errorbar(xpos + 0.2, csr_plot, yerr=csr_std_plot,
                 fmt="none", ecolor="black", capsize=4, lw=1)
    ax.set_xticks(xpos); ax.set_xticklabels(configs, fontsize=7.5)
    ax.set_ylabel("Stress MAE %", fontsize=8, color=LBLUE)
    ax2.set_ylabel("E2E CSR", fontsize=8, color=GREEN)
    ax2.set_ylim(0, 1.2)
    ax.axhline(5, c=RED, lw=1.5, ls="--", alpha=0.6, label="5% target")
    lines, labels = ax.get_legend_handles_labels()
    l2, lb2       = ax2.get_legend_handles_labels()
    ax.legend(lines + l2, labels + lb2, fontsize=7, loc="upper right")

    # ── Row 5: Architecture summary ───────────────────────────────────────────
    ax = fig.add_subplot(gs[5, :])
    ax.set_xlim(0, 12); ax.set_ylim(0, 3.5); ax.axis("off")
    ax.set_title("Closed-loop v3 architecture + novelty map + literature gap",
                 fontsize=11, fontweight="bold", color=BLUE, pad=8)
    boxes = [
        (1.0, 1.9, "[N2] CA-VAE\nLayout Gen",       LBLUE),
        (3.0, 1.9, "[N3] Curriculum\nPPO+GAE-λ",     BLUE),
        (5.0, 1.9, "[N1] Multi-task\nPINN+PDE",      GREEN),
        (7.0, 1.9, "Reward\n4-objective",             GOLD),
        (9.0, 1.9, "Policy\nUpdate",                  PURPLE),
    ]
    for x0, y0, lbl, col in boxes:
        ax.add_patch(mpatches.FancyBboxPatch(
            (x0 - 0.75, y0 - 0.55), 1.5, 1.1,
            boxstyle="round,pad=0.06", lw=1.5, edgecolor=col, facecolor=col + "22",
        ))
        ax.text(x0, y0, lbl, ha="center", va="center",
                fontsize=7.5, fontweight="bold", color=col)
    for x0 in [1.75, 3.75, 5.75, 7.75]:
        ax.annotate("", xy=(x0 + 0.5, 1.9), xytext=(x0, 1.9),
                    arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.5))
    ax.annotate("", xy=(1.0, 1.35), xytext=(9.65, 1.35),
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.5,
                                connectionstyle="arc3,rad=0.3"))
    ax.text(5.3, 0.5, "iterate until convergence",
            ha="center", fontsize=8, color=RED, style="italic")
    novelties = [
        ("[P1] log(σ) target → stress MAE <5% · [P2] KL annealing+free bits → CSR >60% · "
         "[P3] linear+hard-mask reward → stable convergence", BLUE),
        ("[P4] ablation table → isolates N1/N2/N3 contribution · [P5] fair FEM timing "
         "(same n_designs) · [P6] real flow-network GasLib", TEAL),
        ("[N1] shared encoder + task heads + PDE per task · [N2] ASME penalty in ELBO · "
         "[N3] 3-phase curriculum · [N4] thin-wall augment · [N5] kNN correction", PURPLE),
    ]
    for i, (txt, col) in enumerate(novelties):
        ax.text(0.05, 3.2 - i * 0.65, txt, ha="left", va="center",
                fontsize=7.5, color=col, fontweight="bold")

    plt.suptitle(
        "PINN-RL-Generative Framework v3 — Team PINNFlow\n"
        "IEEE IES Generative AI Challenge 2026 — Milestone 1 (Final)",
        fontsize=13, fontweight="bold", color=BLUE, y=0.995,
    )

    path = os.path.join(RESULTS_DIR, "pinnflow_v3_results.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Figure saved → {path}")
    return path
