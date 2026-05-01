"""
Evaluation metrics for PIRN gas pipeline parameter identification.

Metrics:
  1. Parameter identification accuracy: relative error of λ̂_i vs. λ_true_i
  2. State estimation accuracy: RMSE of pressure/flow at internal nodes
  3. Scalability: training time vs. network size
  4. Robustness: accuracy vs. noise level
  5. Dispatch cost comparison (application to Integrated Energy System)

References:
    Wang / Liu et al. (2025), PIRN (arXiv:2502.07230)
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
import time


# ─────────────────────────────────────────────────────────────────────────────
# Parameter identification metrics
# ─────────────────────────────────────────────────────────────────────────────

def parameter_identification_errors(
    lam_estimated: Dict[int, float],
    lam_true:      Dict[int, float],
) -> Dict:
    """
    Compute relative and absolute errors for identified friction factors.

    Args:
        lam_estimated : {pipe_idx: estimated_lambda}
        lam_true      : {pipe_idx: true_lambda}

    Returns:
        metrics dict with per-pipe errors and aggregate statistics
    """
    rel_errors  = {}
    abs_errors  = {}
    for idx in lam_true:
        true = lam_true[idx]
        est  = lam_estimated.get(idx, float("nan"))
        rel_errors[idx] = abs(est - true) / (abs(true) + 1e-12)
        abs_errors[idx] = abs(est - true)

    rel_vals = [v for v in rel_errors.values() if not np.isnan(v)]
    return {
        "per_pipe_rel_error": rel_errors,
        "per_pipe_abs_error": abs_errors,
        "mean_rel_error":     float(np.mean(rel_vals)),
        "max_rel_error":      float(np.max(rel_vals)),
        "median_rel_error":   float(np.median(rel_vals)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# State estimation metrics
# ─────────────────────────────────────────────────────────────────────────────

def pressure_rmse(
    p_predicted: np.ndarray,
    p_true:      np.ndarray,
) -> float:
    """
    RMSE of pressure predictions at all nodes (Pa).

    Args:
        p_predicted : (n_nodes, N_t) predicted pressures
        p_true      : (n_nodes, N_t) true pressures

    Returns:
        rmse : scalar RMSE (Pa)
    """
    return float(np.sqrt(np.mean((p_predicted - p_true) ** 2)))


def flow_rmse(
    q_predicted: np.ndarray,
    q_true:      np.ndarray,
) -> float:
    """
    RMSE of mass flow predictions at all edges (kg/s).
    """
    return float(np.sqrt(np.mean((q_predicted - q_true) ** 2)))


def internal_node_rmse(
    p_predicted: np.ndarray,
    p_true:      np.ndarray,
    terminal_indices: List[int],
) -> float:
    """
    RMSE specifically at internal (non-terminal) nodes.

    These are not observed during training and represent generalisation quality.

    Args:
        terminal_indices : list of terminal (sensor) node indices
    """
    all_idx      = set(range(p_true.shape[0]))
    internal_idx = sorted(all_idx - set(terminal_indices))
    if not internal_idx:
        return float("nan")
    return float(np.sqrt(np.mean((p_predicted[internal_idx, :] -
                                   p_true[internal_idx, :])**2)))


# ─────────────────────────────────────────────────────────────────────────────
# Robustness study: performance vs. noise level
# ─────────────────────────────────────────────────────────────────────────────

def noise_robustness_table(
    results_by_noise: Dict[float, Dict],
) -> None:
    """
    Print a formatted robustness table: noise level vs. parameter/state errors.

    Args:
        results_by_noise : {noise_pct: metrics_dict}
    """
    print("\nNoise robustness study:")
    print(f"{'Noise (%)':<12} {'Mean lam err (%)':>16} {'P_RMSE (kPa)':>15} {'Internal P_RMSE':>17}")
    print("─" * 63)
    for noise_pct in sorted(results_by_noise.keys(), reverse=True):
        m = results_by_noise[noise_pct]
        lam_err  = m.get("mean_rel_lam_error", float("nan")) * 100
        p_rmse   = m.get("p_rmse_kPa", float("nan"))
        p_int    = m.get("internal_p_rmse_kPa", float("nan"))
        print(f"{noise_pct*100:<12.1f} {lam_err:>16.3f} {p_rmse:>15.3f} {p_int:>17.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# Scalability: timing vs. network size
# ─────────────────────────────────────────────────────────────────────────────

def scalability_report(
    network_names: List[str],
    n_pipes_list:  List[int],
    train_times_s: List[float],
) -> None:
    """
    Print scalability report: training time vs. network size (number of pipes).

    Args:
        network_names : list of network name strings (e.g. ['GasLib-11', ...])
        n_pipes_list  : list of pipe counts
        train_times_s : list of training times in seconds
    """
    print("\nScalability report:")
    print(f"{'Network':<15} {'N_pipes':>10} {'Train time (s)':>16} {'Time/pipe (s)':>15}")
    print("─" * 58)
    for name, n_p, t_s in zip(network_names, n_pipes_list, train_times_s):
        print(f"{name:<15} {n_p:>10} {t_s:>16.1f} {t_s/n_p:>15.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# Dispatch cost comparison (IES application)
# ─────────────────────────────────────────────────────────────────────────────

def dispatch_cost_comparison(
    cost_true_model:  float,
    cost_pirn_model:  float,
    cost_naive_model: float,
) -> Dict:
    """
    Compare economic dispatch costs under different pipeline models.

    In the Integrated Energy System (IES) application, the gas pipeline model
    is embedded in an optimal dispatch problem. Errors in pipeline parameters
    lead to suboptimal or infeasible dispatch schedules.

    Args:
        cost_true_model  : optimal dispatch cost using true parameters (£/MWh or $/h)
        cost_pirn_model  : dispatch cost using PIRN-identified parameters
        cost_naive_model : dispatch cost using nominal/default parameters (no identification)

    Returns:
        comparison dict
    """
    delta_pirn  = (cost_pirn_model  - cost_true_model) / (cost_true_model + 1e-12) * 100
    delta_naive = (cost_naive_model - cost_true_model) / (cost_true_model + 1e-12) * 100

    result = {
        "cost_true":         cost_true_model,
        "cost_pirn":         cost_pirn_model,
        "cost_naive":        cost_naive_model,
        "delta_pirn_%":      delta_pirn,
        "delta_naive_%":     delta_naive,
        "pirn_improvement_vs_naive_%": delta_naive - delta_pirn,
    }

    print(f"\nDispatch cost comparison:")
    print(f"  True model:  {cost_true_model:.4f}  (baseline)")
    print(f"  PIRN model:  {cost_pirn_model:.4f}  "
          f"({delta_pirn:+.2f}% vs true)")
    print(f"  Naive model: {cost_naive_model:.4f}  "
          f"({delta_naive:+.2f}% vs true)")
    print(f"  PIRN improvement over naive: {delta_naive - delta_pirn:.2f}%")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Comprehensive evaluation for a trained PIRN
# ─────────────────────────────────────────────────────────────────────────────

def full_evaluation(
    model,
    sim_results: Dict,
    lam_true_dict: Dict[int, float],
    terminal_indices: List[int],
    u_seq: Optional[np.ndarray] = None,
    device: str = "cpu",
) -> Dict:
    """
    Run complete evaluation of a trained PIRNModel.

    Args:
        model           : trained PIRNModel
        sim_results     : simulation results dict from transient_fdm.simulate_network()
        lam_true_dict   : {pipe_idx: true_lambda} reference values
        terminal_indices: list of terminal node indices (sensor locations)
        device          : torch device

    Returns:
        comprehensive metrics dict
    """
    # ── Parameter identification errors ─────────────────────────────────────
    lam_est_raw = model.get_identified_parameters()
    lam_est     = {int(k.split("_")[-1]): v for k, v in lam_est_raw.items()}
    param_errs  = parameter_identification_errors(lam_est, lam_true_dict)

    # ── State estimation: run forward prediction ─────────────────────────────
    model.eval()
    p_true  = sim_results["p_nodes"]        # (n_nodes, N_t)
    N_t     = p_true.shape[1]
    n_nodes = p_true.shape[0]

    # Build input sequences
    x0      = torch.tensor(p_true[:, 0], dtype=torch.float64, device=device)
    if u_seq is None:
        u_seq_t = torch.zeros(
            N_t,
            max(len(model.cell.network_ss.net.compressors), 1),
            dtype=torch.float64,
            device=device,
        )
    else:
        u_seq_t = torch.tensor(u_seq, dtype=torch.float64, device=device)

    with torch.no_grad():
        x_seq, y_seq = model(x0, u_seq_t, N_t)

    p_pred = x_seq[1:].cpu().numpy().T   # (n_nodes, N_t)

    # ── Metrics ──────────────────────────────────────────────────────────────
    p_rmse_pa = pressure_rmse(p_pred, p_true)
    p_int_rmse = internal_node_rmse(p_pred, p_true, terminal_indices)

    metrics = {
        **param_errs,
        "p_rmse_Pa":          p_rmse_pa,
        "p_rmse_kPa":         p_rmse_pa / 1e3,
        "internal_p_rmse_Pa": p_int_rmse,
        "internal_p_rmse_kPa": p_int_rmse / 1e3,
    }
    return metrics
