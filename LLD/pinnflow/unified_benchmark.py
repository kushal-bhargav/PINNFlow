"""
Unified IEEE-style benchmark runner for PINNFlow + the ieeejen paper modules.

This module replaces the old Barlow-formula shortcut with a public FEM/stress-
strain dataset loader and provides one orchestrated entrypoint for:
  1. PINNFlow E2E pipeline evaluation
  2. Paper 1 PINN reliability analysis
  3. Paper 2 ANN/PINN parametric sweep
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

from pinnflow.config import RESULTS_DIR


DEFAULT_ZENODO_RECORD = 15221412
DEFAULT_ZENODO_API = "https://zenodo.org/api/records/{record_id}"


@dataclass
class FEMDatasetBundle:
    frame: pd.DataFrame
    source_url: str
    local_path: Path
    x_col: str
    y_col: str


@dataclass
class BenchmarkSection:
    name: str
    elapsed_s: float
    metrics: Dict[str, Any]


class FEMDatasetLoader:
    """
    Downloads and caches a public FEM/stress-strain dataset from Zenodo.

    The loader uses the Zenodo record API so the file link is discovered at
    runtime rather than hardcoded into the codebase.
    """

    def __init__(
        self,
        cache_dir: str | Path = Path(RESULTS_DIR) / "benchmarks" / "fem_cache",
        record_id: int = DEFAULT_ZENODO_RECORD,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.record_id = record_id

    def load(self) -> FEMDatasetBundle:
        local_path, source_url = self._ensure_local_file()
        frame = self._read_table(local_path)
        x_col, y_col = self._pick_curve_columns(frame)
        frame = frame[[x_col, y_col]].dropna().copy()
        frame = frame.sort_values(by=x_col).reset_index(drop=True)
        return FEMDatasetBundle(
            frame=frame,
            source_url=source_url,
            local_path=local_path,
            x_col=x_col,
            y_col=y_col,
        )

    def _ensure_local_file(self) -> Tuple[Path, str]:
        cached = self._find_cached_file()
        if cached is not None:
            return cached, f"file://{cached}"

        api_url = DEFAULT_ZENODO_API.format(record_id=self.record_id)
        payload = self._fetch_json(api_url)

        file_meta = self._choose_file(payload.get("files", []))
        if not file_meta:
            raise RuntimeError(f"No downloadable file found in Zenodo record {self.record_id}.")

        filename = file_meta["key"]
        download_url = file_meta["links"]["self"]
        cached_path = self.cache_dir / filename
        if cached_path.exists():
            return cached_path, download_url

        cached_path.write_bytes(self._fetch_bytes(download_url))
        return cached_path, download_url

    def _find_cached_file(self) -> Optional[Path]:
        preferred_exts = (".csv", ".xlsx", ".xls", ".json", ".tsv")
        for ext in preferred_exts:
            matches = sorted(self.cache_dir.glob(f"*{ext}"))
            if matches:
                return matches[0]
        return None

    def _fetch_json(self, url: str) -> Dict[str, Any]:
        try:
            import requests  # type: ignore
        except ImportError:
            requests = None

        if requests is not None:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            return response.json()

        from urllib.request import urlopen

        with urlopen(url, timeout=30) as response:
            import json

            return json.loads(response.read().decode("utf-8"))

    def _fetch_bytes(self, url: str) -> bytes:
        try:
            import requests  # type: ignore
        except ImportError:
            requests = None

        if requests is not None:
            response = requests.get(url, timeout=120)
            response.raise_for_status()
            return response.content

        from urllib.request import urlopen

        with urlopen(url, timeout=120) as response:
            return response.read()

    def _choose_file(self, files: Iterable[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        preferred_exts = (".csv", ".xlsx", ".xls", ".json", ".tsv")
        file_list = list(files)
        for ext in preferred_exts:
            for item in file_list:
                if str(item.get("key", "")).lower().endswith(ext):
                    return item
        return file_list[0] if file_list else None

    def _read_table(self, path: Path) -> pd.DataFrame:
        suffix = path.suffix.lower()
        if suffix in {".xlsx", ".xls"}:
            return pd.read_excel(path)
        if suffix == ".json":
            return pd.read_json(path)
        if suffix == ".tsv":
            return pd.read_csv(path, sep="\t")
        return pd.read_csv(path)

    def _pick_curve_columns(self, frame: pd.DataFrame) -> Tuple[str, str]:
        cols = list(frame.columns)
        numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(frame[c])]
        if len(numeric_cols) >= 2:
            x_col, y_col = self._match_named_curve_cols(numeric_cols)
            return x_col, y_col
        raise RuntimeError("The FEM reference dataset does not contain two numeric columns.")

    def _match_named_curve_cols(self, cols: list[str]) -> Tuple[str, str]:
        lower_map = {c.lower(): c for c in cols}
        x_candidates = ["strain", "engineering_strain", "eps", "x"]
        y_candidates = ["stress", "engineering_stress", "sigma", "y"]
        x_col = next((lower_map[name] for name in x_candidates if name in lower_map), cols[0])
        y_col = next((lower_map[name] for name in y_candidates if name in lower_map and lower_map[name] != x_col), cols[1] if len(cols) > 1 else cols[0])
        if x_col == y_col and len(cols) > 1:
            y_col = cols[1]
        return x_col, y_col


def _to_summary_records(rows: Dict[str, Any]) -> pd.DataFrame:
    flat_rows = []
    for section, payload in rows.items():
        if isinstance(payload, dict):
            for metric, value in payload.items():
                flat_rows.append({"Section": section, "Metric": metric, "Value": value})
        else:
            flat_rows.append({"Section": section, "Metric": "value", "Value": payload})
    return pd.DataFrame(flat_rows)


class UnifiedBenchmarkRunner:
    def __init__(self, output_dir: str | Path = Path(RESULTS_DIR) / "benchmarks") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_loader = FEMDatasetLoader()

    def run_all(self) -> Dict[str, Any]:
        sections = {
            "pipeline_e2e": self._safe_stage("pipeline_e2e", self.run_pipeline_e2e_stage),
            "fem_reference": self._safe_stage("fem_reference", self.run_fem_reference_stage),
            "paper1": self._safe_stage("paper1", self.run_paper1_stage),
            "paper2": self._safe_stage("paper2", self.run_paper2_stage),
            "paper5_pirn": self._safe_stage("paper5_pirn", self.run_paper5_stage),
        }
        report_paths = self._write_report(sections)
        sections["report_paths"] = report_paths
        return sections

    def _safe_stage(self, name: str, fn) -> BenchmarkSection:
        t0 = perf_counter()
        try:
            section = fn()
            return section
        except Exception as exc:
            elapsed = perf_counter() - t0
            return BenchmarkSection(name=name, elapsed_s=elapsed, metrics={"error": str(exc)})

    def run_fem_reference_stage(self) -> BenchmarkSection:
        bundle = self.dataset_loader.load()
        frame = bundle.frame
        x = frame[bundle.x_col].to_numpy(dtype=float)
        y = frame[bundle.y_col].to_numpy(dtype=float)

        # The FEM dataset is the source of truth. We benchmark the surrogate
        # against a lightweight interpolated reference curve derived from it.
        ref_start = perf_counter()
        x_sample = np.linspace(float(x.min()), float(x.max()), min(256, len(x)))
        y_ref = np.interp(x_sample, x, y)
        ref_elapsed = perf_counter() - ref_start

        from pinnflow.pinn import MultiTaskPINN
        from pinnflow.environment import PipelineEnv

        pinn = MultiTaskPINN(n_in=10)
        env = PipelineEnv(pinn=pinn)
        X = self._build_pipeline_state_grid(x_sample, env)

        pred_start = perf_counter()
        y_pred = pinn.predict(X)[:, 0]
        pred_elapsed = perf_counter() - pred_start

        mae = float(np.mean(np.abs(y_pred - y_ref)))
        rmse = float(np.sqrt(np.mean((y_pred - y_ref) ** 2)))
        speedup = float(ref_elapsed / max(pred_elapsed, 1e-9))

        metrics = {
            "source_url": bundle.source_url,
            "rows": len(frame),
            "x_col": bundle.x_col,
            "y_col": bundle.y_col,
            "reference_interp_ms": round(ref_elapsed * 1000.0, 4),
            "surrogate_infer_ms": round(pred_elapsed * 1000.0, 4),
            "speedup_factor": round(speedup, 3),
            "mae": round(mae, 5),
            "rmse": round(rmse, 5),
        }
        return BenchmarkSection("fem_reference", ref_elapsed + pred_elapsed, metrics)

    def run_pipeline_e2e_stage(self) -> BenchmarkSection:
        from pinnflow.orchestrator_v2 import UnifiedOrchestrator

        t0 = perf_counter()
        orchestrator = UnifiedOrchestrator()
        result = orchestrator.run_e2e(case_name="refinery_compliance")
        elapsed = perf_counter() - t0
        metrics = {
            "case_name": "refinery_compliance",
            "compliance_score": float(result.get("compliance_score", 0.0)),
            "optimized_state_dim": int(len(result.get("optimized_state", []))),
            "deliverable_count": len(result.get("deliverables", {})),
            "elapsed_s": round(elapsed, 4),
        }
        return BenchmarkSection("pipeline_e2e", elapsed, metrics)

    def run_paper1_stage(self) -> BenchmarkSection:
        from ieeejen.paper1.cross_section import CrossSectionIntegrator
        from ieeejen.paper1.reliability.distributions import UncertaintyModel
        from ieeejen.paper1.reliability.monte_carlo import MonteCarloSimulation
        from ieeejen.paper1.pinn.network import build_pinn

        network = build_pinn()
        cross_section = CrossSectionIntegrator(D=0.914, t=0.0127, n_r=4, n_theta=16)
        uncertainty = UncertaintyModel()
        pipe_params = {
            "D": 0.914,
            "t": 0.0127,
            "P": 10e6,
            "delta_T": 20.0,
            "E": 207e9,
            "nu": 0.3,
            "alpha_T": 1.17e-5,
            "L": 300.0,
            "beta_deg": 15.0,
        }

        mcs = MonteCarloSimulation(
            network=network,
            uncertainty_model=uncertainty,
            cross_section_integrator=cross_section,
            pipe_params=pipe_params,
            n_x=120,
            device="cpu",
        )

        t0 = perf_counter()
        result = mcs.run(n_mcs=2000, batch_size=500, seed=42, verbose=False)
        elapsed = perf_counter() - t0
        metrics = {
            "P_f": float(result["P_f"]),
            "beta_HL": float(result["beta_HL"]),
            "cov_Pf": float(result["cov_Pf"]),
            "n_failures": int(result["n_failures"]),
            "n_mcs": int(result["n_mcs"]),
            "elapsed_s": round(elapsed, 4),
            "avg_inference_ms": round((elapsed / max(result["n_mcs"], 1)) * 1000.0, 5),
        }
        return BenchmarkSection("paper1", elapsed, metrics)

    def run_paper2_stage(self) -> BenchmarkSection:
        from ieeejen.paper2.pinn_base import ElasticPIPENN
        from ieeejen.paper2.parametric_study import parametric_sweep

        model = ElasticPIPENN(EA=1.0e9, EI=1.0e6, ku=5.0e4, kp=2.5e4, L=300.0)
        delta_values = np.linspace(0.05, 0.35, 8)
        beta_values = np.linspace(0.0, 90.0, 7)

        t0 = perf_counter()
        results = parametric_sweep(
            model=model,
            delta_values=delta_values,
            beta_values=beta_values,
            n_x=120,
        )
        elapsed = perf_counter() - t0

        eps_t = results["eps_tension"]
        eps_c = results["eps_compression"]
        metrics = {
            "grid_shape": list(eps_t.shape),
            "elapsed_s": round(elapsed, 4),
            "peak_tension": float(np.max(eps_t)),
            "peak_compression": float(np.min(eps_c)),
            "mean_tension": float(np.mean(eps_t)),
            "mean_compression": float(np.mean(eps_c)),
        }
        return BenchmarkSection("paper2", elapsed, metrics)

    def _build_pirn_control_sequence(self, sim_results: Dict[str, Any], network) -> np.ndarray:
        t = np.asarray(sim_results["t"], dtype=float)
        p_nodes = np.asarray(sim_results["p_nodes"], dtype=float)
        p_ref = p_nodes[network.entry_nodes[0], :]
        p_delta = p_ref - np.mean(p_ref)

        n_steps = len(t)
        n_controls = max(len(network.compressors), 1)
        u_seq = np.zeros((n_steps, n_controls), dtype=float)
        time_scale = max(float(t.max() - t.min()), 1.0)
        phase = 2.0 * np.pi * (t - t.min()) / time_scale

        for idx in range(n_controls):
            amplitude = 0.02 + 0.01 * idx
            offset = 0.005 * idx
            u_seq[:, idx] = (p_delta / 1e6) * (1.0 + 0.05 * idx) + amplitude * np.sin(phase + offset)

        return u_seq

    def run_paper5_stage(self) -> BenchmarkSection:
        import torch

        from ieeejen.paper5.gas_physics.network_assembly import GasLibNetwork, GasNetworkStateSpace
        from ieeejen.paper5.pirn.model import PIRNCell, PIRNModel
        from ieeejen.paper5.pirn.trainer import PIRNTrainer
        from ieeejen.paper5.simulation.transient_fdm import simulate_network

        network = GasLibNetwork("GasLib-11")
        lam_true = [float(props["lam"]) for props in network.pipe_props]
        theta_true = {f"lambda_pipe_{idx}": lam for idx, lam in enumerate(lam_true)}

        sim_results = simulate_network(
            network,
            lam_true_list=lam_true,
            T_total=300.0,
            dt=2.0,
            noise_level=0.01,
            seed=11,
        )
        u_seq = self._build_pirn_control_sequence(sim_results, network)

        y_terminal = np.asarray(sim_results["y_terminal"], dtype=float).T
        p_nodes = np.asarray(sim_results["p_nodes"], dtype=float)
        n_steps = y_terminal.shape[0]
        split_idx = max(4, int(n_steps * 0.7))
        split_idx = min(split_idx, n_steps - 2) if n_steps > 2 else n_steps

        y_train = torch.tensor(y_terminal[:split_idx], dtype=torch.float64)
        y_test = torch.tensor(y_terminal[split_idx:], dtype=torch.float64)
        u_train = torch.tensor(u_seq[:split_idx], dtype=torch.float64)
        u_test = torch.tensor(u_seq[split_idx:], dtype=torch.float64)
        x0_train = torch.tensor(p_nodes[:, 0], dtype=torch.float64)
        x0_test = torch.tensor(p_nodes[:, split_idx], dtype=torch.float64)

        network_ss = GasNetworkStateSpace(network, dt=2.0, n_segments=1)
        model = PIRNModel(PIRNCell(network_ss), tau=min(10, max(2, split_idx // 2)))
        trainer = PIRNTrainer(
            model=model,
            y_measured_train=y_train,
            u_train=u_train,
            x0_train=x0_train,
            y_measured_test=y_test,
            u_test=u_test,
            x0_test=x0_test,
            theta_true=theta_true,
            n_epochs=30,
            lr=1e-3,
            grad_clip=1.0,
            tau=min(10, max(2, split_idx // 2)),
            log_every=10,
            device="cpu",
        )

        t0 = perf_counter()
        history = trainer.train()
        with torch.no_grad():
            _, y_pred_test = model(x0_test.double(), u_test.double(), u_test.shape[0])
        terminal_rmse = float(torch.sqrt(torch.mean((y_pred_test.float() - y_test.float()) ** 2)).item())
        elapsed = perf_counter() - t0

        metrics = {
            "network_name": "GasLib-11",
            "train_samples": int(split_idx),
            "test_samples": int(n_steps - split_idx),
            "elapsed_s": round(elapsed, 4),
            "final_train_loss": float(history["train_loss"][-1]) if history["train_loss"] else float("nan"),
            "final_test_loss": float(history["test_loss"][-1]) if history["test_loss"] else float("nan"),
            "mean_lambda_error": float(history["lambda_errors"][-1]) if history["lambda_errors"] else float("nan"),
            "terminal_rmse": terminal_rmse,
            "terminal_rmse_kPa": terminal_rmse / 1e3,
        }
        return BenchmarkSection("paper5_pirn", elapsed, metrics)

    def _build_pipeline_state_grid(self, x_values: np.ndarray, env) -> np.ndarray:
        template = np.array([273.0, 9.27, 120.0, 5.0, 15.0, 20.0, 2.5, 0.55, 0.0, 0.9], dtype=float)
        states = np.tile(template, (len(x_values), 1))

        x_min = float(np.min(x_values))
        x_max = float(np.max(x_values))
        span = max(x_max - x_min, 1e-9)
        scaled = (x_values - x_min) / span

        states[:, 3] = np.clip(1.0 + 19.0 * scaled, env.BOUNDS[3, 0], env.BOUNDS[3, 1])
        states[:, 8] = 0.0
        states[:, 9] = np.clip(0.6 + 0.2 * scaled, env.BOUNDS[9, 0], env.BOUNDS[9, 1])
        states = np.clip(states, env.BOUNDS[:, 0], env.BOUNDS[:, 1])
        states[:, 8] = np.clip(np.rint(states[:, 8]), env.BOUNDS[8, 0], env.BOUNDS[8, 1])
        states[:, 9] = np.clip(states[:, 9], env.BOUNDS[9, 0], env.BOUNDS[9, 1])
        return states

    def _write_report(self, sections: Dict[str, BenchmarkSection]) -> Dict[str, str]:
        serializable = {name: asdict(section) for name, section in sections.items()}
        df = _to_summary_records({name: section.metrics for name, section in sections.items()})

        csv_path = self.output_dir / "unified_ieee_benchmark.csv"
        md_path = self.output_dir / "unified_ieee_benchmark.md"
        json_path = self.output_dir / "unified_ieee_benchmark.json"

        df.to_csv(csv_path, index=False)
        with open(md_path, "w", encoding="utf-8") as fh:
            fh.write("# Unified IEEE Benchmark Summary\n\n")
            fh.write(df.to_markdown(index=False))
            fh.write("\n")
        with open(json_path, "w", encoding="utf-8") as fh:
            import json

            json.dump(serializable, fh, indent=2)

        print("\n" + "=" * 88)
        print("  UNIFIED IEEE BENCHMARK SUMMARY")
        print("=" * 88)
        print(df.to_string(index=False))
        print("=" * 88)

        return {"csv": str(csv_path), "markdown": str(md_path), "json": str(json_path)}
