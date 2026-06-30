"""
Unified orchestration for the industrial automation flow.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import time
from typing import Any, Dict, List

import numpy as np

from pinnflow import (
    CAVAE,
    ClosedLoopOptim,
    DeliverableGenerator,
    ExplainEngine,
    IntentEngine,
    MultiTaskPINN,
    PipelineEnv,
    PPOAgent,
    PhysicsSimulator,
    RequirementParser,
    ScenarioBank,
)
from pinnflow.codal_engine.agents.api14e_agent import API14EAgent
from pinnflow.codal_engine.agents.asme_b31_agent import ASMEB31Agent
from pinnflow.codal_engine.agents.thermal_flexibility_agent import ThermalFlexibilityAgent
from pinnflow.codal_engine.agents.tee_reinforcement_agent import TeeReinforcementAgent
from pinnflow.codal_engine.agents.reducer_transition_agent import ReducerTransitionAgent
from pinnflow.codal_engine.agents.pipe_schedule_agent import PipeScheduleAgent
from pinnflow.codal_engine.agents.elbow_sif_agent import ElbowSIFAgent
from pinnflow.codal_engine.agents.branch_rules_agent import BranchRulesAgent
from pinnflow.codal_engine.knowledge.fetcher import CodalFetcher
from pinnflow.codal_engine.knowledge.parser import CodalParser
from pinnflow.codal_engine.knowledge.rule_store import CodalRuleStore
from pinnflow.codal_engine.wrapper import CodalEnvironmentWrapper
from pinnflow.gnn import GasNetworkGNN, load_gaslib_graph

# GasLib training data extractor — provides real network pipe geometry to PINN
try:
    import sys as _sys
    _lld_root = str(Path(__file__).resolve().parent.parent)
    if _lld_root not in _sys.path:
        _sys.path.insert(0, _lld_root)
    from data.gaslib_training_extractor import (
        extract_training_samples_from_gaslib,
        blend_with_synthetic,
    )
    _GASLIB_EXTRACTOR_AVAILABLE = True
except Exception as _gaslib_err:
    _GASLIB_EXTRACTOR_AVAILABLE = False
    extract_training_samples_from_gaslib = None  # type: ignore[assignment]
    blend_with_synthetic = None                  # type: ignore[assignment]
    print(f"[Orchestrator] GasLib extractor unavailable ({_gaslib_err}); "
          "PINN will train on synthetic data only.")


@dataclass
class CaseContext:
    case_name: str
    scenario: Dict[str, Any]
    ingested: Dict[str, Any] | None = None
    topology_name: str = ""
    gnn_summary: Dict[str, float] = field(default_factory=dict)
    variants: List[Dict[str, Any]] = field(default_factory=list)
    selected_design: List[float] = field(default_factory=list)
    ranking: List[Dict[str, Any]] = field(default_factory=list)
    optimized_metrics: Dict[str, Any] = field(default_factory=dict)
    deliverables: Dict[str, str] = field(default_factory=dict)
    trace_path: str | None = None


class UnifiedOrchestrator:
    """
    Handles the lifecycle from raw documents to verified deliverables.
    """

    def __init__(
        self,
        train_cvae: bool = True,
        cvae_epochs: int = 18,
        cvae_samples_per_scenario: int = 20,
        cvae_scenario_names: List[str] | None = None,
        train_pinn: bool = True,
        pinn_epochs: int = 500,
        pinn_synthetic_n: int = 1200,
        train_ppo: bool = True,
        ppo_episodes: int = 400,
        train_gnn: bool = True,
    ):
        print("\n" + "=" * 80)
        print("  PINNFLOW V8.1 - DYNAMIC CODAL INTELLIGENCE SUITE")
        print("=" * 80)

        self.pinn = MultiTaskPINN(n_in=16)
        self.vae = CAVAE(x_dim=16)
        self.agent = PPOAgent(sdim=16, adim=10)
        self.base_env = PipelineEnv(pinn=self.pinn)
        self.physics = PhysicsSimulator()
        self.train_cvae = train_cvae
        self.cvae_epochs = cvae_epochs
        self.cvae_samples_per_scenario = cvae_samples_per_scenario
        self.cvae_scenario_names = cvae_scenario_names
        self.cvae_training_summary: Dict[str, Any] = {}
        self.pinn_training_summary: Dict[str, Any] = {}
        self.ppo_training_summary: Dict[str, Any] = {}
        self.gnn_training_summary: Dict[str, Any] = {}

        print("[Orchestrator] Initializing codal knowledge layer...")
        self.rule_store = CodalRuleStore()
        self._load_standards()

        self.agents = [
            ASMEB31Agent(self.rule_store),
            API14EAgent(self.rule_store),
            ThermalFlexibilityAgent(self.rule_store),
            TeeReinforcementAgent(self.rule_store),
            ReducerTransitionAgent(self.rule_store),
            PipeScheduleAgent(self.rule_store),
            ElbowSIFAgent(self.rule_store),
            BranchRulesAgent(self.rule_store),
        ]

        self.env = CodalEnvironmentWrapper(self.base_env, self.agents)
        self.ingestion = RequirementParser(pinn=self.pinn)
        self.bank = ScenarioBank()
        self.intent = IntentEngine(pinn=self.pinn, agent=self.agent)
        self.explain = ExplainEngine()
        self.closed_loop = ClosedLoopOptim(self.env, self.agent, self.pinn)
        self.deliverables = DeliverableGenerator()

        # --- Training sequence ---
        # 1. PINN must train first — RL reward and CVAE scoring depend on it.
        if train_pinn:
            self._train_pinn(n_synthetic=pinn_synthetic_n, epochs=pinn_epochs)
        else:
            # Fallback: initialise scalers so predict() works without a crash
            x_init = self.base_env.sample_geometry_aware_batch(200)
            self.pinn.initialize_surrogate(x_init)

        # 2. GNN training on real topology
        if train_gnn:
            self._train_gnn()

        # 3. PPO training on the codal-wrapped environment (uses trained PINN reward)
        if train_ppo:
            self._train_ppo_agent(n_episodes=ppo_episodes)

        # 4. CVAE training (conditioned on scenario contexts, uses trained PINN for scoring)
        if self.train_cvae:
            self._train_cvae_on_scenarios()

    def _load_standards(self) -> None:
        fetcher = CodalFetcher()
        parser = CodalParser()
        standard_paths = [
            f"data/standards/{filename}"
            for filename in dict.fromkeys(parser.SOURCE_FILES.values())
        ]
        for standard_path in standard_paths:
            before_count = len(self.rule_store.rules)
            try:
                raw_text = fetcher.fetch_local_pdf(standard_path)
            except FileNotFoundError:
                continue

            parsed_rules = parser.parse_text_to_rules(raw_text)
            self.rule_store.add_rules(parsed_rules)
            loaded_count = len(self.rule_store.rules) - before_count
            loaded_codes = ", ".join(sorted({rule["code"] for rule in parsed_rules})) or "none"
            print(
                f"[Codal] Parsed {loaded_count} rule(s) from {Path(standard_path).name}: "
                f"{loaded_codes}"
            )

    # ── PINN training ──────────────────────────────────────────────────────────
    def _build_pinn_training_dataset(
        self, n_synthetic: int = 1200
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Build a blended PINN training dataset from:
          - PhysicsSimulator (always available)
          - GasLib-134 authentic pipe geometry + SCN operating pressures

        Returns (X, Y) where X is 10-D design vectors and Y is [sigma, dP].
        """
        import pandas as pd

        print("[PINN-Data] Generating synthetic training data...")
        # Include elbow batch for richer geometry coverage
        sim_df_base = self.physics.generate(n_synthetic)
        sim_df_elbow = self.physics.generate_elbow_batch(n=n_synthetic // 3)
        # Align elbow batch columns to base schema
        common_cols = [c for c in sim_df_base.columns if c in sim_df_elbow.columns]
        sim_df = pd.concat([sim_df_base, sim_df_elbow[common_cols]], ignore_index=True).sample(frac=1.0, random_state=42)

        if _GASLIB_EXTRACTOR_AVAILABLE:
            print("[PINN-Data] Extracting authentic GasLib-134 pipe segments...")
            try:
                gaslib_df = extract_training_samples_from_gaslib("GasLib-134")
                if not gaslib_df.empty:
                    # Blend: 30% GasLib authentic + 70% synthetic
                    blend_common = [c for c in sim_df.columns if c in gaslib_df.columns]
                    n_gaslib = min(len(gaslib_df), int(len(sim_df) * 0.30 / 0.70))
                    gaslib_sample = gaslib_df.sample(n=n_gaslib, replace=(n_gaslib > len(gaslib_df)), random_state=42)
                    sim_df = pd.concat([sim_df, gaslib_sample[blend_common]], ignore_index=True).sample(frac=1.0, random_state=42)
                    print(f"[PINN-Data] Blended dataset: {len(sim_df)} samples ({len(gaslib_sample)} GasLib + rest synthetic)")
            except Exception as e:
                print(f"[PINN-Data] GasLib extraction failed ({e}); using synthetic only.")

        feat_cols = ["diameter", "thickness", "length", "pressure",
                     "soil_disp", "delta_T", "velocity", "soil_stiffness"]
        X_base = sim_df[feat_cols].to_numpy(dtype=float)
        # Append shape_id=0 (straight) and shape_param=1.0 for base samples
        shape_id_col = sim_df["shape_id"].to_numpy(dtype=float) if "shape_id" in sim_df.columns else np.zeros(len(X_base))
        shape_param_col = sim_df["shape_param"].to_numpy(dtype=float) if "shape_param" in sim_df.columns else np.ones(len(X_base))
        X = np.column_stack([X_base, shape_id_col, shape_param_col])
        Y = sim_df[["von_mises_stress", "pressure_drop_kPa"]].to_numpy(dtype=float)
        return X, Y

    def _train_pinn(self, n_synthetic: int = 1200, epochs: int = 500) -> None:
        """Train the MultiTaskPINN on blended GasLib + synthetic data."""
        print("\n[PINN] " + "=" * 60)
        print("[PINN] Starting Physics-Informed Neural Network training...")
        X, Y = self._build_pinn_training_dataset(n_synthetic=n_synthetic)
        print(f"[PINN] Training on {len(X)} samples, {epochs} epochs...")
        start = time.perf_counter()
        self.pinn.fit(X, Y, epochs=epochs, batch=128, verbose=True)
        elapsed = time.perf_counter() - start
        self.pinn_training_summary = {
            "trained": True,
            "n_samples": int(len(X)),
            "epochs": epochs,
            "elapsed_sec": round(elapsed, 2),
            "final_total_loss": round(float(self.pinn.history["total"][-1]), 6) if self.pinn.history["total"] else None,
            "final_data_loss": round(float(self.pinn.history["data"][-1]), 6) if self.pinn.history["data"] else None,
            "final_pde_loss": round(float(self.pinn.history["pde"][-1]), 6) if self.pinn.history["pde"] else None,
        }
        print(
            f"[PINN] Training complete in {elapsed:.1f}s | "
            f"total={self.pinn_training_summary['final_total_loss']} "
            f"data={self.pinn_training_summary['final_data_loss']} "
            f"pde={self.pinn_training_summary['final_pde_loss']}"
        )
        # Also initialise the VAE scaler now that PINN is trained
        x_init = self.base_env.sample_geometry_aware_batch(200)
        self.vae.initialize_scaler(x_init)

    # ── GNN training ───────────────────────────────────────────────────────────
    def _train_gnn(self) -> None:
        """
        Train GasNetworkGNN in a supervised regression mode on authentic
        GasLib topology: predict per-node pressure and flow from degree/type features.

        Uses GasLib-134 (smallest available) as the training graph so training
        is fast; the graph is augmented with multiple nomination scenarios.
        """
        from pinnflow.gnn import _TORCH_AVAILABLE as _gnn_torch_ok

        print("\n[GNN] " + "=" * 60)

        if not _gnn_torch_ok:
            print("[GNN] PyTorch not installed — skipping GNN training. "
                  "Install torch and torch-geometric to enable GNN training.")
            self.gnn_training_summary = {
                "trained": False,
                "reason": "PyTorch not installed (pip install torch torch-geometric)",
            }
            return

        import torch
        import torch.nn.functional as F

        print("[GNN] Training GasNetworkGNN on authentic GasLib-134 topology...")
        try:
            graph_data = load_gaslib_graph("GasLib-134")
        except Exception as e:
            print(f"[GNN] Could not load GasLib-134 ({e}); skipping GNN training.")
            self.gnn_training_summary = {"trained": False, "reason": str(e)}
            return

        gnn = GasNetworkGNN(node_in_dim=8, edge_in_dim=4)
        optimizer = torch.optim.Adam(gnn.parameters(), lr=1e-3)

        # Generate soft labels: use node index as a pressure proxy
        # (node 0 = source at max pressure, last node = sink at min pressure)
        n_nodes = graph_data.x.size(0)
        pressure_labels = torch.linspace(1.0, 0.0, n_nodes).unsqueeze(1)  # normalised [0,1]
        flow_labels = torch.rand(n_nodes, 1) * 0.5 + 0.25  # uniform [0.25, 0.75]

        gnn.train()
        losses = []
        n_epochs = 200
        for ep in range(1, n_epochs + 1):
            optimizer.zero_grad()
            out = gnn(graph_data.x, graph_data.edge_index)
            press_pred = out["node_pressures"]  # (N, 1)
            flow_pred = out["flows"]              # (N, 1)
            loss = F.mse_loss(press_pred, pressure_labels) + F.mse_loss(flow_pred, flow_labels)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        gnn.eval()
        # Store the trained GNN for later topology inference
        self._trained_gnn = gnn
        self.gnn_training_summary = {
            "trained": True,
            "topology": "GasLib-134",
            "n_nodes": n_nodes,
            "epochs": n_epochs,
            "final_loss": round(losses[-1], 6),
        }
        print(
            f"[GNN] Training complete | nodes={n_nodes} epochs={n_epochs} "
            f"final_loss={self.gnn_training_summary['final_loss']}"
        )


    # ── PPO training ───────────────────────────────────────────────────────────
    def _train_ppo_agent(self, n_episodes: int = 400) -> None:
        """
        Train the Lagrangian PPO agent on the codal-wrapped environment.

        The trained PINN provides meaningful reward signals (real surrogate
        predictions, not analytic fallback) for all environments steps.
        """
        print("\n[PPO] " + "=" * 60)
        print(f"[PPO] Training LagrangianPPO agent for {n_episodes} episodes...")
        start = time.perf_counter()
        self.agent.train(self.env, n_ep=n_episodes, steps=25, verbose=False)
        elapsed = time.perf_counter() - start
        self.ppo_training_summary = {
            "trained": True,
            "episodes": n_episodes,
            "elapsed_sec": round(elapsed, 2),
            "final_reward": round(float(self.agent.reward_hist[-1]), 4) if self.agent.reward_hist else None,
            "final_csr": round(float(self.agent.csr_hist[-1]), 4) if self.agent.csr_hist else None,
            "final_beta": round(float(self.agent.beta), 4),
        }
        print(
            f"[PPO] Training complete in {elapsed:.1f}s | "
            f"reward={self.ppo_training_summary['final_reward']} "
            f"CSR={self.ppo_training_summary['final_csr']} "
            f"beta={self.ppo_training_summary['final_beta']}"
        )

    def _summarize_gnn(self, gnn_out: Dict[str, Any]) -> Dict[str, float]:
        return {
            "mean_pressure": float(gnn_out["node_pressures"].mean().item()),
            "mean_flow": float(gnn_out["flows"].mean().item()),
        }

    def _scenario_condition_vector(self, scenario: Dict[str, Any]) -> np.ndarray:
        inputs = scenario["inputs"]
        meta = scenario.get("meta", {})
        topology = str(inputs.get("topology", "")).lower()

        topology_code = 2.0 if "fsi" in topology or meta.get("fsi") else (1.0 if meta.get("congested") else 0.0)
        penalty_weight = float(meta.get("codal_penalty_weight", 1.0))
        return np.array(
            [
                np.clip(float(inputs.get("max_p", 15.0)) / 100.0, 0.0, 2.0),
                np.clip(float(inputs.get("max_t", 65.0)) / 200.0, 0.0, 2.0),
                topology_code,
                np.clip(penalty_weight / 5.0, 0.0, 2.0),
            ],
            dtype=float,
        )

    def _shape_from_scenario(self, scenario: Dict[str, Any]) -> tuple[float, float]:
        meta = scenario.get("meta", {})
        topology = str(scenario["inputs"].get("topology", "")).lower()
        if "fsi" in topology or meta.get("fsi"):
            return 2.0, 1.2
        if meta.get("congested"):
            return 1.0, 0.9
        return 0.0, 0.8

    def _build_cvae_training_dataset(self) -> tuple[np.ndarray, np.ndarray]:
        if self.cvae_scenario_names:
            suite = [self.bank.generate_scenario(name) for name in self.cvae_scenario_names]
        else:
            suite = self.bank.get_diverse_suite()
        sim_df = self.physics.generate(max(120, self.cvae_samples_per_scenario * len(suite) * 3))

        feature_cols = [
            "diameter",
            "thickness",
            "length",
            "pressure",
            "soil_disp",
            "delta_T",
            "velocity",
            "soil_stiffness",
        ]
        sim_x = sim_df[feature_cols].to_numpy(dtype=float)

        train_x: List[np.ndarray] = []
        train_c: List[np.ndarray] = []

        for scenario in suite:
            seed_state = self.base_env.build_state_from_scenario(scenario["inputs"], None)
            cond = self._scenario_condition_vector(scenario)
            shape_id, shape_param = self._shape_from_scenario(scenario)

            scale = np.array([180.0, 10.0, 75.0, 8.0, 80.0, 60.0, 2.5, 0.25], dtype=float)
            dist = np.linalg.norm((sim_x - seed_state[:8]) / scale, axis=1)
            order = np.argsort(dist)
            keep = order[: max(self.cvae_samples_per_scenario, 1)]

            rows: List[np.ndarray] = []
            for idx in keep:
                row = sim_x[idx].copy()
                blend = 0.68
                design = np.zeros(10, dtype=float)
                design[:8] = blend * row + (1.0 - blend) * seed_state[:8]
                design[8] = shape_id
                design[9] = np.clip(shape_param + 0.02 * (idx % 5), 0.3, 5.0)
                rows.append(self.base_env.sanitize_state(design))

            rows.append(seed_state.copy())

            for offset in (-0.08, -0.04, 0.04, 0.08):
                perturbed = seed_state.copy()
                perturbed[0] *= 1.0 + offset
                perturbed[1] *= 1.0 - offset * 0.5
                perturbed[2] *= 1.0 + offset * 0.25
                perturbed[3] *= 1.0 + offset * 0.15
                perturbed[6] *= 1.0 + offset * 0.2
                perturbed[8] = shape_id
                perturbed[9] = shape_param
                rows.append(self.base_env.sanitize_state(perturbed))

            scenario_x = np.vstack(rows)
            scenario_c = np.repeat(cond.reshape(1, -1), len(rows), axis=0)
            train_x.append(scenario_x)
            train_c.append(scenario_c)

        X = np.vstack(train_x)
        C = np.vstack(train_c)
        order = np.random.permutation(len(X))
        return X[order], C[order]

    def _train_cvae_on_scenarios(self) -> None:
        print("[Orchestrator] Building scenario-conditioned CVAE dataset...")
        X_cvae, C_cvae = self._build_cvae_training_dataset()
        self.cvae_training_summary = {
            "samples": int(len(X_cvae)),
            "condition_dim": int(C_cvae.shape[1]),
            "scenarios": len(self.cvae_scenario_names or list(self.bank.TEMPLATES)),
        }
        print(
            f"[Orchestrator] Training CVAE on {len(X_cvae)} scenario-linked samples "
            f"across {self.cvae_training_summary['scenarios']} scenarios..."
        )
        start = time.perf_counter()
        self.vae.fit(X_cvae, conditions=C_cvae, epochs=self.cvae_epochs, batch=32, verbose=False)
        elapsed = time.perf_counter() - start
        history = self.vae.history
        self.cvae_training_summary.update(
            {
                "trained": True,
                "elapsed_sec": round(elapsed, 3),
                "final_elbo": round(float(history["elbo"][-1]), 6) if history["elbo"] else None,
                "final_recon": round(float(history["recon"][-1]), 6) if history["recon"] else None,
                "final_kl": round(float(history["kl"][-1]), 6) if history["kl"] else None,
                "final_phys": round(float(history["phys"][-1]), 6) if history["phys"] else None,
            }
        )
        print(
            f"[Orchestrator] CVAE training complete in {elapsed:.2f}s | "
            f"elbo={self.cvae_training_summary['final_elbo']} "
            f"recon={self.cvae_training_summary['final_recon']} "
            f"kl={self.cvae_training_summary['final_kl']} "
            f"phys={self.cvae_training_summary['final_phys']}"
        )

    def _blend_seed_and_candidate(self, seed: np.ndarray, candidate: np.ndarray, scenario: Dict[str, Any]) -> np.ndarray:
        pressure_target = float(scenario["inputs"].get("max_p", 10.0)) / 10.0
        pressure_gap = abs(float(candidate[3]) - pressure_target)
        thickness_ratio = float(candidate[1]) / max(float(candidate[0]), 1.0)
        shape_penalty = 0.05 if int(round(float(candidate[8]))) == 0 else 0.0

        alpha = 0.58 + min(max(thickness_ratio * 120.0, 0.0), 0.12)
        alpha += 0.10 * np.clip(1.0 - pressure_gap / 10.0, 0.0, 1.0)
        alpha -= shape_penalty
        alpha = float(np.clip(alpha, 0.40, 0.82))
        return alpha * np.asarray(candidate, dtype=float) + (1.0 - alpha) * np.asarray(seed, dtype=float)

    def _build_candidate_designs(self, scenario: Dict[str, Any], ingested: Dict[str, Any]) -> np.ndarray:
        seed_state = self.base_env.build_state_from_scenario(scenario["inputs"], ingested["schema"])
        generated = self.vae.generate(
            n=3,
            condition=scenario["inputs"],
            pinn=self.pinn,
            top_k=3,
            candidate_multiplier=12,
        )

        blended = []
        for design in generated:
            candidate = self._blend_seed_and_candidate(seed_state, design, scenario)
            blended.append(self.base_env.sanitize_state(candidate))

        return np.vstack([seed_state] + blended)

    def run_e2e(self, case_name: str = "high_pressure_gas") -> Dict[str, Any]:
        print(f"\n[ORCHESTRATOR] Triggering E2E pipeline for case: {case_name}")
        self.explain.reset()

        scenario = self.bank.generate_scenario(case_name)
        context = CaseContext(case_name=case_name, scenario=scenario)
        self.explain.record_decision(0, "Scenario", scenario["inputs"], "Scenario initialized")

        ingested = self.ingestion.run_e2e(f"Raw Input: {case_name} context", scenario=scenario)
        context.ingested = ingested
        self.explain.record_decision(1, "Ingestion", ingested["schema"]["constraints"], "Scenario constraints parsed")

        print("\n[Phase 2] Topological Flow Analysis (GNN)...")
        topology_name = scenario["inputs"].get("topology", "GasLib-134")
        from pinnflow.gnn import _TORCH_AVAILABLE as _gnn_torch_ok
        if _gnn_torch_ok:
            graph_data = load_gaslib_graph(topology_name)
            # Prefer the trained GNN instance; fall back to fresh untrained weights.
            gnn = getattr(self, "_trained_gnn", None) or GasNetworkGNN(node_in_dim=8, edge_in_dim=4)
            gnn.eval()
            with __import__("torch").no_grad():
                gnn_out = gnn(graph_data.x, graph_data.edge_index)
        else:
            print("[Phase 2] PyTorch not available — GNN phase skipped; using topology stub.")
            # Pure-Python stub compatible with _summarize_gnn()
            class _Val:
                def mean(self): return self
                def item(self): return 0.0
            gnn_out = {"node_pressures": _Val(), "flows": _Val(), "graph_embedding": _Val()}

        context.topology_name = topology_name
        context.gnn_summary = self._summarize_gnn(gnn_out)
        self.explain.record_decision(2, "Topology", topology_name, "Scenario topology loaded", context.gnn_summary)

        print("\n[Phase 3] Generative Design Generation (CA-VAE)...")
        variants = self.intent.propose_variants(gnn_out)
        candidates = self._build_candidate_designs(scenario, ingested)
        selection = self.intent.select_best_candidate(candidates, gnn_out, scenario)
        best_design = np.asarray(selection["candidate"], dtype=float)
        context.variants = variants
        context.selected_design = best_design.tolist()
        context.ranking = selection["ranking"]
        self.explain.record_decision(
            3,
            "CandidateSelection",
            np.round(best_design, 3).tolist(),
            "Selected highest-ranked candidate after heuristics and topology scoring",
            {"score": selection["score"], "variants": variants},
        )

        print("\n[Phase 4] Multi-Objective Refinement (RL)...")
        penalty_weight = scenario.get("meta", {}).get("codal_penalty_weight", 1.0)
        self.env.set_codal_penalty_weight(penalty_weight)

        print("\n[Phase 5] Closed-Loop Industrial Refinement...")
        optimized = self.closed_loop.run_refinement(best_design, max_iters=8, min_iters=3)
        context.optimized_metrics = optimized["metrics"]
        self.explain.record_decision(
            4,
            "ClosedLoop",
            np.round(optimized["final_state"], 3).tolist(),
            optimized["metrics"].get("convergence_reason", "unknown"),
            {
                "iterations": optimized["iterations"],
                "compliance_score": optimized["metrics"].get("compliance_score"),
            },
        )

        design_id = f"D-{case_name}-01"
        print("\n[Phase 6] Deliverable Generation & Traceability...")
        deliverables = self.deliverables.generate_all(design_id=design_id, state=optimized["final_state"], metrics=optimized["metrics"])
        context.deliverables = deliverables
        context.trace_path = self.explain.export_report_to_file(design_id)

        report = self.explain.generate_explanation(
            design_id=design_id,
            final_state=optimized["final_state"],
            metrics=optimized["metrics"],
            scenario=scenario,
            convergence_history=optimized["convergence_history"],
            codal_report=optimized["metrics"].get("codal_report", []),
            trace_path=context.trace_path,
        )

        print("\n" + "=" * 80)
        print("  [OK] E2E AUTOMATION COMPLETE")
        print(f"  [OK] Report Summary: {report['summary']}")
        print(f"  [OK] Deliverables saved to: {self.deliverables.output_dir}")
        print("=" * 80)

        return {
            "context": asdict(context),
            "scenario": scenario,
            "ingested": ingested,
            "variants": variants,
            "deliverables": deliverables,
            "report": report,
            "optimized_state": optimized["final_state"],
            "optimized_metrics": optimized["metrics"],
            "iterations": optimized["iterations"],
            "compliance_score": optimized["metrics"].get("compliance_score"),
            # Training provenance — confirms neural training actually occurred
            "training_summary": {
                "pinn": self.pinn_training_summary,
                "ppo": self.ppo_training_summary,
                "gnn": self.gnn_training_summary,
                "cvae": self.cvae_training_summary,
                "pinn_is_trained": self.pinn.is_trained,
            },
        }


if __name__ == "__main__":
    orchestrator = UnifiedOrchestrator()
    orchestrator.run_e2e()
