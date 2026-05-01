# Low-Level Design: PINNFlow

## 1. Purpose

This document describes the internal modules, classes, inputs, outputs, and control flow used by PINNFlow. It is written for engineers and data scientists who need to extend, debug, or benchmark the system.

## 2. Package Map

### 2.1 Entry Point

- `main.py`
  - boots the industrial suite
  - reads environment flags for CVAE training
  - iterates over the built-in scenarios
  - prints high-level execution outcomes

### 2.2 Orchestration

- `pinnflow/orchestrator_v2.py`
  - instantiates the full stack
  - wires the surrogate models, environment, critique agents, and deliverables
  - drives the six-phase pipeline

### 2.3 Scenario and Ingestion

- `pinnflow/scenarios/bank.py`
  - creates reproducible scenario templates
  - injects pressure, temperature, topology, and optional tags
- `pinnflow/ingestion/parser.py`
  - converts raw text context into a structured schema
  - performs PINN-based physics validation on extracted lines

### 2.4 Modeling

- `pinnflow/pinn.py`
  - physics surrogate for stress and pressure-drop estimation
- `pinnflow/vae.py`
  - constraint-aware conditional VAE for design generation
- `pinnflow/gnn.py`
  - graph model for topology and flow summaries
- `pinnflow/agent.py`
  - PPO and related policy logic for closed-loop control

### 2.5 Environment and Optimization

- `pinnflow/environment.py`
  - canonical 10-D state definition
  - reward shaping and state sanitization
- `pinnflow/codal_engine/wrapper.py`
  - wraps the environment with compliance penalties
- `pinnflow/closed_loop/optimizer.py`
  - iterative state refinement loop

### 2.6 Compliance and Deliverables

- `pinnflow/codal_engine/knowledge/fetcher.py`
- `pinnflow/codal_engine/knowledge/parser.py`
- `pinnflow/codal_engine/knowledge/rule_store.py`
- `pinnflow/codal_engine/agents/asme_b31_agent.py`
- `pinnflow/codal_engine/agents/api14e_agent.py`
- `pinnflow/deliverables/generator.py`
- `pinnflow/explainability/trace.py`

## 3. Key Classes and Responsibilities

### 3.1 `UnifiedOrchestrator`

Location: `pinnflow/orchestrator_v2.py`

Responsibilities:

- initialize all core components
- load standards into the codal rule store
- train or warm-start the CVAE
- run the end-to-end workflow for a scenario
- persist trace and deliverable outputs

Important internal data object:

- `CaseContext`

Fields:

- `case_name`
- `scenario`
- `ingested`
- `topology_name`
- `gnn_summary`
- `variants`
- `selected_design`
- `ranking`
- `optimized_metrics`
- `deliverables`
- `trace_path`

### 3.2 `PipelineEnv`

Location: `pinnflow/environment.py`

Responsibilities:

- maintain the current 10-D state
- bound and sanitize all actions
- compute physics-based reward components
- expose a reward signal usable by RL and optimization code

Core state fields:

- `Diameter`
- `Thickness`
- `Length`
- `Pressure`
- `SoilDisp`
- `DeltaT`
- `Velocity`
- `SoilK`
- `ShapeID`
- `ShapeParam`

Important methods:

- `sanitize_state`
- `set_state`
- `build_state_from_scenario`
- `reset`
- `step`
- `_reward`

### 3.3 `RequirementParser`

Location: `pinnflow/ingestion/parser.py`

Responsibilities:

- create structured schemas from text or image-like inputs
- attach scenario context when available
- infer line-level properties and equipment
- run a simple physics validation loop through the PINN

Main outputs:

- `schema`
- `validation`

### 3.4 `IntentEngine`

Location: `pinnflow/design_intent/intent.py`

Responsibilities:

- summarize topology context
- propose heuristic design variants
- apply engineering heuristics to candidate states
- reject invalid designs before optimization
- score candidates with physics and topology context

### 3.5 `CAVAE`

Location: `pinnflow/vae.py`

Responsibilities:

- learn a generative representation of valid design states
- condition generation on scenario features
- score candidate layouts using physics proxies
- keep discrete and bounded fields within valid ranges

Model characteristics:

- supports 8-D and 10-D input variants
- uses standard scaling for training
- maintains ELBO, reconstruction, KL, and physics penalty histories

### 3.6 `CodalEnvironmentWrapper`

Location: `pinnflow/codal_engine/wrapper.py`

Responsibilities:

- run the base environment step
- evaluate codal critique agents
- modify the reward by applying penalties
- attach compliance artifacts to the step info dictionary

Step output additions:

- `raw_codal_penalty`
- `codal_penalty`
- `codal_penalty_weight`
- `codal_report`
- `codal_recommendations`
- `codal_rule_values`
- `compliance_score`

### 3.7 `ClosedLoopOptim`

Location: `pinnflow/closed_loop/optimizer.py`

Responsibilities:

- seed the environment with an initial candidate
- iterate until stabilization or iteration limit
- apply codal guidance between iterations
- collect convergence history and final metrics

Key convergence inputs:

- normalized state delta
- reward delta
- external validation error if available
- compliance score

### 3.8 `DeliverableGenerator`

Location: `pinnflow/deliverables/generator.py`

Responsibilities:

- write per-design BOM CSVs
- generate deterministic isometric JSON
- create compliance matrices from final metrics
- group outputs by design ID

### 3.9 `ExplainEngine`

Location: `pinnflow/explainability/trace.py`

Responsibilities:

- capture step-by-step decisions
- export a JSON trace file
- produce a readable explanation summary

## 4. Data Shapes

### 4.1 Candidate State

The canonical design vector is a NumPy array of shape `(10,)`.

Expected semantics:

| Index | Meaning |
| --- | --- |
| 0 | diameter |
| 1 | thickness |
| 2 | length |
| 3 | pressure |
| 4 | soil displacement |
| 5 | delta T |
| 6 | velocity |
| 7 | soil stiffness |
| 8 | geometry class |
| 9 | geometry parameter |

### 4.2 Ingestion Schema

The ingestion layer returns a nested dictionary with:

- `raw_input`
- `scenario_name`
- `topology`
- `lines`
- `equipment`
- `constraints`
- `meta`

### 4.3 Optimization Metrics

The environment and closed-loop code pass around metric dictionaries containing:

- `sigma`
- `delta_P`
- `fatigue`
- `cost`
- `constraint_ok`
- `violation`
- `shape_id`
- `shape_param`
- `codal_report`
- `codal_recommendations`
- `compliance_score`

## 5. Current Runtime Flow

1. The orchestrator loads standards and initializes models.
2. The scenario bank creates the case context.
3. The ingestion parser builds a structured schema.
4. The graph model summarizes topology and flow.
5. The VAE and intent engine create candidate designs.
6. The selected candidate is refined in the closed loop.
7. The codal wrapper modifies reward and adds compliance metadata.
8. Deliverables and trace files are exported.

## 6. Extension Points

Areas intended for future work:

- richer scenario templates
- stronger external validation in the closed-loop optimizer
- more realistic standards ingestion for multiple code families
- better categorical handling for geometry type
- scenario-scoped result directories for large experiment runs

