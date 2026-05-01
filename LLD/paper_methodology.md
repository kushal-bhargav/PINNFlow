# Paper Methodology and Citation Blueprint

This document is a research-oriented companion to the codebase. It is meant to help write a paper, technical report, or challenge submission that accurately describes what the system does.

## 1. Narrative Angle

The strongest framing for this project is:

> a hybrid industrial intelligence pipeline that combines physics surrogates, graph topology modeling, generative design, reinforcement learning, and standards-based compliance checks.

That framing is more precise than describing the work as "just a neural network" or "just an optimizer."

## 2. Suggested Paper Structure

### 2.1 Introduction

Explain the engineering problem:

- industrial designs must satisfy physics, topology, and regulatory constraints
- brute-force simulation is too expensive for iterative exploration
- isolated ML models are not enough because the workflow needs traceability and compliance

### 2.2 Related Work

Position the project across four literature tracks:

- physics-informed neural networks
- graph neural networks for networked systems
- conditional generative models for design space exploration
- reinforcement learning for constrained optimization

### 2.3 Method

Describe the system as a pipeline:

1. scenario ingestion
2. topology encoding
3. candidate generation
4. heuristic filtering
5. closed-loop refinement
6. compliance scoring
7. deliverable export

### 2.4 Experiments

Report results by scenario family and by ablation. Focus on:

- feasibility rate
- compliance score
- convergence behavior
- stress prediction quality
- pressure-drop response
- artifact generation success

### 2.5 Discussion

Discuss why hybrid systems matter:

- data alone does not encode code compliance
- rules alone do not capture operational tradeoffs
- physics alone does not produce candidate diversity
- optimization alone does not guarantee interpretability

## 3. Literature Mapping

Below is a practical mapping from literature themes to repository modules.

### 3.1 Physics-Informed Surrogates

Relevant code:

- `pinnflow/pinn.py`
- `pinnflow/environment.py`
- `pinnflow/ingestion/parser.py`

Use this layer to discuss:

- PDE-constrained learning
- surrogate modeling
- uncertainty-aware evaluation

### 3.2 Design Optimization

Relevant code:

- `pinnflow/closed_loop/optimizer.py`
- `pinnflow/agent.py`
- `pinnflow/environment.py`

Use this layer to discuss:

- policy optimization
- constrained search
- reward shaping
- iterative refinement

### 3.3 Generative Design

Relevant code:

- `pinnflow/vae.py`

Use this layer to discuss:

- latent-space exploration
- scenario conditioning
- feasibility-preserving generation
- physics-regularized decoding

### 3.4 Topology and Networks

Relevant code:

- `pinnflow/gnn.py`

Use this layer to discuss:

- graph embeddings
- pressure and flow summarization
- topology-conditioned ranking

### 3.5 Compliance and Standards

Relevant code:

- `pinnflow/codal_engine/*`

Use this layer to discuss:

- standards retrieval
- rule-based critique
- explainable compliance penalties

## 4. Recommended Claims That Are Safe to Make

These claims are well aligned with the current code:

- the pipeline is scenario-conditioned
- candidate generation is constrained by scenario metadata
- the system produces traceable deliverables
- the workflow combines learned models with rule-based compliance checks
- the final outputs are design artifacts, not only numeric predictions

## 5. Claims To Phrase Carefully

These statements should be qualified in a paper:

- "physics-accurate" should be replaced with "physics-informed" or "physics-surrogate"
- "fully compliant" should be replaced with "compliance-scored" unless every code source is exhaustively verified
- "real-time" should be used only if benchmarked with measured latency
- "optimal" should be used only relative to the selected objective and constraint set

## 6. Citation Blueprint

The repository already hints at three important reference families:

- PINN-based surrogate modeling for pipelines
- ANN-based piping optimization under ASME constraints
- PPO for constrained control and policy optimization

You can extend the paper bibliography with:

- survey papers on physics-informed machine learning
- graph neural network references for infrastructure systems
- conditional VAE references
- reinforcement learning references for safety-constrained optimization
- standards references for ASME and API-based workflows

## 7. Experiment Tables To Include

A strong paper should include at least these tables:

### 7.1 Scenario Summary

Columns:

- scenario name
- pressure range
- temperature range
- topology
- special tags

### 7.2 Model Performance

Columns:

- PINN loss
- VAE ELBO
- candidate feasibility rate
- mean compliance score
- convergence iterations

### 7.3 Ablation Study

Columns:

- topology on/off
- codal critique on/off
- RL refinement on/off
- heuristic filtering on/off
- deliverable generation on/off

### 7.4 Runtime Study

Columns:

- scenario
- generation time
- refinement time
- total runtime
- artifact export time

## 8. Figures To Include

Recommended figures for a paper:

- overall workflow diagram
- state vector schema
- topology summary plot
- candidate ranking plot
- convergence trace
- deliverable example figure
- compliance score comparison by scenario

## 9. Abstract Template

You can describe the project in one paragraph as:

PINNFlow is a hybrid industrial design intelligence framework that integrates scenario-conditioned ingestion, graph-based topology encoding, conditional generative design, reinforcement-learning refinement, and standards-aware compliance scoring. The system produces traceable engineering deliverables while reducing dependence on repeated expensive simulation.

## 10. Final Writing Guidance

When writing the paper, avoid overclaiming. The strongest, most defensible story is that the project demonstrates a complete, testable pipeline for engineering decision support with reproducible outputs and explicit compliance feedback.

