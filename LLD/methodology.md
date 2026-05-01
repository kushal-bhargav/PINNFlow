# Methodology: PINNFlow

## 1. Research Framing

This project is structured as an industrial data-science pipeline. The core research question is:

> How can we transform scenario context, topology information, physics surrogates, and compliance rules into a reliable design recommendation workflow?

The answer in this repository is a staged hybrid model:

- synthetic scenario generation
- schema extraction
- graph summarization
- conditional generation
- physics-aware ranking
- closed-loop RL refinement
- compliance-aware scoring

## 2. Experimental Design

The system is designed to support controlled experiments across multiple scenario families.

### 2.1 Scenario Families

- high-pressure gas service
- refinery compliance cases
- deep-sea / FSI-inspired cases
- congested refinery topology
- extreme condition recovery tests

These cases are meant to stress different parts of the pipeline:

- pressure envelope
- thermal envelope
- network congestion
- geometry sensitivity
- compliance sensitivity

### 2.2 Data Sources

The repository uses a mix of:

- template-based synthetic scenario metadata
- GasLib-inspired topology inputs
- structured line-list style schemas
- physics-sampled collocation points for surrogate warm-starting
- codal rule text parsed from standards documents

For a data scientist, this is a hybrid of synthetic and semi-real data engineering rather than a single static dataset.

## 3. Feature Engineering

### 3.1 Scenario Features

The scenario bank produces:

- `max_p`
- `max_t`
- `topology`
- scenario tags such as `congested`, `fsi`, and `codal_penalty_weight`

These are converted into condition vectors for the generative model and weighting logic for the optimizer.

### 3.2 Design Features

The 10-D pipeline state captures:

- pipe geometry
- operational pressure
- temperature delta
- flow velocity
- soil or environment stiffness proxies
- categorical geometry state

This is the main feature space shared across models.

### 3.3 Topology Features

The GNN produces topology summaries such as:

- mean pressure
- max pressure
- mean flow
- max flow

These serve as compact descriptors of network context.

## 4. Modeling Strategy

### 4.1 Physics Surrogate

`MultiTaskPINN` is used as a fast approximation layer for stress and pressure-drop predictions. Its role is to:

- reduce repeated simulation cost
- provide a differentiable reward component
- support validation inside the ingestion layer and environment layer

### 4.2 Generative Model

`CAVAE` learns a latent representation of valid designs. It is trained with:

- reconstruction loss
- KL regularization
- physics penalty
- scenario conditioning

The model is used to generate multiple design candidates, not a single deterministic answer.

### 4.3 Policy Optimization

The RL stage uses a PPO-style policy to refine designs under a shaped reward. This makes the problem a multi-objective optimization task with:

- stress minimization
- pressure-drop control
- fatigue control
- cost proxy minimization
- compliance maximization

### 4.4 Compliance Layer

The compliance layer is not treated as a post-processing checklist only. It feeds back into optimization via penalties and recommendations, which makes the workflow more realistic for engineering decision support.

## 5. Training and Calibration

### 5.1 Warm Start

The orchestrator initializes the PINN and VAE using collocation and scenario-linked samples. This gives the models a shared scaling foundation before the main workflow begins.

### 5.2 CVAE Training

The CVAE training set is built from:

- scenario-conditioned seed states
- nearby synthetic simulation rows
- small perturbations around those seed states

This creates a mixed training set that is more diverse than a simple single-scenario sample.

### 5.3 Candidate Ranking

Candidate ranking uses a combination of:

- PINN-predicted stress
- pressure gap to scenario target
- topology summaries
- heuristic intent checks

This is a practical example of hybrid ranking: learned model + rules + physics proxy.

## 6. Evaluation Framework

### 6.1 Core Metrics

The pipeline can be evaluated using:

- `sigma` for stress
- `delta_P` for pressure-drop proxy
- `fatigue`
- `cost`
- `constraint_ok`
- `violation`
- `compliance_score`
- convergence iteration count

### 6.2 Model Quality Metrics

For the generative and surrogate layers, useful evaluation measures include:

- reconstruction error
- ELBO
- KL divergence
- sample diversity
- candidate feasibility rate
- sensitivity to scenario conditioning

### 6.3 System-Level Metrics

For the full pipeline:

- number of feasible candidates produced
- refinement iterations until convergence
- trace completeness
- deliverable generation success rate
- scenario-to-scenario variance

## 7. Reproducibility

The project is easiest to reproduce when the following are controlled:

- random seeds
- scenario selection order
- CVAE training settings
- output directory structure
- standards document availability

Recommended practice:

- save `results/` snapshots for each experiment batch
- retain trace JSONs together with deliverables
- compare the same `design_id` across model versions

## 8. Limitations

Important methodological limitations:

- the workflow depends on surrogate predictions, not a full high-fidelity solver at every step
- synthetic scenarios may not represent all real plant conditions
- categorical geometry is still compacted into a small state field
- compliance rules are only as complete as the loaded standards text

These limitations should be stated explicitly in any paper, presentation, or benchmark report.

## 9. Recommended Experimental Reporting

When reporting results, include:

- scenario name
- topology source
- training settings
- final compliance score
- stress and pressure-drop values
- convergence reason
- trace file path
- deliverable file paths

That gives the result enough context to be interpreted scientifically.

