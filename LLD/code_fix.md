# Code Fix and Maintenance Notes

This document tracks the most important implementation concerns in the current codebase. It is written as a maintenance companion for the repository rather than a bug-only report.

## 1. Current State

The codebase now has a much clearer end-to-end structure than the older prototype:

- scenario generation is explicit
- ingestion uses scenario context
- topology loading is scenario-aware
- candidate generation is conditionally driven
- deliverables are written per design ID
- trace files are exported to disk

That said, the project is still a research stack, so the implementation should be reviewed with a data-validation mindset rather than assuming production-grade completeness.

## 2. What Is Working Well

### 2.1 Scenario Conditioning

Scenario metadata is threaded into:

- requirement parsing
- state construction
- candidate generation
- compliance weighting

This is a strong design choice because it keeps the workflow case-specific.

### 2.2 State Sanitization

The environment now explicitly:

- clips state values to bounds
- rounds the geometry class to a valid discrete value
- keeps the geometry parameter within its valid range

This prevents many accidental invalid states from propagating downstream.

### 2.3 Artifact Separation

Deliverables are written under a per-design directory:

- `results/deliverables/<design_id>/`

That is much better for traceability than a single flat output folder.

### 2.4 Trace Export

The explainability layer writes:

- `results/trace/TRACE_<design_id>.json`

This gives each case an audit trail that can be compared across runs.

## 3. Remaining Areas To Watch

### 3.1 Surrogate Confidence

The pipeline still depends heavily on the PINN for fast scoring. Any future changes to the surrogate should be tested for:

- output variance
- sensitivity to scenario inputs
- numerical stability
- compatibility with the reward function

### 3.2 Discrete Geometry Handling

The geometry index is now rounded in the environment and intent layers, but it should still be treated carefully in any future model changes.

Recommended guardrails:

- keep geometry class discrete in the environment
- avoid learning it as a free continuous scalar unless the representation is intentionally redesigned

### 3.3 Standards Coverage

The codal layer is structured, but standards loading still depends on the availability and completeness of source documents.

Recommended practice:

- load each standard from its own document source
- emit a warning when a critique agent is using fallback behavior
- keep standards parsing unit-tested

### 3.4 Reproducibility

Because the system generates synthetic and sampled data, runs may differ if seeds are not controlled.

For reproducible experiments:

- fix random seeds where possible
- log scenario names and parameters
- save output folders for every benchmark batch

## 4. Testing Priorities

The most valuable tests for this repo are:

1. scenario generation tests
2. ingestion schema tests
3. state sanitization tests
4. candidate selection tests
5. closed-loop convergence tests
6. deliverable generation tests
7. trace export tests

Those tests cover the main data handoffs and are more useful than isolated unit tests that only touch one utility function.

## 5. Suggested Maintenance Tasks

### 5.1 Improve External Validation

If a higher-fidelity simulation callback becomes available, wire it into the closed-loop optimizer as an external convergence check.

### 5.2 Strengthen Experiment Logging

Add richer per-run metadata:

- model version
- scenario parameters
- runtime
- convergence reason
- file paths

### 5.3 Add Scenario-Level Output Isolation

For larger benchmark sweeps, keep an optional scenario batch folder so multiple runs do not mix artifacts.

### 5.4 Extend the Report

The explanation output could include:

- a compact table of metrics
- a list of codal recommendations
- the selected candidate score ranking
- scenario notes and warnings

## 6. Practical Review Checklist

Before merging changes, confirm the following:

- the selected candidate is actually the one refined in the closed loop
- the final compliance score comes from the wrapper metrics
- the outputs are written under the right design ID
- traces are exported for every executed case
- tests still pass after any change to the state vector

## 7. Bottom Line

The strongest implementation principle in this repository is to keep every stage aligned around the same case context. If a future change breaks that alignment, it will hurt the quality of the design, the traceability of the result, and the usefulness of the benchmark data.

