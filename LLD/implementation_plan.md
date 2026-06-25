# Address Reviewer Feedback for PINNFlow

The reviewer's feedback highlights several inconsistencies and presentation issues in `PINNFlow_v8.1_paper.tex`. Additionally, while checking the `results` folder, we confirmed that the updated geometry benchmark and ablation CSVs were not generated or present, meaning the anomalies in the TeX file could not be automatically resolved by inserting new data. 

Here is the detailed implementation plan to address all reviewer concerns and run the missing benchmark scripts.

## User Review Required

- **Geometry Class Reduction**: The text currently claims a 4-expert model (Straight, Elbow, T-Junction, Reducer). However, the benchmark code and tables only evaluate 3 classes. I propose updating the paper text to claim a **3-expert model** to match the empirical data exactly. Please let me know if you would prefer to implement a Reducer class benchmark instead.
- **ANSYS Speedup**: The reviewer requested clarifying that the ANSYS speedup is an estimate. The text already mentions this, but I will reinforce this in the Abstract, Conclusion, and Table captions to ensure complete transparency.

## Open Questions

- Do you have a preferred dataset or benchmark script that should be executed to generate the *missing* `geometry_perf` metrics, or should I just correct the arithmetic in the current Table III based on the existing values? (The plan assumes correcting the arithmetic of existing values).

## Proposed Changes

### `PINNFlow_v8.1_paper.tex`
- **[MODIFY] `PINNFlow_v8.1_paper.tex`**
  - **Anomaly 1 & 2 (Arithmetic & Inconsistency)**: Recalculate the Average Stress MAE in Table III (currently labelled `tab:geometry_perf`) from 196.67 to 102.30. Update the discussion text at line 428 from "410.28 MPa" to "127.17 MPa" to match the table.
  - **Anomaly 3 (Duplicate Tables)**: Remove the duplicate Table II (`tab:perclass`) which erroneously copies Table I. Uncomment and format the actual Per-class Stress MAE table.
  - **Anomaly 4 (4-Expert Claim)**: Update the text in the Abstract, Introduction, and Methodology to claim a **3-expert model** (Straight, Elbow, T-Junction) to align with the reported results. Remove references to the "Reducer" class.
  - **Anomaly 5 (ANSYS Estimate)**: Update Table III's caption and the Abstract to explicitly use the word "estimated" when referring to ANSYS speedups.
  - **Anomaly 6 (Methodological Detail)**: Add explicit details to the Experiment section regarding the Train-Test split, seeds (e.g., 42, 1337, 2026), and sample counts based on `ablation_study.py`.
  - **Anomaly 7 (RL Robustness)**: Add a sentence to the Abstract and Conclusion explicitly stating that the RL refinement exhibits weak robustness under severe environmental noise, aligning with the Discussion.
  - **Anomaly 8 (Duplicate Headings)**: Rename the first `\section{Discussion}` (line 381) to `\section{Results Analysis}`.
  - **Anomaly 9 (Figure 2 Axis)**: Update the PGFPlots code for Figure 2 to use a dual Y-axis. This will plot Reliability (0.0 - 1.0) on the right axis and MAE on the left axis, fixing the readability issue.
  - **Anomaly 10 (Proofread)**: Fix minor run-on sentences and formatting issues identified during the update.

## Verification Plan

### Automated Tests
- N/A (Paper compilation)

### Manual Verification
- Compile the LaTeX file locally or instruct the user to compile it in Overleaf to verify that Figure 2 renders correctly with dual axes, Tables I, II, and III are distinct and mathematically correct, and all reviewer concerns are addressed.
