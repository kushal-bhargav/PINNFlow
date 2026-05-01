# PINNFlow

**PINNFlow** is an end-to-end AI-driven pipeline design and optimization framework that automatically translates high-level industrial requirements into optimal, physics-compliant physical geometries. It leverages a Conditional Auto-Variational Encoder (CA-VAE) and a Graph Neural Network (GNN) to rapidly generate intelligent initial design proposals based on complex network topologies. These designs are then optimized by a Reinforcement Learning agent against real-time stresses from a PINN surrogate and strict regulatory codes (ASME, API) to ensure complete industrial compliance.

## Project Structure

This repository is organized to separate the core logic from data and generated artifacts. **Everything related to the source code, execution, and deep technical documentation is located in the `LLD/` folder.**

*   `LLD/` — **Core Codebase & Documentation**
    *   Contains the main application code (`pinnflow/`), execution scripts (`main.py`), test suites, and deep-dive architectural documents (`HLD.md`, `LLD.md`, `methodology.md`).
    *   *Note: All execution commands should be run from the project root, pointing into this folder.*
*   `data/` — **Raw Data & Assets**
    *   Contains raw network topologies and datasets (e.g., `raw_gaslib` networks) used for ingestion and baseline geometry generation.
*   `results/` — **Artifacts & Deliverables**
    *   The destination for all generated outputs. This includes Bill of Materials (`BOM.csv`), Isometric representations (`ISO.json`), Compliance Matrices, and JSON traces for explainability.
*   `log.txt` — **System Logs**
    *   Captures runtime execution traces, errors, and standard output for debugging.

## Getting Started

All scripts are intended to be executed from the **project root**, but reference the `LLD` directory.

1. **Install Dependencies:**
   ```bash
   pip install -r LLD/requirements.txt
   ```

2. **Run the Main Pipeline:**
   ```bash
   python LLD/main.py
   ```

3. **Run the Test Suite:**
   ```bash
   python -m pytest LLD/tests/
   ```

## Deep Dive & Methodology

For a comprehensive breakdown of the data science architecture, the 6-phase end-to-end execution lifecycle, reward shaping, and ablation studies, please navigate to the detailed documentation inside the `LLD` folder:

*   [LLD/README.md](LLD/README.md) - Main technical overview
*   [LLD/process_flow_guide.md](LLD/process_flow_guide.md) - E2E Execution Lifecycle and Data Architecture Guide
*   [LLD/methodology.md](LLD/methodology.md) - Generative Design & RL Methodology
