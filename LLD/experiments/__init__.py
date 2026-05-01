# experiments/__init__.py
"""
Research Experimentation Framework (v4)
Unified access to all experimental modules.
"""
from .hyperopt import HyperparameterOptimizer
from .pinn_ablation import run_ablation as run_v4_ablation
from .rl_agents import run_rl_comparison
from .simulation_study import run_simulation_study
from .literature_benchmarks import PipelineBenchmarks
