"""Visualization diagnostics for geometry-aware PINNFlow.

The legacy ``plot_all`` implementation lives in the sibling file
``visualization.py``. Because this directory is now a package with the same
public name, load that legacy module by path and re-export its entry point.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

from pinnflow.visualization.attention_maps import plot_curvature_attention
from pinnflow.visualization.error_histograms import plot_per_class_error_histogram
from pinnflow.visualization.stress_heatmap import plot_stress_heatmap

_legacy_path = Path(__file__).resolve().parents[1] / "visualization.py"
_legacy_spec = importlib.util.spec_from_file_location("_pinnflow_legacy_visualization", _legacy_path)
_legacy_module = importlib.util.module_from_spec(_legacy_spec)
if _legacy_spec and _legacy_spec.loader:
    _legacy_spec.loader.exec_module(_legacy_module)
    plot_all = _legacy_module.plot_all
else:
    plot_all = None

__all__ = [
    "plot_all",
    "plot_curvature_attention",
    "plot_per_class_error_histogram",
    "plot_stress_heatmap",
]
