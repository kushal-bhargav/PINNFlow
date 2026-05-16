"""Geometry-aware feature utilities for PINNFlow."""

from pinnflow.geometry.classifier import GeometryGatingNetwork
from pinnflow.geometry.features import (
    CURVATURE_FEATURE_NAMES,
    GEOMETRY_CLASS_NAMES,
    append_curvature_features,
    extract_curvature_features,
    ensure_geometry_state,
    geometry_labels,
)
from pinnflow.geometry.fourier import GeometryAwareFourierLayer

__all__ = [
    "CURVATURE_FEATURE_NAMES",
    "GEOMETRY_CLASS_NAMES",
    "GeometryAwareFourierLayer",
    "GeometryGatingNetwork",
    "append_curvature_features",
    "ensure_geometry_state",
    "extract_curvature_features",
    "geometry_labels",
]
