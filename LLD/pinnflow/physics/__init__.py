"""Physics residual helpers for PINNFlow."""

from pinnflow.physics.loss_balancer import LossBalancer
from pinnflow.physics.pde_elbow import (
    dean_stress_factor,
    elbow_pressure_drop,
    elbow_residuals,
    elbow_stress_multiplier,
    ito_friction_factor,
    stress_concentration_factor,
)
from pinnflow.physics.pde_straight import darcy_pressure_drop, hoop_stress
from pinnflow.physics.pde_tee import tee_residuals
from pinnflow.physics.pde_reducer import reducer_residuals

__all__ = [
    "LossBalancer",
    "darcy_pressure_drop",
    "dean_stress_factor",
    "elbow_pressure_drop",
    "elbow_residuals",
    "elbow_stress_multiplier",
    "hoop_stress",
    "ito_friction_factor",
    "stress_concentration_factor",
    "tee_residuals",
    "reducer_residuals",
]
