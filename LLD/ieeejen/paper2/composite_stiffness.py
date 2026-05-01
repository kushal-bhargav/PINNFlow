"""
Composite pipe stiffness computation for CFRP/FRP-reinforced pipes.

The reinforcement wraps modify axial stiffness (EA) and bending stiffness (EI)
compared to a bare steel pipe. These composite stiffness values are computed
analytically and fed as fixed parameters into the PINN governing PDEs.

References:
    Chen et al. (2025), PINNs + Transfer Learning for Pipe Responses (Computers & Geotechnics)
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Material parameters
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SteelProperties:
    E:       float = 207e9   # Young's modulus (Pa)
    nu:      float = 0.3
    sigma_y: float = 450e6  # yield stress (Pa)
    rho:     float = 7850.0 # density (kg/m³)


@dataclass
class CFRPProperties:
    """
    Unidirectional CFRP lamina properties (axial direction).
    Properties depend on fibre volume fraction and lay-up.
    """
    E_axial:    float = 140e9    # axial Young's modulus (Pa)
    E_hoop:     float = 9e9     # hoop/transverse modulus (Pa)
    G12:        float = 5e9     # in-plane shear modulus (Pa)
    nu12:       float = 0.28    # major Poisson's ratio
    rho:        float = 1600.0  # density (kg/m³)
    t_layer:    float = 0.25e-3 # thickness per CFRP layer (m)


@dataclass
class GFRPProperties:
    """Glass Fibre Reinforced Polymer lamina properties."""
    E_axial:    float = 45e9
    E_hoop:     float = 12e9
    G12:        float = 5e9
    nu12:       float = 0.30
    rho:        float = 1900.0
    t_layer:    float = 0.3e-3


# ─────────────────────────────────────────────────────────────────────────────
# Composite cross-section stiffness
# ─────────────────────────────────────────────────────────────────────────────

class CompositePipeStiffness:
    """
    Computes composite EA and EI for a steel pipe wrapped with N layers of CFRP/GFRP.

    Assumption: reinforcement layers are bonded on the outside of the steel pipe
    (external wrap). Transformed-section method used for EI computation.

    Args:
        D_steel    : steel pipe outer diameter (m)
        t_steel    : steel wall thickness (m)
        n_layers   : number of CFRP wrapping layers
        steel      : SteelProperties
        reinf      : CFRPProperties or GFRPProperties
    """

    def __init__(
        self,
        D_steel: float,
        t_steel: float,
        n_layers: int,
        steel: SteelProperties = None,
        reinf = None,
    ):
        self.D_s    = D_steel
        self.t_s    = t_steel
        self.n_lay  = n_layers
        self.steel  = steel or SteelProperties()
        self.reinf  = reinf or CFRPProperties()

        # Total reinforcement thickness
        self.t_r    = n_layers * self.reinf.t_layer

        # Geometry
        self.r_s_i  = D_steel / 2.0 - t_steel         # steel inner radius
        self.r_s_o  = D_steel / 2.0                   # steel outer radius
        self.r_r_i  = self.r_s_o                       # reinf inner radius
        self.r_r_o  = self.r_s_o + self.t_r           # reinf outer radius

    def area_steel(self) -> float:
        """Annular steel cross-section area (m²)."""
        return np.pi * (self.r_s_o ** 2 - self.r_s_i ** 2)

    def area_reinf(self) -> float:
        """Annular reinforcement cross-section area (m²)."""
        return np.pi * (self.r_r_o ** 2 - self.r_r_i ** 2)

    def EA_composite(self) -> float:
        """
        Composite axial stiffness EA (N).

            EA = E_steel * A_steel + E_CFRP_axial * A_CFRP
        """
        return (self.steel.E * self.area_steel() +
                self.reinf.E_axial * self.area_reinf())

    def EI_composite(self) -> float:
        """
        Composite bending stiffness EI (N·m²), about the centroidal axis.

        For a hollow circular section:
            I = π/4 * (r_o⁴ - r_i⁴)

        The centroid remains at the pipe axis (symmetric cross-section).

        Returns:
            EI : composite bending stiffness (N·m²)
        """
        I_steel = np.pi / 4.0 * (self.r_s_o ** 4 - self.r_s_i ** 4)
        I_reinf = np.pi / 4.0 * (self.r_r_o ** 4 - self.r_r_i ** 4)
        return (self.steel.E * I_steel +
                self.reinf.E_axial * I_reinf)

    def EI_steel_only(self) -> float:
        """Reference EI for bare steel pipe."""
        I_s = np.pi / 4.0 * (self.r_s_o ** 4 - self.r_s_i ** 4)
        return self.steel.E * I_s

    def EA_steel_only(self) -> float:
        """Reference EA for bare steel pipe."""
        return self.steel.E * self.area_steel()

    def stiffness_ratio_EA(self) -> float:
        """Ratio EA_composite / EA_steel — amplification from reinforcement."""
        return self.EA_composite() / self.EA_steel_only()

    def stiffness_ratio_EI(self) -> float:
        """Ratio EI_composite / EI_steel."""
        return self.EI_composite() / self.EI_steel_only()

    def summary(self) -> dict:
        """Return dict of all stiffness quantities."""
        return {
            "EA_steel":    self.EA_steel_only(),
            "EA_composite": self.EA_composite(),
            "EI_steel":    self.EI_steel_only(),
            "EI_composite": self.EI_composite(),
            "EA_ratio":    self.stiffness_ratio_EA(),
            "EI_ratio":    self.stiffness_ratio_EI(),
            "A_steel":     self.area_steel(),
            "A_reinf":     self.area_reinf(),
            "t_reinf":     self.t_r,
            "n_layers":    self.n_lay,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Standard pipe configurations (Table from Paper 2)
# ─────────────────────────────────────────────────────────────────────────────

def get_pipe_configs():
    """
    Return the three pipe configurations tested in Chen et al. (2025):
      A: Steel, no reinforcement
      B: CFRP, 2 layers
      C: CFRP, 4 layers
    All with D=508mm, t=9.5mm.
    """
    D, t = 0.508, 0.0095
    configs = {}

    # Config A: bare steel
    cA = CompositePipeStiffness(D, t, n_layers=0)
    cA.t_r   = 0.0
    cA.r_r_i = cA.r_s_o
    cA.r_r_o = cA.r_s_o
    configs["A_steel"] = cA

    # Config B: 2 CFRP layers
    configs["B_CFRP_2"] = CompositePipeStiffness(D, t, n_layers=2)

    # Config C: 4 CFRP layers
    configs["C_CFRP_4"] = CompositePipeStiffness(D, t, n_layers=4)

    return configs


# ─────────────────────────────────────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    configs = get_pipe_configs()
    print(f"{'Config':<15} {'EA (GN)':<14} {'EI (MNm²)':<14} "
          f"{'EA ratio':<12} {'EI ratio':<12}")
    print("─" * 65)
    for name, cfg in configs.items():
        s = cfg.summary()
        print(f"{name:<15} {s['EA_composite']/1e9:<14.3f} "
              f"{s['EI_composite']/1e6:<14.3f} "
              f"{s['EA_ratio']:<12.3f} {s['EI_ratio']:<12.3f}")
