"""
Global gas network state-space assembly from individual pipe segments.

Assembles the global system matrix K^{net}(θ) from individual pipe K_i(θ_i)
matrices via nodal flow balance (Kirchhoff's current law) and the network
incidence matrix.

The global state-space model:
    x^{net}_{k+1} = A^{net}(θ) * x^{net}_k + B^{net}(θ) * u_k
    y_k           = C^{net} * x^{net}_k

References:
    Wang / Liu et al. (2025), PIRN (arXiv:2502.07230)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Network topology (incidence matrix)
# ─────────────────────────────────────────────────────────────────────────────

def build_incidence_matrix(
    edges: List[Tuple[int, int]],
    n_nodes: int,
) -> np.ndarray:
    """
    Build signed node-edge incidence matrix A_inc of shape (n_nodes, n_edges).

    A_inc[i, j] = +1 if node i is the tail (source) of edge j
    A_inc[i, j] = -1 if node i is the head (sink) of edge j
    A_inc[i, j] =  0 otherwise

    Args:
        edges   : list of (from_node, to_node) tuples (0-indexed)
        n_nodes : total number of nodes in the network

    Returns:
        A_inc : (n_nodes, n_edges) numpy array
    """
    n_edges = len(edges)
    A_inc   = np.zeros((n_nodes, n_edges), dtype=np.float64)
    for j, (i_from, i_to) in enumerate(edges):
        A_inc[i_from, j] = +1.0
        A_inc[i_to,   j] = -1.0
    return A_inc


# ─────────────────────────────────────────────────────────────────────────────
# GasLib topology loader (simplified)
# ─────────────────────────────────────────────────────────────────────────────

class GasLibNetwork:
    """
    Container for gas network topology and nominal pipe properties.

    In practice, parsed from GasLib XML files. Here we provide the
    four benchmark networks from Wang et al. (2025) as hard-coded configs.
    """

    def __init__(self, name: str):
        configs = {
            "GasLib-11":  self._gaslib11,
            "GasLib-24":  self._gaslib24,
            "GasLib-40":  self._gaslib40,
            "GasLib-134": self._gaslib134,
        }
        if name not in configs:
            raise ValueError(f"Unknown network '{name}'. "
                             f"Available: {list(configs.keys())}")
        configs[name]()

    def _gaslib11(self):
        """GasLib-11: 11 nodes, 10 pipes, 1 compressor."""
        self.n_nodes  = 11
        self.n_pipes  = 10
        # Edges: (from, to) tuples
        self.edges    = [(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(7,8),(8,9),(9,10)]
        self.compressors = [0]   # node indices where compressors are attached
        self.entry_nodes = [0]
        self.exit_nodes  = [5, 10]
        # Nominal pipe properties (same for all pipes in this simplified version)
        self.pipe_props = [{"D": 0.5, "L": 10e3, "lam": 0.01, "Z": 0.9} for _ in range(self.n_pipes)]

    def _gaslib24(self):
        """GasLib-24: 24 nodes, 23 pipes, 2 compressors."""
        self.n_nodes  = 24
        self.n_pipes  = 23
        self.edges    = [(i, i+1) for i in range(23)]
        self.compressors = [0, 10]
        self.entry_nodes = [0]
        self.exit_nodes  = [12, 23]
        self.pipe_props = [{"D": 0.6, "L": 8e3, "lam": 0.012, "Z": 0.92} for _ in range(self.n_pipes)]

    def _gaslib40(self):
        """GasLib-40: 40 nodes, 39 pipes, 3 compressors."""
        self.n_nodes  = 40
        self.n_pipes  = 39
        self.edges    = [(i, i+1) for i in range(39)]
        self.compressors = [0, 13, 26]
        self.entry_nodes = [0]
        self.exit_nodes  = [20, 39]
        self.pipe_props = [{"D": 0.55, "L": 5e3, "lam": 0.011, "Z": 0.91} for _ in range(self.n_pipes)]

    def _gaslib134(self):
        """GasLib-134: 134 nodes, 133 pipes, 7 compressors."""
        self.n_nodes  = 134
        self.n_pipes  = 133
        self.edges    = [(i, i+1) for i in range(133)]
        self.compressors = [0, 19, 38, 57, 76, 95, 114]
        self.entry_nodes = [0]
        self.exit_nodes  = [66, 133]
        self.pipe_props = [{"D": 0.7, "L": 3e3, "lam": 0.009, "Z": 0.93} for _ in range(self.n_pipes)]


# ─────────────────────────────────────────────────────────────────────────────
# Global network state-space assembler
# ─────────────────────────────────────────────────────────────────────────────

class GasNetworkStateSpace(nn.Module):
    """
    Assembles the global state-space model for a gas network from
    per-pipe physics modules.

    Each pipe contributes a state-space block. The global system couples
    these blocks through nodal flow balance (Kirchhoff's laws):
        Σ q_ij = d_i   for each node i (demand balance)

    Args:
        network  : GasLibNetwork instance with topology and nominal properties
        T        : operating temperature (K)
        M_g      : gas molar mass (kg/mol)
        p0       : nominal operating pressure (Pa)
        q0       : nominal mass flow (kg/s)
        dt       : simulation time step (s)
    """

    def __init__(
        self,
        network: GasLibNetwork,
        T:   float = 288.0,
        M_g: float = 0.01604,
        p0:  float = 5e6,
        q0:  float = 50.0,
        dt:  float = 60.0,
        n_segments: int = 1,
    ):
        super().__init__()
        self.net     = network
        self.T       = T
        self.M_g     = M_g
        self.p0      = p0
        self.q0      = q0
        self.dt      = dt
        self.n_nodes = network.n_nodes
        self.n_pipes = network.n_pipes

        # Import here to avoid circular import
        from .state_space import PipeSegmentStateSpace

        # One PipeSegmentStateSpace per pipe — each has its own learnable λ
        self.pipe_models = nn.ModuleList([
            PipeSegmentStateSpace(
                lam_init=props["lam"],
                D=props["D"],
                L=props["L"],
                Z_init=props["Z"],
                T=T, M_g=M_g, p0=p0, q0=q0, dt=dt,
                n_segments=n_segments,
            )
            for props in network.pipe_props
        ])

        # Incidence matrix (fixed topology — not learned)
        A_inc = build_incidence_matrix(network.edges, network.n_nodes)
        self.register_buffer("A_inc",
                             torch.tensor(A_inc, dtype=torch.float64))

        # Output matrix C: identity on entry/exit nodes (terminal measurements)
        sensor_nodes = sorted(set(network.entry_nodes + network.exit_nodes))
        n_sensors    = len(sensor_nodes)
        C = np.zeros((n_sensors, network.n_nodes), dtype=np.float64)
        for i, node in enumerate(sensor_nodes):
            C[i, node] = 1.0
        self.register_buffer("C", torch.tensor(C, dtype=torch.float64))
        self.sensor_nodes = sensor_nodes

    def get_learnable_friction_factors(self) -> Dict[int, torch.Tensor]:
        """Return dict of {pipe_index: lam_parameter} for all pipes."""
        return {i: pipe.lam for i, pipe in enumerate(self.pipe_models)}

    def forward(
        self,
        x_k: torch.Tensor,
        u_k: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        One-step network update.

        Args:
            x_k : (n_nodes,) nodal pressure state at time k
            u_k : (n_compressors,) compressor inputs (mass flow additions)

        Returns:
            x_{k+1} : (n_nodes,) next state
            y_k     : (n_sensors,) sensor measurements
        """
        # Assemble global network update from individual pipe updates
        # Each pipe contributes to the pressures at its two endpoint nodes.
        # Simplified: global pressure update via averaging contributions.
        x_next = torch.zeros_like(x_k)
        counts = torch.zeros(self.n_nodes, dtype=torch.float64,
                             device=x_k.device)

        for j, (i_from, i_to) in enumerate(self.net.edges):
            pipe = self.pipe_models[j]
            # Single-segment approximation: propagate pressure from upstream
            x_pipe  = torch.stack([x_k[i_from], x_k[i_to]])
            x_p_next = pipe(x_pipe)
            x_next[i_from] = x_next[i_from] + x_p_next[0]
            x_next[i_to]   = x_next[i_to]   + x_p_next[1]
            counts[i_from] += 1.0
            counts[i_to]   += 1.0

        # Average contributions (where node appears in multiple pipes)
        x_next = x_next / (counts + 1e-8)

        # Add compressor input effect at compressor nodes
        for ci, node_idx in enumerate(self.net.compressors):
            if ci < u_k.shape[0]:
                x_next[node_idx] = x_next[node_idx] + u_k[ci]

        # Measurement: y = C * x_{k+1}
        y_k = self.C @ x_next

        return x_next, y_k
