"""
Base PINN for elastic pipe response under PGD (Paper 2).

Extends Paper 1 to the parametric space [x, δ, β] (continuous δ and β as inputs),
restricted to the linear elastic regime (for CFRP/FRP analysis).

The elastic simplification allows faster training and direct parametric sweeps
over PGD magnitude δ ∈ [0.05, 0.5] m and crossing angle β ∈ [0°, 90°].

References:
    Chen et al. (2025), PINNs + Transfer Learning for Pipe Responses
    (Computers and Geotechnics, doi:10.1016/j.compgeo.2025.106876)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Elastic PINN (3-input parametric version)
# ─────────────────────────────────────────────────────────────────────────────

class ElasticPIPENN(nn.Module):
    """
    Parametric PINN for elastic pipe-on-Winkler-foundation.

    Input:  z₀ = [x, δ, β]     (3 inputs)
    Output: Y  = [û, ŵ]         (axial + lateral displacements)

    This simplified formulation is valid in the pre-yield (elastic) regime.
    For steel pipes in elastic regime and CFRP pipes (which remain elastic
    until brittle failure).

    Args:
        EA, EI       : composite or steel stiffness values (from composite_stiffness.py)
        ku, kp       : linearised axial/lateral soil stiffness (spring moduli)
        L            : pipeline length (m)
        n_hidden     : number of hidden layers
        n_neurons    : neurons per hidden layer
    """

    def __init__(
        self,
        EA: float,
        EI: float,
        ku: float,
        kp: float,
        L: float = 300.0,
        n_hidden: int = 6,
        n_neurons: int = 40,
    ):
        super().__init__()

        self.EA = EA
        self.EI = EI
        self.ku = ku    # axial spring stiffness per unit length (N/m²)
        self.kp = kp    # lateral spring stiffness per unit length (N/m²)
        self.L  = L

        # Input normalisation bounds [x, δ, β]
        lb = torch.tensor([0.0, 0.05,  0.0], dtype=torch.float32)
        ub = torch.tensor([L,   0.5,  90.0], dtype=torch.float32)
        self.register_buffer("lb", lb)
        self.register_buffer("ub", ub)

        # Network
        layers = []
        in_dim = 3
        for _ in range(n_hidden):
            layers += [nn.Linear(in_dim, n_neurons), nn.Tanh()]
            in_dim = n_neurons
        layers.append(nn.Linear(in_dim, 2))   # [û, ŵ]
        self.net = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """X: (N, 3) = [x, δ, β_deg]  →  out: (N, 2) = [û, ŵ]"""
        X_n = 2.0 * (X - self.lb) / (self.ub - self.lb + 1e-8) - 1.0
        return self.net(X_n)

    def pde_residuals(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute elastic PDE residuals at collocation points.

        Elastic governing equations (linearised soil springs):
            EA * u_xx + ku * (Ug - u) = 0    →  R1 = EA*u_xx - ku*u + ku*Ug
            EI * w_xxxx - kp * (Wg - w) = 0  →  R2 = EI*w_xxxx + kp*w - kp*Wg

        (In linear elastic regime, the geometric nonlinearity N·w_xx is
        small and dropped in the first-order elastic formulation.)

        Args:
            X : (N, 3) collocation points [x, δ, β], requires_grad=True

        Returns:
            R1, R2 : axial and lateral PDE residuals (N,)
        """
        if not X.requires_grad:
            X = X.detach().requires_grad_(True)

        out = self.forward(X)
        u   = out[:, 0]
        w   = out[:, 1]

        def grad(y, wrt=X, retain=True):
            return torch.autograd.grad(y, wrt,
                                       grad_outputs=torch.ones_like(y),
                                       create_graph=True,
                                       retain_graph=retain)[0][:, 0]

        u_x   = grad(u)
        u_xx  = grad(u_x)

        w_x   = grad(w)
        w_xx  = grad(w_x)
        w_xxx = grad(w_xx)
        w_xxxx = grad(w_xxx, retain=False)

        # Ground displacement (smooth rectangular profile)
        x       = X[:, 0]
        delta   = X[:, 1]
        beta_deg = X[:, 2]
        beta_rad = beta_deg * torch.pi / 180.0

        x_start = self.L * 0.45
        x_end   = self.L * 0.55
        k       = 50.0
        window  = torch.sigmoid(k * (x - x_start)) - torch.sigmoid(k * (x - x_end))

        Ug = delta * torch.cos(beta_rad) * window
        Wg = delta * torch.sin(beta_rad) * window

        # PDE residuals
        R1 = self.EA * u_xx - self.ku * u + self.ku * Ug
        R2 = self.EI * w_xxxx + self.kp * w - self.kp * Wg

        return R1, R2


def train_base_pinn(
    EA: float,
    EI: float,
    ku: float,
    kp: float,
    L: float = 300.0,
    n_col: int = 8_000,
    n_bc: int = 400,
    n_hidden: int = 6,
    n_neurons: int = 40,
    adam_epochs: int = 15_000,
    lbfgs_iter: int = 3_000,
    lr: float = 1e-3,
    device: str = "cpu",
    seed: int = 0,
) -> Tuple[ElasticPIPENN, Dict]:
    """
    Train a base ElasticPIPENN from scratch.

    Args:
        EA, EI  : pipe axial and bending stiffness
        ku, kp  : soil spring stiffnesses
        L       : pipeline length (m)
        ...

    Returns:
        model   : trained ElasticPIPENN
        history : training loss history dict
    """
    torch.manual_seed(seed)

    model = ElasticPIPENN(EA, EI, ku, kp, L, n_hidden, n_neurons).to(device)
    adam  = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"epoch": [], "L_total": []}

    def sample_col(n, seed_=None):
        if seed_ is not None:
            torch.manual_seed(seed_)
        lb = torch.tensor([0.0, 0.05,  0.0], device=device)
        ub = torch.tensor([L,   0.5,  90.0], device=device)
        X = lb + (ub - lb) * torch.rand(n, 3, device=device)
        return X.requires_grad_(True)

    def bc_loss_elastic(mod, n=200):
        lb = torch.tensor([0.0, 0.05,  0.0], device=device)
        ub = torch.tensor([L,   0.5,  90.0], device=device)
        params = lb[1:] + (ub[1:] - lb[1:]) * torch.rand(n, 2, device=device)
        x0  = torch.zeros(n, 1, device=device)
        xL  = torch.full((n, 1), L, device=device)
        X0  = torch.cat([x0,  params], dim=1).requires_grad_(True)
        XL  = torch.cat([xL,  params], dim=1).requires_grad_(True)
        out0 = mod(X0);  outL = mod(XL)
        loss = (out0 ** 2).mean() + (outL ** 2).mean()
        return loss

    print(f"Training base PINN (EA={EA/1e9:.2f} GN, EI={EI/1e6:.2f} MNm²)")
    print(f"{'─'*50}")

    X_col = sample_col(n_col, seed_=seed)

    for epoch in range(1, adam_epochs + 1):
        adam.zero_grad()
        R1, R2 = model.pde_residuals(X_col)
        L_pde  = R1.pow(2).mean() + R2.pow(2).mean()
        L_bc   = bc_loss_elastic(model, n_bc)
        loss   = L_pde + 10.0 * L_bc
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        adam.step()

        if epoch % 1000 == 0:
            history["epoch"].append(epoch)
            history["L_total"].append(loss.item())
            print(f"  [Adam {epoch:>6d}] total={loss.item():.3e}  "
                  f"pde={L_pde.item():.3e}  bc={L_bc.item():.3e}")

    # L-BFGS phase
    X_col_fixed = sample_col(n_col, seed_=99)
    lbfgs = torch.optim.LBFGS(model.parameters(), max_iter=lbfgs_iter,
                               line_search_fn="strong_wolfe")
    iter_n = [0]

    def closure():
        lbfgs.zero_grad()
        R1, R2 = model.pde_residuals(X_col_fixed)
        L_pde  = R1.pow(2).mean() + R2.pow(2).mean()
        L_bc   = bc_loss_elastic(model)
        loss   = L_pde + 10.0 * L_bc
        loss.backward()
        iter_n[0] += 1
        if iter_n[0] % 500 == 0:
            print(f"  [LBFGS {iter_n[0]:>5d}] total={loss.item():.3e}")
        return loss

    lbfgs.step(closure)
    print("Base PINN training complete.")
    return model, history
