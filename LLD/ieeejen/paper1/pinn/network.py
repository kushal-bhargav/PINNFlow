"""
Parametric Physics-Informed Neural Network (PINN) architecture.

Maps 6-dimensional input [x, δ, c, φ, γ, H] to 2-dimensional output [u, w]
(axial and lateral displacements of the pipeline).

Architecture: Fully-Connected Deep Neural Network (FC-DNN) with tanh activation.

References:
    Haghighat et al. (2025), PINN-RA for Buried Pipeline Reliability Analysis
"""

import torch
import torch.nn as nn
import numpy as np


class PINNNetwork(nn.Module):
    """
    Parametric PINN for buried pipeline displacement prediction.

    Input z₀ = [x, δ, c, φ, γ, H]   (6 inputs)
    Output Y  = [û(x;params), ŵ(x;params)]  (2 outputs)

    Spatial derivatives (u_x, w_x, w_xx) are obtained by calling
    torch.autograd.grad() on the network outputs w.r.t. the input tensor.
    """

    def __init__(
        self,
        n_inputs: int = 6,
        n_outputs: int = 2,
        n_hidden_layers: int = 5,
        n_neurons: int = 40,
        activation: str = "tanh",
        use_input_norm: bool = True,
        input_lb: torch.Tensor = None,
        input_ub: torch.Tensor = None,
    ):
        """
        Args:
            n_inputs        : input dimension (default 6: x + 5 parameters)
            n_outputs       : output dimension (default 2: u, w)
            n_hidden_layers : number of hidden layers L
            n_neurons       : neurons per hidden layer m
            activation      : 'tanh' | 'sigmoid' | 'relu' | 'leaky_relu'
            use_input_norm  : if True, normalise inputs to [-1, 1] using lb/ub
            input_lb        : lower bounds tensor (n_inputs,) for normalisation
            input_ub        : upper bounds tensor (n_inputs,) for normalisation
        """
        super().__init__()

        self.n_inputs  = n_inputs
        self.n_outputs = n_outputs
        self.use_input_norm = use_input_norm

        # Register normalisation constants as buffers (moved with .to(device))
        if use_input_norm:
            if input_lb is None or input_ub is None:
                raise ValueError("Provide input_lb and input_ub for input normalisation.")
            self.register_buffer("input_lb", input_lb.float())
            self.register_buffer("input_ub", input_ub.float())

        # Build activation
        act_map = {
            "tanh"      : nn.Tanh,
            "sigmoid"   : nn.Sigmoid,
            "relu"      : nn.ReLU,
            "leaky_relu": nn.LeakyReLU,
            "softplus"  : nn.Softplus,
        }
        if activation not in act_map:
            raise ValueError(f"Unknown activation '{activation}'. Choose from {list(act_map)}.")
        Act = act_map[activation]

        # Assemble layers: [Input → H1 → H2 → ... → HL → Output]
        layers = []
        in_dim = n_inputs
        for _ in range(n_hidden_layers):
            layers.append(nn.Linear(in_dim, n_neurons))
            layers.append(Act())
            in_dim = n_neurons
        layers.append(nn.Linear(in_dim, n_outputs))
        self.net = nn.Sequential(*layers)

        # Xavier initialisation (suitable for tanh/sigmoid)
        self._initialise_weights()

    def _initialise_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def normalise_input(self, X: torch.Tensor) -> torch.Tensor:
        """
        Normalise input to [-1, 1] range using stored lb/ub.

        Args:
            X : (N, n_inputs) raw input tensor

        Returns:
            X_norm : (N, n_inputs) normalised tensor
        """
        lb = self.input_lb.to(X.device)
        ub = self.input_ub.to(X.device)
        return 2.0 * (X - lb) / (ub - lb + 1e-12) - 1.0

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            X : (N, 6) input tensor [x, δ, c, φ, γ, H], requires_grad=True

        Returns:
            out : (N, 2) predicted [u, w]
        """
        if self.use_input_norm:
            X_in = self.normalise_input(X)
        else:
            X_in = X
        return self.net(X_in)

    def predict_with_grads(self, X: torch.Tensor):
        """
        Forward pass + compute all displacement derivatives via autograd.

        Returns dict:
            u, w           : (N,) displacements
            u_x, w_x, w_xx : (N,) spatial derivatives
        """
        if not X.requires_grad:
            X = X.detach().requires_grad_(True)

        out  = self.forward(X)
        u    = out[:, 0]
        w    = out[:, 1]

        ones_u = torch.ones_like(u)
        ones_w = torch.ones_like(w)

        # First derivatives w.r.t. X (column 0 = x)
        grad_u = torch.autograd.grad(u, X, grad_outputs=ones_u,
                                     create_graph=True, retain_graph=True)[0]
        grad_w = torch.autograd.grad(w, X, grad_outputs=ones_w,
                                     create_graph=True, retain_graph=True)[0]

        u_x = grad_u[:, 0]   # ∂u/∂x
        w_x = grad_w[:, 0]   # ∂w/∂x

        # Second derivative of w w.r.t. x
        grad_wx = torch.autograd.grad(w_x, X, grad_outputs=torch.ones_like(w_x),
                                      create_graph=True, retain_graph=True)[0]
        w_xx = grad_wx[:, 0]  # ∂²w/∂x²

        return {"u": u, "w": w, "u_x": u_x, "w_x": w_x, "w_xx": w_xx}

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_pinn(
    x_lb: float = 0.0,
    x_ub: float = 300.0,
    delta_lb: float = 0.01,
    delta_ub: float = 0.5,
    c_lb: float = 0.0,
    c_ub: float = 50e3,
    phi_lb: float = 20.0,
    phi_ub: float = 45.0,
    gamma_lb: float = 15e3,
    gamma_ub: float = 22e3,
    H_lb: float = 0.9,
    H_ub: float = 2.5,
    n_hidden_layers: int = 5,
    n_neurons: int = 40,
    activation: str = "tanh",
) -> PINNNetwork:
    """
    Convenience constructor that builds a PINNNetwork with input bounds
    corresponding to the 6-parameter space [x, δ, c, φ, γ, H].

    Args:
        x_lb / x_ub         : spatial domain bounds (m)
        delta_lb / delta_ub : ground displacement range (m)
        c_lb / c_ub         : cohesion range (Pa)
        phi_lb / phi_ub     : friction angle range (degrees)
        gamma_lb / gamma_ub : unit weight range (N/m³)
        H_lb / H_ub         : burial depth range (m)
        n_hidden_layers     : number of hidden layers
        n_neurons           : neurons per layer
        activation          : activation function name

    Returns:
        PINNNetwork instance
    """
    lb = torch.tensor([x_lb, delta_lb, c_lb, phi_lb, gamma_lb, H_lb], dtype=torch.float32)
    ub = torch.tensor([x_ub, delta_ub, c_ub, phi_ub, gamma_ub, H_ub], dtype=torch.float32)

    model = PINNNetwork(
        n_inputs=6,
        n_outputs=2,
        n_hidden_layers=n_hidden_layers,
        n_neurons=n_neurons,
        activation=activation,
        use_input_norm=True,
        input_lb=lb,
        input_ub=ub,
    )
    print(f"PINN built: {model.count_parameters()} trainable parameters | "
          f"{n_hidden_layers} hidden layers × {n_neurons} neurons")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model = build_pinn()
    N = 100
    X = torch.rand(N, 6, requires_grad=True)
    out = model(X)
    print(f"Output shape: {out.shape}")   # (100, 2)

    grads = model.predict_with_grads(X)
    for k, v in grads.items():
        print(f"{k}: shape={v.shape}, mean={v.mean().item():.4e}")
