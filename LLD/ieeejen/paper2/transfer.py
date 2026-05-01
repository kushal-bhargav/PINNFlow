"""
Transfer learning for PINN adaptation across pipe configurations.

Strategy:
  1. Train base PINN on a reference steel pipe from scratch.
  2. When adapting to a new configuration (different material/stiffness),
     initialise with the pre-trained weights.
  3. Freeze first k hidden layers (feature extraction), fine-tune last (L-k).
  4. Significantly fewer epochs needed vs. training from scratch.

Claimed benefit: 50–80% reduction in training time at comparable accuracy.

References:
    Chen et al. (2025), PINNs + Transfer Learning for Pipe Responses
"""

import torch
import torch.nn as nn
import copy
import time
from typing import Dict, Tuple, Optional

from .pinn_base import ElasticPIPENN, train_base_pinn


# ─────────────────────────────────────────────────────────────────────────────
# Layer freezing utilities
# ─────────────────────────────────────────────────────────────────────────────

def freeze_first_k_layers(model: ElasticPIPENN, k: int) -> None:
    """
    Freeze the first k hidden layers of the network (set requires_grad=False).

    The network uses nn.Sequential, so layers alternate between Linear and Tanh.
    Each hidden layer = 2 items (Linear + activation). The output layer is the
    last item in the sequence.

    Args:
        model : ElasticPIPENN
        k     : number of hidden layers to freeze (counted from input)
    """
    frozen_count = 0
    for i, layer in enumerate(model.net):
        if isinstance(layer, nn.Linear):
            if frozen_count < k:
                for p in layer.parameters():
                    p.requires_grad = False
                frozen_count += 1
            else:
                break   # remaining layers stay trainable

    n_frozen = sum(1 for p in model.parameters() if not p.requires_grad)
    n_free   = sum(1 for p in model.parameters() if p.requires_grad)
    print(f"  Frozen {n_frozen} params in first {k} layers | "
          f"Fine-tuning {n_free} params in remaining layers")


def unfreeze_all(model: ElasticPIPENN) -> None:
    """Re-enable all model parameters for gradient computation."""
    for p in model.parameters():
        p.requires_grad = True


# ─────────────────────────────────────────────────────────────────────────────
# Transfer learning entry point
# ─────────────────────────────────────────────────────────────────────────────

class TransferLearner:
    """
    Adapts a trained base PINN to a new pipe configuration via transfer learning.

    Args:
        base_model     : pre-trained ElasticPIPENN (source domain)
        target_EA      : axial stiffness of target pipe (N)
        target_EI      : bending stiffness of target pipe (N·m²)
        target_ku      : target axial soil spring stiffness
        target_kp      : target lateral soil spring stiffness
        n_freeze       : number of hidden layers to freeze during fine-tuning
        L              : pipeline length (m)
        device         : torch device
    """

    def __init__(
        self,
        base_model: ElasticPIPENN,
        target_EA: float,
        target_EI: float,
        target_ku: float,
        target_kp: float,
        n_freeze: int = 3,
        L: float = 300.0,
        device: str = "cpu",
    ):
        self.base_model = base_model
        self.target_EA  = target_EA
        self.target_EI  = target_EI
        self.target_ku  = target_ku
        self.target_kp  = target_kp
        self.n_freeze   = n_freeze
        self.L          = L
        self.device     = device

    def _build_target_model(self) -> ElasticPIPENN:
        """
        Create target model with same architecture as base,
        initialised with base model weights.
        """
        base = self.base_model
        target = ElasticPIPENN(
            EA=self.target_EA,
            EI=self.target_EI,
            ku=self.target_ku,
            kp=self.target_kp,
            L=self.L,
            n_hidden=sum(1 for m in base.net if isinstance(m, nn.Linear)) - 1,
            n_neurons=next(m.out_features for m in base.net if isinstance(m, nn.Linear)),
        ).to(self.device)

        # Copy base weights as initialisation
        target.load_state_dict(copy.deepcopy(base.state_dict()))
        return target

    def fine_tune(
        self,
        n_col: int = 4_000,
        n_bc: int = 200,
        adam_epochs: int = 5_000,
        lbfgs_iter: int = 1_000,
        lr: float = 5e-4,
        seed: int = 42,
    ) -> Tuple[ElasticPIPENN, Dict]:
        """
        Fine-tune the transferred model on the target domain physics.

        Args:
            n_col        : interior collocation points
            n_bc         : boundary condition points
            adam_epochs  : fine-tuning epochs (much fewer than full training)
            lbfgs_iter   : L-BFGS iterations for fine-tuning
            lr           : Adam learning rate for fine-tuning
            seed         : random seed

        Returns:
            model   : fine-tuned ElasticPIPENN
            history : training history dict
        """
        torch.manual_seed(seed)
        t0 = time.time()

        model = self._build_target_model()
        freeze_first_k_layers(model, self.n_freeze)

        # Only optimise unfrozen parameters
        trainable = [p for p in model.parameters() if p.requires_grad]
        adam = torch.optim.Adam(trainable, lr=lr)

        history = {"epoch": [], "L_total": [], "mode": "transfer"}

        def sample_col(n):
            lb = torch.tensor([0.0, 0.05,  0.0], device=self.device)
            ub = torch.tensor([self.L, 0.5, 90.0], device=self.device)
            X = lb + (ub - lb) * torch.rand(n, 3, device=self.device)
            return X.requires_grad_(True)

        def bc_loss(mod, n=100):
            lb = torch.tensor([0.05, 0.0], device=self.device)
            ub = torch.tensor([0.5, 90.0], device=self.device)
            params = lb + (ub - lb) * torch.rand(n, 2, device=self.device)
            x0  = torch.zeros(n, 1, device=self.device)
            xL  = torch.full((n, 1), self.L, device=self.device)
            X0  = torch.cat([x0,  params], dim=1).requires_grad_(True)
            XL  = torch.cat([xL,  params], dim=1).requires_grad_(True)
            return (mod(X0).pow(2).mean() + mod(XL).pow(2).mean())

        print(f"\nTransfer fine-tuning  |  freeze={self.n_freeze} layers  |  "
              f"EA_ratio={self.target_EA/self.base_model.EA:.3f}  "
              f"EI_ratio={self.target_EI/self.base_model.EI:.3f}")
        print(f"{'─'*60}")

        X_col = sample_col(n_col)

        for epoch in range(1, adam_epochs + 1):
            adam.zero_grad()
            R1, R2 = model.pde_residuals(X_col)
            L_pde = R1.pow(2).mean() + R2.pow(2).mean()
            L_bc  = bc_loss(model, n_bc)
            loss  = L_pde + 10.0 * L_bc
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            adam.step()

            if epoch % 500 == 0:
                history["epoch"].append(epoch)
                history["L_total"].append(loss.item())
                print(f"  [FT Adam {epoch:>5d}] total={loss.item():.3e}  "
                      f"pde={L_pde.item():.3e}  bc={L_bc.item():.3e}")

        # L-BFGS fine-tuning phase
        unfreeze_all(model)   # unfreeze all for LBFGS refinement
        X_col_f = sample_col(n_col)
        lbfgs = torch.optim.LBFGS(model.parameters(), max_iter=lbfgs_iter,
                                   line_search_fn="strong_wolfe")
        it = [0]
        def closure():
            lbfgs.zero_grad()
            R1, R2 = model.pde_residuals(X_col_f)
            loss   = R1.pow(2).mean() + R2.pow(2).mean() + 10.0 * bc_loss(model)
            loss.backward()
            it[0] += 1
            return loss
        lbfgs.step(closure)

        elapsed = time.time() - t0
        history["elapsed_s"] = elapsed
        print(f"Transfer fine-tuning complete in {elapsed:.1f} s")
        return model, history

    @staticmethod
    def speedup_factor(base_time_s: float, transfer_time_s: float) -> float:
        """Compute speedup ratio of transfer vs. from-scratch training."""
        return base_time_s / transfer_time_s
