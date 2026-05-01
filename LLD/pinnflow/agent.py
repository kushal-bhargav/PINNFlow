"""
pinnflow/agent.py
──────────────────
MODULE 6 — PPO Agent  [N3]

[N3] PPO with GAE-λ advantage, entropy bonus, adaptive LR.
     Action space: 8-dim continuous (one delta per design parameter).

[P2 FIX] train() sets env.episode = ep at the start of every episode so that
         curriculum phases in PipelineEnv transition at the correct episodes
         (100 and 200), not at steps.
"""
from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler
from pinnflow.activations import tanh

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pinnflow.environment import PipelineEnv

class LagrangianPPOAgent:
    """
    [N3] Proximal Policy Optimisation with GAE-λ.

    Actor  : 3-layer tanh network → 8-dim continuous action (tanh-squashed).
    Critic : 2-layer tanh network → scalar value estimate.
    Updates use a simplified last-layer policy gradient (full back-prop
    through all layers is left to future extensions).
    """

    def __init__(
        self,
        sdim: int = 8,
        adim: int = 8,
        hidden: int = 128,
        lr: float = 5e-3,
        gamma: float = 0.99,
        lam: float = 0.95,
        clip: float = 0.2,
        ent: float = 0.02,
        n_ep: int = 4,
        beta_init: float = 0.1,
    ):
        self.gamma = gamma; self.lam = lam; self.clip = clip
        self.ent   = ent;   self.n_ep = n_ep; self.lr_base = lr
        
        # [V5] Lagrangian Multiplier (Saftey Weight)
        self.beta       = beta_init
        self.beta_lr    = 1e-3
        self.cost_limit = 0.02  # Max acceptable ASME violation slack
        
        sc = 0.05

        # ── Actor ─────────────────────────────────────────────────────────────
        self.Wa1 = np.random.randn(sdim,   hidden) * sc; self.ba1 = np.zeros(hidden)
        self.Wa2 = np.random.randn(hidden, hidden) * sc; self.ba2 = np.zeros(hidden)
        self.Wa3 = np.random.randn(hidden,   adim) * sc; self.ba3 = np.zeros(adim)
        self.log_std = np.full(adim, -1.0)

        # ── Critic ────────────────────────────────────────────────────────────
        self.Wc1 = np.random.randn(sdim,   hidden) * sc; self.bc1 = np.zeros(hidden)
        self.Wc2 = np.random.randn(hidden,      1) * sc; self.bc2 = np.zeros(1)

        # ── Adam states ───────────────────────────────────────────────────────
        self._t = 0
        for nm in ["Wa1", "ba1", "Wa2", "ba2", "Wa3", "ba3", "Wc1", "bc1", "Wc2", "bc2"]:
            setattr(self, "m" + nm, np.zeros_like(getattr(self, nm)))
            setattr(self, "v" + nm, np.zeros_like(getattr(self, nm)))

        self.sc_s = StandardScaler(); self._fit = False
        self.reward_hist: list = []
        self.ent_hist:    list = []
        self.csr_hist:    list = []

    # ── Adam step ─────────────────────────────────────────────────────────────
    def _adam(self, nm: str, g: np.ndarray, lr: float,
              b1: float = 0.9, b2: float = 0.999, e: float = 1e-8) -> None:
        self._t += 1
        m = getattr(self, "m" + nm); v = getattr(self, "v" + nm)
        m[:] = b1 * m + (1 - b1) * g
        v[:] = b2 * v + (1 - b2) * g ** 2
        mh = m / (1 - b1 ** self._t)
        vh = v / (1 - b2 ** self._t)
        getattr(self, nm)[:] -= lr * mh / (np.sqrt(vh) + e)

    # ── State normalisation ───────────────────────────────────────────────────
    def _norm(self, s: np.ndarray) -> np.ndarray:
        if not self._fit:
            self.sc_s.fit(s.reshape(1, -1)); self._fit = True
        return (s - self.sc_s.mean_) / (self.sc_s.scale_ + 1e-8)

    # ── Forward passes ────────────────────────────────────────────────────────
    def _actor(self, S: np.ndarray):
        h1 = tanh(S @ self.Wa1 + self.ba1)
        h2 = tanh(h1 @ self.Wa2 + self.ba2)
        mu = np.tanh(h2 @ self.Wa3 + self.ba3)
        return mu, h1, h2

    def _critic(self, S: np.ndarray):
        h1 = tanh(S @ self.Wc1 + self.bc1)
        return (h1 @ self.Wc2 + self.bc2).flatten(), h1

    # ── Action selection ──────────────────────────────────────────────────────
    def select_action(self, state: np.ndarray):
        S  = self._norm(state).reshape(1, -1)
        mu, _, _ = self._actor(S); mu = mu[0]
        std = np.exp(np.clip(self.log_std, -2, -0.5))
        a   = np.clip(mu + 0.5 * std * np.random.randn(len(mu)), -1, 1)
        lp  = -0.5 * np.sum(((a - mu) / (std + 1e-8)) ** 2) - np.sum(np.log(std + 1e-8))
        return a, lp, mu

    # ── GAE-λ ─────────────────────────────────────────────────────────────────
    def _gae(self, R, V, gamma: float, lam: float):
        adv = []; gae = 0; vals = list(V) + [0]
        for r, v, vn in zip(reversed(R), reversed(vals[:-1]), reversed(vals[1:])):
            d = r + gamma * vn - v; gae = d + gamma * lam * gae; adv.insert(0, gae)
        return adv, [a + v for a, v in zip(adv, vals[:-1])]

    # ── PPO update ────────────────────────────────────────────────────────────
    def _update(self, S, A, OLP, ret, adv, viol, lr: float) -> tuple[float, float]:
        adv = np.array(adv); adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        ret = np.array(ret)
        viol = np.array(viol)
        
        # Dual Update for beta (Lagrangian multiplier) [P1.2]
        avg_viol = np.mean(viol)
        self.beta = np.clip(self.beta + self.beta_lr * (avg_viol - self.cost_limit), 0.0, 10.0)

        for _ in range(self.n_ep):
            mu, h1, h2 = self._actor(S)
            std = np.exp(np.clip(self.log_std, -2, -0.5))
            nlp = (
                -0.5 * np.sum(((A - mu) / (std + 1e-8)) ** 2, axis=1)
                - np.sum(np.log(std + 1e-8))
            )
            ratio = np.exp(nlp - OLP)
            ent   = np.mean(0.5 + 0.5 * np.log(2 * np.pi * np.e * (std ** 2)))
            
            # Policy gradient (Lagrangian Augmented: Adv - Beta * Violation)
            # This pushes the agent away from high-violation states.
            aug_adv = adv - self.beta * viol
            pg_g  = aug_adv[:, None] * (A - mu) / (std ** 2 + 1e-8)
            
            dW3   = np.clip(h2.T @ pg_g / max(len(S), 1), -1, 1)
            self._adam("Wa3", -dW3 * 5e-4, lr)
            self.log_std += lr * self.ent * 0.1
            
            # Critic update
            V, ch1 = self._critic(S)
            cvg = 2 * (V - ret)
            self._adam("Wc2", ch1.T @ cvg.reshape(-1, 1) / max(len(S), 1), lr)
            
        return float(ent), float(self.beta)

    # ── Main training loop ────────────────────────────────────────────────────
    def train(
        self,
        env: PipelineEnv,
        n_ep: int = 400,
        steps: int = 25,
        verbose: bool = True,
    ) -> None:
        for ep in range(1, n_ep + 1):
            # [P2 FIX] Drive curriculum phases by episode, not by step() calls.
            env.episode = ep
            lr_ep = max(self.lr_base * (0.5 + 0.5 * (1 - ep / n_ep)), 5e-5)

            s = env.reset()
            S, A, LP, R, V, VIOL = [], [], [], [], [], []
            ep_info = []; ep_r = 0

            for _ in range(steps):
                a, lp, _ = self.select_action(s)
                ns, r, info = env.step(a)
                v, _ = self._critic(self._norm(s).reshape(1, -1))
                S.append(self._norm(s)); A.append(a); LP.append(lp)
                R.append(r); V.append(v[0]); ep_info.append(info)
                VIOL.append(info["violation"])
                ep_r += r; s = ns

            adv, ret = self._gae(R, V, self.gamma, self.lam)
            
            # [P1.4] Dyna-Mix: Add synthetic rollouts if PINN is available
            if hasattr(env, 'pinn'):
                s_S, s_A, s_R, s_V = self.generate_pinn_rollouts(env.pinn, env, n_rollouts=100)
                # Compute advantage for synthetic rollouts (simplified)
                s_adv = np.array(s_R) - np.mean(s_R)
                S_aug = np.vstack([np.stack(S), s_S])
                A_aug = np.vstack([np.stack(A), s_A])
                R_aug = np.concatenate([ret, s_R])
                Adv_aug = np.concatenate([adv, s_adv])
                Viol_aug = np.concatenate([VIOL, s_V])
                LP_aug = np.concatenate([np.array(LP), np.zeros(len(s_S))]) # approx LP for synthetic
                
                ent, b_f = self._update(S_aug, A_aug, LP_aug, R_aug, Adv_aug, Viol_aug, lr_ep)
            else:
                ent, b_f = self._update(np.stack(S), np.stack(A), np.array(LP), ret, adv, VIOL, lr_ep)
            
            self.reward_hist.append(ep_r / steps)
            self.ent_hist.append(float(np.mean(np.exp(np.clip(self.log_std, -2, -0.5)))))
            self.csr_hist.append(float(np.mean([i["constraint_ok"] for i in ep_info])))

    def optimize_pareto(self, env: PipelineEnv, n_steps: int = 500):
        """
        [V7] Pareto Multi-Objective Optimization.
        Tracks the trade-off between multiple rewards (e.g., Stress vs Cost).
        """
        print("\n[V7] Starting Pareto Front Tracking...")
        pareto_front = []
        # Sample different safety weights (beta) to explore the frontier
        betas = np.linspace(0.1, 5.0, 10)
        for b in betas:
            self.beta = b
            self.train(env, n_ep=50, verbose=False) # Quick fine-tune
            s = env.reset()
            for _ in range(25):
                a, _, _ = self.select_action(s)
                s, _, info = env.step(a)
            pareto_front.append({
                "beta": b,
                "stress": info["sigma"],
                "cost": info["cost"],
                "efficiency": info["delta_P"]
            })
        return pareto_front

    def generate_pinn_rollouts(self, pinn, env: PipelineEnv, n_rollouts: int = 100) -> tuple:
        """
        [V7] Enhanced Dyna-style synthetic rollout generation.
        [P3.1] Risk-Averse Reward: Penalise states with high epistemic uncertainty.
        """
        states, actions, rewards, viols = [], [], [], []
        r_scale = env.BOUNDS[:, 1] - env.BOUNDS[:, 0]
        
        # Check if we have an ensemble for UQ
        is_ensemble = hasattr(pinn, 'sample_prediction')
        
        for _ in range(n_rollouts):
            s = env.reset()
            # Random perturb for exploration
            s = np.clip(s + np.random.randn(len(s)) * r_scale * 0.1, env.BOUNDS[:, 0], env.BOUNDS[:, 1])
            a, _, _ = self.select_action(s)
            
            # Synthetic step
            ns = np.clip(s + a * r_scale * 0.05, env.BOUNDS[:, 0], env.BOUNDS[:, 1])
            
            # PINN Inference (with UQ if available)
            if is_ensemble:
                out = pinn.sample_prediction(ns.reshape(1, -1))
                sigma = float(out["mean"][0, 0])
                dP    = float(out["mean"][0, 1])
                unc   = float(np.mean(out["std"])) # Average uncertainty
            else:
                out = pinn.predict(ns.reshape(1, -1))[0]
                sigma, dP = float(out[0]), float(out[1])
                unc = 0.0
                
            # Reward logic with risk-aversion
            viol = max(0.0, sigma - 200.0) / 200.0
            rw = -(0.3 * min(sigma/200, 1) + 0.3 * min(dP/100, 1))
            
            # [P3.1] Risk Penalty: -1.0 * Uncertainty
            rw -= 1.0 * unc 
            
            if viol > 0: rw -= 5.0 * viol
            
            states.append(self._norm(s)); actions.append(a); rewards.append(rw)
            viols.append(viol)
            
        return np.stack(states), np.stack(actions), rewards, viols


# Alias for backward compatibility with evaluation.py and benchmark.py
PPOAgent = LagrangianPPOAgent


class RandomAgent:
    """Baseline Random Agent picking actions uniformly."""
    def __init__(self, adim: int = 8):
        self.adim = adim
        self.reward_hist: list = []
        self.csr_hist: list = []

    def select_action(self, state: np.ndarray):
        a = np.random.uniform(-1, 1, self.adim)
        return a, 0, a
        
    def train(self, env: PipelineEnv, n_ep: int = 400, steps: int = 25, verbose: bool = True):
        self.reward_hist = []
        self.csr_hist = []
        for ep in range(1, n_ep + 1):
            env.episode = ep
            s = env.reset()
            ep_r = 0
            ep_info = []
            for _ in range(steps):
                a, _, _ = self.select_action(s)
                s, r, info = env.step(a)
                ep_r += r
                ep_info.append(info)
            self.reward_hist.append(ep_r / steps)
            self.csr_hist.append(float(np.mean([i["constraint_ok"] for i in ep_info])))


class GreedyAgent:
    """Baseline Greedy Agent picking pure cost minimizing actions in one shot."""
    def __init__(self, adim: int = 8):
        self.adim = adim
        self.reward_hist: list = []
        self.csr_hist: list = []

    def select_action(self, state: np.ndarray):
        a = np.zeros(self.adim)
        a[0] = -1.0 # Decrease D
        a[1] = -1.0 # Decrease T
        return a, 0, a
        
    def train(self, env, n_ep: int = 400, steps: int = 25, verbose: bool = True):
        from pinnflow.environment import PipelineEnv
        env: PipelineEnv = env
        self.reward_hist = []
        self.csr_hist = []
        for ep in range(1, n_ep + 1):
            env.episode = ep
            s = env.reset()
            ep_r = 0
            ep_info = []
            for _ in range(steps):
                a, _, _ = self.select_action(s)
                s, r, info = env.step(a)
                ep_r += r
                ep_info.append(info)
            self.reward_hist.append(ep_r / steps)
            self.csr_hist.append(float(np.mean([i["constraint_ok"] for i in ep_info])))


class EvolutionaryAgent:
    """Simple Cross-Entropy Method (CEM) agent for evolutionary baseline."""
    def __init__(self, adim: int = 8, pop_size: int = 50, elite_frac: float = 0.2):
        self.adim = adim
        self.pop_size = pop_size
        self.elite_frac = elite_frac
        self.mu = np.zeros(adim)
        self.std = np.ones(adim)
        self.reward_hist: list = []
        self.csr_hist: list = []
        
    def select_action(self, state: np.ndarray):
        a = np.clip(self.mu + self.std * np.random.randn(self.adim), -1, 1)
        return a, 0, self.mu
        
    def train(self, env, n_ep: int = 400, steps: int = 25, verbose: bool = True):
        from pinnflow.environment import PipelineEnv
        env: PipelineEnv = env
        self.reward_hist = []
        self.csr_hist = []
        num_elite = max(1, int(self.pop_size * self.elite_frac))
        
        for ep in range(1, n_ep + 1):
            env.episode = ep
            pop_rewards = []
            pop_actions = []
            pop_csrs = []
            
            for p in range(self.pop_size):
                s = env.reset()
                ep_r = 0
                a_seq = []
                info_seq = []
                for _ in range(steps):
                    a = np.clip(self.mu + self.std * np.random.randn(self.adim), -1, 1)
                    s, r, info = env.step(a)
                    ep_r += r
                    a_seq.append(a)
                    info_seq.append(info)
                pop_rewards.append(ep_r / steps)
                pop_actions.append(np.mean(a_seq, axis=0))
                pop_csrs.append(float(np.mean([i["constraint_ok"] for i in info_seq])))
                
            pop_rewards = np.array(pop_rewards)
            elite_idx = np.argsort(pop_rewards)[-num_elite:]
            elite_acts = np.array(pop_actions)[elite_idx]
            
            self.mu = np.mean(elite_acts, axis=0)
            self.std = np.std(elite_acts, axis=0) + 0.1
            
            self.reward_hist.append(np.mean(pop_rewards))
            self.csr_hist.append(np.mean(pop_csrs))
            
            if verbose and ep % 100 == 0:
                print(
                    f"  CEM Ep {ep:4d}/{n_ep} | "
                    f"reward={np.mean(self.reward_hist[-50:]):.4f} "
                    f"csr={np.mean(self.csr_hist[-50:]):.2%}"
                )

