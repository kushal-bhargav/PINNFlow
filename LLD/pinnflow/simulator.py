"""
pinnflow/simulator.py
──────────────────────
MODULE 2 — Physics Simulator
[N4] Physics-guided data augmentation: thin-wall / high-pressure gap-fill.

Synthetic FEM + CFD data generator.  All physics co-located so stress and
pressure drop share the same design vector.
"""
import numpy as np
import pandas as pd
from pinnflow.activations import sigmoid
from pinnflow.geometry.features import extract_curvature_features


class PhysicsSimulator:
    """
    Synthetic FEM + CFD data generator.

    Call ``generate(n)`` to produce a DataFrame with both structural and
    fluid outputs.  Augmented (thin-wall / high-pressure) samples are
    appended at the end — callers must shuffle before splitting.
    """

    RHO   = 1000.0
    E_MOD = 210e3
    ALPHA = 12e-6
    YIELD = 350.0

    DIAMETERS = [114.3, 168.3, 219.1, 273.0, 323.9, 406.4, 457.2, 508.0, 610.0]
    THICK_MAP = {
        114.3: 6.0, 168.3: 7.1, 219.1: 8.2, 273.0: 9.3,
        323.9: 10.3, 406.4: 12.7, 457.2: 14.3, 508.0: 15.9, 610.0: 17.5,
    }

    # ─── Internal FEM/CFD kernel ─────────────────────────────────────────────
    def _fem_cfd(
        self,
        d,
        t,
        L,
        P,
        u,
        dT,
        k_soil,
        augmented: bool = False,
        velocity: float | None = None,
    ) -> dict:
        ro = d / 2
        ri = max(ro - t, 1e-3)
        
        # [V5] Non-ideal gas Z-factor approximation (stochastic)
        Z = 1.0 - (P / 250.0) + np.random.normal(0, 0.05)
        P_eff = P / Z
        
        sig_hoop  = P_eff * ri ** 2 / (ro ** 2 - ri ** 2 + 1e-6) * (1 + ro ** 2 / (ri ** 2 + 1e-6))
        sig_soil  = k_soil * u * (d / 400)
        sig_therm = self.E_MOD * self.ALPHA * dT
        sig_bend  = 0.1 * P_eff * L * d / (8 * max(t ** 2, 1e-6))
        
        # Non-linear coupling: thermal stress depends on Pressure
        sig_therm *= (1.0 + 0.1 * np.sin(P/5))
        
        sv = (
            np.sqrt(sig_hoop ** 2 + sig_bend ** 2 - sig_hoop * sig_bend + 3 * sig_soil ** 2)
            + abs(sig_therm)
        )
        
        # [V5] Plasticity: Ramberg-Osgood approximation for high stress
        if sv > self.YIELD * 0.7:
            sv = self.YIELD * (sv / self.YIELD + 0.02 * (sv / self.YIELD)**10)
            
        noise_scale = max(sv * (0.02 if augmented else 0.04), 1.0)
        sv += np.random.normal(0, noise_scale)
        sv  = max(sv, 5.0)

        v  = float(velocity) if velocity is not None else np.random.uniform(0.5, 9.0)
        mu = np.random.uniform(8e-4, 1.5e-3)
        Re = self.RHO * v * (d / 1000) / mu
        
        # [V5] Advanced friction factor with stochastic roughness
        rough = 0.015 * (1.0 + 0.2 * np.random.randn())
        fd = 0.316 * Re ** -0.25 if Re > 4000 else 64 / max(Re, 1)
        fd *= (1.0 + rough)
        
        dP = fd * (L / (d / 1000)) * self.RHO * v ** 2 / 2 / 1000 * (1 + 0.1 * sv / 300)
        dP = max(dP + np.random.normal(0, dP * 0.05), 0.1)

        return dict(
            diameter=d, thickness=t, length=L, pressure=P,
            soil_disp=u, delta_T=dT,
            velocity=round(v, 3), soil_stiffness=round(k_soil, 3),
            von_mises_stress=round(sv, 3),
            pressure_drop_kPa=round(dP, 4),
            failure_prob=round(float(np.clip(sigmoid((sv - self.YIELD * 0.85) / 30), 1e-3, 0.999)), 5),
            fatigue_life=round(max(1e3, 1e8 * (180 / max(sv, 1)) ** 3.5), 0),
        )

    # ─── Public API ──────────────────────────────────────────────────────────
    def generate(self, n: int = 1200) -> pd.DataFrame:
        """
        Generate *n* base samples + n//4 augmented (thin-wall/high-pressure)
        samples.  Callers must shuffle rows before train/test splitting so
        augmented samples reach the training set.
        """
        rows = []
        # Base samples
        for _ in range(n):
            d = np.random.choice(self.DIAMETERS)
            t = max(self.THICK_MAP[d] + np.random.uniform(-1.0, 1.5), 4.0)
            rows.append(self._fem_cfd(
                d, t,
                np.random.uniform(5, 150),
                np.random.uniform(1.5, 20),
                np.random.uniform(0, 150),
                np.random.uniform(-40, 80),
                np.random.uniform(0.3, 0.8),
            ))
        # [N4] Gap-fill: thin-wall + high-pressure regime
        for _ in range(n // 4):
            d = np.random.choice([406.4, 457.2, 508.0, 610.0])
            t = self.THICK_MAP[d] * np.random.uniform(0.6, 0.85)
            rows.append(self._fem_cfd(
                d, t,
                np.random.uniform(50, 150),
                np.random.uniform(12, 20),
                np.random.uniform(80, 150),
                np.random.uniform(40, 80),
                np.random.uniform(0.3, 0.8),
                augmented=True,
            ))
        df = pd.DataFrame(rows)
        print(
            f"  Dataset: {len(df)} samples | "
            f"sigma_vm in [{df.von_mises_stress.min():.1f}, {df.von_mises_stress.max():.1f}] MPa | "
            f"dP in [{df.pressure_drop_kPa.min():.2f}, {df.pressure_drop_kPa.max():.2f}] kPa"
        )
        return df

    def generate_one(self, d, t, L, P, u, dT, k=None, velocity: float | None = None) -> dict:
        """Single sample — used by FEM baseline timing and fair e2e eval."""
        if k is None:
            k = np.random.uniform(0.3, 0.8)
        return self._fem_cfd(d, t, L, P, u, dT, k, velocity=velocity)

    def generate_elbow_batch(self, n: int = 1600) -> pd.DataFrame:
        """Generate elbow-focused samples with varying bend radius and angle."""
        rows = []
        angles = np.array([30.0, 45.0, 60.0, 90.0])
        for _ in range(n):
            d = float(np.random.choice(self.DIAMETERS))
            t_over_d = np.random.uniform(0.02, 0.08)
            t = max(d * t_over_d, 4.0)
            r_over_d = float(np.random.uniform(1.0, 5.0))
            angle = float(np.random.choice(angles))
            v = float(np.random.uniform(0.5, 12.0))
            L = max((np.deg2rad(angle) * r_over_d * d / 1000.0), 1.0)
            base = self._fem_cfd(
                d,
                t,
                L,
                np.random.uniform(1.5, 20.0),
                np.random.uniform(0.0, 150.0),
                np.random.uniform(-40.0, 80.0),
                np.random.uniform(0.3, 0.8),
                augmented=True,
                velocity=v,
            )
            X = np.array(
                [[base["diameter"], base["thickness"], base["length"], base["pressure"],
                  base["soil_disp"], base["delta_T"], base["velocity"], base["soil_stiffness"],
                  1.0, r_over_d]],
                dtype=float,
            )
            curve = extract_curvature_features(X)[0]
            dean_factor = 1.0 + 0.033 * np.log10(max(curve[-1], 1.0)) ** 2
            scf = 0.9 / max(t_over_d, 1e-4) ** (2.0 / 3.0) * r_over_d ** (1.0 / 3.0)
            turbulence = np.random.lognormal(mean=np.log(0.06), sigma=0.35)
            mesh_factor = np.random.lognormal(mean=0.0, sigma=0.2)
            stress_multiplier = np.clip(0.35 * scf * dean_factor, 1.05, 4.0)
            base["von_mises_stress"] = round(max(base["von_mises_stress"] * stress_multiplier, 5.0), 3)
            base["pressure_drop_kPa"] = round(max(base["pressure_drop_kPa"] * (1.0 + 0.25 / r_over_d + turbulence), 0.1), 4)
            base.update({
                "shape_id": 1.0,
                "shape_param": r_over_d,
                "elbow_angle_deg": angle,
                "turbulence_intensity": round(float(turbulence), 5),
                "mesh_resolution_factor": round(float(mesh_factor), 4),
                "bend_radius": curve[0],
                "curvature_ratio": curve[1],
                "sin_elbow_angle": curve[2],
                "cos_elbow_angle": curve[3],
                "thickness_ratio": curve[4],
                "dean_number": curve[5],
            })
            rows.append(base)
        return pd.DataFrame(rows)
