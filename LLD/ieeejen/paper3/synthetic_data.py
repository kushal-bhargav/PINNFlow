"""
Synthetic dataset generation for the dynamic load estimation inverse problem.

Pipeline:
  1. Define known ground-truth load f_true(x,t)
  2. Solve the forward Euler-Bernoulli PDE via modal superposition (exact) or FDM
  3. Sample the response field at N_s sparse sensor locations
  4. Add Gaussian noise at specified SNR levels
  5. Return sensor data as training observations for the PINN

References:
    Patel et al. (2025), PINNs for Dynamic Load Estimation in Pipework
"""

import numpy as np
import torch
from typing import Dict, Tuple, Optional, List, Callable


# ─────────────────────────────────────────────────────────────────────────────
# Ground-truth load profiles
# ─────────────────────────────────────────────────────────────────────────────

def single_harmonic_load(
    x: np.ndarray,
    t: np.ndarray,
    F0: float = 1000.0,
    omega: float = 20.0,
    x0: float = 1.0,
    width: float = 0.1,
) -> np.ndarray:
    """
    Localised harmonic point load: F₀ * sin(ωt) * Gaussian_x(x₀, width).

    Args:
        x, t    : spatial/temporal arrays (meshgrid-style, shape (N_x,) and (N_t,))
        F0      : load amplitude (N/m)
        omega   : angular frequency (rad/s)
        x0      : spatial location of load (m)
        width   : Gaussian spatial width (m)

    Returns:
        f : (N_x, N_t) load field
    """
    X, T   = np.meshgrid(x, t, indexing="ij")
    spatial = np.exp(-0.5 * ((X - x0) / width) ** 2)
    temporal= np.sin(omega * T)
    return F0 * spatial * temporal


def multi_harmonic_load(
    x: np.ndarray,
    t: np.ndarray,
    harmonics: List[Dict],
) -> np.ndarray:
    """
    Sum of multiple harmonic loads at different locations and frequencies.

    Args:
        harmonics : list of dicts, each with keys {F0, omega, x0, width}

    Returns:
        f : (N_x, N_t) combined load field
    """
    f = np.zeros((len(x), len(t)))
    for h in harmonics:
        f += single_harmonic_load(x, t, **h)
    return f


def broadband_random_load(
    x: np.ndarray,
    t: np.ndarray,
    F_rms: float = 500.0,
    f_low: float = 5.0,
    f_high: float = 100.0,
    seed: int = 0,
) -> np.ndarray:
    """
    Broadband random load with flat PSD between f_low and f_high Hz.

    Uses inverse FFT of filtered white noise.

    Args:
        F_rms  : RMS amplitude (N/m)
        f_low  : lower frequency bound (Hz)
        f_high : upper frequency bound (Hz)
        seed   : random seed

    Returns:
        f : (N_x, N_t) load field
    """
    rng = np.random.default_rng(seed)
    N_t = len(t)
    N_x = len(x)
    dt  = t[1] - t[0]
    freqs = np.fft.rfftfreq(N_t, d=dt)

    # Spatial mode shapes (first few sinusoidal modes)
    L = x[-1]
    f_field = np.zeros((N_x, N_t))
    for mode in range(1, 4):
        phi_x = np.sin(mode * np.pi * x / L)           # (N_x,)
        # Random phase signal with bandpass filter
        noise     = rng.standard_normal(N_t)
        noise_fft = np.fft.rfft(noise)
        bandpass  = ((freqs >= f_low) & (freqs <= f_high)).astype(float)
        filtered  = np.fft.irfft(noise_fft * bandpass, n=N_t)
        # Normalise to unit RMS
        filtered /= (filtered.std() + 1e-12)
        f_field += np.outer(phi_x, filtered) * (F_rms / 3.0)

    return f_field


# ─────────────────────────────────────────────────────────────────────────────
# Forward solver: modal superposition (Euler-Bernoulli, simply supported)
# ─────────────────────────────────────────────────────────────────────────────

def modal_superposition_solver(
    x: np.ndarray,
    t: np.ndarray,
    f_field: np.ndarray,
    E: float,
    I: float,
    rho: float,
    A: float,
    L: float,
    n_modes: int = 20,
    zeta: float = 0.01,
) -> np.ndarray:
    """
    Solve Euler-Bernoulli beam response via modal superposition (simply supported BC).

    Natural frequencies: ω_n = (nπ/L)² * sqrt(EI / ρA)
    Mode shapes:        φ_n(x) = sin(nπx/L)

    Args:
        x, t     : spatial (N_x,) and temporal (N_t,) arrays
        f_field  : (N_x, N_t) forcing field
        E, I     : Young's modulus and second moment of area
        rho, A   : density and cross-section area
        L        : pipe length (m)
        n_modes  : number of modes included
        zeta     : modal damping ratio

    Returns:
        w : (N_x, N_t) displacement field
    """
    dt = t[1] - t[0]
    N_t = len(t)
    N_x = len(x)

    EI   = E * I
    rhoA = rho * A
    w_resp = np.zeros((N_x, N_t))

    for n in range(1, n_modes + 1):
        # Natural frequency
        kn   = n * np.pi / L
        wn   = kn ** 2 * np.sqrt(EI / rhoA)   # rad/s
        wd   = wn * np.sqrt(max(1 - zeta**2, 1e-10))

        # Mode shape
        phi_n = np.sin(kn * x)                 # (N_x,)

        # Generalised force: F_n(t) = ∫₀ᴸ f(x,t)*φ_n(x) dx
        F_n = np.trapz(f_field * phi_n[:, None], x, axis=0)   # (N_t,)

        # Modal mass (normalised mode shapes)
        M_n = rhoA * L / 2.0

        # Duhamel integral (convolution with impulse response)
        q_n = np.zeros(N_t)
        h_conv = np.zeros(N_t)
        for k in range(N_t):
            tau = t[k]
            # Impulse response function (causal)
            h_conv[k] = (1.0 / (M_n * wd)) * np.exp(-zeta * wn * tau) * np.sin(wd * tau)

        # Convolution via FFT
        q_n = np.convolve(F_n, h_conv)[:N_t] * dt

        w_resp += np.outer(phi_n, q_n)

    return w_resp


# ─────────────────────────────────────────────────────────────────────────────
# Sensor placement and noise addition
# ─────────────────────────────────────────────────────────────────────────────

def place_sensors(
    x: np.ndarray,
    n_sensors: int,
    strategy: str = "uniform",
    seed: int = 0,
) -> np.ndarray:
    """
    Select sensor locations along the pipe.

    Args:
        x         : spatial array (N_x,)
        n_sensors : number of sensors
        strategy  : 'uniform' | 'random' | 'optimal' (ends + midpoints)

    Returns:
        sensor_x_indices : array of integer indices into x
    """
    rng = np.random.default_rng(seed)
    if strategy == "uniform":
        return np.linspace(0, len(x)-1, n_sensors, dtype=int)
    elif strategy == "random":
        return rng.choice(len(x), size=n_sensors, replace=False)
    elif strategy == "optimal":
        # Ends + evenly spaced interior sensors
        interior = np.linspace(1, len(x)-2, n_sensors-2, dtype=int)
        return np.array([0] + list(interior) + [len(x)-1])
    else:
        raise ValueError(f"Unknown strategy '{strategy}'")


def add_gaussian_noise(
    signal: np.ndarray,
    snr_db: float,
    seed: int = 0,
) -> np.ndarray:
    """
    Add Gaussian white noise to a signal at specified SNR level.

    Args:
        signal : clean signal array
        snr_db : signal-to-noise ratio in dB (np.inf = noiseless)
        seed   : random seed

    Returns:
        noisy signal
    """
    if np.isinf(snr_db):
        return signal.copy()
    rng = np.random.default_rng(seed)
    signal_power = np.mean(signal ** 2)
    noise_power  = signal_power / (10 ** (snr_db / 10.0))
    noise        = rng.normal(0, np.sqrt(noise_power), signal.shape)
    return signal + noise


# ─────────────────────────────────────────────────────────────────────────────
# Full synthetic dataset generator
# ─────────────────────────────────────────────────────────────────────────────

class SyntheticDataset:
    """
    Generate complete synthetic training/test datasets for the inverse problem.

    Attributes:
        x_grid, t_grid : spatial and temporal discretisation
        w_true         : (N_x, N_t) ground-truth displacement field
        f_true         : (N_x, N_t) ground-truth load field
        X_sensor       : (N_s * N_t, 2) sensor point coordinates [x, t]
        w_sensor_noisy : (N_s * N_t,) noisy sensor measurements
    """

    def __init__(
        self,
        L: float = 5.0,
        T: float = 1.0,
        N_x: int = 200,
        N_t: int = 500,
        E: float = 207e9,
        I: float = 1.7e-5,
        rho: float = 7850.0,
        A: float = 3.6e-3,
        n_sensors: int = 5,
        sensor_strategy: str = "uniform",
        snr_db: float = 20.0,
        load_type: str = "single_harmonic",
        load_kwargs: Optional[Dict] = None,
        n_modes: int = 20,
        seed: int = 0,
    ):
        self.params = dict(L=L, T=T, E=E, I=I, rho=rho, A=A)

        x_grid = np.linspace(0.0, L,  N_x)
        t_grid = np.linspace(0.0, T,  N_t)
        self.x_grid = x_grid
        self.t_grid = t_grid

        # Generate ground-truth load
        kw = load_kwargs or {}
        if load_type == "single_harmonic":
            kw.setdefault("F0", 1000.0); kw.setdefault("omega", 20.0)
            kw.setdefault("x0", L / 2); kw.setdefault("width", 0.05 * L)
            self.f_true = single_harmonic_load(x_grid, t_grid, **kw)
        elif load_type == "broadband":
            self.f_true = broadband_random_load(x_grid, t_grid, **kw)
        elif load_type == "multi_harmonic":
            self.f_true = multi_harmonic_load(x_grid, t_grid, **kw)
        else:
            raise ValueError(f"Unknown load_type '{load_type}'")

        # Solve forward problem
        self.w_true = modal_superposition_solver(
            x_grid, t_grid, self.f_true, E, I, rho, A, L, n_modes=n_modes
        )

        # Place sensors
        sensor_idx = place_sensors(x_grid, n_sensors, sensor_strategy, seed=seed)
        self.sensor_x_indices = sensor_idx
        self.sensor_x = x_grid[sensor_idx]

        # Extract sensor measurements + noise
        X_s, T_s = np.meshgrid(self.sensor_x, t_grid, indexing="ij")  # (n_s, N_t)
        w_s_clean = self.w_true[sensor_idx, :]                          # (n_s, N_t)
        w_s_noisy = add_gaussian_noise(w_s_clean, snr_db, seed=seed+1)

        self.X_sensor = np.column_stack([X_s.ravel().astype(np.float32),
                                          T_s.ravel().astype(np.float32)])
        self.w_sensor_noisy = w_s_noisy.ravel().astype(np.float32)

    def to_tensors(self, device: str = "cpu") -> Dict[str, torch.Tensor]:
        """Convert all arrays to PyTorch tensors."""
        X_s = torch.tensor(self.X_sensor, device=device)
        w_s = torch.tensor(self.w_sensor_noisy, device=device)

        # Collocation: random (x,t) in domain
        N_col = 10_000
        rng = np.random.default_rng(0)
        xc  = rng.uniform(0, self.params["L"], N_col).astype(np.float32)
        tc  = rng.uniform(0, self.params["T"], N_col).astype(np.float32)
        X_col = torch.tensor(np.column_stack([xc, tc]), device=device).requires_grad_(True)

        # IC points (t = 0)
        x_ic = self.x_grid.astype(np.float32)
        t_ic = np.zeros_like(x_ic)
        X_ic = torch.tensor(np.column_stack([x_ic, t_ic]), device=device).requires_grad_(True)
        w0   = torch.zeros(len(x_ic), device=device)
        wd0  = torch.zeros(len(x_ic), device=device)

        # BC points (x=0, x=L)
        t_bc  = np.linspace(0, self.params["T"], 200, dtype=np.float32)
        x0_bc = np.zeros_like(t_bc)
        xL_bc = np.full_like(t_bc, self.params["L"])
        X_bc  = torch.tensor(
            np.column_stack([np.hstack([x0_bc, xL_bc]),
                              np.hstack([t_bc,   t_bc])]),
            device=device
        ).requires_grad_(True)

        return {
            "X_col":    X_col,
            "X_sensor": X_s,
            "w_sensor": w_s,
            "X_ic":     X_ic,
            "w0":       w0,
            "wdot0":    wd0,
            "X_bc":     X_bc,
        }
