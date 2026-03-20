"""
AWY-1: Physics-Based Ultrasonic Sensor Simulation
==================================================
Generates synthetic FIUS distance trajectories as specified in the exposé:

  Moving Away  : object starts at 0.5–2 m, moves to 3–5 m at 0.3–3 m/s
  Stationary   : object fixed at 0.5–5 m with measurement noise only
  Approaching  : object starts at 3–5 m, moves to 0.5–2 m at 0.3–3 m/s

Noise model:
  • Gaussian noise ±5–8% of true distance  (realistic ultrasonic uncertainty)
  • 1–2% random spike outliers              (ultrasonic dropouts / multi-path)

Zone definition (per exposé):
  Occupied : d < 3.0 m
  Clear    : d ≥ 3.0 m

Usage (standalone test):
    python simulation.py
"""

import numpy as np


ZONE_THRESHOLD   = 3.0    # metres — zone is clear when d ≥ this value
MEAS_RATE        = 38     # Hz  (FIUS measurement repetition rate)
TIME_PER_READING = 1.0 / MEAS_RATE   # ≈ 26.3 ms per reading


LABEL_APPROACHING = 0
LABEL_AWAY        = 1
LABEL_STATIONARY  = 2



def _add_noise(d_clean: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Add realistic sensor noise to a clean distance trajectory.

    1. Gaussian noise: std = uniform(5%, 8%) × mean distance
    2. Spike outliers: 1–2% of readings get a random large jump ±0.3–1.0 m
    """
    n         = len(d_clean)
    noise_std = rng.uniform(0.05, 0.08) * float(np.mean(d_clean))
    d         = d_clean + rng.normal(0.0, noise_std, n)

    # Spike outliers
    n_spikes = max(1, int(0.015 * n))
    idx      = rng.choice(n, n_spikes, replace=False)
    d[idx]  += rng.choice([-1, 1], n_spikes) * rng.uniform(0.3, 1.0, n_spikes)

    return np.clip(d, 0.05, 6.0)   # valid sensor range


# ── Trajectory generators ───────────────────────────────────────────────────
def generate_away(seed=None, speed=None, d_start=None, d_end=None):
    """Generate one Moving Away trajectory.

    Parameters
    ----------
    speed   : float, m/s  (default: random 0.3–2.0)
    d_start : float, m    (default: random 0.5–2.0)
    d_end   : float, m    (default: random 3.5–5.0)
    """
    rng     = np.random.default_rng(seed)
    d_start = d_start if d_start is not None else rng.uniform(0.5, 2.0)
    d_end   = d_end   if d_end   is not None else rng.uniform(3.5, 5.0)
    spd     = speed   if speed   is not None else rng.uniform(0.3, 2.0)

    n       = max(200, int((d_end - d_start) / spd * MEAS_RATE))
    d_clean = np.linspace(d_start, d_end, n)
    return _add_noise(d_clean, rng)


def generate_approaching(seed=None, speed=None, d_start=None, d_end=None):
    """Generate one Approaching trajectory."""
    rng     = np.random.default_rng(seed)
    d_start = d_start if d_start is not None else rng.uniform(3.5, 5.0)
    d_end   = d_end   if d_end   is not None else rng.uniform(0.5, 2.0)
    spd     = speed   if speed   is not None else rng.uniform(0.3, 2.0)

    n       = max(200, int((d_start - d_end) / spd * MEAS_RATE))
    d_clean = np.linspace(d_start, d_end, n)
    return _add_noise(d_clean, rng)


def generate_stationary(seed=None, d_fixed=None):
    """Generate one Stationary trajectory."""
    rng     = np.random.default_rng(seed)
    d_fixed = d_fixed if d_fixed is not None else rng.uniform(0.5, 5.0)
    n       = int(rng.integers(250, 550))
    d_clean = np.full(n, d_fixed, dtype=float)
    return _add_noise(d_clean, rng)


# ── Dataset builder ─────────────────────────────────────────────────────────
def generate_dataset(n_per_class: int = 25, seed: int = 42) -> dict:
    """Generate a balanced dataset for ML training/testing.

    Returns
    -------
    dict  {label (int) → list[ndarray]}
          Labels:  0=Approaching, 1=Away, 2=Stationary
    """
    rng   = np.random.default_rng(seed)
    seeds = rng.integers(0, 999_999, (3, n_per_class))

    return {
        LABEL_APPROACHING: [generate_approaching(seed=int(s)) for s in seeds[0]],
        LABEL_AWAY:        [generate_away(seed=int(s))        for s in seeds[1]],
        LABEL_STATIONARY:  [generate_stationary(seed=int(s))  for s in seeds[2]],
    }


# ── Standalone test ──────────────────────────────────────────────────────────
if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('AWY-1 Simulation — Sample Trajectories (one per class)',
                 fontweight='bold')

    for ax, (fn, label, color) in zip(axes, [
        (generate_approaching, 'Approaching', '#e74c3c'),
        (generate_away,        'Moving Away', '#2ecc71'),
        (generate_stationary,  'Stationary',  '#3498db'),
    ]):
        traj = fn(seed=7)
        t    = np.arange(len(traj)) * TIME_PER_READING
        ax.plot(t, traj, color=color, linewidth=1.2, alpha=0.9)
        ax.axhline(ZONE_THRESHOLD, color='orange', linestyle='--',
                   linewidth=1.8, label=f'Zone boundary ({ZONE_THRESHOLD} m)')
        ax.fill_between(t, 0, ZONE_THRESHOLD, alpha=0.07, color='red')
        ax.fill_between(t, ZONE_THRESHOLD, 6, alpha=0.05, color='green')
        ax.set_title(label, fontsize=12, fontweight='bold', color=color)
        ax.set_xlabel('Time (s)'); ax.set_ylabel('Distance (m)')
        ax.set_ylim(0, 6); ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig('results/figures/00_simulation_samples.png', dpi=150, bbox_inches='tight')
    print("Saved: results/figures/00_simulation_samples.png")
