"""
Particle Filter Post-Clustering Estimator for GeoFusion.

For each cluster, runs a Sequential Importance Resampling (SIR) particle
filter over all member observations to produce a refined position estimate.

--- State and motion model ---

State        : x = [N, E]  (local metres relative to cluster centroid)
Transition   : x_{t+1} = x_t + w,   w ~ N(0, Q)   (static target)
Likelihood   : p(z_t | x_t) = N(z_t; x_t, R_t)
               where R_t = avg_rawPrUnc_t^2 * I_2  (metres^2)
Init particles: drawn from N([0,0], P_0),  P_0 = hDop_first^2 * I_2

Resampling   : Systematic resampling when effective sample size
               N_eff = 1 / sum(w^2) drops below n_particles / 2.

The final estimate is the weighted mean of particles, converted back to
degrees via the WGS84 factors at the cluster centroid.

--- Interface ---

Input  : clustering output CSV — must contain predicted_cluster,
         latDeg_phone, lngDeg_phone, avg_rawPrUnc, hDop, epoch_unix_s.
Output : same CSV + column `predicted_location_pf` as (lat, lon) tuples.

--- Notebook usage ---

    from particle_filter_estimator import run_pf
    df_out = run_pf("kmeans_output.csv", "pf_output.csv", n_particles=500)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_REQUIRED = ["predicted_cluster", "latDeg_phone", "lngDeg_phone",
             "avg_rawPrUnc", "hDop", "epoch_unix_s"]


# ---------------------------------------------------------------------------
# WGS84 helpers (shared with EKF)
# ---------------------------------------------------------------------------

def _meters_per_degree(lat_deg: float) -> tuple[float, float]:
    phi   = np.radians(lat_deg)
    m_lat = (111132.92 - 559.82 * np.cos(2 * phi)
             + 1.175 * np.cos(4 * phi) - 0.0023 * np.cos(6 * phi))
    m_lon = (111412.84 * np.cos(phi)
             - 93.5 * np.cos(3 * phi) + 0.118 * np.cos(5 * phi))
    return float(m_lat), float(m_lon)


def _to_local_meters(
    lats: np.ndarray, lons: np.ndarray,
    ref_lat: float, ref_lon: float,
) -> tuple[np.ndarray, np.ndarray]:
    m_lat, m_lon = _meters_per_degree(ref_lat)
    return (lats - ref_lat) * m_lat, (lons - ref_lon) * m_lon


def _to_degrees(
    N_m: float, E_m: float,
    ref_lat: float, ref_lon: float,
) -> tuple[float, float]:
    m_lat, m_lon = _meters_per_degree(ref_lat)
    return float(ref_lat + N_m / m_lat), float(ref_lon + E_m / m_lon)


# ---------------------------------------------------------------------------
# Systematic resampling
# ---------------------------------------------------------------------------

def _systematic_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Systematic resampling (Kitagawa 1996).
    More deterministic than multinomial resampling: reduces variance while
    preserving the same expected counts. Returns indices of selected particles.
    """
    n  = len(weights)
    cs = np.cumsum(weights)
    cs[-1] = 1.0                          # guard against floating-point drift
    u  = (rng.random() + np.arange(n)) / n
    return np.searchsorted(cs, u)


# ---------------------------------------------------------------------------
# Core particle filter over one cluster
# ---------------------------------------------------------------------------

def _pf_cluster(
    N_obs: np.ndarray,
    E_obs: np.ndarray,
    pr_uncs: np.ndarray,
    init_hdop: float,
    Q_var: float,
    n_particles: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """
    SIR particle filter over one cluster in local metre coordinates.

    Parameters
    ----------
    N_obs, E_obs : (n,) observations in local metres (temporal order)
    pr_uncs      : (n,) avg_rawPrUnc values (metres, measurement noise std)
    init_hdop    : hDop of first member (initialises particle spread, metres)
    Q_var        : process noise variance (metres^2)
    n_particles  : number of particles
    rng          : numpy Generator for reproducibility

    Returns
    -------
    (N_est, E_est) weighted mean position in local metres
    """
    # --- initialise particles around origin (cluster centroid) ---
    particles = rng.normal(
        loc=0.0,
        scale=init_hdop,
        size=(n_particles, 2),
    )                                                 # (P, 2)
    weights = np.full(n_particles, 1.0 / n_particles)

    Q_std = np.sqrt(Q_var)

    for N_z, E_z, pr_unc in zip(N_obs, E_obs, pr_uncs):
        # --- propagate: static model + process noise ---
        particles += rng.normal(0.0, Q_std, size=(n_particles, 2))

        # --- weight: Gaussian likelihood p(z|x) ---
        sigma = float(pr_unc)                        # measurement noise std (m)
        diff  = particles - np.array([N_z, E_z])     # (P, 2)
        # log-likelihood: -0.5 * ||diff||^2 / sigma^2  (isotropic R = sigma^2 I)
        log_w = -0.5 * np.sum(diff ** 2, axis=1) / (sigma ** 2)
        log_w -= log_w.max()                         # numerical stability
        weights = np.exp(log_w)
        weights /= weights.sum()

        # --- resample if N_eff is too low ---
        n_eff = 1.0 / np.sum(weights ** 2)
        if n_eff < n_particles / 2:
            idx       = _systematic_resample(weights, rng)
            particles = particles[idx]
            weights   = np.full(n_particles, 1.0 / n_particles)

    # --- final estimate: weighted mean ---
    N_est = float(np.sum(weights * particles[:, 0]))
    E_est = float(np.sum(weights * particles[:, 1]))
    return N_est, E_est


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def run_pf(
    input_path: str,
    output_path: str,
    Q_var: float = 1.0,
    n_particles: int = 500,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Apply a Particle Filter to each cluster in a GeoFusion clustering output.

    Filtering is performed in local (North, East) metre coordinates centred
    on each cluster's mean phone position. Measurement noise is driven
    directly by avg_rawPrUnc (metres), and initial particle spread by hDop.

    Parameters
    ----------
    input_path : str
        Path to clustering output CSV.
    output_path : str
        Path to write result CSV.
    Q_var : float
        Process noise variance (metres^2). Default 1.0 m^2.
    n_particles : int
        Number of particles. Default 500; increase for smoother estimates
        at higher computational cost.
    random_state : int
        Seed for reproducibility. Default 42.

    Returns
    -------
    pd.DataFrame
        Original dataframe + `predicted_location_pf` column.
    """
    df = pd.read_csv(input_path)
    missing = [c for c in _REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    rng = np.random.default_rng(random_state)

    pf_lat = np.empty(len(df))
    pf_lon = np.empty(len(df))

    for cluster_id, grp in df.groupby("predicted_cluster", sort=False):
        grp_sorted = grp.sort_values("epoch_unix_s")

        lats    = grp_sorted["latDeg_phone"].to_numpy(dtype=float)
        lons    = grp_sorted["lngDeg_phone"].to_numpy(dtype=float)
        pr_uncs = grp_sorted["avg_rawPrUnc"].to_numpy(dtype=float)

        ref_lat  = float(lats.mean())
        ref_lon  = float(lons.mean())
        N_obs, E_obs = _to_local_meters(lats, lons, ref_lat, ref_lon)
        init_hdop    = float(grp_sorted["hDop"].iloc[0])

        N_est, E_est = _pf_cluster(
            N_obs, E_obs, pr_uncs,
            init_hdop=init_hdop,
            Q_var=Q_var,
            n_particles=n_particles,
            rng=rng,
        )

        lat_est, lon_est = _to_degrees(N_est, E_est, ref_lat, ref_lon)
        pf_lat[grp.index] = lat_est
        pf_lon[grp.index] = lon_est

    df["predicted_location_pf"] = list(zip(pf_lat.tolist(), pf_lon.tolist()))
    df.to_csv(output_path, index=False)
    print(f"Particle filter estimates written to {output_path}")
    return df
