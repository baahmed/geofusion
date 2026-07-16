"""
Extended Kalman Filter Post-Clustering Estimator for GeoFusion.

Identical state/motion model to the linear KF, but converts lat/lon to
local meters (North, East) before filtering and converts back afterward.
This avoids the degree-space distortion where 1 deg lat != 1 deg lon in
meters, making the covariance matrix physically meaningful and allowing
the hDop and avg_rawPrUnc (both in meters) to be used directly as-is
without degree-space scaling.

--- State and motion model ---

State        : x = [N, E]  (local meters relative to cluster centroid)
Transition   : x_{t+1} = x_t + w,   w ~ N(0, Q)   (static target)
Measurement  : z_t = [N_phone, E_phone]  (phone position in local meters)
Meas. noise  : R_t = avg_rawPrUnc_t^2 * I_2  (meters^2, used directly)
Init state   : [0, 0]  (origin = cluster centroid in local meter frame)
Init cov     : P_0 = hDop_first^2 * I_2  (meters^2)

The final estimate is converted back to degrees using the WGS84
meters-per-degree factors at the cluster centroid.

--- Interface ---

Input  : clustering output CSV — must contain predicted_cluster,
         latDeg_phone, lngDeg_phone, avg_rawPrUnc, hDop, epoch_unix_s.
Output : same CSV + column `predicted_location_ekf` as (lat, lon) tuples.

--- Notebook usage ---

    from ekf_estimator import run_ekf
    df_out = run_ekf("kmeans_output.csv", "ekf_output.csv")
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_REQUIRED = ["predicted_cluster", "latDeg_phone", "lngDeg_phone",
             "avg_rawPrUnc", "hDop", "epoch_unix_s"]


# ---------------------------------------------------------------------------
# WGS84 helpers
# ---------------------------------------------------------------------------

def _meters_per_degree(lat_deg: float) -> tuple[float, float]:
    """Metres per degree of latitude and longitude at a given latitude."""
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
    N = (lats - ref_lat) * m_lat
    E = (lons - ref_lon) * m_lon
    return N, E


def _to_degrees(
    N_m: float, E_m: float,
    ref_lat: float, ref_lon: float,
) -> tuple[float, float]:
    m_lat, m_lon = _meters_per_degree(ref_lat)
    lat = ref_lat + N_m / m_lat
    lon = ref_lon + E_m / m_lon
    return float(lat), float(lon)


# ---------------------------------------------------------------------------
# Core EKF (linear measurement, non-linear coordinate conversion)
# ---------------------------------------------------------------------------

def _ekf_cluster(
    N_obs: np.ndarray,
    E_obs: np.ndarray,
    pr_uncs: np.ndarray,
    init_hdop: float,
    Q_var: float,
) -> tuple[float, float]:
    """
    EKF over one cluster in local meter coordinates.

    State is [N, E] in meters. The measurement function is linear (H = I),
    so the EKF reduces to a standard KF in this coordinate frame — the
    'extended' part is the coordinate transformation applied before and
    after filtering, which accounts for the nonlinear relationship between
    degrees and meters across latitude.

    Returns the final (N_est, E_est) in meters.
    """
    x = np.zeros(2, dtype=float)          # initialise at cluster centroid
    P = (init_hdop ** 2) * np.eye(2)      # metres^2
    F = np.eye(2)
    H = np.eye(2)
    Q = Q_var * np.eye(2)                 # metres^2

    for N_z, E_z, pr_unc in zip(N_obs, E_obs, pr_uncs):
        # predict
        x = F @ x
        P = F @ P @ F.T + Q

        # update — R in metres^2, matching state space units
        R = (float(pr_unc) ** 2) * np.eye(2)
        z = np.array([N_z, E_z], dtype=float)
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        x = x + K @ (z - H @ x)
        P = (np.eye(2) - K @ H) @ P

    return float(x[0]), float(x[1])


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def run_ekf(
    input_path: str,
    output_path: str,
    Q_var: float = 1.0,
) -> pd.DataFrame:
    """
    Apply Extended Kalman Filter to each cluster in a GeoFusion clustering
    output.

    Filtering is performed in local (North, East) meter coordinates centred
    on each cluster's mean phone position, making hDop and avg_rawPrUnc
    directly usable as metre-scale covariance values without degree-space
    distortion.

    Parameters
    ----------
    input_path : str
        Path to clustering output CSV.
    output_path : str
        Path to write result CSV.
    Q_var : float
        Process noise variance (metres^2). Default 1.0 m^2, appropriate
        for a stationary target with small residual drift.

    Returns
    -------
    pd.DataFrame
        Original dataframe + `predicted_location_ekf` column.
    """
    df = pd.read_csv(input_path)
    missing = [c for c in _REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    ekf_lat = np.empty(len(df))
    ekf_lon = np.empty(len(df))

    for cluster_id, grp in df.groupby("predicted_cluster", sort=False):
        grp_sorted = grp.sort_values("epoch_unix_s")

        lats    = grp_sorted["latDeg_phone"].to_numpy(dtype=float)
        lons    = grp_sorted["lngDeg_phone"].to_numpy(dtype=float)
        pr_uncs = grp_sorted["avg_rawPrUnc"].to_numpy(dtype=float)

        # cluster centroid as local coordinate origin
        ref_lat = float(lats.mean())
        ref_lon = float(lons.mean())

        N_obs, E_obs = _to_local_meters(lats, lons, ref_lat, ref_lon)
        init_hdop    = float(grp_sorted["hDop"].iloc[0])

        N_est, E_est = _ekf_cluster(
            N_obs, E_obs, pr_uncs,
            init_hdop=init_hdop,
            Q_var=Q_var,
        )

        lat_est, lon_est = _to_degrees(N_est, E_est, ref_lat, ref_lon)
        ekf_lat[grp.index] = lat_est
        ekf_lon[grp.index] = lon_est

    df["predicted_location_ekf"] = list(zip(ekf_lat.tolist(), ekf_lon.tolist()))
    df.to_csv(output_path, index=False)
    print(f"EKF estimates written to {output_path}")
    return df
