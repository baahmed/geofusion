"""
Kalman Filter Post-Clustering Estimator for GeoFusion.

For each cluster (identified by `predicted_cluster`), runs a linear Kalman
filter over all member observations to produce a refined position estimate.

--- State and motion model ---

State        : x = [lat, lon]  (degrees)
Transition   : x_{t+1} = x_t + w,   w ~ N(0, Q)   (static target)
Measurement  : z_t = [latDeg_phone, lngDeg_phone]
Meas. noise  : R_t = avg_rawPrUnc_t^2 * I_2
Init state   : cluster representative mean(latDeg_phone), mean(lngDeg_phone)
Init cov     : P_0 = hDop_first^2 * I_2

--- Interface ---

Input  : clustering output CSV — must contain predicted_cluster,
         latDeg_phone, lngDeg_phone, avg_rawPrUnc, hDop, epoch_unix_s,
         latDeg_gt, lngDeg_gt, collectionName.
Output : same CSV + column `predicted_location_kf` as (lat, lon) tuples.

--- Notebook usage ---

    from kalman_estimator import run_kalman
    df_out = run_kalman("kmeans_output.csv", "kalman_output.csv")
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_REQUIRED = ["predicted_cluster", "latDeg_phone", "lngDeg_phone",
             "avg_rawPrUnc", "hDop", "epoch_unix_s"]


def _kalman_cluster(
    lats: np.ndarray,
    lons: np.ndarray,
    pr_uncs: np.ndarray,
    init_lat: float,
    init_lon: float,
    init_hdop: float,
    Q_var: float,
) -> tuple[float, float]:
    """
    Linear Kalman filter over one cluster's observations.

    Observations are processed in temporal order (sorted by epoch_unix_s
    before this function is called). Returns the final state estimate.
    """
    x = np.array([init_lat, init_lon], dtype=float)
    P = (init_hdop ** 2) * np.eye(2)
    F = np.eye(2)          # static: no motion
    H = np.eye(2)          # direct position measurement
    Q = Q_var * np.eye(2)

    for lat_z, lon_z, pr_unc in zip(lats, lons, pr_uncs):
        # predict
        x = F @ x
        P = F @ P @ F.T + Q

        # update
        R = (float(pr_unc) ** 2) * np.eye(2)
        z = np.array([lat_z, lon_z], dtype=float)
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        x = x + K @ (z - H @ x)
        P = (np.eye(2) - K @ H) @ P

    return float(x[0]), float(x[1])


def run_kalman(
    input_path: str,
    output_path: str,
    Q_var: float = 1e-8,
) -> pd.DataFrame:
    """
    Apply Kalman filter to each cluster in a GeoFusion clustering output.

    Parameters
    ----------
    input_path : str
        Path to clustering output CSV.
    output_path : str
        Path to write result CSV.
    Q_var : float
        Process noise variance (degrees^2). Default 1e-8.

    Returns
    -------
    pd.DataFrame
        Original dataframe + `predicted_location_kf` column.
    """
    df = pd.read_csv(input_path)
    missing = [c for c in _REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    kf_lat = np.empty(len(df))
    kf_lon = np.empty(len(df))

    for cluster_id, grp in df.groupby("predicted_cluster", sort=False):
        grp_sorted = grp.sort_values("epoch_unix_s")

        lats    = grp_sorted["latDeg_phone"].to_numpy(dtype=float)
        lons    = grp_sorted["lngDeg_phone"].to_numpy(dtype=float)
        pr_uncs = grp_sorted["avg_rawPrUnc"].to_numpy(dtype=float)

        # initialise at cluster representative (mean phone position)
        init_lat  = float(lats.mean())
        init_lon  = float(lons.mean())
        init_hdop = float(grp_sorted["hDop"].iloc[0])

        lat_est, lon_est = _kalman_cluster(
            lats, lons, pr_uncs,
            init_lat, init_lon, init_hdop,
            Q_var=Q_var,
        )

        kf_lat[grp.index] = lat_est
        kf_lon[grp.index] = lon_est

    df["predicted_location_kf"] = list(zip(kf_lat.tolist(), kf_lon.tolist()))
    df.to_csv(output_path, index=False)
    print(f"Kalman filter estimates written to {output_path}")
    return df
