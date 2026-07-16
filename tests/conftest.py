"""
Shared pytest fixtures for GeoFusion tests.

All tests use a small synthetic dataset so they run quickly without the
real GSDC sample files. The synthetic data has the correct column schema
and 3 drives with 5 clusters each (150 observations total).
"""

import numpy as np
import pandas as pd
import pytest


def _make_synthetic_df(
    n_drives: int = 3,
    n_clusters_per_drive: int = 5,
    obs_per_cluster: int = 10,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Generate a minimal synthetic GeoFusion dataset.

    Each true GT location is a fixed point; phone observations are scattered
    around it with ~3 m of noise. All observations within a drive share the
    same GNSS quality statistics (constant features per drive).
    """
    rng = np.random.default_rng(seed)

    # Scatter true GT locations across a ~1 km² patch around Mountain View, CA
    base_lat, base_lon = 37.3861, -122.0839
    m_lat, m_lon = 111132.0, 87843.0   # approximate metres per degree at this lat

    drives   = [f"drive_{i:02d}" for i in range(n_drives)]
    phones   = [f"phone_{i:02d}" for i in range(n_drives)]
    rows     = []
    epoch_t  = 1_600_000_000.0

    for drive_idx, (drive, phone) in enumerate(zip(drives, phones)):
        # Per-drive GNSS quality (constant within drive for simplicity)
        j_avg      = rng.uniform(-12, -6)
        speed      = rng.uniform(0, 15)
        n_signals  = int(rng.integers(25, 40))
        pr_unc     = rng.uniform(2, 10)
        hdop       = rng.uniform(0.8, 2.5)
        vdop       = rng.uniform(1.0, 3.0)
        iono       = rng.uniform(0.5, 3.0)
        tropo      = rng.uniform(0.2, 1.0)

        # True GT cluster centres for this drive
        gt_lats = base_lat + rng.uniform(-0.005, 0.005, size=n_clusters_per_drive)
        gt_lons = base_lon + rng.uniform(-0.005, 0.005, size=n_clusters_per_drive)

        for c_idx in range(n_clusters_per_drive):
            for obs_idx in range(obs_per_cluster):
                noise_n = rng.normal(0, 3.0)   # metres
                noise_e = rng.normal(0, 3.0)
                phone_lat = gt_lats[c_idx] + noise_n / m_lat
                phone_lon = gt_lons[c_idx] + noise_e / m_lon
                rows.append({
                    "collectionName":          drive,
                    "phoneName":               phone,
                    "epoch_unix_s":            epoch_t,
                    "latDeg_gt":               gt_lats[c_idx],
                    "lngDeg_gt":               gt_lons[c_idx],
                    "latDeg_phone":            phone_lat,
                    "lngDeg_phone":            phone_lon,
                    "radial_error_m":          np.sqrt(noise_n**2 + noise_e**2),
                    "error_N_m":               noise_n,
                    "error_E_m":               noise_e,
                    "speedMps":                speed,
                    "courseDegree":            rng.uniform(0, 360),
                    "hDop":                    hdop,
                    "vDop":                    vdop,
                    "timeSinceFirstFixSeconds":float(obs_idx),
                    "avg_iono":                iono,
                    "avg_tropo":               tropo,
                    "avg_rawPrUnc":            pr_unc,
                    "n_signals":               float(n_signals),
                    "j_avg":                   j_avg,
                })
                epoch_t += 1.0

    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def synthetic_df():
    """Small 3-drive, 15-cluster, 150-obs synthetic dataset."""
    return _make_synthetic_df()


@pytest.fixture(scope="session")
def synthetic_df_large():
    """Slightly larger dataset for SDSGC (needs > k observations)."""
    return _make_synthetic_df(n_drives=4, n_clusters_per_drive=8, obs_per_cluster=15)


@pytest.fixture(scope="session")
def dnn_drives(synthetic_df):
    """Consistent 3-way drive split for DNN tests."""
    drives = synthetic_df["collectionName"].unique().tolist()
    return {
        "test_drives": [drives[0]],
        "val_drives":  [drives[1]],
        # train: drives[2]
    }
