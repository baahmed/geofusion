"""
Unit tests for metric computation in geofusion.core.

Tests that V, H, MAE are computed correctly without running the full pipeline,
using manually constructed dataframes with known ground-truth answers.
"""

import math
import numpy as np
import pandas as pd
import pytest

from geofusion.core import _compute_metrics, _compute_metrics_test, _meters_per_degree


# ── WGS84 helper ─────────────────────────────────────────────────────────────

def test_meters_per_degree_lat_approx_mountain_view():
    """m_lat at Mountain View should be ~111,100 m/deg (standard WGS84 value)."""
    m_lat, _ = _meters_per_degree(37.386, -122.084)
    assert 110_900 < m_lat < 111_400


def test_meters_per_degree_uses_lon_arg():
    """
    m_lon uses cos(lon_deg) (theta term in the benchmark formula), NOT cos(lat).
    At lon=0: cos(0)=1 → m_lon ≈ 111,319.
    At lon=60: cos(60°)=0.5 → m_lon ≈ 55,660.
    The ratio should be approximately 2.
    """
    _, m_lon_0  = _meters_per_degree(0.0, 0.0)
    _, m_lon_60 = _meters_per_degree(0.0, 60.0)
    assert abs(m_lon_0) > abs(m_lon_60) * 1.8


# ── _compute_metrics ─────────────────────────────────────────────────────────

def _make_df(clusters):
    """
    clusters: list of (pred_lat, pred_lon, gt_lat, gt_lon, drive, n_obs)
    Returns df_out and df_orig matching the _compute_metrics interface.
    """
    out_rows, orig_rows = [], []
    cluster_id = 0
    ref_lat, ref_lon = 37.386, -122.084

    for pred_lat, pred_lon, gt_lat, gt_lon, drive, n_obs in clusters:
        for _ in range(n_obs):
            out_rows.append({
                "predicted_cluster":  cluster_id,
                "predicted_location": (pred_lat, pred_lon),
                "collectionName":     drive,
            })
            orig_rows.append({
                "latDeg_phone":   ref_lat,
                "lngDeg_phone":   ref_lon,
                "latDeg_gt":      gt_lat,
                "lngDeg_gt":      gt_lon,
                "collectionName": drive,
            })
        cluster_id += 1

    return pd.DataFrame(out_rows), pd.DataFrame(orig_rows)


def test_perfect_prediction_gives_zero_error():
    """When predicted location == GT, all errors should be zero."""
    lat, lon = 37.386, -122.084
    df_out, df_orig = _make_df([
        (lat, lon, lat, lon, "d0", 5),
        (lat, lon, lat, lon, "d1", 5),
    ])
    m = _compute_metrics(df_out, df_orig)
    assert m["V"]   < 1e-6
    assert m["H"]   < 1e-6
    assert m["MAE"] < 1e-6


def test_known_north_error():
    """A 1 m northward prediction error should give V ≈ 1 m, H ≈ 0."""
    ref_lat, ref_lon = 37.386, -122.084
    m_lat, _ = _meters_per_degree(ref_lat, ref_lon)
    offset_lat = 1.0 / m_lat

    df_out, df_orig = _make_df([
        (ref_lat + offset_lat, ref_lon, ref_lat, ref_lon, "d0", 4),
    ])
    m = _compute_metrics(df_out, df_orig)
    assert abs(m["V"] - 1.0) < 0.01
    assert m["H"] < 0.01
    assert abs(m["MAE"] - 1.0) < 0.01


def test_mae_is_euclidean():
    """MAE = sqrt(V² + H²) per cluster, then averaged."""
    ref_lat, ref_lon = 37.386, -122.084
    m_lat, m_lon = _meters_per_degree(ref_lat, ref_lon)
    n_lat = 3.0 / m_lat
    e_lon = 4.0 / abs(m_lon)   # use abs() — m_lon may be negative at lon=-122

    df_out, df_orig = _make_df([
        (ref_lat + n_lat, ref_lon + e_lon, ref_lat, ref_lon, "d0", 4),
    ])
    m = _compute_metrics(df_out, df_orig)
    assert abs(m["MAE"] - 5.0) < 0.05
    assert abs(m["V"]   - 3.0) < 0.03
    assert abs(m["H"]   - 4.0) < 0.04


def test_average_is_per_cluster_not_per_row():
    """
    Averaging is per cluster then across clusters — not per observation.
    Two clusters with 1 m and 3 m error → MAE = 2 m regardless of obs count.
    """
    ref_lat, ref_lon = 37.386, -122.084
    m_lat, _ = _meters_per_degree(ref_lat, ref_lon)
    off1 = 1.0 / m_lat
    off3 = 3.0 / m_lat

    df_out, df_orig = _make_df([
        (ref_lat + off1, ref_lon, ref_lat, ref_lon, "d0", 10),
        (ref_lat + off3, ref_lon, ref_lat, ref_lon, "d0",  2),
    ])
    m = _compute_metrics(df_out, df_orig)
    assert abs(m["V"]   - 2.0) < 0.02
    assert abs(m["MAE"] - 2.0) < 0.02


# ── _compute_metrics_test ─────────────────────────────────────────────────────

def test_metrics_test_filters_by_drive():
    """Only clusters whose observations include test drives should be counted."""
    ref_lat, ref_lon = 37.386, -122.084
    m_lat, _ = _meters_per_degree(ref_lat, ref_lon)
    off = 2.0 / m_lat

    df_out, df_orig = _make_df([
        (ref_lat + off,     ref_lon, ref_lat, ref_lon, "drive_test",  5),
        (ref_lat + 10*off,  ref_lon, ref_lat, ref_lon, "drive_train", 5),
    ])
    m = _compute_metrics_test(df_out, df_orig, test_drives=["drive_test"])
    assert abs(m["V"] - 2.0) < 0.02
    assert m["n_test_clusters"] == 1


def test_metrics_test_all_clusters_when_all_drives_are_test():
    ref_lat, ref_lon = 37.386, -122.084
    m_lat, _ = _meters_per_degree(ref_lat, ref_lon)
    off = 1.0 / m_lat

    df_out, df_orig = _make_df([
        (ref_lat + off, ref_lon, ref_lat, ref_lon, "d0", 4),
        (ref_lat + off, ref_lon, ref_lat, ref_lon, "d1", 4),
    ])
    m_all  = _compute_metrics(df_out, df_orig)
    m_test = _compute_metrics_test(df_out, df_orig, test_drives=["d0", "d1"])
    assert abs(m_all["V"] - m_test["V"]) < 1e-6
    assert m_test["n_test_clusters"] == 2
