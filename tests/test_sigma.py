"""
Unit tests for uncertainty model sigma computation in geofusion.core.

Critical correctness invariant: sigma floor must be np.maximum(s, 0.01),
NOT np.abs(s). Using abs() changes which observations get large uncertainty
and produces different cluster assignments that break benchmark reproducibility.
"""

import numpy as np
import pandas as pd
import pytest

from geofusion.core import _build_enriched_df


def _make_row(**kwargs):
    defaults = dict(
        j_avg=-9.0, speedMps=5.0, n_signals=30.0,
        avg_rawPrUnc=5.0, latDeg_phone=37.386, lngDeg_phone=-122.084,
    )
    defaults.update(kwargs)
    return pd.DataFrame([defaults])


# ── volcano model ─────────────────────────────────────────────────────────────

def test_volcano_sigma_formula():
    """sigma = max(0.787 * j_avg + 2.192, 0.01)."""
    j = -9.0
    expected = max(0.787 * j + 2.192, 0.01)   # = max(-4.891, 0.01) = 0.01
    # wait — that's a negative j which gives negative sigma; floor catches it
    j2 = -1.0
    expected2 = max(0.787 * j2 + 2.192, 0.01)  # = 1.405

    df1 = _build_enriched_df(_make_row(j_avg=j),  "volcano")
    df2 = _build_enriched_df(_make_row(j_avg=j2), "volcano")

    assert abs(df1["reported_std_N"].iloc[0] - max(0.787*j  + 2.192, 0.01)) < 1e-6
    assert abs(df2["reported_std_N"].iloc[0] - max(0.787*j2 + 2.192, 0.01)) < 1e-6


def test_volcano_sigma_isotropic():
    """North and East sigmas should be equal for the volcano model."""
    df = _build_enriched_df(_make_row(), "volcano")
    assert df["reported_std_N"].iloc[0] == df["reported_std_E"].iloc[0]


def test_volcano_sigma_floor():
    """Very negative j_avg (bad geometry) should floor at 0.01, not go negative."""
    df = _build_enriched_df(_make_row(j_avg=-100.0), "volcano")
    assert df["reported_std_N"].iloc[0] >= 0.01
    assert df["reported_std_E"].iloc[0] >= 0.01


# ── mountain model ────────────────────────────────────────────────────────────

def test_mountain_sigma_floor_not_abs():
    """
    THE CRITICAL TEST.
    When the mountain regression formula produces a negative value,
    the floor must be 0.01, NOT abs(value).

    With extreme inputs (high j_avg, high n_signals, high speed),
    the regression formula for sN can go negative.
    max(s, 0.01) clips to 0.01.
    abs(s) would return |s| which is a large positive number.
    The two behaviours are identical for positive s, but diverge for negative s.
    """
    # Choose inputs that make sN clearly negative
    # sN ∝ (7.398 - 0.090*j - 0.103*speed - 0.151*n_signals + 0.069*unc)
    # large n_signals dominates and drives sN negative
    df = _build_enriched_df(
        _make_row(j_avg=0.0, speedMps=0.0, n_signals=200.0, avg_rawPrUnc=0.0),
        "mountain"
    )
    nu_N = 2.209
    raw_sN = (7.398 - 0.151 * 200.0) * np.sqrt((nu_N - 2) / nu_N)
    assert raw_sN < 0, "Test setup: raw_sN must be negative for this test to be meaningful"

    floored = df["reported_std_N"].iloc[0]
    assert floored == pytest.approx(0.01, abs=1e-9), (
        f"sigma floor should be 0.01, got {floored:.6f}. "
        f"If this is {abs(raw_sN):.4f}, the code is using abs() instead of max()."
    )


def test_mountain_sigma_directional():
    """North and East sigmas should differ (different regression coefficients)."""
    df = _build_enriched_df(_make_row(n_signals=30.0, avg_rawPrUnc=5.0), "mountain")
    # They are not required to be different in all cases, but should be for typical inputs
    sN = df["reported_std_N"].iloc[0]
    sE = df["reported_std_E"].iloc[0]
    # For typical GPS inputs, sN and sE will differ because their regression
    # intercepts (7.398 vs 7.612) and coefficient signs differ
    assert sN != sE or (sN == 0.01 and sE == 0.01)   # both floored is ok


def test_mountain_sigma_positive_for_typical_inputs():
    """For typical GSDC inputs, mountain sigma should be well above the floor."""
    df = _build_enriched_df(
        _make_row(j_avg=-9.0, speedMps=5.0, n_signals=30.0, avg_rawPrUnc=5.0),
        "mountain"
    )
    assert df["reported_std_N"].iloc[0] > 0.01
    assert df["reported_std_E"].iloc[0] > 0.01


def test_mountain_sigma_decreases_with_more_signals():
    """More signals → lower uncertainty (negative coefficient on n_signals)."""
    df_few  = _build_enriched_df(_make_row(n_signals=20.0), "mountain")
    df_many = _build_enriched_df(_make_row(n_signals=40.0), "mountain")
    # If both are above floor, more signals should give lower sigma
    sN_few  = df_few["reported_std_N"].iloc[0]
    sN_many = df_many["reported_std_N"].iloc[0]
    if sN_few > 0.01 and sN_many > 0.01:
        assert sN_many < sN_few


# ── enriched column names ─────────────────────────────────────────────────────

def test_enriched_df_has_required_columns():
    for model in ("volcano", "mountain"):
        df = _build_enriched_df(_make_row(), model)
        assert "reported_std_N"        in df.columns
        assert "reported_std_E"        in df.columns
        assert "reported_location_lat" in df.columns
        assert "reported_location_lon" in df.columns


def test_enriched_location_columns_equal_phone_coords():
    df_orig = _make_row()
    df_enr  = _build_enriched_df(df_orig, "mountain")
    assert df_enr["reported_location_lat"].iloc[0] == df_orig["latDeg_phone"].iloc[0]
    assert df_enr["reported_location_lon"].iloc[0] == df_orig["lngDeg_phone"].iloc[0]
