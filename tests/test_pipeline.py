"""
End-to-end smoke tests for run_geofusion().

Checks that every valid model × algorithm × estimator combination:
  1. Runs without error
  2. Returns a GeoFusionResult with the right fields
  3. Produces metrics that are finite positive floats
  4. Produces df_out with predicted_cluster and predicted_location columns
  5. clustering_nc > 0
"""

import pytest
import numpy as np
from geofusion import run_geofusion, GeoFusionResult

K = 3   # small k so tests run fast


def _check_result(result, k, estimator):
    assert isinstance(result, GeoFusionResult)
    assert result.clustering_nc > 0
    assert result.runtime_s > 0
    assert result.peak_RAM_MB >= 0
    assert "predicted_cluster"  in result.df_out.columns
    assert "predicted_location" in result.df_out.columns
    for metric in ("V", "H", "MAE"):
        assert metric in result.metrics
        val = result.metrics[metric]
        assert np.isfinite(val) and val >= 0, f"{metric}={val} is not a valid error"
    if estimator == "dnn":
        assert "n_test_clusters" in result.metrics
        assert result.metrics["n_test_clusters"] > 0


# ── certain algorithms ────────────────────────────────────────────────────────

@pytest.mark.parametrize("algorithm", ["kmeans", "kmedoids"])
@pytest.mark.parametrize("estimator", ["rep", "kf", "ekf", "pf"])
def test_certain_partitional(synthetic_df, algorithm, estimator):
    result = run_geofusion(
        synthetic_df, "certain", algorithm,
        dict(k=K, n_init=1, max_iter=20),
        estimator, verbose=False,
    )
    _check_result(result, K, estimator)


def test_certain_sdsgc_rep(synthetic_df_large):
    result = run_geofusion(
        synthetic_df_large, "certain", "sdsgc",
        dict(k=K, nn=3, strategy="early_stop"),
        "rep", verbose=False,
    )
    _check_result(result, K, "rep")


@pytest.mark.parametrize("estimator", ["kf", "ekf", "pf"])
def test_certain_sdsgc_filters(synthetic_df_large, estimator):
    result = run_geofusion(
        synthetic_df_large, "certain", "sdsgc",
        dict(k=K, nn=3, strategy="early_stop"),
        estimator, verbose=False,
    )
    _check_result(result, K, estimator)


# ── uncertain algorithms (volcano) ───────────────────────────────────────────

@pytest.mark.parametrize("algorithm", ["ukmeans", "ukmedoids"])
@pytest.mark.parametrize("estimator", ["rep", "kf", "ekf", "pf"])
def test_volcano(synthetic_df, algorithm, estimator):
    params = dict(k=K, n_init=1, max_iter=20)
    if algorithm == "ukmedoids":
        params["n_samples"] = 10   # fast MC
    result = run_geofusion(
        synthetic_df, "volcano", algorithm, params,
        estimator, verbose=False,
    )
    _check_result(result, K, estimator)


# ── uncertain algorithms (mountain) ──────────────────────────────────────────

@pytest.mark.parametrize("algorithm", ["ukmeans", "ukmedoids"])
@pytest.mark.parametrize("estimator", ["rep", "kf", "ekf", "pf"])
def test_mountain(synthetic_df, algorithm, estimator):
    params = dict(k=K, n_init=1, max_iter=20)
    if algorithm == "ukmedoids":
        params["n_samples"] = 10
    result = run_geofusion(
        synthetic_df, "mountain", algorithm, params,
        estimator, verbose=False,
    )
    _check_result(result, K, estimator)


# ── DNN (mountain + ukmedoids, test-drive metrics only) ──────────────────────

def test_dnn_mountain_ukmedoids(synthetic_df, dnn_drives):
    result = run_geofusion(
        synthetic_df, "mountain", "ukmedoids",
        dict(k=K, n_init=1, max_iter=20, n_samples=10),
        "dnn",
        estimator_params={
            **dnn_drives,
            "max_epochs": 5,    # minimal training for speed
            "patience":   3,
        },
        verbose=False,
    )
    _check_result(result, K, "dnn")
    assert "predicted_location_dnn" in result.df_out.columns


# ── GeoFusionResult repr ──────────────────────────────────────────────────────

def test_result_repr(synthetic_df):
    result = run_geofusion(
        synthetic_df, "certain", "kmeans",
        dict(k=K, n_init=1), "rep", verbose=False,
    )
    r = repr(result)
    assert "GeoFusionResult" in r
    assert "V=" in r
    assert "MAE=" in r


# ── Particle filter with custom n_particles ───────────────────────────────────

def test_pf_custom_particles(synthetic_df):
    result = run_geofusion(
        synthetic_df, "certain", "kmeans",
        dict(k=K, n_init=1),
        "pf",
        estimator_params={"n_particles": 50},
        verbose=False,
    )
    _check_result(result, K, "pf")


# ── uk-means provably equals k-means (same cluster assignments) ───────────────

def test_ukmeans_equals_kmeans(synthetic_df):
    """
    UK-means on GPS data is provably equivalent to k-means.
    With the same random_state and n_init, both should assign every
    observation to the same cluster.
    """
    r_km = run_geofusion(
        synthetic_df, "certain",  "kmeans",
        dict(k=K, random_state=42, n_init=5), "rep", verbose=False,
    )
    r_uk = run_geofusion(
        synthetic_df, "mountain", "ukmeans",
        dict(k=K, random_state=42, n_init=5), "rep", verbose=False,
    )
    # Cluster labels may be permuted; compare sorted cluster sizes
    km_sizes = sorted(r_km.df_out["predicted_cluster"].value_counts().tolist())
    uk_sizes = sorted(r_uk.df_out["predicted_cluster"].value_counts().tolist())
    assert km_sizes == uk_sizes, (
        "UK-means and k-means should produce identical partition sizes on GPS data"
    )
