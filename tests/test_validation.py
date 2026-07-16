"""
Tests for run_geofusion() input validation.
All invalid combinations must raise ValueError with a descriptive message.
"""

import pytest
from geofusion import run_geofusion


# ── invalid model ─────────────────────────────────────────────────────────────

def test_unknown_model_raises(synthetic_df):
    with pytest.raises(ValueError, match="Unknown model"):
        run_geofusion(synthetic_df, "gaussian", "kmeans", dict(k=3), "rep")


# ── invalid algorithm ─────────────────────────────────────────────────────────

def test_unknown_algorithm_raises(synthetic_df):
    with pytest.raises(ValueError, match="Unknown algorithm"):
        run_geofusion(synthetic_df, "certain", "geoopt", dict(k=3), "rep")


# ── invalid estimator ─────────────────────────────────────────────────────────

def test_unknown_estimator_raises(synthetic_df):
    with pytest.raises(ValueError, match="Unknown estimator"):
        run_geofusion(synthetic_df, "certain", "kmeans", dict(k=3), "lstm")


# ── model × algorithm compatibility ──────────────────────────────────────────

@pytest.mark.parametrize("model,algorithm", [
    ("volcano",  "kmeans"),
    ("volcano",  "kmedoids"),
    ("volcano",  "sdsgc"),
    ("mountain", "kmeans"),
    ("mountain", "kmedoids"),
    ("mountain", "sdsgc"),
    ("certain",  "ukmeans"),
    ("certain",  "ukmedoids"),
])
def test_incompatible_model_algorithm_raises(synthetic_df, model, algorithm):
    with pytest.raises(ValueError, match="Incompatible combination"):
        run_geofusion(synthetic_df, model, algorithm, dict(k=3), "rep")


# ── valid combinations do NOT raise ──────────────────────────────────────────

@pytest.mark.parametrize("model,algorithm", [
    ("certain",  "kmeans"),
    ("certain",  "kmedoids"),
    ("certain",  "sdsgc"),
    ("volcano",  "ukmeans"),
    ("volcano",  "ukmedoids"),
    ("mountain", "ukmeans"),
    ("mountain", "ukmedoids"),
])
def test_valid_combinations_do_not_raise(synthetic_df, model, algorithm):
    result = run_geofusion(
        synthetic_df, model, algorithm,
        dict(k=3, n_init=1, max_iter=10),
        "rep", verbose=False,
    )
    assert result.clustering_nc > 0


# ── missing dataset columns ───────────────────────────────────────────────────

def test_missing_required_column_raises(synthetic_df):
    df_bad = synthetic_df.drop(columns=["j_avg"])
    with pytest.raises(ValueError, match="missing required columns"):
        run_geofusion(df_bad, "certain", "kmeans", dict(k=3), "rep")


# ── DNN: missing test/val drives ─────────────────────────────────────────────

def test_dnn_missing_test_drives_raises(synthetic_df, dnn_drives):
    with pytest.raises(ValueError, match="test_drives"):
        run_geofusion(
            synthetic_df, "mountain", "ukmedoids", dict(k=3, n_init=1),
            "dnn",
            estimator_params={"val_drives": dnn_drives["val_drives"]},
            verbose=False,
        )


def test_dnn_missing_val_drives_raises(synthetic_df, dnn_drives):
    with pytest.raises(ValueError, match="val_drives"):
        run_geofusion(
            synthetic_df, "mountain", "ukmedoids", dict(k=3, n_init=1),
            "dnn",
            estimator_params={"test_drives": dnn_drives["test_drives"]},
            verbose=False,
        )


def test_dnn_overlapping_drives_raises(synthetic_df):
    drives = synthetic_df["collectionName"].unique().tolist()
    with pytest.raises(ValueError, match="overlap"):
        run_geofusion(
            synthetic_df, "mountain", "ukmedoids", dict(k=3, n_init=1),
            "dnn",
            estimator_params={
                "test_drives": [drives[0]],
                "val_drives":  [drives[0]],   # same drive in both
            },
            verbose=False,
        )


def test_dnn_unknown_test_drive_raises(synthetic_df, dnn_drives):
    with pytest.raises(ValueError, match="not found in dataset"):
        run_geofusion(
            synthetic_df, "mountain", "ukmedoids", dict(k=3, n_init=1),
            "dnn",
            estimator_params={
                "test_drives": ["nonexistent_drive"],
                "val_drives":  dnn_drives["val_drives"],
            },
            verbose=False,
        )
