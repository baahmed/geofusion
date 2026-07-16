"""
GeoFusion core pipeline module.

See geofusion/__init__.py for the public API and full documentation.
"""

from __future__ import annotations

import ast
import json
import math
import os
import subprocess
import sys
import tempfile
import time
import tracemalloc
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import cdist
import scipy.sparse.linalg as spla

# ── relative imports (package modules) ───────────────────────────────────────
from .clustering.kmeans_clustering    import run_kmeans
from .clustering.kmedoids_clustering  import run_kmedoids
from .clustering.ukmeans_clustering   import run_ukmeans
from .clustering.ukmedoids_clustering import run_ukmedoids
from .clustering.sdsgc_clustering     import (
    _l2_distance, _sym_neighbors, _eig_laplacian, _proj_simplex,
)
from .estimators.kalman_estimator          import run_kalman
from .estimators.ekf_estimator             import run_ekf
from .estimators.particle_filter_estimator import run_pf
from .estimators.dnn_estimator             import (
    build_cluster_dataset, train_dnn, predict_dnn,
)


# ─────────────────────────────────────────────────────────────────────────────
# Compatibility table
# ─────────────────────────────────────────────────────────────────────────────

_ALGO_MODELS: Dict[str, Tuple[str, ...]] = {
    "kmeans":    ("certain",),
    "kmedoids":  ("certain",),
    "ukmeans":   ("volcano", "mountain"),
    "ukmedoids": ("volcano", "mountain"),
    "sdsgc":     ("certain",),
}

_VALID_MODELS     = {"certain", "volcano", "mountain"}
_VALID_ALGORITHMS = set(_ALGO_MODELS)
_VALID_ESTIMATORS = {"rep", "kf", "ekf", "pf", "dnn"}

_REQUIRED_COLS = [
    "collectionName", "latDeg_gt", "lngDeg_gt",
    "latDeg_phone", "lngDeg_phone",
    "j_avg", "speedMps", "n_signals", "avg_rawPrUnc",
    "hDop", "vDop", "avg_iono", "avg_tropo",
]


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GeoFusionResult:
    """
    Return value of run_geofusion().

    Attributes
    ----------
    df_out           : output dataframe — original columns plus
                       'predicted_cluster' (int) and
                       'predicted_location' (lat, lon tuple).
                       For DNN, also 'predicted_location_dnn'.
    metrics          : dict — V (north error m), H (east error m),
                       MAE (Euclidean m), all averaged per cluster then
                       across clusters. For DNN: test-drive clusters only.
    runtime_s        : total wall-clock seconds
    peak_RAM_MB      : peak tracemalloc allocation in MB
    clustering_nc    : actual number of clusters produced
    model            : uncertainty model used
    algorithm        : clustering algorithm used
    estimator        : estimator used
    algo_params      : hyperparameters passed to the clustering algorithm
    estimator_params : hyperparameters passed to the estimator
    """
    df_out:           pd.DataFrame
    metrics:          Dict[str, Any]
    runtime_s:        float
    peak_RAM_MB:      float
    clustering_nc:    int
    model:            str
    algorithm:        str
    estimator:        str
    algo_params:      Dict[str, Any] = field(default_factory=dict)
    estimator_params: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        m = self.metrics
        dnn_note = " (test-drive clusters only)" if self.estimator == "dnn" else ""
        return (
            f"GeoFusionResult(\n"
            f"  model={self.model!r}  algorithm={self.algorithm!r}  "
            f"estimator={self.estimator!r}\n"
            f"  nc={self.clustering_nc}  "
            f"V={m.get('V', float('nan')):.4f}m  "
            f"H={m.get('H', float('nan')):.4f}m  "
            f"MAE={m.get('MAE', float('nan')):.4f}m{dnn_note}\n"
            f"  runtime={self.runtime_s:.2f}s  peak_RAM={self.peak_RAM_MB:.1f}MB\n"
            f")"
        )


# ─────────────────────────────────────────────────────────────────────────────
# WGS84 helpers (benchmark-exact: uses both lat and lon)
# ─────────────────────────────────────────────────────────────────────────────

def _meters_per_degree(lat_deg: float, lon_deg: float) -> Tuple[float, float]:
    phi   = math.radians(lat_deg)
    theta = math.radians(lon_deg)
    m_lat = (111132.92 - 559.82 * math.cos(2 * phi)
             + 1.175 * math.cos(4 * phi) - 0.0023 * math.cos(6 * phi))
    m_lon = (111412.84 * math.cos(theta)
             - 93.5 * math.cos(3 * theta) + 0.118 * math.cos(5 * theta))
    return float(m_lat), float(m_lon)


def _parse_loc(val) -> Tuple[float, float]:
    if isinstance(val, (tuple, list)):
        return float(val[0]), float(val[1])
    return tuple(float(x) for x in str(val).strip("()").split(","))


def _compute_metrics(
    df_out: pd.DataFrame,
    df_orig: pd.DataFrame,
) -> Dict[str, float]:
    """Per-cluster V/H/MAE averaged across all clusters."""
    pred    = df_out["predicted_cluster"].values
    ref_lat = df_orig["latDeg_phone"].mean()
    ref_lon = df_orig["lngDeg_phone"].mean()
    m_lat, m_lon = _meters_per_degree(ref_lat, ref_lon)
    gt_lat  = df_orig["latDeg_gt"].values
    gt_lon  = df_orig["lngDeg_gt"].values
    h, v, mae = [], [], []
    for c in np.unique(pred):
        mask = pred == c
        rl, rn = _parse_loc(df_out.loc[mask, "predicted_location"].iloc[0])
        dn = (rl - gt_lat[mask].mean()) * m_lat
        de = (rn - gt_lon[mask].mean()) * m_lon
        v.append(abs(dn)); h.append(abs(de))
        mae.append(math.sqrt(dn ** 2 + de ** 2))
    return {
        "V":   float(np.mean(v)),
        "H":   float(np.mean(h)),
        "MAE": float(np.mean(mae)),
    }


def _compute_metrics_test(
    df_out: pd.DataFrame,
    df_orig: pd.DataFrame,
    test_drives: List[str],
) -> Dict[str, Any]:
    """Per-cluster V/H/MAE restricted to test-drive clusters."""
    pred    = df_out["predicted_cluster"].values
    ref_lat = df_orig["latDeg_phone"].mean()
    ref_lon = df_orig["lngDeg_phone"].mean()
    m_lat, m_lon = _meters_per_degree(ref_lat, ref_lon)
    gt_lat  = df_orig["latDeg_gt"].values
    gt_lon  = df_orig["lngDeg_gt"].values
    drives  = df_orig["collectionName"].values
    h, v, mae = [], [], []
    for c in np.unique(pred):
        mask = pred == c
        if not np.any(np.isin(drives[mask], test_drives)):
            continue
        rl, rn = _parse_loc(df_out.loc[mask, "predicted_location"].iloc[0])
        dn = (rl - gt_lat[mask].mean()) * m_lat
        de = (rn - gt_lon[mask].mean()) * m_lon
        v.append(abs(dn)); h.append(abs(de))
        mae.append(math.sqrt(dn ** 2 + de ** 2))
    return {
        "V":               float(np.mean(v)),
        "H":               float(np.mean(h)),
        "MAE":             float(np.mean(mae)),
        "n_test_clusters": len(v),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Uncertainty model: enriched DataFrame builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_enriched_df(df: pd.DataFrame, model: str) -> pd.DataFrame:
    """
    Add reported_std_N, reported_std_E, reported_location_lat/lon columns.

    Sigma floor is np.maximum(s, 0.01) — NOT np.abs(s).
    Using np.abs() changes which observations get large uncertainty,
    producing different cluster assignments that break benchmark reproducibility.
    """
    dfe = df.copy()
    j   = df["j_avg"].to_numpy(float)
    spd = df["speedMps"].to_numpy(float)
    ns  = df["n_signals"].to_numpy(float)
    unc = df["avg_rawPrUnc"].to_numpy(float)

    if model == "volcano":
        s = np.maximum(0.787 * j + 2.192, 0.01)
        dfe["reported_std_N"] = s
        dfe["reported_std_E"] = s

    elif model == "mountain":
        nu_N, nu_E = 2.209, 2.743
        sN = (7.398 - 0.090*j - 0.103*spd
              - 0.151*ns + 0.069*unc) * np.sqrt((nu_N - 2) / nu_N)
        sE = (7.612 + 0.031*j - 0.099*spd
              - 0.116*ns + 0.012*unc) * np.sqrt((nu_E - 2) / nu_E)
        dfe["reported_std_N"] = np.maximum(sN, 0.01)
        dfe["reported_std_E"] = np.maximum(sE, 0.01)

    dfe["reported_location_lat"] = dfe["latDeg_phone"]
    dfe["reported_location_lon"] = dfe["lngDeg_phone"]
    return dfe


# ─────────────────────────────────────────────────────────────────────────────
# Clustering
# ─────────────────────────────────────────────────────────────────────────────

def _run_clustering(
    df: pd.DataFrame,
    model: str,
    algorithm: str,
    algo_params: Dict[str, Any],
) -> Tuple[pd.DataFrame, int]:
    k            = algo_params.get("k", 10)
    random_state = algo_params.get("random_state", 42)
    n_init       = algo_params.get("n_init", 10)
    max_iter     = algo_params.get("max_iter", 300)

    in_tmp  = tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w")
    out_tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w")
    in_path, out_path = in_tmp.name, out_tmp.name
    in_tmp.close(); out_tmp.close()

    if algorithm in ("kmeans", "kmedoids", "sdsgc"):
        df.to_csv(in_path, index=False)
        cols = ("latDeg_phone", "lngDeg_phone")
    else:
        _build_enriched_df(df, model).to_csv(in_path, index=False)
        cols = ("reported_location_lat", "reported_location_lon")

    try:
        if algorithm == "kmeans":
            df_out = run_kmeans(in_path, out_path,
                                columns=cols, k=k,
                                random_state=random_state, n_init=n_init)
            nc = int(df_out["predicted_cluster"].nunique())

        elif algorithm == "kmedoids":
            df_out = run_kmedoids(in_path, out_path,
                                  columns=cols, k=k,
                                  random_state=random_state,
                                  n_init=n_init, max_iter=max_iter)
            nc = int(df_out["predicted_cluster"].nunique())

        elif algorithm == "ukmeans":
            df_out = run_ukmeans(in_path, out_path,
                                 columns=cols, k=k,
                                 random_state=random_state, n_init=n_init,
                                 std_n_col="reported_std_N",
                                 std_e_col="reported_std_E",
                                 max_iter=max_iter)
            nc = int(df_out["predicted_cluster"].nunique())

        elif algorithm == "ukmedoids":
            n_samples = algo_params.get("n_samples", 50)
            df_out = run_ukmedoids(in_path, out_path,
                                   columns=cols, k=k,
                                   random_state=random_state,
                                   n_init=n_init, n_samples=n_samples,
                                   std_n_col="reported_std_N",
                                   std_e_col="reported_std_E",
                                   max_iter=max_iter)
            nc = int(df_out["predicted_cluster"].nunique())

        elif algorithm == "sdsgc":
            df_out, nc = _run_sdsgc(df, algo_params)

        else:
            raise ValueError(f"Unknown algorithm: {algorithm!r}")

    finally:
        for p in (in_path, out_path):
            if os.path.exists(p):
                os.unlink(p)

    # Restore columns that may not survive CSV round-trip
    for col in _REQUIRED_COLS:
        if col in df.columns and col not in df_out.columns:
            df_out[col] = df[col].values

    return df_out, nc


# ─────────────────────────────────────────────────────────────────────────────
# SDSGC implementation
# ─────────────────────────────────────────────────────────────────────────────

def _default_sdsgc_strategy(k: int) -> str:
    if k == 10:
        return "early_stop"
    if k == 50:
        return "best_of_n"
    return "threshold"


def _run_sdsgc(
    df: pd.DataFrame,
    algo_params: Dict[str, Any],
) -> Tuple[pd.DataFrame, int]:
    """
    SDSGC medoid clustering.

    Strategies
    ----------
    early_stop  : iterate until nc == k (default for k=10, nn=5)
    best_of_n   : run max_iter iters, keep nc closest to k (default for k=50)
    threshold   : 1 ALM iter + W-matrix threshold sweep (default for k≥100, nn=4)
    """
    k         = algo_params.get("k", 10)
    nn        = algo_params.get("nn", 5 if k <= 50 else 4)
    strategy  = algo_params.get("strategy", _default_sdsgc_strategy(k))
    threshold = algo_params.get("threshold", None)
    max_iter  = algo_params.get(
        "max_iter",
        1 if strategy == "threshold" else 10 if strategy == "best_of_n" else 300,
    )
    eigsh_tol     = algo_params.get("eigsh_tol", 1e-6)
    eigsh_maxiter = algo_params.get("eigsh_maxiter", 501)
    gamma, eta, rho = 0.001, 0.1, 1.3

    lat = df["latDeg_phone"].to_numpy(float)
    lon = df["lngDeg_phone"].to_numpy(float)
    X   = np.column_stack([lat, lon])
    n   = len(X)

    dist_sq     = _l2_distance(X)
    W, knn_sets = _sym_neighbors(dist_sq, nn)

    # Spectral embedding with relaxed tolerance when needed (n ≥ 6000)
    try:
        D_ = np.diag(W.sum(1))
        L_ = csr_matrix(D_ - W)
        vals_, vecs_ = spla.eigsh(
            L_, k=k + 1, which="SM", tol=eigsh_tol, maxiter=eigsh_maxiter)
        ord_ = np.argsort(vals_)
        F    = vecs_[:, ord_[1:k + 1]]
    except Exception:
        F, _ = _eig_laplacian(W, k)

    mu  = 1.0
    lam = np.zeros((n, n))
    best_labels = None
    best_nc     = 999 if strategy == "best_of_n" else 0

    for _ in range(max_iter):
        dist_F = _l2_distance(F)
        Z = (mu * W.T
             - 0.5 * lam + 0.5 * lam.T
             - 0.5 * dist_sq
             - 0.5 * eta * dist_F)
        W_new = np.zeros((n, n))
        for i in range(n):
            ad = Z[i, knn_sets[i]] / (gamma + mu)
            W_new[i, knn_sets[i]] = _proj_simplex(ad)
        lam   = lam + mu * (W_new - W_new.T)
        mu   *= rho
        W_new = (W_new + W_new.T) / 2.0
        F, _  = _eig_laplacian(W_new, k)
        W     = W_new

        if strategy == "threshold":
            break   # only 1 ALM iteration

        nc, lbl = connected_components(csr_matrix(W > 1e-6), directed=False)

        if strategy == "early_stop":
            if nc <= k and nc > best_nc:
                best_nc = nc; best_labels = lbl.copy()
            if nc == k:
                break
        elif strategy == "best_of_n":
            if abs(nc - k) < abs(best_nc - k):
                best_nc = nc; best_labels = lbl.copy()
            if nc == k:
                break

    # ── Threshold strategy: sweep W to extract exactly nc == k ────────────
    if strategy == "threshold":
        t0 = threshold if threshold is not None else 0.1
        nc, lbl = connected_components(csr_matrix(W > t0), directed=False)
        if nc != k:
            found = False
            for t in np.arange(t0 - 0.005, t0 + 0.015, 0.00001):
                nc_t, lbl_t = connected_components(
                    csr_matrix(W > t), directed=False)
                if nc_t == k:
                    lbl = lbl_t; nc = nc_t; found = True; break
            if not found:
                import warnings
                best_nc_local = 999
                for t in np.arange(t0 - 0.02, t0 + 0.02, 0.0001):
                    nc_t, lbl_t = connected_components(
                        csr_matrix(W > t), directed=False)
                    if abs(nc_t - k) < abs(best_nc_local - k):
                        best_nc_local = nc_t; lbl = lbl_t; nc = nc_t
                warnings.warn(
                    f"SDSGC: nc={k} not reachable; using best achievable nc={nc}",
                    RuntimeWarning, stacklevel=4,
                )
        best_labels = lbl; best_nc = nc

    labels = best_labels

    # ── Medoid representatives ─────────────────────────────────────────────
    pred_lat = np.empty(n); pred_lon = np.empty(n)
    for cid in np.unique(labels):
        mask = labels == cid
        midx = np.where(mask)[0]
        if midx.size == 1:
            pred_lat[mask] = lat[midx[0]]
            pred_lon[mask] = lon[midx[0]]
        else:
            members = np.column_stack([lat[midx], lon[midx]])
            D = cdist(members, members)
            best = midx[np.argmin(D.sum(axis=1))]
            pred_lat[mask] = lat[best]
            pred_lon[mask] = lon[best]

    df_out = df.copy()
    df_out["predicted_cluster"]  = labels
    df_out["predicted_location"] = [
        (float(pred_lat[i]), float(pred_lon[i])) for i in range(n)
    ]
    return df_out, int(best_nc)


# ─────────────────────────────────────────────────────────────────────────────
# Estimation
# ─────────────────────────────────────────────────────────────────────────────

def _run_estimation(
    df_clustered: pd.DataFrame,
    df_orig: pd.DataFrame,
    estimator: str,
    estimator_params: Dict[str, Any],
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    if estimator == "rep":
        return _compute_metrics(df_clustered, df_orig), df_clustered

    in_tmp  = tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w")
    df_clustered.to_csv(in_tmp, index=False)
    tmp_path = in_tmp.name; in_tmp.close()
    out_path = tmp_path + "_est_out.csv"

    try:
        if estimator == "kf":
            df_out = run_kalman(tmp_path, out_path)
            lats, lons = _extract_arrays(df_out, "predicted_location_kf")

        elif estimator == "ekf":
            df_out = run_ekf(tmp_path, out_path)
            lats, lons = _extract_arrays(df_out, "predicted_location_ekf")

        elif estimator == "pf":
            n_particles = estimator_params.get("n_particles", 500)
            df_out = run_pf(tmp_path, out_path, n_particles=n_particles)
            lats, lons = _extract_arrays(df_out, "predicted_location_pf")

        elif estimator == "dnn":
            metrics, df_clustered = _run_dnn_estimation(
                df_clustered, df_orig, tmp_path, estimator_params)
            return metrics, df_clustered

        else:
            raise ValueError(f"Unknown estimator: {estimator!r}")

    finally:
        for p in (tmp_path, out_path):
            if os.path.exists(p):
                os.unlink(p)

    df_result = df_clustered.copy()
    df_result["predicted_location"] = list(zip(lats.tolist(), lons.tolist()))
    return _compute_metrics(df_result, df_orig), df_result


def _extract_arrays(
    df: pd.DataFrame, col: str,
) -> Tuple[np.ndarray, np.ndarray]:
    vals = df[col].tolist()
    if vals and isinstance(vals[0], str):
        vals = [ast.literal_eval(v) for v in vals]
    return (np.array([v[0] for v in vals], float),
            np.array([v[1] for v in vals], float))


# ─────────────────────────────────────────────────────────────────────────────
# DNN estimation (in-process and subprocess modes)
# ─────────────────────────────────────────────────────────────────────────────

def _run_dnn_estimation(
    df_clustered: pd.DataFrame,
    df_orig: pd.DataFrame,
    clustered_csv_path: str,
    estimator_params: Dict[str, Any],
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    test_drives = estimator_params["test_drives"]
    val_drives  = estimator_params["val_drives"]
    random_state = estimator_params.get("random_state", 42)
    max_epochs   = estimator_params.get("max_epochs", 200)
    patience     = estimator_params.get("patience", 20)
    model_path   = estimator_params.get("model_path", "/tmp/_gf_dnn.pt")
    scaler_path  = estimator_params.get("scaler_path", "/tmp/_gf_scaler.pkl")
    use_subprocess = estimator_params.get("subprocess", False)

    if use_subprocess:
        return _run_dnn_subprocess(
            df_clustered, df_orig, clustered_csv_path,
            test_drives, val_drives, random_state, max_epochs, patience,
        )

    df = pd.read_csv(clustered_csv_path)
    if isinstance(df["predicted_location"].iloc[0], str):
        df["predicted_location"] = df["predicted_location"].apply(
            ast.literal_eval)

    df_cl = build_cluster_dataset(df)
    all_drives   = df_cl["collectionName"].unique().tolist()
    train_drives = [d for d in all_drives
                    if d not in val_drives and d not in test_drives]

    model, scalers = train_dnn(
        df_cl, train_drives, val_drives,
        model_path=model_path, scaler_path=scaler_path,
        max_epochs=max_epochs, patience=patience,
        random_state=random_state,
    )
    df_out = predict_dnn(df, model, scalers)

    dnn_vals = df_out["predicted_location_dnn"].tolist()
    if dnn_vals and isinstance(dnn_vals[0], str):
        dnn_vals = [ast.literal_eval(v) for v in dnn_vals]

    df_tmp = df_clustered.copy()
    df_tmp["predicted_location"] = dnn_vals
    metrics = _compute_metrics_test(df_tmp, df_orig, test_drives)

    df_clustered = df_clustered.copy()
    df_clustered["predicted_location_dnn"] = dnn_vals
    return metrics, df_clustered


# Worker script string — runs in subprocess for clean RAM measurement
_DNN_WORKER_SCRIPT = """\
import sys, time, tracemalloc, json, ast, os
import pandas as pd
# insert package parent so relative imports work when called as __main__
_pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _pkg_root not in sys.path:
    sys.path.insert(0, _pkg_root)
from geofusion.estimators.dnn_estimator import (
    build_cluster_dataset, train_dnn, predict_dnn)

clustered_path = sys.argv[1]
orig_path      = sys.argv[2]
out_path       = sys.argv[3]
test_drives    = json.loads(sys.argv[4])
val_drives     = json.loads(sys.argv[5])
random_state   = int(sys.argv[6])
max_epochs     = int(sys.argv[7])
patience       = int(sys.argv[8])

df      = pd.read_csv(clustered_path)
df_orig = pd.read_csv(orig_path)
if isinstance(df["predicted_location"].iloc[0], str):
    df["predicted_location"] = df["predicted_location"].apply(ast.literal_eval)

df_cl = build_cluster_dataset(df)
all_drives   = df_cl["collectionName"].unique().tolist()
train_drives = [d for d in all_drives
                if d not in val_drives and d not in test_drives]

tracemalloc.start()
t0 = time.perf_counter()
model, scalers = train_dnn(
    df_cl, train_drives, val_drives,
    model_path="/tmp/_gf_dnn_sub.pt",
    scaler_path="/tmp/_gf_scaler_sub.pkl",
    max_epochs=max_epochs, patience=patience, random_state=random_state,
)
df_out = predict_dnn(df, model, scalers)
rt = time.perf_counter() - t0
_, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
df_out.to_csv(out_path, index=False)
print("RESULT:" + json.dumps({
    "runtime_s": rt, "peak_RAM_MB": peak / 1e6}), flush=True)
"""


def _run_dnn_subprocess(
    df_clustered: pd.DataFrame,
    df_orig: pd.DataFrame,
    clustered_csv_path: str,
    test_drives: List[str],
    val_drives: List[str],
    random_state: int,
    max_epochs: int,
    patience: int,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    # Write worker script next to this file so relative imports resolve
    pkg_dir     = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(pkg_dir, "_dnn_worker_tmp.py")
    with open(script_path, "w") as f:
        f.write(_DNN_WORKER_SCRIPT)

    orig_tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w")
    df_orig.to_csv(orig_tmp, index=False)
    orig_path = orig_tmp.name; orig_tmp.close()

    out_path = clustered_csv_path + "_dnn_sub_out.csv"
    cmd = [
        sys.executable, script_path,
        clustered_csv_path, orig_path, out_path,
        json.dumps(test_drives), json.dumps(val_drives),
        str(random_state), str(max_epochs), str(patience),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    os.unlink(script_path)
    os.unlink(orig_path)

    if proc.returncode != 0:
        raise RuntimeError(f"DNN subprocess failed:\n{proc.stderr}")

    result_lines = [l for l in proc.stdout.strip().split("\n")
                    if l.startswith("RESULT:")]
    meta = json.loads(result_lines[0].replace("RESULT:", ""))

    df_out = pd.read_csv(out_path)
    if os.path.exists(out_path):
        os.unlink(out_path)

    dnn_vals = df_out["predicted_location_dnn"].tolist()
    if dnn_vals and isinstance(dnn_vals[0], str):
        dnn_vals = [ast.literal_eval(v) for v in dnn_vals]

    df_tmp = df_clustered.copy()
    df_tmp["predicted_location"] = dnn_vals
    metrics = _compute_metrics_test(df_tmp, df_orig, test_drives)
    metrics["peak_RAM_MB"] = meta["peak_RAM_MB"]

    df_clustered = df_clustered.copy()
    df_clustered["predicted_location_dnn"] = dnn_vals
    return metrics, df_clustered


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────

def _validate(
    df: pd.DataFrame,
    model: str,
    algorithm: str,
    estimator: str,
    estimator_params: Dict[str, Any],
) -> None:
    if model not in _VALID_MODELS:
        raise ValueError(
            f"Unknown model {model!r}. Valid: {sorted(_VALID_MODELS)}")
    if algorithm not in _VALID_ALGORITHMS:
        raise ValueError(
            f"Unknown algorithm {algorithm!r}. Valid: {sorted(_VALID_ALGORITHMS)}")
    if estimator not in _VALID_ESTIMATORS:
        raise ValueError(
            f"Unknown estimator {estimator!r}. Valid: {sorted(_VALID_ESTIMATORS)}")

    permitted = _ALGO_MODELS[algorithm]
    if model not in permitted:
        raise ValueError(
            f"Incompatible combination: model={model!r} with algorithm={algorithm!r}.\n"
            f"  {algorithm!r} requires model in {list(permitted)}.\n"
            f"  Certain model (no uncertainty): kmeans, kmedoids, sdsgc.\n"
            f"  Volcano/mountain model: ukmeans, ukmedoids."
        )

    missing = [c for c in _REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Input DataFrame is missing required columns: {missing}")

    if estimator == "dnn":
        if "test_drives" not in estimator_params:
            raise ValueError(
                "estimator='dnn' requires 'test_drives' in estimator_params.")
        if "val_drives" not in estimator_params:
            raise ValueError(
                "estimator='dnn' requires 'val_drives' in estimator_params.")
        all_drives  = set(df["collectionName"].unique())
        test_drives = estimator_params["test_drives"]
        val_drives  = estimator_params["val_drives"]
        unknown_test = [d for d in test_drives if d not in all_drives]
        unknown_val  = [d for d in val_drives  if d not in all_drives]
        if unknown_test:
            raise ValueError(
                f"test_drives not found in dataset: {unknown_test}")
        if unknown_val:
            raise ValueError(
                f"val_drives not found in dataset: {unknown_val}")
        overlap = set(test_drives) & set(val_drives)
        if overlap:
            raise ValueError(
                f"test_drives and val_drives overlap: {overlap}")


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_geofusion(
    df: pd.DataFrame,
    model: str,
    algorithm: str,
    algo_params: Dict[str, Any],
    estimator: str,
    estimator_params: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
) -> GeoFusionResult:
    """
    Run the full GeoFusion pipeline: represent → cluster → estimate → evaluate.

    Parameters
    ----------
    df               : Input dataframe. Required columns listed in
                       geofusion.REQUIRED_COLS.
    model            : Uncertainty model — 'certain', 'volcano', or 'mountain'.
    algorithm        : Clustering algorithm — 'kmeans', 'kmedoids', 'ukmeans',
                       'ukmedoids', or 'sdsgc'.
    algo_params      : Hyperparameter dict for the clustering algorithm.
                       Must include 'k'. See package docstring for full list.
    estimator        : Post-clustering estimator — 'rep', 'kf', 'ekf', 'pf',
                       or 'dnn'.
    estimator_params : Hyperparameter dict for the estimator.
                       Required for 'dnn': must include 'test_drives' and
                       'val_drives' (lists of collectionName values).
    verbose          : Print progress messages (default True).

    Returns
    -------
    GeoFusionResult

    Raises
    ------
    ValueError
        On invalid model/algorithm/estimator combination, missing required
        columns, or missing required estimator_params for DNN.
    """
    if estimator_params is None:
        estimator_params = {}

    _validate(df, model, algorithm, estimator, estimator_params)

    if verbose:
        k = algo_params.get("k", "?")
        print(f"GeoFusion | model={model!r}  algorithm={algorithm!r}  "
              f"k={k}  estimator={estimator!r}")

    tracemalloc.start()
    t_start = time.perf_counter()

    if verbose:
        print(f"  [1/2] Clustering ({algorithm})...", flush=True)

    df_clustered, actual_nc = _run_clustering(df, model, algorithm, algo_params)

    if verbose:
        print(f"        nc={actual_nc}", flush=True)
        print(f"  [2/2] Estimation ({estimator})...", flush=True)

    metrics, df_out = _run_estimation(
        df_clustered, df, estimator, estimator_params)

    _, peak  = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    runtime = time.perf_counter() - t_start

    if verbose:
        m = metrics
        dnn_note = (f" (test: {m.get('n_test_clusters', '?')} clusters)"
                    if estimator == "dnn" else "")
        print(f"  Done | "
              f"V={m['V']:.4f}m  H={m['H']:.4f}m  MAE={m['MAE']:.4f}m"
              f"{dnn_note}  RT={runtime:.2f}s  RAM={peak/1e6:.1f}MB")

    return GeoFusionResult(
        df_out           = df_out,
        metrics          = metrics,
        runtime_s        = runtime,
        peak_RAM_MB      = peak / 1e6,
        clustering_nc    = actual_nc,
        model            = model,
        algorithm        = algorithm,
        estimator        = estimator,
        algo_params      = dict(algo_params),
        estimator_params = dict(estimator_params),
    )
