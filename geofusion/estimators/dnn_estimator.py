"""
Post-Clustering DNN Estimator for GeoFusion.

For each cluster (identified by `predicted_cluster`), trains and applies a
lightweight MLP that predicts a position correction (delta_lat, delta_lon)
relative to the cluster representative (mean phone position), using the
mean GNSS quality features of the cluster's members as input.

--- Pipeline ---

1. Group rows by `predicted_cluster`.
2. Per cluster:
     - Cluster representative : mean(latDeg_phone), mean(lngDeg_phone)
     - Quality features        : mean of FEATURE_COLS across members
     - Training label          : mean(latDeg_gt) - rep_lat,
                                 mean(lngDeg_gt) - rep_lon  (delta correction)
3. Train MLP on cluster-level dataset split by drive (collectionName).
4. At inference: corrected position = representative + predicted delta.

--- Why predict a delta? ---

The cluster representative is already within metres of ground truth.
Predicting (delta_lat, delta_lon) keeps targets near zero (std ~1e-5 deg)
making training fast and well-conditioned. Predicting absolute coordinates
(~37 deg lat, ~-122 deg lon) would require the model to learn those large
offsets before learning anything useful.

--- Feature columns (8 quality features + 2 position = 10 inputs) ---

    rep_lat, rep_lon            cluster representative (degrees)
    j_avg                       satellite geometry cost
    speedMps                    vehicle speed (m/s)
    n_signals                   number of satellite signals
    avg_rawPrUnc                average pseudorange uncertainty (m)
    hDop                        horizontal dilution of precision
    vDop                        vertical dilution of precision
    avg_iono                    average ionospheric delay (m)
    avg_tropo                   average tropospheric delay (m)

--- Notebook usage ---

    from dnn_estimator import build_cluster_dataset, train_dnn, predict_dnn

    df_clustered = pd.read_csv("kmeans_output.csv")
    df_clusters  = build_cluster_dataset(df_clustered)

    all_drives   = df_clusters['collectionName'].unique().tolist()
    val_drives   = all_drives[:4]
    train_drives = all_drives[4:]

    model, scalers = train_dnn(df_clusters, train_drives, val_drives,
                               model_path="dnn_model.pt",
                               scaler_path="dnn_scaler.pkl")

    df_out  = predict_dnn(df_clustered, model, scalers,
                          output_path="dnn_predictions.csv")
"""

from __future__ import annotations

import argparse
import pickle
import warnings
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FEATURE_COLS: List[str] = [
    "j_avg", "speedMps", "n_signals", "avg_rawPrUnc",
    "hDop", "vDop", "avg_iono", "avg_tropo",
]
GT_COLS:    List[str] = ["latDeg_gt", "lngDeg_gt"]
DRIVE_COL:  str       = "collectionName"
INPUT_COLS: List[str] = ["rep_lat", "rep_lon"] + FEATURE_COLS   # dim = 10

_REQUIRED = ["predicted_cluster"] + FEATURE_COLS + GT_COLS + [DRIVE_COL,
             "latDeg_phone", "lngDeg_phone"]


def _meters_per_degree(lat_deg: float) -> Tuple[float, float]:
    phi   = np.radians(lat_deg)
    m_lat = (111132.92 - 559.82 * np.cos(2 * phi)
             + 1.175 * np.cos(4 * phi) - 0.0023 * np.cos(6 * phi))
    m_lon = (111412.84 * np.cos(phi)
             - 93.5 * np.cos(3 * phi) + 0.118 * np.cos(5 * phi))
    return float(m_lat), float(m_lon)


# ---------------------------------------------------------------------------
# Step 1 — Build cluster-level dataset
# ---------------------------------------------------------------------------

def build_cluster_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse a per-row clustering output into a cluster-level dataframe.

    Groups by `predicted_cluster`. Per cluster:
      - rep_lat / rep_lon : mean(latDeg_phone), mean(lngDeg_phone)
      - FEATURE_COLS      : mean across all members
      - delta_lat/lon     : mean(latDeg_gt) - rep_lat  (correction target)
      - collectionName    : majority drive among members
      - n_members         : number of rows

    Parameters
    ----------
    df : pd.DataFrame
        Output of any GeoFusion clustering module.

    Returns
    -------
    pd.DataFrame — one row per cluster.
    """
    missing = [c for c in _REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    records = []
    for cluster_id, grp in df.groupby("predicted_cluster", sort=False):
        rep_lat = float(grp["latDeg_phone"].mean())
        rep_lon = float(grp["lngDeg_phone"].mean())
        gt_lat  = float(grp["latDeg_gt"].mean())
        gt_lon  = float(grp["lngDeg_gt"].mean())

        rec = {
            "predicted_cluster": cluster_id,
            "rep_lat":    rep_lat,
            "rep_lon":    rep_lon,
            "n_members":  len(grp),
            "label_lat":  gt_lat,
            "label_lon":  gt_lon,
            "delta_lat":  gt_lat - rep_lat,
            "delta_lon":  gt_lon - rep_lon,
            DRIVE_COL:    grp[DRIVE_COL].mode().iloc[0],
        }
        for col in FEATURE_COLS:
            rec[col] = float(grp[col].mean())
        records.append(rec)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Step 2 — MLP architecture
# ---------------------------------------------------------------------------

class _PostClusterMLP(nn.Module):
    """
    MLP predicting (delta_lat, delta_lon).
    Input  : 10-d [rep_lat, rep_lon, 8 quality features]  (standardised)
    Output : 2-d  [delta_lat, delta_lon]                   (standardised)
    Hidden : [64, 32, 16] with ReLU + Dropout(p=dropout).
    """

    def __init__(self, input_dim: int = 10,
                 hidden: Tuple[int, ...] = (64, 32, 16),
                 dropout: float = 0.1):
        super().__init__()
        layers: list = []
        prev = input_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Step 3 — Training
# ---------------------------------------------------------------------------

def train_dnn(
    df_clusters: pd.DataFrame,
    train_drives: List[str],
    val_drives: List[str],
    model_path: str = "dnn_model.pt",
    scaler_path: str = "dnn_scaler.pkl",
    hidden: Tuple[int, ...] = (64, 32, 16),
    dropout: float = 0.1,
    lr: float = 1e-3,
    batch_size: int = 64,
    max_epochs: int = 200,
    patience: int = 20,
    random_state: int = 42,
    device: Optional[str] = None,
) -> Tuple[_PostClusterMLP, dict]:
    """
    Train the Post-C DNN on the cluster-level dataset.

    Both inputs and correction targets are standardised independently using
    training-split statistics so gradients are well-conditioned.

    Parameters
    ----------
    df_clusters  : output of build_cluster_dataset()
    train_drives : collectionName values for training
    val_drives   : collectionName values for early stopping
    model_path   : path to save model state dict (.pt)
    scaler_path  : path to save scalers dict (.pkl)
                   Keys: 'input' -> StandardScaler, 'target' -> StandardScaler
    hidden       : hidden layer sizes. Default (64, 32, 16)
    dropout      : dropout rate. Default 0.1
    lr           : Adam learning rate. Default 1e-3
    batch_size   : mini-batch size. Default 64
    max_epochs   : maximum epochs. Default 200
    patience     : early stopping patience. Default 20
    random_state : reproducibility seed. Default 42
    device       : 'cuda', 'cpu', or None (auto). Default None

    Returns
    -------
    model   : _PostClusterMLP (CPU, eval mode)
    scalers : dict with keys 'input' and 'target'
    """
    torch.manual_seed(random_state)
    np.random.seed(random_state)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    df_tr = df_clusters[df_clusters[DRIVE_COL].isin(train_drives)].copy()
    df_va = df_clusters[df_clusters[DRIVE_COL].isin(val_drives)].copy()

    if len(df_tr) == 0:
        raise ValueError("No clusters found for train_drives.")
    if len(df_va) == 0:
        warnings.warn("No clusters found for val_drives; monitoring train loss.")
        df_va = df_tr

    X_tr = df_tr[INPUT_COLS].to_numpy(dtype=np.float32)
    y_tr = df_tr[["delta_lat", "delta_lon"]].to_numpy(dtype=np.float32)
    X_va = df_va[INPUT_COLS].to_numpy(dtype=np.float32)
    y_va = df_va[["delta_lat", "delta_lon"]].to_numpy(dtype=np.float32)

    scaler_x = StandardScaler().fit(X_tr)
    scaler_y = StandardScaler().fit(y_tr)

    X_tr = scaler_x.transform(X_tr).astype(np.float32)
    y_tr = scaler_y.transform(y_tr).astype(np.float32)
    X_va = scaler_x.transform(X_va).astype(np.float32)
    y_va = scaler_y.transform(y_va).astype(np.float32)

    Xt = torch.from_numpy(X_tr.copy()).to(device)
    yt = torch.from_numpy(y_tr.copy()).to(device)
    Xv = torch.from_numpy(X_va.copy()).to(device)
    yv = torch.from_numpy(y_va.copy()).to(device)

    model     = _PostClusterMLP(input_dim=len(INPUT_COLS), hidden=hidden,
                                dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.L1Loss()

    n_tr              = Xt.shape[0]
    best_val_loss     = np.inf
    best_state        = None
    epochs_no_improve = 0

    for epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(n_tr, device=device)
        for start in range(0, n_tr, batch_size):
            idx  = perm[start:start + batch_size]
            loss = criterion(model(Xt[idx]), yt[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(Xv), yv).item()

        if val_loss < best_val_loss:
            best_val_loss     = val_loss
            best_state        = {k: v.cpu().clone()
                                 for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch + 1}  "
                  f"(best val loss: {best_val_loss:.6f} std-units)")
            break
    else:
        print(f"Completed {max_epochs} epochs  "
              f"(best val loss: {best_val_loss:.6f} std-units)")

    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.cpu().eval()

    scalers = {"input": scaler_x, "target": scaler_y}
    torch.save(model.state_dict(), model_path)
    with open(scaler_path, "wb") as f:
        pickle.dump(scalers, f)

    print(f"Model  saved → {model_path}")
    print(f"Scaler saved → {scaler_path}")
    return model, scalers


# ---------------------------------------------------------------------------
# Step 4 — Inference
# ---------------------------------------------------------------------------

def predict_dnn(
    df: pd.DataFrame,
    model: _PostClusterMLP,
    scalers: dict,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Apply trained Post-C DNN to a clustering output dataframe.

    Corrected position = cluster representative + predicted (delta_lat, delta_lon).
    Result stored in `predicted_location_dnn` as (lat, lon) tuples.

    Parameters
    ----------
    df          : clustering output dataframe
    model       : trained _PostClusterMLP (eval mode)
    scalers     : dict {'input': scaler, 'target': scaler} from train_dnn
    output_path : if provided, write result CSV here

    Returns
    -------
    pd.DataFrame with additional column `predicted_location_dnn`.
    """
    df_cl = build_cluster_dataset(df)

    X      = df_cl[INPUT_COLS].to_numpy(dtype=np.float32)
    X_sc   = scalers["input"].transform(X).astype(np.float32)

    model.eval()
    with torch.no_grad():
        delta_sc = model(torch.from_numpy(X_sc)).numpy()

    delta = scalers["target"].inverse_transform(delta_sc)

    df_cl["dnn_lat"] = df_cl["rep_lat"].to_numpy() + delta[:, 0]
    df_cl["dnn_lon"] = df_cl["rep_lon"].to_numpy() + delta[:, 1]

    # lookup by predicted_cluster integer — clean and unambiguous
    lookup = dict(zip(
        df_cl["predicted_cluster"].tolist(),
        zip(df_cl["dnn_lat"].tolist(), df_cl["dnn_lon"].tolist()),
    ))

    df = df.copy()
    df["predicted_location_dnn"] = df["predicted_cluster"].map(lookup)

    if output_path is not None:
        df.to_csv(output_path, index=False)
        print(f"DNN predictions written to {output_path}")

    return df


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

def evaluate_dnn(
    df: pd.DataFrame,
    pred_col: str = "predicted_location_dnn",
) -> pd.DataFrame:
    """
    Per-cluster radial error in metres between DNN predictions and mean GT.

    Returns pd.DataFrame with columns:
        predicted_cluster, rep_lat, rep_lon, dnn_lat, dnn_lon,
        gt_lat, gt_lon, error_m, collectionName.
    """
    df = df.copy()
    df["_dnn_lat"] = df[pred_col].apply(lambda t: float(t[0]))
    df["_dnn_lon"] = df[pred_col].apply(lambda t: float(t[1]))

    records = []
    for cluster_id, grp in df.groupby("predicted_cluster", sort=False):
        dnn_lat = float(grp["_dnn_lat"].iloc[0])
        dnn_lon = float(grp["_dnn_lon"].iloc[0])
        gt_lat  = float(grp["latDeg_gt"].mean())
        gt_lon  = float(grp["lngDeg_gt"].mean())
        rep_lat = float(grp["latDeg_phone"].mean())
        rep_lon = float(grp["lngDeg_phone"].mean())

        m_lat, m_lon = _meters_per_degree(gt_lat)
        error_m = np.sqrt(
            ((dnn_lat - gt_lat) * m_lat) ** 2 +
            ((dnn_lon - gt_lon) * m_lon) ** 2
        )
        records.append({
            "predicted_cluster": cluster_id,
            "rep_lat": rep_lat, "rep_lon": rep_lon,
            "dnn_lat": dnn_lat, "dnn_lon": dnn_lon,
            "gt_lat":  gt_lat,  "gt_lon":  gt_lon,
            "error_m": error_m,
            DRIVE_COL: grp[DRIVE_COL].mode().iloc[0],
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Convenience loader
# ---------------------------------------------------------------------------

def load_model(
    model_path: str,
    scaler_path: str,
    hidden: Tuple[int, ...] = (64, 32, 16),
    dropout: float = 0.1,
) -> Tuple[_PostClusterMLP, dict]:
    """Load a previously saved model and scalers from disk."""
    with open(scaler_path, "rb") as f:
        scalers = pickle.load(f)
    model = _PostClusterMLP(input_dim=len(INPUT_COLS), hidden=hidden,
                            dropout=dropout)
    model.load_state_dict(
        torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()
    return model, scalers


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GeoFusion Post-Clustering DNN Estimator")
    sub = parser.add_subparsers(dest="command", required=True)

    tr = sub.add_parser("train")
    tr.add_argument("--input",      required=True)
    tr.add_argument("--model",      default="dnn_model.pt")
    tr.add_argument("--scaler",     default="dnn_scaler.pkl")
    tr.add_argument("--val-drives", default="")
    tr.add_argument("--max-epochs", type=int,   default=200)
    tr.add_argument("--patience",   type=int,   default=20)
    tr.add_argument("--lr",         type=float, default=1e-3)
    tr.add_argument("--batch-size", type=int,   default=64)
    tr.add_argument("--seed",       type=int,   default=42)

    pr = sub.add_parser("predict")
    pr.add_argument("--input",  required=True)
    pr.add_argument("--output", required=True)
    pr.add_argument("--model",  default="dnn_model.pt")
    pr.add_argument("--scaler", default="dnn_scaler.pkl")

    args = parser.parse_args()

    if args.command == "train":
        df_raw   = pd.read_csv(args.input)
        df_cl    = build_cluster_dataset(df_raw)
        all_dr   = df_cl[DRIVE_COL].unique().tolist()
        val_dr   = [d.strip() for d in args.val_drives.split(",") if d.strip()]
        train_dr = [d for d in all_dr if d not in val_dr]
        print(f"Drives — train: {len(train_dr)}, val: {len(val_dr)}")
        train_dnn(df_cl, train_dr, val_dr,
                  model_path=args.model, scaler_path=args.scaler,
                  max_epochs=args.max_epochs, patience=args.patience,
                  lr=args.lr, batch_size=args.batch_size,
                  random_state=args.seed)

    elif args.command == "predict":
        model, scalers = load_model(args.model, args.scaler)
        df_raw = pd.read_csv(args.input)
        predict_dnn(df_raw, model, scalers, output_path=args.output)
