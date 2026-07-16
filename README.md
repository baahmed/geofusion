# GeoFusion

An uncertain machine learning framework for GPS/GNSS sensor fusion.

GeoFusion estimates true location events from noisy phone-reported GPS coordinates by modelling each observation as an uncertain object, clustering by location event, and fusing the cluster into a refined position estimate.

## Installation

```bash
pip install geofusion
```

## Quick start

```python
import pandas as pd
from geofusion import run_geofusion

df = pd.read_csv("sample_top100_groups.csv")

# UK-medoids (mountain uncertainty model) + Kalman filter
result = run_geofusion(
    df          = df,
    model       = "mountain",
    algorithm   = "ukmedoids",
    algo_params = dict(k=100, random_state=42, n_init=10, n_samples=50),
    estimator   = "kf",
)
print(result)
# GeoFusionResult(
#   model='mountain'  algorithm='ukmedoids'  estimator='kf'
#   nc=100  V=2.75m  H=1.54m  MAE=3.43m
#   runtime=4.5s  peak_RAM=152.1MB
# )

# Access the output dataframe and metrics
df_out  = result.df_out       # original columns + predicted_cluster + predicted_location
metrics = result.metrics       # {'V': ..., 'H': ..., 'MAE': ...}
```

## Uncertainty models

| `model`    | Description | Compatible algorithms |
|------------|-------------|----------------------|
| `certain`  | Raw phone coordinates, no uncertainty | `kmeans`, `kmedoids`, `sdsgc` |
| `volcano`  | Isotropic sigma from satellite geometry cost J_avg (El Abbous & Samanta, 2017) | `ukmeans`, `ukmedoids` |
| `mountain` | Directional Student-t scale from multivariate regression on GSDC dataset | `ukmeans`, `ukmedoids` |

## Clustering algorithms

| `algorithm`  | Description |
|--------------|-------------|
| `kmeans`     | k-means++ (sklearn) |
| `kmedoids`   | k-medoids with k-medoids++ initialisation |
| `ukmeans`    | UK-means (Chau et al., 2006) — provably equivalent to k-means on GPS data |
| `ukmedoids`  | UK-medoids (Gullo et al., 2008) with Monte Carlo expected-distance estimation |
| `sdsgc`      | Structured Doubly Stochastic Graph-Based Clustering (Wang et al., TNNLS 2025) |

## Estimators

| `estimator` | Description |
|-------------|-------------|
| `rep`       | Cluster representative (centroid or medoid) |
| `kf`        | Linear Kalman filter (static target, degree space) |
| `ekf`       | Extended Kalman filter (local metre space — corrects degree-space distortion) |
| `pf`        | Sequential Importance Resampling particle filter |
| `dnn`       | Post-clustering MLP predicting a position correction from cluster-level GNSS features |

## Required dataset columns

| Column | Description |
|--------|-------------|
| `collectionName` | Drive identifier (used for DNN drive-level split) |
| `latDeg_gt` | NovAtel reference latitude (degrees) |
| `lngDeg_gt` | NovAtel reference longitude (degrees) |
| `latDeg_phone` | Phone-reported latitude (degrees) |
| `lngDeg_phone` | Phone-reported longitude (degrees) |
| `j_avg` | Average satellite geometry cost |
| `speedMps` | Vehicle speed (m/s) |
| `n_signals` | Number of satellite signals |
| `avg_rawPrUnc` | Average pseudorange uncertainty (m) |
| `hDop` | Horizontal dilution of precision |
| `vDop` | Vertical dilution of precision |
| `avg_iono` | Average ionospheric delay (m) |
| `avg_tropo` | Average tropospheric delay (m) |

The dataset is publicly available on [Kaggle](https://www.kaggle.com/code/basantmounir/geofusion-dataset).

## algo_params reference

All algorithms accept `k`, `random_state`, `n_init`, `max_iter`.

Additional parameters:
- **`ukmedoids`**: `n_samples` (Monte Carlo samples for expected distance, default 50)
- **`sdsgc`**: `nn` (nearest neighbours, default 5 for k≤50, 4 for k≥100), `strategy` (`early_stop` / `best_of_n` / `threshold`), `threshold` (W-matrix threshold for component extraction), `eigsh_tol`, `eigsh_maxiter`

## estimator_params reference

- **`pf`**: `n_particles` (default 500)
- **`dnn`**: `test_drives` (required), `val_drives` (required), `max_epochs` (200), `patience` (20), `random_state` (42), `subprocess` (False — set True for clean RAM measurement)

## DNN example

```python
result = run_geofusion(
    df               = df,
    model            = "mountain",
    algorithm        = "ukmedoids",
    algo_params      = dict(k=300, random_state=42, n_init=10, n_samples=50),
    estimator        = "dnn",
    estimator_params = dict(
        test_drives = ["2020-08-03-US-MTV-1", "2020-07-08-US-MTV-1", "2021-04-15-US-MTV-1"],
        val_drives  = ["2021-04-28-US-MTV-1", "2021-04-28-US-SJC-1"],
        max_epochs  = 200,
        patience    = 20,
    ),
)
# DNN metrics are evaluated on test-drive clusters only
print(result.metrics)   # {'V': 2.05, 'H': 1.40, 'MAE': 2.71, 'n_test_clusters': 69}
```

## License

MIT
