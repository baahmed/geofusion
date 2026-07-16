"""
GeoFusion
=========

An uncertain machine learning framework for GPS/GNSS sensor fusion.

Implements the two-phase pipeline from Mounir & Abdel-Hamid (IEEE Access):

    1. **Representation** — each GPS observation is modelled as an uncertain
       object (volcano model or mountain model) reflecting the positional
       uncertainty given satellite geometry, speed, and signal quality.

    2. **Clustering** — observations are grouped by location event using
       either certain or uncertain clustering algorithms.

    3. **Estimation** — observations within each cluster are fused into a
       single refined position estimate.

Quick start
-----------
    import pandas as pd
    from geofusion import run_geofusion

    df = pd.read_csv("sample_top100_groups.csv")

    # UK-medoids (mountain) + Kalman filter
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

    # SDSGC (certain) + particle filter
    result = run_geofusion(
        df          = df,
        model       = "certain",
        algorithm   = "sdsgc",
        algo_params = dict(k=100, nn=4, strategy="threshold", threshold=0.1),
        estimator   = "pf",
    )

    # Post-clustering DNN (test-drive evaluation)
    result = run_geofusion(
        df               = df,
        model            = "mountain",
        algorithm        = "ukmedoids",
        algo_params      = dict(k=100, random_state=42, n_init=10, n_samples=50),
        estimator        = "dnn",
        estimator_params = dict(
            test_drives = ["2020-08-03-US-MTV-1", "2020-07-08-US-MTV-1"],
            val_drives  = ["2021-04-28-US-MTV-1"],
        ),
    )

Uncertainty models
------------------
    certain   No uncertainty. Raw phone coordinates.
              Algorithms: kmeans, kmedoids, sdsgc

    volcano   Model A — isotropic sigma from satellite geometry cost J_avg
              (El Abbous & Samanta, 2017). Zero-shot transfer, no fitting.
              sigma = max(0.787 * j_avg + 2.192, 0.01)
              Algorithms: ukmeans, ukmedoids

    mountain  Model B — directional Student-t scale from multivariate
              regression on the GSDC dataset (Eqs 11-12 in the paper).
              Sigma floor: max(s, 0.01) — NOT abs(s).
              Algorithms: ukmeans, ukmedoids

Clustering algorithms
---------------------
    kmeans    k-means++ (sklearn) — certain only
    kmedoids  k-medoids (Lloyd-style, k-medoids++ init) — certain only
    ukmeans   UK-means (Chau et al., 2006) — volcano or mountain
              Note: provably equivalent to k-means on GPS data
    ukmedoids UK-medoids (Gullo et al., 2008) — volcano or mountain
    sdsgc     Structured Doubly Stochastic Graph-Based Clustering
              (Wang et al., TNNLS 2025) — certain only

Estimators
----------
    rep   Cluster representative (centroid or medoid) — no post-processing
    kf    Linear Kalman filter (degree-space, static target)
    ekf   Extended Kalman filter (local metre-space, corrects degree distortion)
    pf    Sequential Importance Resampling particle filter
    dnn   Post-clustering MLP (Siemuri et al., 2021, adapted for GeoFusion)
          Requires 'test_drives' and 'val_drives' in estimator_params.

Required dataset columns
------------------------
    collectionName   drive identifier (used for DNN drive-level split)
    latDeg_gt        NovAtel reference latitude (degrees)
    lngDeg_gt        NovAtel reference longitude (degrees)
    latDeg_phone     phone-reported latitude (degrees)
    lngDeg_phone     phone-reported longitude (degrees)
    j_avg            average satellite geometry cost
    speedMps         vehicle speed (m/s)
    n_signals        number of satellite signals
    avg_rawPrUnc     average pseudorange uncertainty (m)
    hDop             horizontal dilution of precision
    vDop             vertical dilution of precision
    avg_iono         average ionospheric delay (m)
    avg_tropo        average tropospheric delay (m)

References
----------
    Mounir, B. & Abdel-Hamid, A.T. (2025). GeoFusion: An Uncertain Machine
        Learning Framework for Sensor Fusion. IEEE Access.
    El Abbous, A. & Samanta, N. (2017). A Modeling of GPS Error Distributions.
        EURONAV. DOI: 10.1109/EURONAV.2017.7954200
    Chau, M. et al. (2006). Uncertain Data Mining: An Example in Clustering
        Location Data. PAKDD. DOI: 10.1007/11731139_24
    Gullo, F. et al. (2008). Clustering Uncertain Data Via K-Medoids. MDAI.
        DOI: 10.1007/978-3-540-87993-0_19
    Wang, N. et al. (2025). Structured Doubly Stochastic Graph-Based
        Clustering. IEEE TNNLS. DOI: 10.1109/TNNLS.2025.3531987
    Siemuri, A. et al. (2021). Improving Precision GNSS Positioning and
        Navigation Accuracy on Smartphones Using Machine Learning. ION GNSS+.
"""

from .core import run_geofusion, GeoFusionResult
from .core import _REQUIRED_COLS as REQUIRED_COLS

__all__ = [
    "run_geofusion",
    "GeoFusionResult",
    "REQUIRED_COLS",
]

__version__ = "0.1.0"
__author__  = "Basant Mounir, Amr T. Abdel-Hamid"
