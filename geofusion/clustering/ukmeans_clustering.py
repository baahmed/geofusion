"""
UK-means clustering for GeoFusion simulation data.

Implements the uncertain proximity-based partitional algorithm (GeoFusion
paper's Algorithm 1, UK-means flavor) using the expected squared distance
between each uncertain object (elliptical volcano distribution) and the
(certain) cluster centroid.

--- Why this provably reduces to certain k-means ---

Each uncertain object X = (x, y) is independently Gaussian-distributed in
local (North, East) meters:
    x ~ N(mu_N, sigma_N^2),  y ~ N(mu_E, sigma_E^2)
with (mu_N, mu_E) = the reported location (proven: the mean of the volcano
distribution is the reported location itself).

For a deterministic centroid rho = (rho_N, rho_E), the expected squared
distance is:
    E[||X - rho||^2] = E[(x - rho_N)^2] + E[(y - rho_E)^2]
                      = Var(x) + (mu_N - rho_N)^2 + Var(y) + (mu_E - rho_E)^2
                      = (sigma_N^2 + sigma_E^2) + ||mu - rho||^2

The term (sigma_N^2 + sigma_E^2) is a per-object constant: it does not
depend on which cluster/centroid the object is being compared to. So:
  - Assignment step: argmin_rho E[||X-rho||^2] = argmin_rho ||mu-rho||^2,
    i.e. identical to assigning the certain reported location to its
    nearest centroid.
  - Centroid step: per GeoFusion Eq. 6, the UK-means centroid is the mean
    of the member means, i.e. exactly the certain k-means centroid (mean
    of reported locations).

So UK-means and k-means converge to the identical clustering. This module
still implements the explicit uncertain formulation (rather than silently
calling k-means) so the derivation stays visible and verifiable in code,
and so the per-object uncertainty term is available if you want to report
it (e.g. as an "expected within-cluster dispersion" diagnostic).

--- sigma_N / sigma_E source ---

sigma_N and sigma_E are read directly as two independent columns rather
than derived from a single sigma via a fixed ellipticity ratio. They are
expected to already be populated using the empirically-fitted exponential
model driven by sensor speed and visible satellite count:

    sigma_N(speed, n_sat) = 14.0726 * exp(-0.02441*speed - 0.05900*n_sat)
    sigma_E(speed, n_sat) = 12.1145 * exp(-0.02778*speed - 0.06770*n_sat)

(both fit on real GPS observations; R^2 = 0.889 and 0.920 respectively).
This module does not recompute sigma_N/sigma_E from speed/n_sat itself --
it assumes the input CSV's std columns already hold the correct values,
keeping this module agnostic to whichever sigma model produced them.

Usage (in a notebook cell):
    df_out = run_ukmeans(
        "real_data_elliptical.csv",
        "real_data_elliptical_ukmeans.csv",
        columns=("reported_location_lat", "reported_location_lon"),
        k=10,
        random_state=42,
    )
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


def _kmeans_plusplus_init(X: np.ndarray, k: int, rng):
    """
    k-means++ seeding (Arthur & Vassilvitskii, 2007): pick the first
    centroid uniformly at random, then each subsequent centroid with
    probability proportional to its squared distance from the nearest
    centroid already chosen. This spreads initial centroids out and is
    what sklearn's KMeans uses by default -- using uniform random
    initialization here instead would make UK-means and k-means converge
    to different local optima purely due to a weaker starting point, not
    because of any real difference between the two objectives (which are
    proven equivalent in the assignment step; see module docstring).
    """
    n = X.shape[0]
    centroids = np.empty((k, X.shape[1]))

    first_idx = rng.integers(n)
    centroids[0] = X[first_idx]

    closest_sq_dist = np.sum((X - centroids[0]) ** 2, axis=1)

    for i in range(1, k):
        probs = closest_sq_dist / closest_sq_dist.sum()
        next_idx = rng.choice(n, p=probs)
        centroids[i] = X[next_idx]
        new_sq_dist = np.sum((X - centroids[i]) ** 2, axis=1)
        closest_sq_dist = np.minimum(closest_sq_dist, new_sq_dist)

    return centroids


def _ukmeans_single_run(X: np.ndarray, uncertainty_term: np.ndarray, k: int, rng, max_iter: int):
    """
    One Lloyd-style run of UK-means from a k-means++ centroid initialization.

    Returns (labels, centroids, expected_inertia).
    """
    n = X.shape[0]
    centroids = _kmeans_plusplus_init(X, k, rng)
    labels = np.zeros(n, dtype=int)

    for _ in range(max_iter):
        sq_dist = cdist(X, centroids, metric="sqeuclidean")          # ||mu - rho||^2
        expected_sq_dist = sq_dist + uncertainty_term[:, None]        # + constant per row
        new_labels = np.argmin(expected_sq_dist, axis=1)

        # vectorized centroid recompute: mean of X per cluster (Eq. 6)
        sums = np.zeros((k, X.shape[1]))
        counts = np.zeros(k)
        np.add.at(sums, new_labels, X)
        np.add.at(counts, new_labels, 1)
        nonempty = counts > 0
        new_centroids = centroids.copy()
        new_centroids[nonempty] = sums[nonempty] / counts[nonempty, None]

        converged = np.array_equal(new_labels, labels) and np.allclose(new_centroids, centroids)
        labels = new_labels
        centroids = new_centroids
        if converged:
            break

    final_sq_dist = cdist(X, centroids, metric="sqeuclidean")
    final_expected_sq_dist = final_sq_dist + uncertainty_term[:, None]
    expected_inertia = final_expected_sq_dist[np.arange(n), labels].sum()

    return labels, centroids, expected_inertia


def run_ukmeans(
    input_path: str,
    output_path: str,
    columns=("reported_location_lat", "reported_location_lon"),
    k: int = 10,
    random_state: int = 42,
    n_init: int = 10,
    std_n_col: str = "reported_std_N",
    std_e_col: str = "reported_std_E",
    max_iter: int = 300,
) -> pd.DataFrame:
    """
    Run UK-means clustering on the given columns of a CSV file and write
    the result (original data + predicted_cluster + predicted_location)
    to output_path.

    Parameters
    ----------
    input_path : str
        Path to the input CSV file.
    output_path : str
        Path to write the output CSV file to.
    columns : tuple of str
        (lat, lon) column names to cluster on, default
        ('reported_location_lat', 'reported_location_lon').
    k : int
        Number of clusters, default 10.
    random_state : int
        Seed for reproducibility.
    n_init : int
        Number of random centroid initializations; the run with lowest
        expected inertia is kept (mirrors sklearn KMeans' n_init).
    std_n_col : str
        Column holding sigma_N (North-South standard deviation, meters),
        read directly -- no fixed ellipticity ratio is applied. Default
        'reported_std_N'.
    std_e_col : str
        Column holding sigma_E (East-West standard deviation, meters),
        read directly. Default 'reported_std_E'.
    max_iter : int
        Maximum number of assign/recompute iterations per init.

    Returns
    -------
    pd.DataFrame
        The output dataframe (also written to output_path).
    """
    df = pd.read_csv(input_path)

    lat_col, lon_col = columns
    missing = [c for c in (lat_col, lon_col, std_n_col, std_e_col) if c not in df.columns]
    if missing:
        raise ValueError(f"Column(s) not found in input data: {missing}")

    X = df[[lat_col, lon_col]].to_numpy(dtype=float)

    sigma_n = df[std_n_col].to_numpy(dtype=float)
    sigma_e = df[std_e_col].to_numpy(dtype=float)
    uncertainty_term = sigma_n**2 + sigma_e**2  # Var(N) + Var(E), per-object constant

    rng = np.random.default_rng(random_state)

    best_labels, best_centroids, best_inertia = None, None, np.inf
    for _ in range(n_init):
        labels, centroids, inertia = _ukmeans_single_run(X, uncertainty_term, k, rng, max_iter)
        if inertia < best_inertia:
            best_labels, best_centroids, best_inertia = labels, centroids, inertia

    df["predicted_cluster"] = best_labels
    df["predicted_location"] = [
        tuple(float(v) for v in best_centroids[lbl]) for lbl in best_labels
    ]

    df.to_csv(output_path, index=False)
    return df
