"""
UK-medoids clustering for GeoFusion simulation data.

Implements the uncertain proximity-based partitional algorithm (Gullo,
Ponti & Tagarelli's UK-medoids / GeoFusion paper's Algorithm 1) using
expected distance between uncertain objects, where each object's
uncertainty is represented by the elliptical volcano distribution:

    g(x, y) = 1 / (2*pi*sigma_N*sigma_E) *
              exp(-x^2 / (2*sigma_N^2) - y^2 / (2*sigma_E^2))

i.e. an independent bivariate Gaussian in local (North, East) meters,
centered at the reported location (proven to be the mean of the volcano
distribution), with direction-specific scales sigma_N and sigma_E.

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

Because there is no closed form for the expected Euclidean distance
between two independent (and, in general, differently-scaled-per-axis)
Gaussians, the expected distance delta(o_i, o_j) is estimated via Monte
Carlo sampling (paired iid samples from each object's distribution),
matching the methodology in the UK-medoids paper.

The clustering loop itself is NOT classic PAM (Kaufman & Rousseeuw swap
optimization). It follows the simpler Lloyd-style scheme used by both
Gullo et al.'s UK-medoids and the GeoFusion paper's "Algorithm 1":
    repeat:
        assign each object to the cluster of its nearest medoid (by δ)
        recompute each cluster's medoid as the member minimizing total
        δ to the other members of its own cluster
    until the medoid set stops changing

Usage (in a notebook cell):
    df_out = run_ukmedoids(
        "real_data_elliptical.csv",
        "real_data_elliptical_ukmedoids.csv",
        columns=("reported_location_lat", "reported_location_lon"),
        k=10,
        random_state=42,
        n_samples=50,
    )
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


# ---------------------------------------------------------------------------
# WGS84 -> local meters conversion (paper Eqs. 17-18), used here only to
# place reported locations on a common local (North, East) meter plane so
# that Gaussian noise (also in meters) can be added/compared consistently.
# ---------------------------------------------------------------------------
def _meters_per_degree(lat_deg, lon_deg):
    phi = np.radians(lat_deg)
    theta = np.radians(lon_deg)
    meters_per_deg_lat = (
        111132.92
        - 559.82 * np.cos(2 * phi)
        + 1.175 * np.cos(4 * phi)
        - 0.0023 * np.cos(6 * phi)
    )
    meters_per_deg_lon = (
        111412.84 * np.cos(theta)
        - 93.5 * np.cos(3 * theta)
        + 0.118 * np.cos(5 * theta)
    )
    return meters_per_deg_lat, meters_per_deg_lon


def _to_local_meters(lat, lon, ref_lat, ref_lon):
    """Convert (lat, lon) arrays to local (N, E) meters relative to a reference point."""
    m_per_lat, m_per_lon = _meters_per_degree(ref_lat, ref_lon)
    n_m = (lat - ref_lat) * m_per_lat
    e_m = (lon - ref_lon) * m_per_lon
    return n_m, e_m


# ---------------------------------------------------------------------------
# Monte Carlo expected-distance matrix
# ---------------------------------------------------------------------------
def _expected_distance_matrix(n_m, e_m, sigma_n, sigma_e, n_samples, random_state):
    """
    Estimate the full n x n matrix of expected (Euclidean) distances
    delta(o_i, o_j) = E[||X_i - X_j||], X_i ~ N((n_m_i, e_m_i), diag(sigma_n_i^2, sigma_e_i^2)),
    via paired Monte Carlo sampling: for each of n_samples draws s, sample
    one point from each object's distribution, compute the pairwise
    Euclidean distance matrix for that sample, and average over samples.
    Since each object's samples are drawn independently of every other
    object's, pairing by sample index still gives an unbiased estimate.

    sigma_n and sigma_e are independent per-object arrays (no fixed ratio
    is assumed between them), so each object can have its own North/East
    scale driven by, e.g., its own speed and satellite count.
    """
    rng = np.random.default_rng(random_state)
    n = n_m.shape[0]

    z_n = rng.standard_normal((n_samples, n))
    z_e = rng.standard_normal((n_samples, n))
    samples_n = n_m[None, :] + z_n * sigma_n[None, :]  # (S, n)
    samples_e = e_m[None, :] + z_e * sigma_e[None, :]  # (S, n)

    delta = np.zeros((n, n))
    for s in range(n_samples):
        points_s = np.column_stack([samples_n[s], samples_e[s]])  # (n, 2)
        delta += cdist(points_s, points_s, metric="euclidean")
    delta /= n_samples

    return delta


# ---------------------------------------------------------------------------
# Lloyd-style uncertain K-medoids loop (assign -> recompute medoid)
# ---------------------------------------------------------------------------
def _kmedoidspp_init(delta: np.ndarray, k: int, rng):
    """
    k-medoids++ seeding, the medoid analogue of k-means++ (Arthur &
    Vassilvitskii, 2007), using the precomputed expected-distance matrix
    delta in place of Euclidean distance: pick the first medoid uniformly
    at random, then each subsequent medoid with probability proportional
    to its squared expected distance from the nearest medoid already
    chosen. Using uniform random initialization instead (the previous
    behavior) makes UK-medoids converge to a noticeably different, often
    worse, local optimum than a k-medoids implementation using smarter
    seeding (e.g. sklearn-extra's KMedoids, which defaults to k-medoids++),
    which is not a real difference in the underlying uncertain-distance
    objective -- only in how well each search procedure explores it.
    """
    n = delta.shape[0]
    medoid_indices = np.empty(k, dtype=int)

    medoid_indices[0] = rng.integers(n)
    closest_sq_dist = delta[:, medoid_indices[0]] ** 2

    for i in range(1, k):
        probs = closest_sq_dist / closest_sq_dist.sum()
        next_idx = rng.choice(n, p=probs)
        medoid_indices[i] = next_idx
        new_sq_dist = delta[:, next_idx] ** 2
        closest_sq_dist = np.minimum(closest_sq_dist, new_sq_dist)

    return medoid_indices


def _uk_medoids_single_run(delta: np.ndarray, k: int, rng, max_iter: int):
    """One Lloyd-style run of UK-medoids from a k-medoids++ initialization."""
    n = delta.shape[0]

    medoid_indices = _kmedoidspp_init(delta, k, rng)
    labels = np.zeros(n, dtype=int)

    for _ in range(max_iter):
        # --- assignment step: each object goes to its nearest medoid ---
        d_to_medoids = delta[:, medoid_indices]
        labels = np.argmin(d_to_medoids, axis=1)

        # --- recompute medoid of each cluster ---
        new_medoid_indices = medoid_indices.copy()
        for cluster_idx in range(k):
            member_mask = labels == cluster_idx
            member_indices = np.where(member_mask)[0]
            if member_indices.size == 0:
                continue  # keep previous medoid if cluster is empty
            sub_delta = delta[np.ix_(member_indices, member_indices)]
            total_dist = sub_delta.sum(axis=1)
            best_local = member_indices[np.argmin(total_dist)]
            new_medoid_indices[cluster_idx] = best_local

        if np.array_equal(np.sort(new_medoid_indices), np.sort(medoid_indices)):
            medoid_indices = new_medoid_indices
            break

        medoid_indices = new_medoid_indices

    # final assignment with converged medoids
    d_to_medoids = delta[:, medoid_indices]
    labels = np.argmin(d_to_medoids, axis=1)

    total_expected_dist = d_to_medoids[np.arange(n), labels].sum()

    return labels, medoid_indices, total_expected_dist


def _uk_medoids_loop(delta: np.ndarray, k: int, random_state: int, n_init: int, max_iter: int):
    """
    Run UK-medoids n_init times from independent k-medoids++ seeds and
    keep the run with lowest total expected distance to medoids (mirrors
    sklearn KMeans' n_init / best-of-restarts behavior, and gives
    UK-medoids the same number of search attempts as run_kmeans gets via
    its own n_init -- the original single-restart version was not a fair
    comparison against a multi-restart k-means/k-medoids baseline).
    """
    rng = np.random.default_rng(random_state)

    best_labels, best_medoids, best_dist = None, None, np.inf
    for _ in range(n_init):
        labels, medoid_indices, total_dist = _uk_medoids_single_run(delta, k, rng, max_iter)
        if total_dist < best_dist:
            best_labels, best_medoids, best_dist = labels, medoid_indices, total_dist

    return best_labels, best_medoids


def run_ukmedoids(
    input_path: str,
    output_path: str,
    columns=("reported_location_lat", "reported_location_lon"),
    k: int = 10,
    random_state: int = 42,
    n_init: int = 10,
    n_samples: int = 50,
    std_n_col: str = "reported_std_N",
    std_e_col: str = "reported_std_E",
    max_iter: int = 300,
) -> pd.DataFrame:
    """
    Run UK-medoids clustering on the given columns of a CSV file and write
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
        Seed for reproducibility (medoid initialization and MC sampling).
    n_init : int
        Number of independent k-medoids++ initializations; the run with
        lowest total expected distance to medoids is kept (mirrors
        sklearn KMeans' n_init), default 10.
    n_samples : int
        Number of Monte Carlo samples used to estimate each expected
        distance, default 50.
    std_n_col : str
        Column holding sigma_N (North-South standard deviation, meters),
        read directly -- no fixed ellipticity ratio is applied. Default
        'reported_std_N'.
    std_e_col : str
        Column holding sigma_E (East-West standard deviation, meters),
        read directly. Default 'reported_std_E'.
    max_iter : int
        Maximum number of assign/recompute iterations.

    Returns
    -------
    pd.DataFrame
        The output dataframe (also written to output_path). predicted_location
        is the (lat, lon) of the medoid -- an actual reported location, not
        an average.
    """
    df = pd.read_csv(input_path)

    lat_col, lon_col = columns
    missing = [c for c in (lat_col, lon_col, std_n_col, std_e_col) if c not in df.columns]
    if missing:
        raise ValueError(f"Column(s) not found in input data: {missing}")

    lat = df[lat_col].to_numpy(dtype=float)
    lon = df[lon_col].to_numpy(dtype=float)
    sigma_n = df[std_n_col].to_numpy(dtype=float)
    sigma_e = df[std_e_col].to_numpy(dtype=float)

    ref_lat, ref_lon = lat.mean(), lon.mean()
    n_m, e_m = _to_local_meters(lat, lon, ref_lat, ref_lon)

    delta = _expected_distance_matrix(n_m, e_m, sigma_n, sigma_e, n_samples, random_state)
    labels, medoid_indices = _uk_medoids_loop(
        delta, k=k, random_state=random_state, n_init=n_init, max_iter=max_iter
    )

    medoid_lat = lat[medoid_indices]
    medoid_lon = lon[medoid_indices]

    df["predicted_cluster"] = labels
    df["predicted_location"] = [
        (float(medoid_lat[lbl]), float(medoid_lon[lbl])) for lbl in labels
    ]

    df.to_csv(output_path, index=False)
    return df
