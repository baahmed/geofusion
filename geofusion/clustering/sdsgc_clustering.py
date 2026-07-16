"""
SDSGC clustering for GeoFusion.

Implements the Structured Doubly Stochastic Graph-Based Clustering (SDSGC)
model of Wang et al. [TNNLS 2025, doi:10.1109/TNNLS.2025.3531987], ported
from the authors' public MATLAB release (github.com/NianWang-HJJGCDX/SDSGC)
to Python, and wrapped in the standard GeoFusion run_* interface.

--- What SDSGC is ---

SDSGC is a graph-based spectral clustering method. It takes a set of certain
(deterministic) data points, builds an affinity graph over them, and learns a
structured doubly stochastic matrix W such that:
  - W >= 0 (non-negative)
  - W 1 = 1 (rows sum to one)
  - W = W' (symmetric)
  - rank(L_W) = n - k (W has exactly k connected components)

Cluster membership is read directly from the connected components of the
converged W, requiring no separate k-means post-processing step. The
optimization is solved via an augmented Lagrangian multiplier (ALM) scheme
that enforces all doubly stochastic conditions simultaneously.

--- Why SDSGC does NOT use uncertain objects ---

Unlike UK-means, UK-medoids, and GEO-OPT, SDSGC operates exclusively on
certain feature vectors. It has no notion of a data point as a probability
distribution, no expected-distance computation, and makes no use of the
sigma_N / sigma_E uncertainty columns. This is by design: SDSGC and the
four related works cited by reviewer R6 (patcog.2026.113698,
neucom.2025.132099, sigpro.2025.110144, neucom.2025.130571) are all
graph-based spectral clustering methods for certain high-dimensional data.
They belong to a different algorithmic paradigm than the uncertain object
clustering used in GeoFusion. SDSGC is included here to empirically
demonstrate the performance gap when uncertainty information is discarded,
in response to reviewer R6's request to discuss graph uncertain clustering
models and clarify GeoFusion's distinctiveness.

--- predicted_location convention ---

SDSGC's output is cluster membership from connected components; it produces
no medoid or centroid internally. For consistency with the GeoFusion
evaluation framework, predicted_location is set to the centroid (mean of
latDeg_phone / lngDeg_phone) of each connected component.

--- Parameters ---

The MATLAB implementation exposes:
  k         : number of *neighbors* for the initial graph (not clusters)
  gamma     : regularization weight on ||W||_F^2
  eta       : weight on the spectral embedding term
  local     : whether to enforce sparsity (1) or use all neighbors (0)

In our wrapper, `k` follows the GeoFusion convention (number of clusters).
`n_neighbors` is the graph construction parameter (MATLAB's k), defaulting
to 5 as used in the authors' own benchmark experiments.

Usage (in a notebook cell):
    df_out = run_sdsgc(
        "real_data_elliptical.csv",
        "real_data_elliptical_sdsgc.csv",
        columns=("latDeg_phone", "lngDeg_phone"),
        k=10,
        random_state=42,
    )
"""

import warnings

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import eigsh
from scipy.spatial.distance import cdist
from sklearn.preprocessing import normalize


# ---------------------------------------------------------------------------
# Core subroutines (ported from MATLAB)
# ---------------------------------------------------------------------------

def _l2_distance(X: np.ndarray) -> np.ndarray:
    """
    Pairwise squared Euclidean distance matrix (MATLAB: L2_distance_1).
    X : (n, d) array of data points.
    Returns (n, n) matrix D where D[i,j] = ||x_i - x_j||^2.
    """
    sq = np.sum(X ** 2, axis=1, keepdims=True)
    D = sq + sq.T - 2.0 * (X @ X.T)
    np.fill_diagonal(D, 0.0)
    D = np.maximum(D, 0.0)   # numerical safety: kill tiny negatives
    return D


def _sym_neighbors(dist_sq: np.ndarray, n_neighbors: int):
    """
    Build a symmetric k-nearest-neighbor affinity graph (MATLAB: symNeighbors).
    Uses a self-tuned Gaussian kernel: sigma_i = dist to k-th neighbor of i.

    Returns
    -------
    W0 : (n, n) initial doubly-stochastic-normalized affinity matrix
    knn_sets : list of length n; knn_sets[i] = indices of i's k nearest neighbors
               (excluding self), used for local sparsity enforcement
    """
    n = dist_sq.shape[0]
    # k+1 because argpartition includes self at distance 0
    knn_idx = np.argpartition(dist_sq, n_neighbors + 1, axis=1)[:, :n_neighbors + 1]

    # self-tuned sigma: distance to the k-th nearest neighbor (excluding self)
    knn_sets = []
    sigma = np.zeros(n)
    for i in range(n):
        neighbors = knn_idx[i][knn_idx[i] != i][:n_neighbors]
        knn_sets.append(neighbors)
        sigma[i] = np.sqrt(dist_sq[i, neighbors[-1]]) + 1e-10

    # Gaussian affinity with self-tuned sigma
    W0 = np.exp(-dist_sq / (sigma[:, None] * sigma[None, :]))
    np.fill_diagonal(W0, 0.0)

    # symmetrize and normalize to doubly stochastic via Sinkhorn-Knopp
    W0 = (W0 + W0.T) / 2.0
    W0 = _sinkhorn(W0, n_iter=30)
    return W0, knn_sets


def _sinkhorn(W: np.ndarray, n_iter: int = 30) -> np.ndarray:
    """
    Sinkhorn-Knopp normalization: iteratively normalize rows then columns
    to produce an approximately doubly stochastic matrix.
    """
    W = np.maximum(W, 0.0)
    for _ in range(n_iter):
        row_sums = W.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        W = W / row_sums
        col_sums = W.sum(axis=0, keepdims=True)
        col_sums[col_sums == 0] = 1.0
        W = W / col_sums
    return W


def _eig_laplacian(W: np.ndarray, k: int):
    """
    Compute the k smallest eigenvectors of the graph Laplacian L = D - W
    (MATLAB: eig1). Returns (F, eigenvalues) where F is (n, k).
    """
    D = np.diag(W.sum(axis=1))
    L = D - W
    # eigsh finds smallest algebraic eigenvalues; sigma=0 uses shift-invert
    try:
        vals, vecs = eigsh(csr_matrix(L), k=k + 1, which='SM', tol=1e-6,
                           maxiter=2000)
    except Exception:
        # fallback: dense eigen-decomposition
        vals, vecs = np.linalg.eigh(L)
        vecs = vecs[:, :k + 1]
        vals = vals[:k + 1]

    # sort ascending
    order = np.argsort(vals)
    vals = vals[order]
    vecs = vecs[:, order]
    # drop the trivial zero eigenvector (first), keep k non-trivial ones
    F = vecs[:, 1:k + 1]
    ev = vals
    return F, ev


def _proj_simplex(v: np.ndarray) -> np.ndarray:
    """
    Project vector v onto the probability simplex {x >= 0, sum(x) = 1}
    (MATLAB: EProjSimplex_new). Uses the O(n log n) algorithm of
    Duchi et al. (2008).
    """
    n = len(v)
    if n == 0:
        return v.copy()
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    condition = u * np.arange(1, n + 1) > (cssv - 1)
    nonzero = np.nonzero(condition)[0]
    if len(nonzero) == 0:
        # all mass goes to the largest element
        result = np.zeros(n)
        result[np.argmax(v)] = 1.0
        return result
    rho = nonzero[-1]
    theta = (cssv[rho] - 1.0) / (rho + 1.0)
    return np.maximum(v - theta, 0.0)


# ---------------------------------------------------------------------------
# Main SDSGC loop (ported from MATLAB)
# ---------------------------------------------------------------------------

def _sdsgc(X: np.ndarray, k: int, gamma: float, eta: float,
           n_neighbors: int, local: bool, max_iter: int):
    """
    Core SDSGC optimization loop.

    Parameters
    ----------
    X          : (n, d) certain data matrix (each row is a data point)
    k          : number of clusters
    gamma      : regularization on ||W||_F^2
    eta        : weight on spectral embedding distance term
    n_neighbors: number of nearest neighbors for initial graph
    local      : if True, enforce row-sparsity to knn support only
    max_iter   : maximum ALM iterations

    Returns
    -------
    labels : (n,) integer cluster labels
    W      : (n, n) learned doubly stochastic matrix
    """
    n = X.shape[0]

    dist_sq = _l2_distance(X)
    W, knn_sets = _sym_neighbors(dist_sq, n_neighbors)

    D = np.diag(W.sum(axis=1))
    L = D - W
    F, ev = _eig_laplacian(W, k)

    rho = 1.3
    mu = 1.0
    lam = np.zeros((n, n))   # Lagrange multiplier for symmetry W = W'

    dist_F_prev = None

    for iteration in range(max_iter):
        # --- compute embedding distance matrix ---
        dist_F = _l2_distance(F)

        # --- update W row by row ---
        Z = mu * W.T - 0.5 * lam + 0.5 * lam.T - 0.5 * dist_sq - 0.5 * eta * dist_F
        W_new = np.zeros((n, n))
        for i in range(n):
            if local:
                neighbors = knn_sets[i]
            else:
                neighbors = np.concatenate([np.arange(i), np.arange(i + 1, n)])
            ad = Z[i, neighbors] / (gamma + mu)
            W_new[i, neighbors] = _proj_simplex(ad)

        # --- update Lagrange multiplier (symmetry residual) ---
        h = W_new - W_new.T
        lam = lam + mu * h

        # --- update penalty ---
        mu = rho * mu

        # --- symmetrize ---
        W_new = (W_new + W_new.T) / 2.0

        # --- update spectral embedding F ---
        D_new = np.diag(W_new.sum(axis=1))
        L_new = D_new - W_new
        F_old = F.copy()
        F, ev = _eig_laplacian(W_new, k)

        fn1 = np.sum(ev[1:k + 1])      # sum of k smallest non-trivial eigenvalues
        fn2 = np.sum(ev[1:k + 2]) if len(ev) > k + 1 else np.inf

        # doubly stochastic check: diagonal of D should be ~1
        diag_D = np.diag(D_new)
        is_ds = np.sum(np.abs(diag_D - 1.0) < 0.01) == n

        W = W_new

        # --- convergence: structured + doubly stochastic ---
        if fn1 < 1e-10 and fn2 > 1e-10 and is_ds:
            break

        # --- eta adaptation (mirrors MATLAB logic) ---
        if fn1 > 1e-10:
            eta = 2.0 * eta
        elif fn2 < 1e-10:
            eta = eta / 2.0
            F = F_old

    # --- extract clusters from connected components of sparse W ---
    W_sparse = csr_matrix(W > 1e-6)
    n_components, labels = connected_components(W_sparse, directed=False)

    if n_components != k:
        warnings.warn(
            f"SDSGC found {n_components} connected components instead of "
            f"the requested {k} clusters. Consider adjusting gamma, eta, or "
            f"n_neighbors. Returning {n_components} clusters.",
            RuntimeWarning,
        )

    return labels, W


# ---------------------------------------------------------------------------
# GeoFusion-compatible wrapper
# ---------------------------------------------------------------------------

def run_sdsgc(
    input_path: str,
    output_path: str,
    columns=("latDeg_phone", "lngDeg_phone"),
    k: int = 10,
    random_state: int = 42,
    n_neighbors: int = 50,
    gamma: float = 0.1,
    eta: float = 0.1,
    local: bool = True,
    max_iter: int = 300,
    representative: str = "centroid",
) -> pd.DataFrame:
    """
    Run SDSGC clustering on the given columns of a CSV file and write the
    result (original data + predicted_cluster + predicted_location) to
    output_path.

    SDSGC operates on certain (deterministic) data points. It does NOT use
    the sigma_N / sigma_E uncertainty columns. See module docstring for the
    rationale and the distinction from uncertain object clustering.

    Parameters
    ----------
    input_path : str
        Path to the input CSV file.
    output_path : str
        Path to write the output CSV file.
    columns : tuple of str
        (lat, lon) column names to cluster on. Default ('latDeg_phone',
        'lngDeg_phone').
    k : int
        Number of clusters (connected components) to find. Default 10.
    random_state : int
        Unused (SDSGC is deterministic given the data and parameters), kept
        for interface consistency with all other GeoFusion clustering modules.
    n_neighbors : int
        Number of nearest neighbors for initial affinity graph construction.
        Default 50, selected empirically on the GSDC dataset to minimize
        the gap between the number of connected components found and the
        target k on 2D GPS data.
    gamma : float
        Regularization weight on ||W||_F^2. Controls graph sparsity.
        Default 0.1.
    eta : float
        Initial weight on the spectral embedding distance term. Adapted
        automatically during optimization. Default 0.1.
    local : bool
        If True, enforce row-sparsity of W to the k-nearest-neighbor support
        (recommended for compact spatial clusters). Default True.
    max_iter : int
        Maximum ALM iterations. Default 300.
    representative : str
        Post-hoc cluster representative to compute for each connected
        component. Either 'centroid' (mean lat/lon of cluster members,
        default) or 'medoid' (the cluster member with the minimum total
        Euclidean distance to all other members).

    Returns
    -------
    pd.DataFrame
        Output dataframe (also written to output_path). predicted_location is
        the centroid (mean lat/lon) of each connected component, for
        consistency with the GeoFusion evaluation framework.
    """
    df = pd.read_csv(input_path)

    lat_col, lon_col = columns
    missing = [c for c in (lat_col, lon_col) if c not in df.columns]
    if missing:
        raise ValueError(f"Column(s) not found in input data: {missing}")

    X = df[[lat_col, lon_col]].to_numpy(dtype=float)

    labels, W = _sdsgc(
        X, k=k, gamma=gamma, eta=eta,
        n_neighbors=n_neighbors, local=local, max_iter=max_iter,
    )

    if representative not in ("centroid", "medoid"):
        raise ValueError(f"representative must be 'centroid' or 'medoid', got '{representative}'")

    # --- predicted_location: post-hoc representative per connected component ---
    lat = X[:, 0]
    lon = X[:, 1]
    n = len(df)
    pred_lat = np.empty(n)
    pred_lon = np.empty(n)

    for cluster_id in np.unique(labels):
        mask = labels == cluster_id
        member_idx = np.where(mask)[0]
        if representative == "centroid" or member_idx.size == 1:
            pred_lat[mask] = lat[mask].mean()
            pred_lon[mask] = lon[mask].mean()
        else:
            # medoid: member with minimum total Euclidean distance to all others
            members = np.column_stack([lat[member_idx], lon[member_idx]])
            dist_matrix = cdist(members, members, metric="euclidean")
            best_local = np.argmin(dist_matrix.sum(axis=1))
            pred_lat[mask] = lat[member_idx[best_local]]
            pred_lon[mask] = lon[member_idx[best_local]]

    df["predicted_cluster"] = labels
    df["predicted_location"] = [
        (float(pred_lat[i]), float(pred_lon[i])) for i in range(n)
    ]

    df.to_csv(output_path, index=False)
    return df
