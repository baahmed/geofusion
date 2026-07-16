import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

def _kmeans_plusplus_init(X, k, rng):
    n = X.shape[0]
    first_idx = rng.integers(n)
    center_indices = [first_idx]
    closest_sq_dist = np.sum((X - X[first_idx])**2, axis=1)
    for _ in range(1, k):
        probs = closest_sq_dist / closest_sq_dist.sum()
        idx = rng.choice(n, p=probs)
        center_indices.append(idx)
        new_sq = np.sum((X - X[idx])**2, axis=1)
        closest_sq_dist = np.minimum(closest_sq_dist, new_sq)
    return np.array(center_indices)

def _lloyd_kmedoids_single(X, k, rng, max_iter):
    n = X.shape[0]
    medoid_idx = _kmeans_plusplus_init(X, k, rng)
    labels = np.zeros(n, dtype=int)
    for _ in range(max_iter):
        d_to_medoids = cdist(X, X[medoid_idx], metric='euclidean')
        new_labels = np.argmin(d_to_medoids, axis=1)
        new_medoid_idx = medoid_idx.copy()
        for c in range(k):
            members = np.where(new_labels == c)[0]
            if len(members) == 0: continue
            if len(members) == 1: new_medoid_idx[c] = members[0]; continue
            sub_dist = cdist(X[members], X[members], metric='euclidean')
            new_medoid_idx[c] = members[np.argmin(sub_dist.sum(axis=1))]
        if np.array_equal(new_labels, labels) and np.array_equal(np.sort(new_medoid_idx), np.sort(medoid_idx)):
            labels = new_labels; medoid_idx = new_medoid_idx; break
        labels = new_labels; medoid_idx = new_medoid_idx
    d_final = cdist(X, X[medoid_idx], metric='euclidean')
    total_cost = d_final[np.arange(n), labels].sum()
    return labels, medoid_idx, total_cost

def _lloyd_kmedoids(X, k, random_state, n_init, max_iter):
    rng = np.random.default_rng(random_state)
    best_labels, best_medoid_idx, best_cost = None, None, np.inf
    for _ in range(n_init):
        labels, medoid_idx, cost = _lloyd_kmedoids_single(X, k, rng, max_iter)
        if cost < best_cost:
            best_labels, best_medoid_idx, best_cost = labels, medoid_idx, cost
    return best_labels, best_medoid_idx

def run_kmedoids(input_path, output_path, columns=("reported_location_lat","reported_location_lon"),
                 k=10, random_state=42, n_init=10, max_iter=300):
    df = pd.read_csv(input_path)
    X = df[list(columns)].to_numpy(dtype=float)
    labels, medoid_indices = _lloyd_kmedoids(X, k=k, random_state=random_state, n_init=n_init, max_iter=max_iter)
    medoids = X[medoid_indices]
    df["predicted_cluster"] = labels
    df["predicted_location"] = [tuple(float(v) for v in medoids[lbl]) for lbl in labels]
    df.to_csv(output_path, index=False)
    return df
