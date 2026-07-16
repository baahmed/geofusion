"""
k-means clustering for GeoFusion.

Wraps sklearn's KMeans with k-means++ initialization and the standard
GeoFusion function signature, for a clean like-for-like comparison with
all other clustering methods in the framework.

The cluster representative is the centroid (mean) of reported locations
within each cluster -- a computed average, not an actual data point.
This is what distinguishes k-means from k-medoids in terms of the
representative: centroids may not correspond to any real observation.

Usage (in a notebook cell):
    df_out = run_kmeans(
        "real_data_sample_10events_5mapart.csv",
        "real_data_sample_10events_kmeans.csv",
        columns=("reported_location_lat", "reported_location_lon"),
        k=10,
        random_state=42,
    )
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


def run_kmeans(
    input_path: str,
    output_path: str,
    columns=("reported_location_lat", "reported_location_lon"),
    k: int = 10,
    random_state: int = 42,
    n_init: int = 10,
) -> pd.DataFrame:
    """
    Run k-means clustering on the given columns of a CSV file and write
    the result (original data + predicted_cluster + predicted_location)
    to output_path.

    Parameters
    ----------
    input_path : str
        Path to the input CSV file.
    output_path : str
        Path to write the output CSV file to.
    columns : tuple of str
        Column names to cluster on.
    k : int
        Number of clusters, default 10.
    random_state : int
        Seed for reproducibility (passed to sklearn KMeans).
    n_init : int
        Number of k-means++ initializations; best run by inertia is kept.
        Default 10, matching sklearn's default and all other GeoFusion
        algorithms for a fair comparison.

    Returns
    -------
    pd.DataFrame
        The output dataframe (also written to output_path).
    """
    df = pd.read_csv(input_path)

    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Column(s) not found in input data: {missing}")

    X = df[list(columns)].to_numpy(dtype=float)

    model = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
    labels = model.fit_predict(X)
    centroids = model.cluster_centers_

    df["predicted_cluster"] = labels
    df["predicted_location"] = [
        tuple(float(v) for v in centroids[lbl]) for lbl in labels
    ]

    df.to_csv(output_path, index=False)
    return df
