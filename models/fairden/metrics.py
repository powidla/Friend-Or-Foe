from scipy.spatial.distance import cdist
import numpy as np


def dcsi_score(X, labels):
    """
    Density-Based Clustering Selection Index (DCSI)

    """
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)

    # Handle edge cases
    if n_clusters < 2:
        return 0.0

    # Calculate intra-cluster compactness
    intra_dists = []
    for k in unique_labels:
        if k == -1:
            continue
        cluster_points = X[labels == k]
        if len(cluster_points) > 1:
            centroid = np.mean(cluster_points, axis=0)
            intra_dists.extend(cdist(cluster_points, [centroid], 'euclidean').flatten())

    centroids = []
    for k in unique_labels:
        if k == -1:
            continue
        centroids.append(np.mean(X[labels == k], axis=0))

    if len(centroids) < 2:
        return 0.0

    inter_dists = cdist(centroids, centroids, 'euclidean')
    np.fill_diagonal(inter_dists, np.inf)
    avg_intra = np.mean(intra_dists) if intra_dists else 0
    min_inter = np.min(inter_dists) if inter_dists.size > 0 else 0

    # Norm DCSI
    if avg_intra == 0:
        return 1.0
    if min_inter == 0:
        return 0.0

    max_dist = np.max(cdist(X, X, 'euclidean'))
    if max_dist == 0:
        return 0.0

    normalized_dcsi = (min_inter / max_dist) * (1 - (avg_intra / max_dist))
    return np.clip(normalized_dcsi, 0.0, 1.0)