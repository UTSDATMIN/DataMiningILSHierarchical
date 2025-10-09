import numpy as np
import pandas as pd

def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b)**2))

def region_query(X, point_idx, eps):
    """Find all points within eps distance from point_idx."""
    neighbors = []
    for i in range(len(X)):
        if euclidean_distance(X[point_idx], X[i]) <= eps:
            neighbors.append(i)
    return neighbors

def expand_cluster(X, labels, point_idx, cluster_id, eps, min_pts, visited):
    """Expand cluster from a core point."""
    neighbors = region_query(X, point_idx, eps)
    if len(neighbors) < min_pts:
        labels[point_idx] = -1  # Noise
        return False
    else:
        labels[point_idx] = cluster_id
        for n in neighbors:
            labels[n] = cluster_id

        while neighbors:
            current_point = neighbors.pop(0)
            if not visited[current_point]:
                visited[current_point] = True
                new_neighbors = region_query(X, current_point, eps)
                if len(new_neighbors) >= min_pts:
                    for n in new_neighbors:
                        if n not in neighbors:
                            neighbors.append(n)
            if labels[current_point] == -1:
                labels[current_point] = cluster_id
        return True

def dbscan_manual(X, eps=0.5, min_pts=3):
    """
    Manual implementation of DBSCAN (no sklearn).
    Returns cluster labels (noise = -1).
    """
    n_points = len(X)
    labels = np.full(n_points, -1)  # start with all as noise
    visited = np.zeros(n_points, dtype=bool)
    cluster_id = 0

    for i in range(n_points):
        if visited[i]:
            continue
        visited[i] = True
        neighbors = region_query(X, i, eps)
        if len(neighbors) < min_pts:
            labels[i] = -1  # still noise
        else:
            cluster_id += 1
            labels[i] = cluster_id
            for n in neighbors:
                labels[n] = cluster_id
            expand_cluster(X, labels, i, cluster_id, eps, min_pts, visited)
    return labels


def dbscan_summary(labels):
    """
    Summarize DBSCAN results in a more detailed way.
    Returns:
        {
            "clusters": <int>,              # number of non-noise clusters
            "noise_points": <int>,          # count of noise points (-1)
            "unique_labels": <list>,        # list of all labels (including -1)
            "cluster_sizes": <dict>         # {cluster_label: number_of_points}
        }
    """
    labels_series = pd.Series(labels)
    # Count points per cluster (including noise)
    cluster_counts = labels_series.value_counts().sort_index().to_dict()
    
    # Number of clusters (excluding noise)
    n_clusters = len([lbl for lbl in cluster_counts.keys() if lbl != -1])
    
    # Count of noise points
    n_noise = cluster_counts.get(-1, 0)
    
    return {
        "clusters": n_clusters,
        "noise_points": n_noise,
        "unique_labels": sorted(cluster_counts.keys()),
        "cluster_sizes": cluster_counts
    }
