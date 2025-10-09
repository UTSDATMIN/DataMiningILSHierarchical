import numpy as np

def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b)**2))

def ward_distance(cluster_a, cluster_b, X):
    points_a = X[cluster_a]
    points_b = X[cluster_b]
    merged = np.vstack([points_a, points_b])
    mean_a, mean_b, mean_m = points_a.mean(axis=0), points_b.mean(axis=0), merged.mean(axis=0)
    sse_a = ((points_a - mean_a)**2).sum()
    sse_b = ((points_b - mean_b)**2).sum()
    sse_m = ((merged - mean_m)**2).sum()
    return sse_m - (sse_a + sse_b)

def agglomerative_clustering(X, n_clusters=2):
    clusters = [[i] for i in range(len(X))]
    while len(clusters) > n_clusters:
        min_dist = float("inf")
        to_merge = (None, None)
        for i in range(len(clusters)):
            for j in range(i+1, len(clusters)):
                dist = ward_distance(clusters[i], clusters[j], X)
                if dist < min_dist:
                    min_dist = dist
                    to_merge = (i, j)
        i, j = to_merge
        new_cluster = clusters[i] + clusters[j]
        clusters = [c for k, c in enumerate(clusters) if k not in (i, j)]
        clusters.append(new_cluster)

    labels = np.zeros(len(X), dtype=int)
    for cluster_id, cluster in enumerate(clusters):
        for idx in cluster:
            labels[idx] = cluster_id
    return labels

def agglomerative_with_history(X):
    clusters = [[i] for i in range(len(X))]
    history = []
    while len(clusters) > 1:
        min_dist = float("inf")
        to_merge = (None, None)
        for i in range(len(clusters)):
            for j in range(i+1, len(clusters)):
                dist = ward_distance(clusters[i], clusters[j], X)
                if dist < min_dist:
                    min_dist = dist
                    to_merge = (i, j)
        i, j = to_merge
        history.append((clusters[i], clusters[j], min_dist))
        new_cluster = clusters[i] + clusters[j]
        clusters = [c for k, c in enumerate(clusters) if k not in (i, j)]
        clusters.append(new_cluster)
    return history
