
import numpy as np
import pandas as pd

def dbscan_manual(X, eps=0.5, min_pts=3):
    """
    Implementasi manual DBSCAN yang sudah diperbaiki.
    
    Status Label:
      -2: Unvisited (Belum Dikunjungi)
      -1: Noise
      >=0: Cluster ID
    """
    n_points = X.shape[0]
    # Inisialisasi semua titik sebagai 'Unvisited'
    labels = np.full(n_points, -2) ## <-- PERUBAHAN: Mulai dengan -2 (Unvisited)
    cluster_id = 0

    # Loop utama untuk setiap titik data
    for point_idx in range(n_points):
        # Lewati titik yang sudah menjadi bagian dari cluster lain
        if labels[point_idx] != -2: ## <-- PERUBAHAN: Cek jika titik sudah punya label
            continue

        # Temukan tetangga untuk titik saat ini
        distances = np.linalg.norm(X - X[point_idx], axis=1)
        neighbors_indices = np.where(distances <= eps)[0]

        # Cek apakah titik ini adalah noise atau core point
        if len(neighbors_indices) < min_pts:
            labels[point_idx] = -1 # Tandai sebagai Noise
            continue
        
        # Titik ini adalah CORE POINT. Mulai perluasan cluster.
        # Tandai titik awal dengan ID cluster baru
        labels[point_idx] = cluster_id
        
        # Buat antrean (queue) yang berisi semua tetangga dari core point
        queue = list(neighbors_indices)
        
        # Proses antrean sampai habis
        head = 0
        while head < len(queue):
            current_point_idx = queue[head]
            head += 1

            # Jika tetangga adalah noise, selamatkan dan jadikan border point
            if labels[current_point_idx] == -1:
                labels[current_point_idx] = cluster_id
            
            # Jika tetangga belum dikunjungi, proses lebih lanjut
            if labels[current_point_idx] == -2:
                labels[current_point_idx] = cluster_id # Tambahkan ke cluster
                
                # Cari tetangga dari titik ini
                new_distances = np.linalg.norm(X - X[current_point_idx], axis=1)
                new_neighbors = np.where(new_distances <= eps)[0]
                
                # Jika tetangga ini juga core point, tambahkan semua tetangganya ke antrean
                if len(new_neighbors) >= min_pts:
                    queue.extend(new_neighbors) ## <-- PERUBAHAN: Logika perluasan inti ada di sini
        
        # Setelah cluster selesai diperluas, siapkan ID untuk cluster berikutnya
        cluster_id += 1

    return labels


def dbscan_summary(labels):
    """
    Fungsi ini sudah benar, tidak perlu diubah.
    """
    labels_series = pd.Series(labels)
    cluster_counts = labels_series.value_counts().sort_index().to_dict()
    
    n_clusters = len([lbl for lbl in cluster_counts.keys() if lbl != -1])
    n_noise = cluster_counts.get(-1, 0)
    
    return {
        "clusters": n_clusters,
        "noise_points": n_noise,
        "unique_labels": sorted(cluster_counts.keys()),
        "cluster_sizes": cluster_counts
    }