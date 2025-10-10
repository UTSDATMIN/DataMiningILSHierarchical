import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.cluster.hierarchy import dendrogram, fcluster

def render_heatmap_tab(X_scaled, scores, agglomerative_with_history, n_clusters):
    """
    Render a clustered heatmap consistent with user-selected n_clusters.
    Converts your custom agglomerative history into a valid SciPy linkage matrix.
    """
    st.header("🔥 Heatmap Gaya Belajar")
    st.markdown("Heatmap ini diurutkan berdasarkan hasil dari proses hierarchical clustering.")

    with st.spinner("🔄 Mengurutkan data dan membuat heatmap..."):
        # --- Step 1: Generate your custom history ---
        history = agglomerative_with_history(X_scaled)

        # --- Step 2: Convert to valid linkage matrix (SciPy format) ---
        Z_list = []
        n_samples = len(X_scaled)
        next_id = n_samples

        for step in history:
            if len(step) == 3:
                id1, id2, dist = step
                count = 1
            elif len(step) >= 4:
                id1, id2, dist, count = step[:4]
            else:
                continue

            def flatten_id(x):
                if isinstance(x, (list, tuple, np.ndarray)):
                    return float(np.ravel(x)[0])
                try:
                    return float(x)
                except Exception:
                    return np.nan

            id1, id2 = flatten_id(id1), flatten_id(id2)
            dist = abs(dist)
            Z_list.append([id1, id2, float(dist), float(count)])

        # --- Step 3: Ensure cluster IDs are unique ---
        used_ids = set()
        fixed_Z = []
        for row in Z_list:
            a, b, d, c = row
            if a in used_ids:
                a = next_id
                next_id += 1
            if b in used_ids:
                b = next_id
                next_id += 1
            used_ids.update([a, b, next_id])
            fixed_Z.append([a, b, d, c])

        Z_fixed = np.array(fixed_Z, dtype=float)

        # --- Step 4: Guard clause ---
        if Z_fixed.ndim != 2 or Z_fixed.shape[1] != 4:
            st.error(f"⚠️ Invalid linkage matrix shape: {Z_fixed.shape}")
            return

        # --- Step 5: Cut tree to n_clusters ---
        try:
            cluster_assignments = fcluster(Z_fixed, t=n_clusters, criterion="maxclust")
        except Exception as e:
            st.error(f"❌ Gagal membuat cluster: {e}")
            cluster_assignments = np.ones(len(scores))  # fallback

        # --- Step 6: Add cluster info safely ---
        scores_for_heatmap = scores.copy()
        scores_for_heatmap["Cluster"] = cluster_assignments
        scores_for_heatmap = scores_for_heatmap.sort_values("Cluster")

        # --- Step 7: Prepare reordered data ---
        try:
            dendro_data = dendrogram(Z_fixed, no_plot=True)
            reordered_indices = dendro_data["leaves"]
            reordered_scores = scores_for_heatmap.iloc[reordered_indices]
        except Exception:
            reordered_scores = scores_for_heatmap  # fallback if dendrogram fails

        # --- Step 8: Ensure 'Cluster' exists before dropping ---
        columns_to_plot = [col for col in reordered_scores.columns if col != "Cluster"]

        # --- Step 9: Plot ---
        fig, ax = plt.subplots(figsize=(10, 15))
        sns.heatmap(
            reordered_scores[columns_to_plot],
            ax=ax,
            cmap="viridis",
            cbar=True,
            yticklabels=True
        )

        ax.set_title(f"Clustered Heatmap (k = {n_clusters})", fontsize=14)
        ax.set_xlabel("Features")
        ax.set_ylabel("Mahasiswa (diurutkan berdasarkan cluster)")
        st.pyplot(fig)
