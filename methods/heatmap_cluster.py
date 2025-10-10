import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.cluster.hierarchy import dendrogram

def render_heatmap_tab(X_scaled, scores, agglomerative_with_history):
    """
    Safely render a clustered heatmap without modifying your existing
    agglomerative_with_history() function.
    Converts the custom history output into a SciPy-compatible linkage matrix.
    """
    st.header("🔥 Heatmap Gaya Belajar")
    st.markdown("Heatmap ini diurutkan berdasarkan hasil dari proses hierarchical clustering.")

    with st.spinner("Mengurutkan data dan membuat heatmap..."):
        # --- Run your existing function ---
        history = agglomerative_with_history(X_scaled)

        # --- Convert custom format → valid linkage matrix ---
        Z_list = []
        n_samples = len(X_scaled)
        next_id = n_samples

        for step in history:
            # expect something like [id1, id2, distance, count]
            if len(step) == 3:
                id1, id2, dist = step
                count = 1
            elif len(step) >= 4:
                id1, id2, dist, count = step[:4]
            else:
                continue

            # Make sure IDs are numeric (flatten if list)
            def flatten_id(x):
                if isinstance(x, (list, tuple, np.ndarray)):
                    # take the first element if nested list
                    return float(np.ravel(x)[0])
                try:
                    return float(x)
                except Exception:
                    return np.nan

            id1, id2 = flatten_id(id1), flatten_id(id2)

            # Ward distance should be positive
            if dist < 0:
                dist = abs(dist)

            # Append to linkage list with unique cluster index
            Z_list.append([id1, id2, float(dist), float(count)])

        # --- Fix duplicated cluster IDs problem ---
        # Assign unique cluster IDs for each merge
        used_ids = set()
        fixed_Z = []
        for i, row in enumerate(Z_list):
            a, b, d, c = row
            # if already used, remap to next available ID
            if a in used_ids:
                a = next_id
                next_id += 1
            if b in used_ids:
                b = next_id
                next_id += 1
            used_ids.update([a, b, next_id])
            fixed_Z.append([a, b, d, c])

        Z_fixed = np.array(fixed_Z, dtype=float)

        # --- Guard for valid shape ---
        if Z_fixed.ndim != 2 or Z_fixed.shape[1] != 4:
            st.error(f"Invalid linkage matrix shape: {Z_fixed.shape}")
            return

        # --- Generate dendrogram & reorder ---
        dendro_data = dendrogram(Z_fixed, no_plot=True)
        reordered_indices = dendro_data['leaves']

        # --- Prepare heatmap data ---
        scores_for_heatmap = scores.drop(columns=['Cluster'], errors='ignore')
        reordered_scores = scores_for_heatmap.iloc[reordered_indices]

        # --- Plot ---
        fig, ax = plt.subplots(figsize=(10, 15))
        sns.heatmap(
            reordered_scores,
            ax=ax,
            cmap='viridis',
            cbar=True,
            yticklabels=True
        )

        ax.set_title("Clustered Heatmap (Converted Linkage)", fontsize=14)
        ax.set_yticklabels(reordered_scores.index, rotation=0)
        st.pyplot(fig)
