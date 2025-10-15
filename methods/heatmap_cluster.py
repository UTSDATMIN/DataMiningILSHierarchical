import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def render_heatmap_tab(X_scaled, scores, agglomerative_with_history, n_clusters):
    """
    Render a clustered heatmap without using SciPy.
    Clustering order is derived from the custom agglomerative history.
    """
    st.header("🔥 Heatmap Gaya Belajar")
    st.markdown("Heatmap ini diurutkan berdasarkan hasil dari proses hierarchical clustering (manual).")

    with st.spinner("🔄 Mengurutkan data dan membuat heatmap..."):
        # --- Step 1: Ambil riwayat penggabungan cluster dari fungsi agglomerative ---
        history = agglomerative_with_history(X_scaled)
        n_samples = len(X_scaled)

        # --- Step 2: Bangun urutan gabungan manual berdasarkan riwayat ---
        # history diharapkan berisi tuple seperti (id1, id2, dist, count)
        cluster_order = list(range(n_samples))
        try:
            for step in history:
                id1, id2, *rest = step
                # Flatten nested ids
                if isinstance(id1, (list, tuple, np.ndarray)):
                    id1 = id1[0]
                if isinstance(id2, (list, tuple, np.ndarray)):
                    id2 = id2[0]

                # Pastikan keduanya dalam urutan
                if id1 in cluster_order and id2 in cluster_order:
                    idx1 = cluster_order.index(id1)
                    idx2 = cluster_order.index(id2)
                    if idx1 > idx2:
                        cluster_order.pop(idx1)
                        cluster_order.insert(idx2 + 1, id1)
                    else:
                        cluster_order.pop(idx2)
                        cluster_order.insert(idx1 + 1, id2)
        except Exception as e:
            st.warning(f"⚠️ Gagal memproses urutan history: {e}")
            cluster_order = list(range(n_samples))

        # --- Step 3: Buat assignment cluster sederhana ---
        # (ini bisa kamu ubah sesuai output clustering kamu)
        cluster_assignments = np.repeat(range(1, n_clusters + 1), n_samples // n_clusters + 1)[:n_samples]

        # --- Step 4: Tambahkan hasil cluster ke DataFrame ---
        scores_for_heatmap = scores.copy()
        scores_for_heatmap["Cluster"] = cluster_assignments

        # --- Step 5: Urutkan berdasarkan cluster dan urutan manual ---
        reordered_scores = scores_for_heatmap.iloc[cluster_order]
        columns_to_plot = [col for col in reordered_scores.columns if col != "Cluster"]

        # --- Step 6: Plot heatmap ---
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
        ax.set_ylabel("Mahasiswa (diurutkan manual berdasarkan cluster)")
        st.pyplot(fig)
