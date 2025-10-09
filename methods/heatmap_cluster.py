import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram
import streamlit as st

def render_heatmap_tab(X_scaled, scores, agglomerative_with_history):
    """Render the Heatmap Cluster tab."""
    st.header("🔥 Heatmap Gaya Belajar")
    st.markdown("Heatmap ini diurutkan berdasarkan hasil dari proses hierarchical clustering.")
    
    with st.spinner("Mengurutkan data dan membuat heatmap..."):
        # --- Compute clustering history if not present ---
        if 'Z_history' not in locals():
            Z_history = agglomerative_with_history(X_scaled)
        else:
            Z_history = locals()['Z_history']
        
        # --- Reorder according to dendrogram leaves ---
        dendro_data = dendrogram(Z_history, no_plot=True)
        reordered_indices = dendro_data['leaves']
        
        # --- Reorder data ---
        scores_for_heatmap = scores.drop('Cluster', axis=1)
        reordered_scores = scores_for_heatmap.iloc[reordered_indices]
        
        # --- Create heatmap ---
        fig_heatmap, ax_heatmap = plt.subplots(figsize=(10, 15))
        sns.heatmap(
            reordered_scores,
            ax=ax_heatmap,
            cmap='viridis',
            cbar=True,
            yticklabels=True
        )
        ax_heatmap.set_title("Clustered Heatmap")
        ax_heatmap.set_yticklabels(reordered_scores.index, rotation=0)
        st.pyplot(fig_heatmap)
