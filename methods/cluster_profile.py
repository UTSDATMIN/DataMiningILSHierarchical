import streamlit as st
import plotly.express as px
import pandas as pd

def render_cluster_profile_tab(scores: pd.DataFrame):
    """
    Render the 'Analisis Profil Cluster' tab.
    Displays mean (centroid) values for each cluster
    and a grouped bar chart comparing learning-style scores.
    """
    st.header("🎯 Analisis Profil Rata-rata (Centroid) per Cluster")

    # Compute cluster centroids (mean per dimension per cluster)
    cluster_centroids = scores.groupby("Cluster").mean().round(2)

    # Display the centroid table with color gradients
    st.subheader("📋 Rata-rata Skor per Cluster")
    st.dataframe(
        cluster_centroids.style.background_gradient(cmap='viridis').format("{:.2f}"),
        use_container_width=True
    )

    # Create a grouped bar plot for centroid comparison
    st.subheader("📊 Perbandingan Profil Antar Cluster")
    fig_centroids = px.bar(
        cluster_centroids.T,
        barmode='group',
        title="<b>Perbandingan Profil Rata-rata Antar Cluster</b>",
        labels={'value': 'Skor Rata-rata', 'index': 'Dimensi Gaya Belajar'}
    )
    st.plotly_chart(fig_centroids, use_container_width=True)
