import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots
from scipy.cluster.hierarchy import dendrogram
from sklearn.preprocessing import StandardScaler
from methods.dbscan_manual import dbscan_manual, dbscan_summary
from methods.heatmap_cluster import render_heatmap_tab
from methods.clustering_manual import agglomerative_with_history


# --- our modules ---
from methods.data_loader import load_data
from methods.scoring import encode_answers, build_features
from methods.clustering_manual import agglomerative_clustering
from methods.interpretation import interpret_cluster, describe_cluster
from methods.visualization import (
    figures_cluster_distribution,
    figure_learning_style_subplots,
    figure_dendrogram_with_library,
    figure_custom_dendrogram,
)

# ---------------------- DATA LOAD  -----------------------
df, meta_cols, q_cols = load_data(
    "data/Kuesioner Identifikasi Pola Gaya Belajar Mahasiswa melalui Metode Clustering (Responses) - Form responses 1.csv"
)

# ---------------------- ENCODE & SCORES  -----------------
enc = encode_answers(df, q_cols)
scores, labels, app_cols, kuisoner_data, X = build_features(df, enc)

# ---------------------- SCALER  --------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(scores)

# ---- Jalankan manual hierarchical clustering (Ward) ----
labels4 = agglomerative_clustering(X_scaled, n_clusters=4)
kuisoner_data["Cluster"] = ["Cluster " + str(l) for l in labels4]

# ---- Ringkasan rata-rata skor per cluster ----
cluster_means = scores.copy()
cluster_means["Cluster"] = kuisoner_data["Cluster"]
cluster_summary = cluster_means.groupby("Cluster").mean().reset_index()

# ---- Interpretasi  ----
cluster_summary["Karakteristik"] = cluster_summary.apply(interpret_cluster, axis=1)

# ---- Process beforehand (keep this near the top or before tab section) ----
cluster_summary_view = cluster_summary.copy()
kuisoner_view = kuisoner_data.copy()
total = len(kuisoner_view)

def get_cluster_descriptions():
    texts = []
    for _, row in cluster_summary_view.iterrows():
        texts.append(describe_cluster(row, total, kuisoner_view))
    return texts

interpretations = get_cluster_descriptions()

# prepare questions (keep it global)
questions = {
    "AR": ("Saat belajar, saya lebih suka:", ["Diskusi kelompok", "Belajar sendiri"]),
    "SI": ("Saya lebih suka mempelajari:", ["Fakta nyata", "Konsep abstrak"]),
    "VV": ("Saya lebih mudah memahami materi melalui:", ["Diagram/gambar", "Teks/lisan"]),
    "QG": ("Saya lebih suka belajar dengan:", ["Langkah runtut", "Gambaran besar dulu"]),
}
# ---- App Layout (same title) ----
st.title("ILS Hierarchical Clustering Demo")

st.subheader("Data with Clusters")
st.dataframe(kuisoner_data)

# ---- Sidebar clustering method (same logic & names) ----
st.sidebar.header("⚙️ Pilih Metode Clustering")
clustering_method = st.sidebar.selectbox("Metode:", ["Hierarchical (Ward)", "DBSCAN (Manual)"])

if clustering_method == "Hierarchical (Ward)":
    labels2 = agglomerative_clustering(X, n_clusters=2)  # (kept raw X per your code)
    kuisoner_data["Cluster"] = ["Cluster " + str(l+1) for l in labels2]
else:  # DBSCAN manual
    eps = st.sidebar.slider("DBSCAN eps (radius)", 0.1, 2.0, 0.6, 0.1)
    min_pts = st.sidebar.slider("minPts", 2, 10, 3, 1)
    labels_db = dbscan_manual(X, eps=eps, min_pts=min_pts)
    kuisoner_data["Cluster"] = [f"Cluster {l}" if l != -1 else "Noise" for l in labels_db]
    summary = dbscan_summary(labels_db)
    st.sidebar.write("**DBSCAN Summary (Manual)**")
    st.sidebar.write(f"- Clusters: {summary['clusters']}")
    st.sidebar.write(f"- Noise points: {summary['noise_points']}")

# ---- Visualizations (same tabs and content) ----
st.subheader("📊 Cluster Visualizations")
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Cluster Distribution", "Learning Styles", "Dendrogram (with library)", "Dendogram", "Automatic Interpretation", "Heatmap Cluster"])

with tab1:
    fig_pie, fig_bar = figures_cluster_distribution(kuisoner_data)
    st.plotly_chart(fig_pie, use_container_width=True)
    st.plotly_chart(fig_bar, use_container_width=True)

with tab2:
    fig_styles = figure_learning_style_subplots(kuisoner_data)
    st.plotly_chart(fig_styles, use_container_width=True)

with tab3:
    fig = figure_dendrogram_with_library(X, kuisoner_data['ID'].tolist())
    st.pyplot(fig)

with tab4:
    fig_custom = figure_custom_dendrogram(X)
    st.pyplot(fig_custom)

with tab5:
    st.subheader("Interpretasi Otomatis (4 Cluster)")
    for t in interpretations:
        st.write(t)
    st.dataframe(cluster_summary_view.set_index("Cluster"))     

with tab6:
    render_heatmap_tab(X_scaled, scores, agglomerative_with_history)

# ---- User Input  ----
st.subheader("Jawab Pertanyaan")
answers = {}
for key, (qtext, opts) in questions.items():
    answers[key] = st.radio(qtext, opts, key=key)

# ---- Predict Cluster  ----
if st.button("Submit"):
    user_vector = [
        1 if answers["AR"] == "Diskusi kelompok" else 0,
        1 if answers["SI"] == "Fakta nyata" else 0,
        1 if answers["VV"] == "Diagram/gambar" else 0,
        1 if answers["QG"] == "Langkah runtut" else 0
    ]
    dists = np.linalg.norm(X - user_vector, axis=1)
    nearest_idx = np.argmin(dists)
    cluster = kuisoner_data.iloc[nearest_idx]['Cluster']

    st.success("Hasil Profil Anda:")
    st.write(f"Active–Reflective: {answers['AR']}")
    st.write(f"Sensing–Intuitive: {answers['SI']}")
    st.write(f"Visual–Verbal: {answers['VV']}")
    st.write(f"Sequential–Global: {answers['QG']}")
    st.write(f"**Cluster yang sesuai: {cluster}**")
