import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from methods.dbscan_manual import dbscan_manual, dbscan_summary
from methods.heatmap_cluster import render_heatmap_tab
from methods.clustering_manual import agglomerative_with_history


# --- our modules ---
from methods.data_loader import load_data
from methods.scoring import encode_answers, build_features
from methods.clustering_manual import agglomerative_clustering
from methods.interpretation import interpret_cluster, describe_cluster
from methods.cluster_profile import render_cluster_profile_tab
from methods.visualization import (
    figures_cluster_distribution,
    figure_learning_style_subplots,
    figure_dendrogram_with_library,
    figure_custom_dendrogram,
)

st.set_page_config(
    page_title="ILS Clustering Dashboard",
    layout="wide",           
    initial_sidebar_state="expanded"
)

# ---------------------- DATA LOAD  -----------------------
# ---------------------- DATA LOAD  -----------------------
st.sidebar.header("📂 Data Input")

uploaded_file = st.sidebar.file_uploader(
    "Upload file CSV (optional):",
    type=["csv"],
    help="Upload custom dataset anda, jika tidak, dataset default akan digunakan.",
)

if uploaded_file is not None:
    st.sidebar.success("Custom dataset berhasil digunakan.")
    df, meta_cols, q_cols = load_data(uploaded_file)
else:
    st.sidebar.info("Menggunakan dataset default.")
    df, meta_cols, q_cols = load_data(
        "data/Kuesioner Identifikasi Pola Gaya Belajar Mahasiswa melalui Metode Clustering (Responses) - Form responses 1.csv"
    )


# ---------------------- ENCODE & SCORES  -----------------
enc = encode_answers(df, q_cols)
scores, labels, app_cols, kuisoner_data, X = build_features(df, enc)

# ---------------------- SCALER  --------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(scores)


# ---- Sidebar clustering method (same logic & names) ----
st.sidebar.header("⚙️ Pilih Metode Clustering")
clustering_method = st.sidebar.selectbox("Metode:", ["Hierarchical (Ward)", "DBSCAN (Manual)"])
if clustering_method == "Hierarchical (Ward)":
    n_clusters = st.sidebar.slider("Jumlah Cluster (k):", 2, 10, 3)
else:
    eps = st.sidebar.slider("DBSCAN eps (radius)", 0.1, 2.0, 0.6, 0.1)
    min_pts = st.sidebar.slider("minPts", 2, 10, 3, 1)

# ---- Jalankan manual hierarchical clustering (Ward) ----
# --- Prevent NameError ---
n_clusters, eps, min_pts = 3, 0.6, 3
labels4 = agglomerative_clustering(X_scaled, n_clusters=n_clusters)
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
st.title("Dashboard Analisis Pola Gaya Belajar")

# ---- Run clustering based on selected method ----
if clustering_method == "Hierarchical (Ward)":
    # Use the number of clusters specified in the sidebar
    labels = agglomerative_clustering(X_scaled, n_clusters=n_clusters)
    kuisoner_data["Cluster"] = ["Cluster " + str(l + 1) for l in labels]

else:  # --- DBSCAN (Manual) ---
    labels = dbscan_manual(X_scaled, eps=eps, min_pts=min_pts)
    kuisoner_data["Cluster"] = [
        f"Cluster {l + 1}" if l != -1 else "Noise" for l in labels
    ]

    # Sidebar summary
    summary = dbscan_summary(labels)
    st.sidebar.write("**📊 DBSCAN Summary (Manual)**")
    st.sidebar.write(f"- Clusters: {summary['clusters']}")
    st.sidebar.write(f"- Noise points: {summary['noise_points']}")


# ---- Visualizations (same tabs and content) ----
st.subheader("📊 Cluster Visualizations")
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Distribusi Cluster", "Gaya Belajar", "Dendrogram", "Analisis Profil Cluster", "Interpretasi Otomatis", "Cluster Heatmap"])

with tab1:
    # --- Cluster Distribution Section ---
    st.subheader("📊 Cluster Distribution Overview")

    # Create both figures
    fig_pie, fig_bar = figures_cluster_distribution(kuisoner_data)

    # Place charts side by side
    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(fig_pie, config={"responsive": True})

    with col2:
        st.plotly_chart(fig_bar, config={"responsive": True})

    # Show the data table below the charts
    st.markdown("---")
    st.subheader("📋 Data with Cluster Assignments")
    st.dataframe(
        kuisoner_data,
        width='stretch',
        hide_index=True
    )

with tab2:
    st.subheader("🎨 Learning Style Preferences")

    # --- Create a wide two-column layout ---
    col1, col2 = st.columns([1, 3])  # left (dropdown) narrow, right (chart) wide

    with col1:
        st.markdown("### 🎛️ Select Dimension")
        style_choice = st.selectbox(
            "Learning Style Dimension:",
            [
                "All (2×2 Subplot)",
                "Active vs Reflective (AR)",
                "Sensing vs Intuitive (SI)",
                "Visual vs Verbal (VV)",
                "Sequential vs Global (QG)"
            ],
            label_visibility="collapsed"  # hides the "Learning Style Dimension:" label for a cleaner look
        )

    with col2:
        if style_choice == "All (2×2 Subplot)":
            fig_styles = figure_learning_style_subplots(kuisoner_data)
            st.plotly_chart(fig_styles, config={"responsive": True})
        else:
            mapping = {
                "Active vs Reflective (AR)": "AR",
                "Sensing vs Intuitive (SI)": "SI",
                "Visual vs Verbal (VV)": "VV",
                "Sequential vs Global (QG)": "QG"
            }
            dim = mapping[style_choice]
            counts = kuisoner_data[dim].value_counts()

            # Create single bar dynamically
            fig_single = go.Figure()
            fig_single.add_trace(go.Bar(
                x=counts.index,
                y=counts.values,
                marker_color=['#FF6B6B', '#4ECDC4'],
                showlegend=False
            ))

            fig_single.update_layout(
                title=f"Distribution of {style_choice}",
                xaxis_title="Style",
                yaxis_title="Number of Students",
                height=500,
                margin=dict(t=60, l=40, r=40, b=40)
            )
            st.plotly_chart(fig_single, config={"responsive": True})


with tab3:
    fig = figure_dendrogram_with_library(X, kuisoner_data['ID'].tolist())
    st.pyplot(fig)

with tab4:
    # make a copy to avoid modifying original
    scores_with_cluster = scores.copy()
    scores_with_cluster["Cluster"] = kuisoner_data["Cluster"].values
    render_cluster_profile_tab(scores_with_cluster)


with tab5:
    st.subheader("Interpretasi Otomatis (4 Cluster)")
    for t in interpretations:
        st.write(t)
    st.dataframe(cluster_summary_view.set_index("Cluster"))     

with tab6:
    render_heatmap_tab(X_scaled, scores, agglomerative_with_history, n_clusters)

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
