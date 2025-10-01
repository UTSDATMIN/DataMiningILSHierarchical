import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# ---- Dummy dataset ----
dummy_data = pd.DataFrame([
    {"ID": 1, "AR": "Active", "SI": "Sensing", "VV": "Visual", "QG": "Sequential"},
    {"ID": 2, "AR": "Reflective", "SI": "Intuitive", "VV": "Verbal", "QG": "Global"},
    {"ID": 3, "AR": "Active", "SI": "Sensing", "VV": "Visual", "QG": "Global"},
    {"ID": 4, "AR": "Reflective", "SI": "Intuitive", "VV": "Verbal", "QG": "Sequential"},
    {"ID": 5, "AR": "Active", "SI": "Sensing", "VV": "Visual", "QG": "Sequential"},
    {"ID": 6, "AR": "Reflective", "SI": "Intuitive", "VV": "Visual", "QG": "Global"},
    {"ID": 7, "AR": "Active", "SI": "Intuitive", "VV": "Verbal", "QG": "Sequential"},
    {"ID": 8, "AR": "Reflective", "SI": "Sensing", "VV": "Verbal", "QG": "Global"},
    {"ID": 9, "AR": "Active", "SI": "Sensing", "VV": "Verbal", "QG": "Sequential"},
    {"ID": 10, "AR": "Reflective", "SI": "Sensing", "VV": "Visual", "QG": "Sequential"},
    {"ID": 11, "AR": "Reflective", "SI": "Intuitive", "VV": "Visual", "QG": "Global"},
    {"ID": 12, "AR": "Active", "SI": "Intuitive", "VV": "Verbal", "QG": "Sequential"},
    {"ID": 13, "AR": "Reflective", "SI": "Sensing", "VV": "Verbal", "QG": "Global"},
    {"ID": 14, "AR": "Active", "SI": "Sensing", "VV": "Visual", "QG": "Sequential"},
    {"ID": 15, "AR": "Reflective", "SI": "Intuitive", "VV": "Verbal", "QG": "Global"},
    {"ID": 16, "AR": "Active", "SI": "Sensing", "VV": "Visual", "QG": "Global"},
    {"ID": 17, "AR": "Active", "SI": "Intuitive", "VV": "Verbal", "QG": "Sequential"},
    {"ID": 18, "AR": "Reflective", "SI": "Sensing", "VV": "Visual", "QG": "Sequential"},
    {"ID": 19, "AR": "Active", "SI": "Sensing", "VV": "Verbal", "QG": "Global"},
    {"ID": 20, "AR": "Reflective", "SI": "Intuitive", "VV": "Visual", "QG": "Sequential"},
    {"ID": 21, "AR": "Active", "SI": "Sensing", "VV": "Visual", "QG": "Sequential"},
    {"ID": 22, "AR": "Reflective", "SI": "Intuitive", "VV": "Verbal", "QG": "Global"},
    {"ID": 23, "AR": "Active", "SI": "Sensing", "VV": "Visual", "QG": "Global"},
    {"ID": 24, "AR": "Reflective", "SI": "Sensing", "VV": "Verbal", "QG": "Sequential"},
    {"ID": 25, "AR": "Active", "SI": "Intuitive", "VV": "Visual", "QG": "Sequential"},
    {"ID": 26, "AR": "Reflective", "SI": "Intuitive", "VV": "Verbal", "QG": "Global"},
    {"ID": 27, "AR": "Active", "SI": "Sensing", "VV": "Verbal", "QG": "Global"},
    {"ID": 28, "AR": "Reflective", "SI": "Intuitive", "VV": "Visual", "QG": "Sequential"},
    {"ID": 29, "AR": "Active", "SI": "Sensing", "VV": "Visual", "QG": "Sequential"},
    {"ID": 30, "AR": "Reflective", "SI": "Sensing", "VV": "Verbal", "QG": "Global"},
    {"ID": 31, "AR": "Active", "SI": "Intuitive", "VV": "Visual", "QG": "Sequential"},
    {"ID": 32, "AR": "Reflective", "SI": "Intuitive", "VV": "Verbal", "QG": "Global"},
    {"ID": 33, "AR": "Active", "SI": "Sensing", "VV": "Visual", "QG": "Sequential"},
    {"ID": 34, "AR": "Reflective", "SI": "Intuitive", "VV": "Verbal", "QG": "Sequential"},
    {"ID": 35, "AR": "Active", "SI": "Sensing", "VV": "Verbal", "QG": "Global"},
    {"ID": 36, "AR": "Reflective", "SI": "Intuitive", "VV": "Visual", "QG": "Global"},
    {"ID": 37, "AR": "Active", "SI": "Intuitive", "VV": "Visual", "QG": "Sequential"},
    {"ID": 38, "AR": "Reflective", "SI": "Sensing", "VV": "Verbal", "QG": "Sequential"},
    {"ID": 39, "AR": "Active", "SI": "Sensing", "VV": "Visual", "QG": "Global"},
    {"ID": 40, "AR": "Reflective", "SI": "Intuitive", "VV": "Verbal", "QG": "Sequential"},
    {"ID": 41, "AR": "Active", "SI": "Sensing", "VV": "Visual", "QG": "Sequential"},
    {"ID": 42, "AR": "Reflective", "SI": "Sensing", "VV": "Verbal", "QG": "Global"},
    {"ID": 43, "AR": "Active", "SI": "Intuitive", "VV": "Visual", "QG": "Sequential"},
    {"ID": 44, "AR": "Reflective", "SI": "Intuitive", "VV": "Verbal", "QG": "Global"},
    {"ID": 45, "AR": "Active", "SI": "Sensing", "VV": "Verbal", "QG": "Sequential"},
    {"ID": 46, "AR": "Reflective", "SI": "Sensing", "VV": "Visual", "QG": "Global"},
    {"ID": 47, "AR": "Active", "SI": "Intuitive", "VV": "Visual", "QG": "Sequential"},
    {"ID": 48, "AR": "Reflective", "SI": "Intuitive", "VV": "Verbal", "QG": "Sequential"},
    {"ID": 49, "AR": "Active", "SI": "Sensing", "VV": "Visual", "QG": "Global"},
    {"ID": 50, "AR": "Reflective", "SI": "Sensing", "VV": "Verbal", "QG": "Sequential"},
    {"ID": 51, "AR": "Active", "SI": "Intuitive", "VV": "Visual", "QG": "Global"},
    {"ID": 52, "AR": "Reflective", "SI": "Intuitive", "VV": "Verbal", "QG": "Sequential"},
    {"ID": 53, "AR": "Active", "SI": "Sensing", "VV": "Visual", "QG": "Sequential"},
    {"ID": 54, "AR": "Reflective", "SI": "Sensing", "VV": "Verbal", "QG": "Global"},
    {"ID": 55, "AR": "Active", "SI": "Intuitive", "VV": "Visual", "QG": "Sequential"},
    {"ID": 56, "AR": "Reflective", "SI": "Intuitive", "VV": "Verbal", "QG": "Global"},
    {"ID": 57, "AR": "Active", "SI": "Sensing", "VV": "Visual", "QG": "Global"},
    {"ID": 58, "AR": "Reflective", "SI": "Intuitive", "VV": "Verbal", "QG": "Sequential"},
    {"ID": 59, "AR": "Active", "SI": "Sensing", "VV": "Verbal", "QG": "Sequential"},
    {"ID": 60, "AR": "Reflective", "SI": "Sensing", "VV": "Visual", "QG": "Global"},
    {"ID": 61, "AR": "Active", "SI": "Intuitive", "VV": "Visual", "QG": "Sequential"},
    {"ID": 62, "AR": "Reflective", "SI": "Intuitive", "VV": "Verbal", "QG": "Global"},
    {"ID": 63, "AR": "Active", "SI": "Sensing", "VV": "Visual", "QG": "Sequential"},
    {"ID": 64, "AR": "Reflective", "SI": "Sensing", "VV": "Verbal", "QG": "Global"},
    {"ID": 65, "AR": "Active", "SI": "Intuitive", "VV": "Verbal", "QG": "Sequential"},
    {"ID": 66, "AR": "Reflective", "SI": "Intuitive", "VV": "Visual", "QG": "Sequential"},
    {"ID": 67, "AR": "Active", "SI": "Sensing", "VV": "Visual", "QG": "Global"},
    {"ID": 68, "AR": "Reflective", "SI": "Intuitive", "VV": "Verbal", "QG": "Sequential"},
])

# ---- Mapping categorical to numeric ----
mappings = {
    'AR': {'Active': 1, 'Reflective': 0},
    'SI': {'Sensing': 1, 'Intuitive': 0},
    'VV': {'Visual': 1, 'Verbal': 0},
    'QG': {'Sequential': 1, 'Global': 0}
}
numerical_data = dummy_data.copy()
for col, mapping in mappings.items():
    numerical_data[f"{col}_num"] = numerical_data[col].map(mapping)

# ---- Hierarchical clustering ----
X = numerical_data[['AR_num', 'SI_num', 'VV_num', 'QG_num']]
model = AgglomerativeClustering(n_clusters=2, metric='euclidean', linkage='ward')
dummy_data['Cluster'] = model.fit_predict(X).astype(str)
dummy_data['Cluster'] = "Cluster " + (dummy_data['Cluster'].astype(int) + 1).astype(str)

# ---- Questions ----
questions = {
    "AR": ("Saat belajar, saya lebih suka:", ["Diskusi kelompok", "Belajar sendiri"]),
    "SI": ("Saya lebih suka mempelajari:", ["Fakta nyata", "Konsep abstrak"]),
    "VV": ("Saya lebih mudah memahami materi melalui:", ["Diagram/gambar", "Teks/lisan"]),
    "QG": ("Saya lebih suka belajar dengan:", ["Langkah runtut", "Gambaran besar dulu"]),
}

# ---- App Layout ----
st.title("ILS Hierarchical Clustering Demo")

st.subheader("Dummy Data with Clusters")
st.dataframe(dummy_data)

# ---- Visualizations ----
st.subheader("📊 Cluster Visualizations")

tab1, tab2, tab3 = st.tabs(["Cluster Distribution", "Learning Styles", "Dendrogram"])

with tab1:
    cluster_counts = dummy_data['Cluster'].value_counts()
    fig_pie = px.pie(
        values=cluster_counts.values,
        names=cluster_counts.index,
        title="Distribution of Students Across Clusters",
        color_discrete_sequence=['#FF6B6B', '#4ECDC4']
    )
    st.plotly_chart(fig_pie, use_container_width=True)

    fig_bar = px.bar(
        x=cluster_counts.index,
        y=cluster_counts.values,
        title="Number of Students per Cluster",
        labels={'x': 'Cluster', 'y': 'Number of Students'},
        color=cluster_counts.index,
        color_discrete_sequence=['#FF6B6B', '#4ECDC4']
    )
    st.plotly_chart(fig_bar, use_container_width=True)

with tab2:
    dimensions = ['AR', 'SI', 'VV', 'QG']
    fig_styles = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Active vs Reflective', 'Sensing vs Intuitive',
                        'Visual vs Verbal', 'Sequential vs Global'),
        specs=[[{'type': 'bar'}, {'type': 'bar'}],
               [{'type': 'bar'}, {'type': 'bar'}]]
    )

    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    positions = [(1,1), (1,2), (2,1), (2,2)]

    for i, dim in enumerate(dimensions):
        counts = dummy_data[dim].value_counts()
        row, col = positions[i]
        fig_styles.add_trace(
            go.Bar(x=counts.index, y=counts.values, marker_color=colors[i], showlegend=False),
            row=row, col=col
        )

    fig_styles.update_layout(height=600, title_text="Learning Style Preferences Distribution")
    st.plotly_chart(fig_styles, use_container_width=True)

with tab3:
    # Generate linkage matrix for dendrogram
    Z = linkage(X.values, method='ward', metric='euclidean')

    st.subheader("🔗 Hierarchical Clustering Dendrogram")

    fig, ax = plt.subplots(figsize=(8, 5))
    dendrogram(Z, labels=dummy_data['ID'].tolist(), ax=ax, leaf_rotation=90)
    plt.title("Dendrogram of Learning Style Clusters")
    plt.xlabel("Student ID")
    plt.ylabel("Distance")

    st.pyplot(fig)

# ---- User Input ----
st.subheader("Jawab Pertanyaan")
answers = {}
for key, (qtext, opts) in questions.items():
    answers[key] = st.radio(qtext, opts, key=key)

# ---- Predict Cluster ----
if st.button("Submit"):
    # Convert answers to numeric
    user_vector = [
        1 if answers["AR"] == "Diskusi kelompok" else 0,
        1 if answers["SI"] == "Fakta nyata" else 0,
        1 if answers["VV"] == "Diagram/gambar" else 0,
        1 if answers["QG"] == "Langkah runtut" else 0
    ]
    # Predict cluster manually using same model
    # Since Agglomerative doesn't have predict, find nearest existing sample
    dists = np.linalg.norm(X.values - user_vector, axis=1)
    nearest_idx = np.argmin(dists)
    cluster = dummy_data.iloc[nearest_idx]['Cluster']

    st.success("Hasil Profil Anda:")
    st.write(f"Active–Reflective: {answers['AR']}")
    st.write(f"Sensing–Intuitive: {answers['SI']}")
    st.write(f"Visual–Verbal: {answers['VV']}")
    st.write(f"Sequential–Global: {answers['QG']}")
    st.write(f"**Cluster yang sesuai: {cluster}**")
