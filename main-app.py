import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage

# # ---- Dummy dataset ----
# dummy_data = pd.read_json("dummy_data.json")

# # ---- Mapping categorical to numeric ----
# mappings = {
#     'AR': {'Active': 1, 'Reflective': 0},
#     'SI': {'Sensing': 1, 'Intuitive': 0},
#     'VV': {'Visual': 1, 'Verbal': 0},
#     'QG': {'Sequential': 1, 'Global': 0}
# }
# numerical_data = dummy_data.copy()
# for col, mapping in mappings.items():
#     numerical_data[f"{col}_num"] = numerical_data[col].map(mapping)

df = pd.read_csv("Kuesioner Identifikasi Pola Gaya Belajar Mahasiswa melalui Metode Clustering (Responses) - Form responses 1.csv", sep=None, engine="python")

# Example: keep the first 5 meta columns as-is, rename the rest to Q1..Q20
meta_cols = df.columns[:5].tolist()
q_cols = [f"Q{i}" for i in range(1, 21)]
rename_map = {old: new for old, new in zip(df.columns[5:5+20], q_cols)}
df = df.rename(columns=rename_map)

def norm(s):
    return str(s).strip()

for c in q_cols:
    df[c] = df[c].map(norm)

ANS_MAP = {
    # A. Active – Reflective (Q1–Q5)
    "Q1": {"Diskusi kelompok": 1, "Belajar sendiri dengan merenung": 0},
    "Q2": {"Langsung mencoba mengerjakan": 1, "Memikirkan dulu sebelum mencoba": 0},
    "Q3": {"Menjelaskannya pada orang lain": 1, "Membaca/merenungkannya dalam diam": 0},
    "Q4": {"Terlibat aktif dalam diskusi": 1, "Mendengarkan lalu memikirkan sendiri": 0},
    "Q5": {"Mengerjakan proyek bersama teman": 1, "Mengerjakan tugas secara mandiri": 0},

    # B. Sensing – Intuitive (Q6–Q10)
    "Q6": {"Fakta dan contoh nyata": 1, "Teori dan konsep abstrak": 0},
    "Q7": {"Menggunakan metode yang sudah jelas": 1, "Mencoba pendekatan baru yang kreatif": 0},
    "Q8": {"Detail praktis": 1, "Hubungan antar konsep": 0},
    "Q9": {"Yang banyak aplikasinya dalam kehidupan nyata": 1, "Yang penuh ide baru dan inovatif": 0},
    "Q10": {"Mengetahui langkah-langkah pasti": 1, "Mengeksplorasi kemungkinan lain": 0},

    # C. Visual – Verbal (Q11–Q15)
    "Q11": {"Diagram, grafik, gambar": 1, "Penjelasan teks atau lisan": 0},
    "Q12": {"Membuat skema/mindmap": 1, "Menulis dalam bentuk kalimat/paragraf": 0},
    "Q13": {"Menggunakan media visual (slide, gambar)": 1, "Menjelaskan panjang lebar dengan kata-kata": 0},
    "Q14": {"Belajar dengan melihat ilustrasi": 1, "Belajar dengan membaca/menyimak penjelasan": 0},
    "Q15": {"Visual (gambar, warna)": 1, "Verbal (kata, suara)": 0},

    # D. Sequential – Global (Q16–Q20)
    "Q16": {"Langkah demi langkah yang teratur": 1, "Melihat gambaran besar terlebih dahulu": 0},
    "Q17": {"Mengikuti urutan dari awal ke akhir": 1, "Membaca bagian yang saya anggap penting dulu": 0},
    "Q18": {"Dijelaskan secara runtut dan sistematis": 1, "Dijelaskan secara garis besar dulu": 0},
    "Q19": {"Mengikuti prosedur langkah demi langkah": 1, "Melompat ke solusi dengan memahami konsep besar": 0},
    "Q20": {"Dengan urutan yang jelas dan logis": 1, "Dengan cara bebas dan menyeluruh": 0},
}

enc = df.copy()
for q in q_cols:
    enc[q] = enc[q].map(ANS_MAP[q]).astype("Int64")  # Int64 allows NA if something didn't match

def sumcols(cols): return enc[cols].astype("float").sum(axis=1)

score_AR = sumcols(["Q1","Q2","Q3","Q4","Q5"])   # 1=Active
score_SI = sumcols(["Q6","Q7","Q8","Q9","Q10"])  # 1=Sensing
score_VV = sumcols(["Q11","Q12","Q13","Q14","Q15"]) # 1=Visual
score_QG = sumcols(["Q16","Q17","Q18","Q19","Q20"]) # 1=Sequential

scores = pd.DataFrame({
    "AR_num": score_AR,
    "SI_num": score_SI,
    "VV_num": score_VV,
    "QG_num": score_QG
})

def label_dim(score, a_label, b_label):
    if score >= 4: return f"Strong {a_label}"
    if score == 3: return f"Lean {a_label}"
    if score == 2: return f"Lean {b_label}"
    return f"Strong {b_label}"

labels = pd.DataFrame({
    "AR_label": scores["AR_num"].map(lambda s: label_dim(s, "Active","Reflective")),
    "SI_label": scores["SI_num"].map(lambda s: label_dim(s, "Sensing","Intuitive")),
    "VV_label": scores["VV_num"].map(lambda s: label_dim(s, "Visual","Verbal")),
    "QG_label": scores["QG_num"].map(lambda s: label_dim(s, "Sequential","Global")),
})

# Also the categorical poles your app expects:
def pole(score, a_label, b_label):
    return a_label if score >= 3 else b_label

app_cols = pd.DataFrame({
    "AR": scores["AR_num"].map(lambda s: pole(s,"Active","Reflective")),
    "SI": scores["SI_num"].map(lambda s: pole(s,"Sensing","Intuitive")),
    "VV": scores["VV_num"].map(lambda s: pole(s,"Visual","Verbal")),
    "QG": scores["QG_num"].map(lambda s: pole(s,"Sequential","Global")),
})

# Prefer NIM if present; otherwise make an incremental ID
if "NIM" in df.columns:
    ID = df["NIM"]
else:
    ID = pd.RangeIndex(start=1, stop=len(df)+1, step=1)

dummy_data = pd.DataFrame({
    "ID": ID,
    "AR": app_cols["AR"],
    "SI": app_cols["SI"],
    "VV": app_cols["VV"],
    "QG": app_cols["QG"],
})


# ---- Hierarchical clustering ----
X = scores[["AR_num", "SI_num", "VV_num", "QG_num"]].values

# ---- Helper functions ----Z = linkage(X, method='ward')
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b)**2))

def ward_distance(cluster_a, cluster_b, X):
    """Ward's method: increase in variance when merging clusters"""
    points_a = X[cluster_a]
    points_b = X[cluster_b]
    merged = np.vstack([points_a, points_b])
    # SSE (sum of squared errors) before and after merging
    mean_a, mean_b, mean_m = points_a.mean(axis=0), points_b.mean(axis=0), merged.mean(axis=0)
    sse_a = ((points_a - mean_a)**2).sum()
    sse_b = ((points_b - mean_b)**2).sum()
    sse_m = ((merged - mean_m)**2).sum()
    return sse_m - (sse_a + sse_b)

# ---- Agglomerative Clustering ----
def agglomerative_clustering(X, n_clusters=2):
    # Start with each point as its own cluster
    clusters = [[i] for i in range(len(X))]

    while len(clusters) > n_clusters:
        min_dist = float("inf")
        to_merge = (None, None)

        # Find closest two clusters
        for i in range(len(clusters)):
            for j in range(i+1, len(clusters)):
                dist = ward_distance(clusters[i], clusters[j], X)
                if dist < min_dist:
                    min_dist = dist
                    to_merge = (i, j)

        # Merge the closest pair
        i, j = to_merge
        new_cluster = clusters[i] + clusters[j]

        # Rebuild cluster list
        clusters = [c for k, c in enumerate(clusters) if k not in (i, j)]
        clusters.append(new_cluster)

    # Assign cluster labels
    labels = np.zeros(len(X), dtype=int)
    for cluster_id, cluster in enumerate(clusters):
        for idx in cluster:
            labels[idx] = cluster_id
    return labels

# ---- Run clustering ----
labels = agglomerative_clustering(X, n_clusters=2)
dummy_data["Cluster"] = ["Cluster " + str(l+1) for l in labels]

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

tab1, tab2, tab3, tab4 = st.tabs(["Cluster Distribution", "Learning Styles", "Dendrogram (with library)", "Dendogram"])

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
    Z = linkage(X, method='ward', metric='euclidean')

    st.subheader("🔗 Hierarchical Clustering Dendrogram")

    fig, ax = plt.subplots(figsize=(8, 5))
    dendrogram(Z, labels=dummy_data['ID'].tolist(), ax=ax, leaf_rotation=90)
    plt.title("Dendrogram of Learning Style Clusters")
    plt.xlabel("Student ID")
    plt.ylabel("Distance")

    st.pyplot(fig)

with tab4:
    st.subheader("🔗 Hierarchical Clustering Dendrogram")

    # ---- Custom Agglomerative with history ----
    def agglomerative_with_history(X):
        clusters = [[i] for i in range(len(X))]
        history = []

        while len(clusters) > 1:
            min_dist = float("inf")
            to_merge = (None, None)

            # find closest pair
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

    history = agglomerative_with_history(X)

    # ---- Plot dendrogram ----
    fig, ax = plt.subplots(figsize=(8, 5))
    cluster_positions = {}

    # place leaves
    for idx in range(len(X)):
        cluster_positions[frozenset([idx])] = (idx, 0)
        ax.text(idx, -0.2, str(idx), ha='center')

    # plot merges
    for c1, c2, dist in history:
        pos1 = cluster_positions[frozenset(c1)]
        pos2 = cluster_positions[frozenset(c2)]
        new_x = (pos1[0] + pos2[0]) / 2
        new_y = dist
        ax.plot([pos1[0], pos1[0]], [pos1[1], new_y], c='k')
        ax.plot([pos2[0], pos2[0]], [pos2[1], new_y], c='k')
        ax.plot([pos1[0], pos2[0]], [new_y, new_y], c='k')
        cluster_positions[frozenset(c1+c2)] = (new_x, new_y)

    ax.set_xlabel("Samples")
    ax.set_ylabel("Distance")
    ax.set_title("Custom Dendrogram (Ward’s Method)")
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
    dists = np.linalg.norm(X - user_vector, axis=1)
    nearest_idx = np.argmin(dists)
    cluster = dummy_data.iloc[nearest_idx]['Cluster']

    st.success("Hasil Profil Anda:")
    st.write(f"Active–Reflective: {answers['AR']}")
    st.write(f"Sensing–Intuitive: {answers['SI']}")
    st.write(f"Visual–Verbal: {answers['VV']}")
    st.write(f"Sequential–Global: {answers['QG']}")
    st.write(f"**Cluster yang sesuai: {cluster}**")
