import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
from scipy.cluster.hierarchy import dendrogram # Hanya dipakai untuk visualisasi
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# --- Konfigurasi Halaman ---
st.set_page_config(layout="wide", page_title="Analisis Gaya Belajar")

# --- Bagian 1: Logika Clustering ---
# Fungsi-fungsi ini sekarang menjadi inti dari aplikasi
def ward_distance(cluster_a_indices, cluster_b_indices, X):
    points_a = X[cluster_a_indices]
    points_b = X[cluster_b_indices]
    
    if points_a.shape[0] == 0 or points_b.shape[0] == 0:
        return float('inf')

    merged = np.vstack([points_a, points_b])
    
    mean_a, mean_b, mean_m = points_a.mean(axis=0), points_b.mean(axis=0), merged.mean(axis=0)
    
    sse_a = np.sum((points_a - mean_a)**2)
    sse_b = np.sum((points_b - mean_b)**2)
    sse_m = np.sum((merged - mean_m)**2)
    
    return sse_m - (sse_a + sse_b)

def agglomerative_clustering(X, n_clusters):
    clusters = [[i] for i in range(len(X))]
    while len(clusters) > n_clusters:
        min_dist = float("inf")
        to_merge = (None, None)
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                dist = ward_distance(clusters[i], clusters[j], X)
                if dist < min_dist:
                    min_dist = dist
                    to_merge = (i, j)
        i, j = to_merge
        new_cluster = clusters[i] + clusters[j]
        clusters = [c for k, c in enumerate(clusters) if k not in (i, j)]
        clusters.append(new_cluster)
    
    labels = np.zeros(len(X), dtype=int)
    for cluster_id, cluster in enumerate(clusters):
        for idx in cluster:
            labels[idx] = cluster_id
    return labels + 1 # Label mulai dari 1

def agglomerative_with_history(X):
    clusters = {i: [i] for i in range(len(X))}
    history = []
    next_cluster_id = len(X)
    
    while len(clusters) > 1:
        min_dist = float("inf")
        to_merge = (None, None)
        
        cluster_ids = list(clusters.keys())
        for i in range(len(cluster_ids)):
            for j in range(i + 1, len(cluster_ids)):
                id1, id2 = cluster_ids[i], cluster_ids[j]
                dist = ward_distance(clusters[id1], clusters[id2], X)
                if dist < min_dist:
                    min_dist = dist
                    to_merge = (id1, id2)
                    
        id1, id2 = to_merge
        
        new_cluster_content = clusters[id1] + clusters[id2]
        history.append([id1, id2, min_dist, len(new_cluster_content)])
        
        del clusters[id1]
        del clusters[id2]
        clusters[next_cluster_id] = new_cluster_content
        next_cluster_id += 1
        
    return np.array(history)

# --- Bagian 2: Pemuatan dan Pemrosesan Data ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("Kuesioner Identifikasi Pola Gaya Belajar Mahasiswa melalui Metode Clustering (Responses) - Form responses 1.csv", sep=None, engine="python")
    except FileNotFoundError:
        st.error("File CSV tidak ditemukan. Pastikan file 'Kuesioner ... .csv' berada di folder yang sama.")
        return None, None
    
    q_cols = [f"Q{i}" for i in range(1, 21)]
    rename_map = {old: new for old, new in zip(df.columns[5:25], q_cols)}
    df = df.rename(columns=rename_map)
    for c in q_cols: df[c] = df[c].map(lambda s: str(s).strip())

    ANS_MAP = {
        "Q1": {"Diskusi kelompok": 1, "Belajar sendiri dengan merenung": 0}, "Q2": {"Langsung mencoba mengerjakan": 1, "Memikirkan dulu sebelum mencoba": 0}, "Q3": {"Menjelaskannya pada orang lain": 1, "Membaca/merenungkannya dalam diam": 0}, "Q4": {"Terlibat aktif dalam diskusi": 1, "Mendengarkan lalu memikirkan sendiri": 0}, "Q5": {"Mengerjakan proyek bersama teman": 1, "Mengerjakan tugas secara mandiri": 0},
        "Q6": {"Fakta dan contoh nyata": 1, "Teori dan konsep abstrak": 0}, "Q7": {"Menggunakan metode yang sudah jelas": 1, "Mencoba pendekatan baru yang kreatif": 0}, "Q8": {"Detail praktis": 1, "Hubungan antar konsep": 0}, "Q9": {"Yang banyak aplikasinya dalam kehidupan nyata": 1, "Yang penuh ide baru dan inovatif": 0}, "Q10": {"Mengetahui langkah-langkah pasti": 1, "Mengeksplorasi kemungkinan lain": 0},
        "Q11": {"Diagram, grafik, gambar": 1, "Penjelasan teks atau lisan": 0}, "Q12": {"Membuat skema/mindmap": 1, "Menulis dalam bentuk kalimat/paragraf": 0}, "Q13": {"Menggunakan media visual (slide, gambar)": 1, "Menjelaskan panjang lebar dengan kata-kata": 0}, "Q14": {"Belajar dengan melihat ilustrasi": 1, "Belajar dengan membaca/menyimak penjelasan": 0}, "Q15": {"Visual (gambar, warna)": 1, "Verbal (kata, suara)": 0},
        "Q16": {"Langkah demi langkah yang teratur": 1, "Melihat gambaran besar terlebih dahulu": 0}, "Q17": {"Mengikuti urutan dari awal ke akhir": 1, "Membaca bagian yang saya anggap penting dulu": 0}, "Q18": {"Dijelaskan secara runtut dan sistematis": 1, "Dijelaskan secara garis besar dulu": 0}, "Q19": {"Mengikuti prosedur langkah demi langkah": 1, "Melompat ke solusi dengan memahami konsep besar": 0}, "Q20": {"Dengan urutan yang jelas dan logis": 1, "Dengan cara bebas dan menyeluruh": 0},
    }

    enc = df.copy()
    for q in q_cols: enc[q] = enc[q].map(ANS_MAP[q]).astype("Int64")
    if enc[q_cols].isnull().values.any():
        st.warning("Beberapa jawaban tidak dapat di-encode dan diabaikan.")
        enc = enc.dropna(subset=q_cols); df = df.loc[enc.index]

    scores = pd.DataFrame({
        "AR_num": enc[q_cols[0:5]].sum(axis=1), "SI_num": enc[q_cols[5:10]].sum(axis=1),
        "VV_num": enc[q_cols[10:15]].sum(axis=1), "QG_num": enc[q_cols[15:20]].sum(axis=1)
    }).astype(float) 

    app_cols = pd.DataFrame({
        "AR": scores["AR_num"].map(lambda s: "Active" if s >= 3 else "Reflective"),
        "SI": scores["SI_num"].map(lambda s: "Sensing" if s >= 3 else "Intuitive"),
        "VV": scores["VV_num"].map(lambda s: "Visual" if s >= 3 else "Verbal"),
        "QG": scores["QG_num"].map(lambda s: "Sequential" if s >= 3 else "Global"),
    })
    
    ID = df["NIM"].fillna(f"Mhs-{df.index}") if "NIM" in df.columns else pd.RangeIndex(start=1, stop=len(df)+1, step=1)
    display_data = pd.DataFrame({"ID": ID}).join(app_cols)
    display_data.set_index('ID', inplace=True)
    return scores, display_data

scores, display_data = load_data()
if scores is None: st.stop()

# --- Bagian 3: Pengaturan Interaktif di Sidebar ---
st.sidebar.header("⚙️ Pengaturan Clustering")
n_clusters = st.sidebar.slider("Jumlah Cluster:", 2, 10, 3)

# --- Bagian 4: Pre-processing & Clustering ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(scores)

# Menjalankan clustering untuk mendapatkan label
labels = agglomerative_clustering(X_scaled, n_clusters=n_clusters)
display_data["Cluster"] = [f"Cluster {l}" for l in labels]
scores["Cluster"] = [f"Cluster {l}" for l in labels]

# --- Bagian 5: Tampilan Utama Aplikasi ---
st.title("👨‍🎓 Dashboard Analisis Pola Gaya Belajar")
tab_list = ["Distribusi Data", "Analisis Profil Cluster", "Dendrogram", "Heatmap Cluster", "Prediksi Gaya Belajar Anda"]
tabs = st.tabs(tab_list)

with tabs[0]: # Distribusi Data
    st.header("📊 Distribusi Awal Data")
    col1, col2 = st.columns(2)
    with col1:
        cluster_counts = display_data['Cluster'].value_counts().sort_index()
        fig_pie = px.pie(cluster_counts, values=cluster_counts.values, names=cluster_counts.index, title=f"<b>Distribusi Mahasiswa per Cluster (Total: {len(display_data)})</b>", color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig_pie, use_container_width=True)
    with col2:
        dimensions = ['AR', 'SI', 'VV', 'QG']; dim_map = {'AR': 'Active/Reflective', 'SI': 'Sensing/Intuitive', 'VV': 'Visual/Verbal', 'QG': 'Sequential/Global'}
        selected_dim = st.selectbox("Pilih Dimensi Gaya Belajar:", options=dimensions, format_func=lambda x: dim_map[x])
        counts = display_data[selected_dim].value_counts()
        fig_bar_dim = px.bar(counts, x=counts.index, y=counts.values, title=f"<b>Distribusi Preferensi {dim_map[selected_dim]}</b>", labels={'y': 'Jumlah Mahasiswa', 'index': 'Preferensi'})
        st.plotly_chart(fig_bar_dim, use_container_width=True)
    st.dataframe(display_data)

with tabs[1]: # Analisis Profil Cluster
    st.header("🎯 Analisis Profil Rata-rata (Centroid) per Cluster")
    cluster_centroids = scores.groupby("Cluster").mean()
    st.dataframe(cluster_centroids.style.background_gradient(cmap='viridis').format("{:.2f}"))
    fig_centroids = px.bar(cluster_centroids.T, barmode='group', title="<b>Perbandingan Profil Rata-rata Antar Cluster</b>", labels={'value': 'Skor Rata-rata', 'index': 'Dimensi Gaya Belajar'})
    st.plotly_chart(fig_centroids, use_container_width=True)

with tabs[2]: # Dendrogram
    st.header("🌳 Dendrogram Hierarchical Clustering")
    st.markdown("Visualisasi pohon ini sepenuhnya dihasilkan dari algoritma yang diimplementasikan dalam program ini.")
    
    with st.spinner("Menghitung linkage matrix dari histori..."):
        Z_history = agglomerative_with_history(X_scaled)

    fig, ax = plt.subplots(figsize=(12, 8))
    dendrogram(Z_history, labels=display_data.index.tolist(), orientation='top', leaf_rotation=90, ax=ax)
    plt.title("Dendrogram (Metode: Ward)", fontsize=16)
    plt.ylabel("Jarak (Peningkatan Varians)")
    plt.tight_layout()
    st.pyplot(fig)

with tabs[3]: # Heatmap Cluster
    st.header("🔥 Heatmap Gaya Belajar")
    st.markdown("Heatmap ini diurutkan berdasarkan hasil dari proses hierarchical clustering.")
    
    with st.spinner("Mengurutkan data dan membuat heatmap..."):
        if 'Z_history' not in locals():
            Z_history = agglomerative_with_history(X_scaled)
            
        dendro_data = dendrogram(Z_history, no_plot=True)
        reordered_indices = dendro_data['leaves']
        
        scores_for_heatmap = scores.drop('Cluster', axis=1)
        reordered_scores = scores_for_heatmap.iloc[reordered_indices]
        
        fig_heatmap, ax_heatmap = plt.subplots(figsize=(10, 15))
        sns.heatmap(reordered_scores, ax=ax_heatmap, cmap='viridis', cbar=True, yticklabels=True)
        ax_heatmap.set_title("Clustered Heatmap")
        ax_heatmap.set_yticklabels(reordered_scores.index, rotation=0)
        st.pyplot(fig_heatmap)
        
with tabs[4]: # Prediksi Gaya Belajar
    st.header("🤔 Cek Profil Gaya Belajar Anda")
    questions = {
        "AR": ("Saat belajar, saya lebih suka:", ["Diskusi kelompok", "Belajar sendiri dengan merenung"]),
        "SI": ("Saya lebih suka mempelajari:", ["Fakta dan contoh nyata", "Teori dan konsep abstrak"]),
        "VV": ("Saya lebih mudah memahami materi melalui:", ["Diagram, grafik, gambar", "Penjelasan teks atau lisan"]),
        "QG": ("Saya lebih suka belajar dengan:", ["Langkah demi langkah yang teratur", "Melihat gambaran besar terlebih dahulu"]),
    }
    
    answers = {}
    with st.form("quiz_form"):
        st.write("Jawablah pertanyaan singkat berikut untuk memprediksi kelompok gaya belajar Anda.")
        for key, (qtext, opts) in questions.items():
            answers[key] = st.radio(qtext, opts, key=key)
        
        submitted = st.form_submit_button("Lihat Hasil Prediksi")

    if submitted:
        user_scores = {
            "AR_num": 5 if answers["AR"] == "Diskusi kelompok" else 0,
            "SI_num": 5 if answers["SI"] == "Fakta dan contoh nyata" else 0,
            "VV_num": 5 if answers["VV"] == "Diagram, grafik, gambar" else 0,
            "QG_num": 5 if answers["QG"] == "Langkah demi langkah yang teratur" else 0
        }
        
        user_vector = np.array(list(user_scores.values())).reshape(1, -1)
        user_vector_scaled = scaler.transform(user_vector)
        
        dists = np.linalg.norm(X_scaled - user_vector_scaled, axis=1)
        nearest_idx = np.argmin(dists)
        predicted_cluster = display_data.iloc[nearest_idx]['Cluster']
        
        st.success(f"**Prediksi Selesai! Anda paling cocok dengan profil {predicted_cluster}.**")
        
        profil_cluster = cluster_centroids.loc[predicted_cluster]
        
        ar_style = "Aktif (Active)" if profil_cluster['AR_num'] >= 2.5 else "Reflektif (Reflective)"
        si_style = "Sensing" if profil_cluster['SI_num'] >= 2.5 else "Intuitif (Intuitive)"
        vv_style = "Visual" if profil_cluster['VV_num'] >= 2.5 else "Verbal"
        qg_style = "Sekuensial (Sequential)" if profil_cluster['QG_num'] >= 2.5 else "Global"
        
        st.markdown("---")
        st.markdown("#### Karakteristik Utama Gaya Belajar Anda:")
        st.write(f"""
        Berdasarkan profil **{predicted_cluster}**, tipe belajar Anda yang dominan adalah:
        - **{ar_style}**: Cenderung belajar melalui tindakan, diskusi, dan eksperimen.
        - **{si_style}**: Fokus pada fakta, data konkret, dan prosedur yang sudah terbukti.
        - **{vv_style}**: Lebih mudah memahami informasi melalui gambar, diagram, dan media visual lainnya.
        - **{qg_style}**: Memproses informasi secara bertahap, langkah demi langkah, dan teratur.
        """)
        
        st.info("Catatan: Deskripsi ini adalah interpretasi dari karakteristik rata-rata mahasiswa di dalam cluster tersebut.")
