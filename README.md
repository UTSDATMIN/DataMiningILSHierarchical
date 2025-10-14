## 📊 ILS Clustering Dashboard

**ILS Clustering Dashboard** adalah aplikasi berbasis **Streamlit** untuk menganalisis **pola gaya belajar mahasiswa** menggunakan metode **Hierarchical (Ward)** dan **DBSCAN (Manual)**.
Aplikasi ini memungkinkan pengguna melakukan **eksperimen, eksplorasi, dan penarikan kesimpulan manual** terhadap hasil clustering melalui **beragam visualisasi interaktif dan interpretatif**.

---

### 🧩 Fitur Utama

* **📂 Upload Dataset Sendiri**
  Pengguna dapat mengunggah file CSV alternatif melalui sidebar untuk menggantikan dataset default bawaan aplikasi.
  (Jika tidak diunggah, aplikasi otomatis memuat dataset default dari folder `data/`.)

* **⚙️ Pilihan Metode Clustering**

  * **Hierarchical (Ward)** untuk melihat struktur hirarki data dan hubungan antar-responden.
  * **DBSCAN (Manual)** untuk mengelompokkan data berdasarkan kerapatan dan mendeteksi outlier atau noise.

* **📈 Visualisasi Lengkap untuk Analisis Manual**
  Aplikasi ini menyediakan **beragam grafik interaktif** guna mempermudah pengguna dalam **melihat pola, memahami hasil clustering, dan menarik kesimpulan secara mandiri**, meliputi:

  * **Distribusi Cluster (Pie & Bar Chart):** Menunjukkan proporsi tiap cluster.
  * **Analisis Gaya Belajar (AR, SI, VV, QG):** Menampilkan kecenderungan responden berdasarkan dimensi gaya belajar.
  * **Dendrogram Interaktif:** Memvisualisasikan proses penggabungan cluster pada metode Hierarchical.
  * **Profil Tiap Cluster:** Rata-rata skor tiap dimensi gaya belajar pada masing-masing cluster.
  * **Interpretasi Otomatis:** Deskripsi karakteristik cluster berdasarkan data agregat.
  * **Cluster Heatmap:** Representasi visual hubungan antar-fitur dalam tiap cluster.

  Visualisasi-visualisasi ini **tidak hanya menampilkan hasil otomatis**, tetapi juga dirancang agar pengguna dapat **melakukan interpretasi manual**—misalnya dengan membandingkan pola antar-cluster, mengevaluasi kedekatan antar dimensi, atau menganalisis kecenderungan responden tertentu.

* **🧠 Prediksi Gaya Belajar**
  Pengguna dapat menjawab 4 pertanyaan sederhana (AR, SI, VV, QG) untuk melihat **cluster gaya belajar yang paling mirip** dengan profilnya.

---

### 📦 Persyaratan

Pastikan sudah terinstal:

* Python ≥ 3.8
* pip atau conda
* Koneksi internet (untuk Streamlit)

---

### 🚀 Instalasi & Menjalankan Aplikasi

1. **Kloning repositori**

   ```bash
   git clone https://github.com/username/ils-clustering-dashboard.git
   cd ils-clustering-dashboard
   ```

2. **Buat environment (opsional)**

   ```bash
   conda create -n ils python=3.10
   conda activate ils
   ```

   atau dengan venv:

   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   source venv/bin/activate  # Mac/Linux
   ```

3. **Instal dependensi**

   ```bash
   pip install -r requirements.txt
   ```

4. **Jalankan aplikasi**

   ```bash
   streamlit run main-app.py
   ```

---

### 🗂️ Struktur Folder

```
📁 project-root/
│
├── main-app.py
├── requirements.txt
├── README.md
│
├── data/
│   └── Kuesioner Identifikasi Pola Gaya Belajar Mahasiswa.csv
│
├── methods/
│   ├── data_loader.py
│   ├── scoring.py
│   ├── clustering_manual.py
│   ├── dbscan_manual.py
│   ├── heatmap_cluster.py
│   ├── visualization.py
│   ├── interpretation.py
│   └── cluster_profile.py
```

---

### 🧾 Cara Menggunakan

1. Jalankan aplikasi → browser akan otomatis terbuka di `http://localhost:8501`.
2. Di **sidebar kiri**, pilih:

   * Metode clustering (Ward / DBSCAN)
   * Parameter jumlah cluster atau nilai eps & minPts
3. (Opsional) Upload dataset CSV sendiri di bagian **📂 Upload CSV file (optional)**.
4. Lihat hasil di 6 tab utama:

   * Distribusi Cluster
   * Gaya Belajar
   * Dendrogram
   * Profil Cluster
   * Interpretasi Otomatis
   * Cluster Heatmap
5. Isi pertanyaan gaya belajar di bagian bawah → tekan **Submit** untuk melihat hasil prediksi cluster Anda.

---

### 📘 Tujuan Analisis Manual

Aplikasi ini tidak hanya memberikan hasil otomatis, tetapi juga mendorong **pengguna untuk memahami data secara mendalam**.
Dengan melihat berbagai visualisasi yang disediakan, pengguna dapat:

* Menarik kesimpulan berdasarkan pola visual.
* Membandingkan antar cluster secara mandiri.
* Mengamati perbedaan dimensi gaya belajar antar kelompok.
* Mengevaluasi hasil clustering dengan pendekatan eksploratif.

Pendekatan ini sangat berguna dalam konteks **penelitian pendidikan**, **analisis perilaku belajar**, maupun **pengembangan sistem rekomendasi pembelajaran adaptif**.

---

### 👨‍💻 Pengembang

**Nama:** Muhammad Fachri, Michael Christianto, Aidan Ismail, Atharik Putra
**Proyek:** ILS Clustering Dashboard
**Framework:** Streamlit, Plotly, Scikit-learn, NumPy, Pandas