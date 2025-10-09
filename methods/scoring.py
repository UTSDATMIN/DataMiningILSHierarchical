import pandas as pd
import numpy as np

ANS_MAP = {
    "Q1": {"Diskusi kelompok": 1, "Belajar sendiri dengan merenung": 0},
    "Q2": {"Langsung mencoba mengerjakan": 1, "Memikirkan dulu sebelum mencoba": 0},
    "Q3": {"Menjelaskannya pada orang lain": 1, "Membaca/merenungkannya dalam diam": 0},
    "Q4": {"Terlibat aktif dalam diskusi": 1, "Mendengarkan lalu memikirkan sendiri": 0},
    "Q5": {"Mengerjakan proyek bersama teman": 1, "Mengerjakan tugas secara mandiri": 0},

    "Q6": {"Fakta dan contoh nyata": 1, "Teori dan konsep abstrak": 0},
    "Q7": {"Menggunakan metode yang sudah jelas": 1, "Mencoba pendekatan baru yang kreatif": 0},
    "Q8": {"Detail praktis": 1, "Hubungan antar konsep": 0},
    "Q9": {"Yang banyak aplikasinya dalam kehidupan nyata": 1, "Yang penuh ide baru dan inovatif": 0},
    "Q10": {"Mengetahui langkah-langkah pasti": 1, "Mengeksplorasi kemungkinan lain": 0},

    "Q11": {"Diagram, grafik, gambar": 1, "Penjelasan teks atau lisan": 0},
    "Q12": {"Membuat skema/mindmap": 1, "Menulis dalam bentuk kalimat/paragraf": 0},
    "Q13": {"Menggunakan media visual (slide, gambar)": 1, "Menjelaskan panjang lebar dengan kata-kata": 0},
    "Q14": {"Belajar dengan melihat ilustrasi": 1, "Belajar dengan membaca/menyimak penjelasan": 0},
    "Q15": {"Visual (gambar, warna)": 1, "Verbal (kata, suara)": 0},

    "Q16": {"Langkah demi langkah yang teratur": 1, "Melihat gambaran besar terlebih dahulu": 0},
    "Q17": {"Mengikuti urutan dari awal ke akhir": 1, "Membaca bagian yang saya anggap penting dulu": 0},
    "Q18": {"Dijelaskan secara runtut dan sistematis": 1, "Dijelaskan secara garis besar dulu": 0},
    "Q19": {"Mengikuti prosedur langkah demi langkah": 1, "Melompat ke solusi dengan memahami konsep besar": 0},
    "Q20": {"Dengan urutan yang jelas dan logis": 1, "Dengan cara bebas dan menyeluruh": 0},
}

def encode_answers(df, q_cols):
    enc = df.copy()
    for q in q_cols:
        enc[q] = enc[q].map(ANS_MAP[q]).astype("Int64")
    return enc

def _sumcols(enc, cols):
    return enc[cols].astype("float").sum(axis=1)

def label_dim(score, a_label, b_label):
    if score >= 4: return f"Strong {a_label}"
    if score == 3: return f"Lean {a_label}"
    if score == 2: return f"Lean {b_label}"
    return f"Strong {b_label}"

def pole(score, a_label, b_label):
    return a_label if score >= 3 else b_label

def build_features(df, enc):
    score_AR = _sumcols(enc, ["Q1","Q2","Q3","Q4","Q5"])
    score_SI = _sumcols(enc, ["Q6","Q7","Q8","Q9","Q10"])
    score_VV = _sumcols(enc, ["Q11","Q12","Q13","Q14","Q15"])
    score_QG = _sumcols(enc, ["Q16","Q17","Q18","Q19","Q20"])

    scores = pd.DataFrame({
        "AR_num": score_AR,
        "SI_num": score_SI,
        "VV_num": score_VV,
        "QG_num": score_QG
    })

    labels = pd.DataFrame({
        "AR_label": scores["AR_num"].map(lambda s: label_dim(s, "Active","Reflective")),
        "SI_label": scores["SI_num"].map(lambda s: label_dim(s, "Sensing","Intuitive")),
        "VV_label": scores["VV_num"].map(lambda s: label_dim(s, "Visual","Verbal")),
        "QG_label": scores["QG_num"].map(lambda s: label_dim(s, "Sequential","Global")),
    })

    app_cols = pd.DataFrame({
        "AR": scores["AR_num"].map(lambda s: pole(s,"Active","Reflective")),
        "SI": scores["SI_num"].map(lambda s: pole(s,"Sensing","Intuitive")),
        "VV": scores["VV_num"].map(lambda s: pole(s,"Visual","Verbal")),
        "QG": scores["QG_num"].map(lambda s: pole(s,"Sequential","Global")),
    })

    if "NIM" in df.columns:
        ID = df["NIM"]
    else:
        ID = pd.RangeIndex(start=1, stop=len(df)+1, step=1)

    kuisoner_data = pd.DataFrame({
        "ID": ID,
        "AR": app_cols["AR"],
        "SI": app_cols["SI"],
        "VV": app_cols["VV"],
        "QG": app_cols["QG"],
    })

    X = scores[["AR_num", "SI_num", "VV_num", "QG_num"]].values
    return scores, labels, app_cols, kuisoner_data, X
