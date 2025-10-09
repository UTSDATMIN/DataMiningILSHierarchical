def interpret_cluster(row, threshold=3.0):
    parts = []
    parts.append("Active" if row["AR_num"] >= threshold else "Reflective")
    parts.append("Sensing" if row["SI_num"] >= threshold else "Intuitive")
    parts.append("Visual" if row["VV_num"] >= threshold else "Verbal")
    parts.append("Sequential" if row["QG_num"] >= threshold else "Global")
    return ", ".join(parts)

def describe_cluster(row, total_count, kuisoner_data):
    cluster = row["Cluster"]
    n = int((kuisoner_data["Cluster"] == cluster).sum())
    pct = n / total_count * 100
    chars = row["Karakteristik"]
    return (f"{cluster}: berisi {n} peserta ({pct:.1f}% dari total). "
            f"Karakteristik dominan: {chars}. "
            f"Rata-rata skor — AR={row['AR_num']:.2f}, "
            f"SI={row['SI_num']:.2f}, VV={row['VV_num']:.2f}, "
            f"QG={row['QG_num']:.2f}.")
