import pandas as pd

def load_data(file_path):
    df = pd.read_csv(file_path, sep=None, engine="python")

    meta_cols = df.columns[:5].tolist()
    q_cols = [f"Q{i}" for i in range(1, 21)]
    rename_map = {old: new for old, new in zip(df.columns[5:5+20], q_cols)}
    df = df.rename(columns=rename_map)

    def norm(s):
        return str(s).strip()

    for c in q_cols:
        df[c] = df[c].map(norm)

    return df, meta_cols, q_cols
