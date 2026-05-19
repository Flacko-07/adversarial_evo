# src/data_os.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_binary_30k(csv_path, test_size=0.2, seed=42):
    """Load Binary-30K CSV and return X, y, input_dim."""
    df = pd.read_csv(csv_path, low_memory=False)

    # Auto‑detect the label column
    possible_labels = ['malware', 'label', 'is_malware', 'class', 'benign']
    label_col = None
    for col in possible_labels:
        if col in df.columns:
            label_col = col
            break
    if label_col is None:
        # Last resort: print columns and ask
        raise KeyError(f"No known label column found. Columns: {list(df.columns)}")

    # Determine which value means 'malware'
    # Usually 1, True, or 'malware'
    if label_col == 'class':
        y = (df[label_col] != 'benign').astype(int).values.astype(np.float32)
    else:
        y = df[label_col].astype(int).values.astype(np.float32)

    # Drop label and non‑numeric columns (filenames, hashes)
    drop_cols = ['filename', 'hash', 'sha256', 'md5', label_col]
    drop_cols = [c for c in drop_cols if c in df.columns]
    X_df = df.drop(columns=drop_cols, errors='ignore')
    X_df = X_df.select_dtypes(include=[np.number])

    # Handle inf/nan
    X_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_df.fillna(0, inplace=True)

    X = X_df.values.astype(np.float32)
    input_dim = X.shape[1]

    # Normalise
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    print(f"[✓] Loaded Binary-30K: {len(X)} samples, {input_dim} features")
    print(f"    Benign: {(y==0).sum()}, Malicious: {(y==1).sum()}")
    return X, y, input_dim