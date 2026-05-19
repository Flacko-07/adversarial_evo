# src/data_unified.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_unified_dataset(
    binary_csv=None,
    ids_csv_dir=None,
    os_csv=None,
    test_size=0.2,
    seed=42
):
    all_X = []
    all_y = []
    all_features = []

    # 1. Binary‑30K
    if binary_csv:
        df = pd.read_csv(binary_csv, low_memory=False)
        label_col = 'is_malware'
        y_bin = df[label_col].astype(int).values.astype(np.float32)
        drop_cols = ['file_id','file_path','file_name','sha256','md5','tokens', label_col]
        drop_cols = [c for c in drop_cols if c in df.columns]
        X_bin = df.select_dtypes(include=[np.number]).drop(columns=drop_cols, errors='ignore')
        X_bin.replace([np.inf, -np.inf], np.nan, inplace=True)
        X_bin.fillna(0, inplace=True)
        all_X.append(X_bin.values.astype(np.float32))
        all_y.append(y_bin)
        all_features.append(('binary', list(X_bin.columns)))
        print(f"Binary samples: {len(y_bin)}, features: {X_bin.shape[1]}")

    # 2. IDS
    if ids_csv_dir:
        from src.data_ids import load_cic_ids2017
        X_ids, y_ids = load_cic_ids2017(ids_csv_dir)
        all_X.append(X_ids.astype(np.float32))
        all_y.append(y_ids.astype(np.float32))
        all_features.append(('ids', [f'ids_feat_{i}' for i in range(X_ids.shape[1])]))
        print(f"IDS samples: {len(y_ids)}, features: {X_ids.shape[1]}")

    # 3. OS fingerprints
    if os_csv:
        df_os = pd.read_csv(os_csv)
        if 'label' not in df_os.columns:
            raise KeyError("OS CSV must contain a 'label' column (0/1)")
        y_os = df_os['label'].astype(int).values.astype(np.float32)
        drop_cols = ['label', 'os', 'timestamp', 'listen_ports']
        drop_cols = [c for c in drop_cols if c in df_os.columns]
        X_os = df_os.select_dtypes(include=[np.number]).drop(columns=drop_cols, errors='ignore')
        X_os.replace([np.inf, -np.inf], np.nan, inplace=True)
        X_os.fillna(0, inplace=True)
        all_X.append(X_os.values.astype(np.float32))
        all_y.append(y_os)
        all_features.append(('os', list(X_os.columns)))
        print(f"OS samples: {len(y_os)}, features: {X_os.shape[1]}")

    # ── Concatenate with zero‑padding ──
    total_features = sum(X.shape[1] for X in all_X)
    total_samples = sum(len(y) for y in all_y)
    X_combined = np.zeros((total_samples, total_features), dtype=np.float32)
    y_combined = np.concatenate(all_y)

    col_start = 0
    row_start = 0
    for X in all_X:
        n_feat = X.shape[1]
        n_rows = X.shape[0]
        X_combined[row_start:row_start + n_rows, col_start:col_start + n_feat] = X
        row_start += n_rows
        col_start += n_feat

    # Normalise after merging
    scaler = StandardScaler()
    X_combined = scaler.fit_transform(X_combined)

    print(f"\nUnified dataset: {X_combined.shape[0]} samples, {X_combined.shape[1]} features")
    print(f"Benign: {(y_combined==0).sum()}, Malicious/Vulnerable: {(y_combined==1).sum()}")
    return X_combined, y_combined, total_features, all_features