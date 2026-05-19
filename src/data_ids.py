# src/data_ids.py
"""Real‑world IDS data loaders for adversarial evolution on Arch/BlackArch."""

import json
import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder

# ──────────────────────────────────────────────────────────────────
# 1 – Suricata eve.json parser
# ──────────────────────────────────────────────────────────────────

def parse_eve_json(eve_path, max_lines=None):
    """
    Parse a Suricata eve.json file into a pandas DataFrame.

    Parameters
    ----------
    eve_path : str
        Path to /var/log/suricata/eve.json or offline copy.
    max_lines : int or None
        Limit number of lines to parse (None = all).

    Returns
    -------
    pd.DataFrame
        One row per event. Columns: timestamp, event_type, src_ip,
        src_port, dest_ip, dest_port, proto, severity, signature, etc.
    """
    rows = []
    with open(eve_path, "r") as fh:
        for i, line in enumerate(fh):
            if max_lines and i >= max_lines:
                break
            try:
                event = json.loads(line.strip())
            except json.JSONDecodeError:
                continue

            row = {
                "timestamp":    event.get("timestamp"),
                "event_type":   event.get("event_type"),
                "src_ip":       event.get("src_ip"),
                "src_port":     event.get("src_port", 0),
                "dest_ip":      event.get("dest_ip"),
                "dest_port":    event.get("dest_port", 0),
                "proto":        event.get("proto", "").upper(),
            }

            # severity & signature from 'alert' sub-dict
            alert = event.get("alert", {})
            row["severity"]  = alert.get("severity", 0)
            row["signature"] = alert.get("signature", "")
            row["category"]  = alert.get("category", "")
            row["action"]    = alert.get("action", "allowed")

            # flow stats if present
            flow = event.get("flow", {})
            row["flow_pkts_toserver"] = flow.get("pkts_toserver", 0)
            row["flow_pkts_toclient"] = flow.get("pkts_toclient", 0)
            row["flow_bytes_toserver"] = flow.get("bytes_toserver", 0)
            row["flow_bytes_toclient"] = flow.get("bytes_toclient", 0)
            row["flow_start"] = flow.get("start", "")
            row["flow_end"]   = flow.get("end", "")
            row["flow_age"]   = flow.get("age", 0)

            rows.append(row)

    return pd.DataFrame(rows)


def extract_features_from_eve(eve_path, max_lines=None):
    """
    Generate feature vectors + labels from Suricata eve.json.

    Labels are derived from event_type: 'alert' → 1 (attack),
    everything else → 0 (benign).

    Returns
    -------
    X : np.ndarray, shape (n_samples, 15)
    y : np.ndarray, shape (n_samples,)
    """
    df = parse_eve_json(eve_path, max_lines)

    # numeric features
    df["severity"]      = pd.to_numeric(df["severity"], errors="coerce").fillna(0)
    df["src_port"]      = pd.to_numeric(df["src_port"], errors="coerce").fillna(0)
    df["dest_port"]     = pd.to_numeric(df["dest_port"], errors="coerce").fillna(0)
    df["flow_age"]      = pd.to_numeric(df["flow_age"], errors="coerce").fillna(0)
    df["flow_pkts_toserver"]  = pd.to_numeric(df["flow_pkts_toserver"], errors="coerce").fillna(0)
    df["flow_pkts_toclient"]  = pd.to_numeric(df["flow_pkts_toclient"], errors="coerce").fillna(0)
    df["flow_bytes_toserver"] = pd.to_numeric(df["flow_bytes_toserver"], errors="coerce").fillna(0)
    df["flow_bytes_toclient"] = pd.to_numeric(df["flow_bytes_toclient"], errors="coerce").fillna(0)

    # encode protocol
    proto_le = LabelEncoder()
    df["proto_enc"] = proto_le.fit_transform(df["proto"].fillna("UNKNOWN"))

    # whether action was "blocked"
    df["blocked"] = (df["action"] == "blocked").astype(int)

    feature_cols = [
        "src_port", "dest_port", "severity",
        "flow_age", "flow_pkts_toserver", "flow_pkts_toclient",
        "flow_bytes_toserver", "flow_bytes_toclient",
        "proto_enc", "blocked"
    ]

    X = df[feature_cols].values.astype(np.float32)
    # label: alert=1, everything else=0
    y = (df["event_type"] == "alert").astype(int).values.astype(np.float32)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y


# ──────────────────────────────────────────────────────────────────
# 2 – CIC‑IDS‑2017 CSV loader
# ──────────────────────────────────────────────────────────────────

CIC_IDS2017_FEATURES = [
    "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets",
    "Fwd Packet Length Max", "Fwd Packet Length Min",
    "Fwd Packet Length Mean", "Fwd Packet Length Std",
    "Bwd Packet Length Max", "Bwd Packet Length Min",
    "Bwd Packet Length Mean", "Bwd Packet Length Std",
    "Flow Bytes/s", "Flow Packets/s",
    "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
    "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min",
    "Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min",
    "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags",
    "Fwd Header Length", "Bwd Header Length",
    "Fwd Packets/s", "Bwd Packets/s",
    "Min Packet Length", "Max Packet Length",
    "Packet Length Mean", "Packet Length Std", "Packet Length Variance",
    "FIN Flag Count", "SYN Flag Count", "RST Flag Count",
    "PSH Flag Count", "ACK Flag Count", "URG Flag Count",
    "CWE Flag Count", "ECE Flag Count",
    "Down/Up Ratio", "Average Packet Size",
    "Avg Fwd Segment Size", "Avg Bwd Segment Size",
    "Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate",
    "Subflow Fwd Packets", "Subflow Fwd Bytes",
    "Subflow Bwd Packets", "Subflow Bwd Bytes",
    "Init_Win_bytes_forward", "Init_Win_bytes_backward",
    "act_data_pkt_fwd", "min_seg_size_forward",
    "Active Mean", "Active Std", "Active Max", "Active Min",
    "Idle Mean", "Idle Std", "Idle Max", "Idle Min",
]


def load_cic_ids2017(csv_dir, benign_label="BENIGN"):
    """
    Load CIC‑IDS‑2017 CSV files into X, y arrays.

    Parameters
    ----------
    csv_dir : str
        Directory containing CSV files (e.g. "Friday-WorkingHours-Morning.pcap_ISCX.csv").
    benign_label : str
        String that marks benign traffic in the Label column.

    Returns
    -------
    X : np.ndarray
        Feature vectors (79 columns).
    y : np.ndarray
        Binary labels (1 = attack, 0 = benign).
    """
    csv_dir = Path(csv_dir)
    csv_files = list(csv_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {csv_dir}")

    frames = []
    for f in csv_files:
        df = pd.read_csv(f, low_memory=False)
        # Strip whitespace from column names
        df.columns = df.columns.str.strip()
        # Merge possible multi-line labels
        if "Label" in df.columns:
            df["Label"] = df["Label"].str.strip()
        frames.append(df)

    full = pd.concat(frames, ignore_index=True)

    # Use only the standard 79 features
    available = [c for c in CIC_IDS2017_FEATURES if c in full.columns]
    missing = set(CIC_IDS2017_FEATURES) - set(available)
    if missing:
        print(f"⚠  Missing features: {missing}")

    X = full[available].copy()
    # Replace inf / nan
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(0, inplace=True)

    # Labels
    y = (full["Label"] != benign_label).astype(int).values.astype(np.float32)

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.values.astype(np.float32))

    print(f"Loaded {len(X_scaled)} samples, {X_scaled.shape[1]} features")
    print(f"  Benign: {(y == 0).sum()}, Attack: {(y == 1).sum()}")
    return X_scaled, y


# ──────────────────────────────────────────────────────────────────
# 3 – Unified loader (drop-in replacement for synthetic data)
# ──────────────────────────────────────────────────────────────────

def load_ids_data(source="auto", csv_dir=None, eve_path=None, n_features=None):
    """
    Load IDS data from the best available source.

    1. If csv_dir is given → CIC‑IDS‑2017 CSV
    2. If eve_path is given → Suricata eve.json
    3. If source == "auto" → try default locations on Arch

    Returns
    -------
    X : np.ndarray
    y : np.ndarray
    """
    # Option 1 – explicit CSV
    if csv_dir and Path(csv_dir).exists():
        print(f"[*] Loading CIC-IDS2017 from {csv_dir}")
        return load_cic_ids2017(csv_dir)

    # Option 2 – explicit eve.json
    if eve_path and Path(eve_path).exists():
        print(f"[*] Loading Suricata eve.json from {eve_path}")
        return extract_features_from_eve(eve_path)

    # Option 3 – auto‑detect on Arch
    default_paths = [
        "/var/log/suricata/eve.json",
        os.path.expanduser("~/suricata/logs/eve.json"),
        "./logs/eve.json",
    ]
    for p in default_paths:
        if Path(p).exists():
            print(f"[*] Auto‑detected Suricata log: {p}")
            return extract_features_from_eve(p)

    raise FileNotFoundError(
        "No IDS data found. Run Suricata first or pass csv_dir/eve_path."
    )
