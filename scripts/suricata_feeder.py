#!/usr/bin/env python3
"""Tail eve.json and write padded feature vectors to a CSV file."""
import json, csv, time, os, sys
from pathlib import Path
import numpy as np
import pandas as pd

# ---- Adjust these paths ----
EVE_PATH = "/var/log/suricata/eve.json"   # or ./logs/eve.json
OUT_CSV = "data/live_suricata.csv"
# Feature dimensions from your unified dataset
N_BINARY_FEATURES = 13
N_IDS_FEATURES = 76
N_OS_FEATURES = 10
TOTAL_FEATURES = N_BINARY_FEATURES + N_IDS_FEATURES + N_OS_FEATURES

# ---- Feature extraction for a single eve.json line ----
def extract_features_from_line(line: dict) -> dict:
    """Extract the 10 Suricata features used in your IDS loader."""
    return {
        "src_port": line.get("src_port", 0) or 0,
        "dest_port": line.get("dest_port", 0) or 0,
        "severity": line.get("alert", {}).get("severity", 0) or 0,
        "flow_age": line.get("flow", {}).get("age", 0) or 0,
        "flow_pkts_toserver": line.get("flow", {}).get("pkts_toserver", 0) or 0,
        "flow_pkts_toclient": line.get("flow", {}).get("pkts_toclient", 0) or 0,
        "flow_bytes_toserver": line.get("flow", {}).get("bytes_toserver", 0) or 0,
        "flow_bytes_toclient": line.get("flow", {}).get("bytes_toclient", 0) or 0,
        "proto_enc": {"TCP": 0, "UDP": 1, "ICMP": 2}.get(line.get("proto", "").upper(), 3),
        "blocked": 1 if line.get("action") == "blocked" else 0,
    }

def pad_to_unified(suricata_features: dict, label: int) -> list:
    """Return a list of length TOTAL_FEATURES, ready for CSV."""
    # Binary part → all zeros
    binary_pad = [0.0] * N_BINARY_FEATURES
    # IDS part → the 10 Suricata features, padded with zeros to 76
    ids_feat = [
        suricata_features["src_port"],
        suricata_features["dest_port"],
        suricata_features["severity"],
        suricata_features["flow_age"],
        suricata_features["flow_pkts_toserver"],
        suricata_features["flow_pkts_toclient"],
        suricata_features["flow_bytes_toserver"],
        suricata_features["flow_bytes_toclient"],
        suricata_features["proto_enc"],
        suricata_features["blocked"],
    ]
    ids_pad = ids_feat + [0.0] * (N_IDS_FEATURES - len(ids_feat))
    # OS part → all zeros
    os_pad = [0.0] * N_OS_FEATURES
    return binary_pad + ids_pad + os_pad + [label]

def main():
    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    # Write header if file is new
    if not Path(OUT_CSV).exists():
        header = [f"feat_{i}" for i in range(TOTAL_FEATURES)] + ["label"]
        with open(OUT_CSV, "w", newline="") as f:
            csv.writer(f).writerow(header)

    print(f"Monitoring {EVE_PATH} ...")
    # Follow the file (like tail -f)
    with open(EVE_PATH, "r") as f:
        # Go to end of file
        f.seek(0, 2)
        while True:
            line = f.readline()
            if not line:
                time.sleep(0.5)
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            # Only process flow or alert events
            etype = event.get("event_type", "")
            if etype not in ("alert", "flow", "http", "dns", "tls"):
                continue
            features = extract_features_from_line(event)
            label = 1 if etype == "alert" else 0
            row = pad_to_unified(features, label)
            with open(OUT_CSV, "a", newline="") as outf:
                csv.writer(outf).writerow(row)
            print(f"Added {etype} (label={label})")

if __name__ == "__main__":
    main()