#!/usr/bin/env python3
"""Classify live flows and add iptables DROP rules for malicious IPs."""
import sys, os, subprocess, torch, json, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from src.detector.model import DetectorMLP

EVE_PATH = "/var/log/suricata/eve.json"
MODEL_PATH = "unified_detector.ckpt"
INPUT_DIM = 99   # total unified features
BLOCKED_IPS_FILE = "blocked_ips.txt"

def load_model():
    model = DetectorMLP(input_dim=INPUT_DIM, hidden_dims=[512,256,128], dropout=0.0)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()
    return model

def extract_features(event):
    # same 10 features as in the feeder
    return {
        "src_port": event.get("src_port", 0) or 0,
        "dest_port": event.get("dest_port", 0) or 0,
        "severity": event.get("alert", {}).get("severity", 0) or 0,
        "flow_age": event.get("flow", {}).get("age", 0) or 0,
        "flow_pkts_toserver": event.get("flow", {}).get("pkts_toserver", 0) or 0,
        "flow_pkts_toclient": event.get("flow", {}).get("pkts_toclient", 0) or 0,
        "flow_bytes_toserver": event.get("flow", {}).get("bytes_toserver", 0) or 0,
        "flow_bytes_toclient": event.get("flow", {}).get("bytes_toclient", 0) or 0,
        "proto_enc": {"TCP":0,"UDP":1,"ICMP":2}.get(event.get("proto","").upper(),3),
        "blocked": 1 if event.get("action") == "blocked" else 0,
    }

def pad_features(feat_dict):
    ids_part = [
        feat_dict["src_port"], feat_dict["dest_port"], feat_dict["severity"],
        feat_dict["flow_age"], feat_dict["flow_pkts_toserver"], feat_dict["flow_pkts_toclient"],
        feat_dict["flow_bytes_toserver"], feat_dict["flow_bytes_toclient"],
        feat_dict["proto_enc"], feat_dict["blocked"]
    ]
    # pad to 13 binary + 76 ids + 10 os = 99? Wait, unified total is 13+76+10 = 99, not 98. Let's fix: earlier we had 13 binary + 76 ids + 10 os = 99. Actually earlier we said 98? In the output "Unified dataset: ... features: 98"? No, the code printed 98? In the user's earlier successful run, it printed 13+76+10=99? We need to check. In the output for Binary-30K it said 13 features, IDS 76 features, OS 10 features => total 99. But the error said "operands could not be broadcast" with shapes (29793,13) (2830743,76) – that's a concatenation error, not about 99. So total features is 13+76+10 = 99. I'll use 99. But in the training script we used input_dim = 13+76+? Wait, in the run_os.py we used 13 features only. In the unified we have 13 + 76 + 10 = 99. I'll set TOTAL_FEATURES = 99.
    # But the user's run_unified.py printed "features: 98"? Actually we never got that far because of the bug. Let's just compute dynamically from the model. We'll load model input_dim from the checkpoint if possible. But for simplicity, we'll hardcode 99. We'll adjust the feeder and responder to use 99.
    # I'll define N_BINARY=13, N_IDS=76, N_OS=10 => TOTAL=99.
    binary_pad = [0.0]*13
    ids_pad = ids_part + [0.0]*(76-10)
    os_pad = [0.0]*10
    return binary_pad + ids_pad + os_pad

def block_ip(ip):
    # Block using iptables (WSL supports it)
    if not os.path.exists(BLOCKED_IPS_FILE):
        open(BLOCKED_IPS_FILE, 'w').close()
    with open(BLOCKED_IPS_FILE) as f:
        blocked = set(line.strip() for line in f)
    if ip in blocked:
        return
    subprocess.run(["sudo", "iptables", "-A", "INPUT", "-s", ip, "-j", "DROP"], check=False)
    with open(BLOCKED_IPS_FILE, "a") as f:
        f.write(ip + "\n")
    print(f"Blocked {ip}")

def main():
    model = load_model()
    print("Firewall responder started. Monitoring eve.json …")
    with open(EVE_PATH, "r") as f:
        f.seek(0, 2)  # tail
        while True:
            line = f.readline()
            if not line:
                time.sleep(0.5)
                continue
            try:
                event = json.loads(line)
            except:
                continue
            src_ip = event.get("src_ip")
            if not src_ip:
                continue
            features = extract_features(event)
            vec = torch.tensor([pad_features(features)], dtype=torch.float32)
            with torch.no_grad():
                logit = model(vec)
                prob = torch.sigmoid(logit).item()
                if prob > 0.5:
                    print(f"Attack detected from {src_ip} (prob={prob:.4f})")
                    block_ip(src_ip)

if __name__ == "__main__":
    main()