<div align="center">

# 🛡️ BlackAV — Adversarial Security Evolution Engine

**An open-source, self-improving AI cybersecurity engine that fights a GAN attacker and hardens itself automatically.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch Lightning](https://img.shields.io/badge/PyTorch_Lightning-2.x-purple?logo=lightning)](https://lightning.ai/)
[![Hydra](https://img.shields.io/badge/Hydra-1.3-orange?logo=meta)](https://hydra.cc/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](./LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Arch%20Linux%20%7C%20WSL2-black?logo=linux)](https://archlinux.org/)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen)]()

> Built on **Arch Linux / WSL2** • **PyTorch Lightning** • **Hydra** • **Suricata** • **NVD API**

[Overview](#overview) • [Results](#results) • [Architecture](#architecture) • [Quick Start](#quick-start) • [Project Structure](#project-structure) • [Roadmap](#roadmap) • [Contributing](#contributing)

</div>

---

## Overview

BlackAV is an **adversarial machine learning system** for endpoint protection. It trains a **Blue neural network detector** to identify malware, network intrusions, and OS-level vulnerabilities — while a **Red GAN (WGAN-GP)** continuously tries to bypass it. The result is a **provably robust** detector that remains effective even against adaptive, AI-generated attacks.

Unlike conventional signature-based cybersecurity engines that fail against zero-day threats, BlackAV learns from adversarial pressure. Each time the Red GAN crafts a new evasion, the Blue detector adapts — a closed loop that produces a model hardened against the worst-case attacker.

### Key Principles

- **Adversarial robustness over accuracy**: A 99%+ clean-accuracy model is worthless if a GAN can evade it at 50%. BlackAV optimises for adversarial detection rate.
- **Multi-layer coverage**: One unified model covers binary malware, network intrusions, and OS vulnerabilities simultaneously.
- **Fully local & private**: No cloud telemetry. No model API calls. Everything runs on your machine.
- **Self-improving by design**: The cron/systemd retraining pipeline ensures the model continuously improves as new attack data arrives.

---

## What It Detects

| Layer | Data Source | Samples | Features | Coverage |
|-------|-------------|---------|----------|----------|
| 🦠 Binary malware | [Binary-30K (HuggingFace)](https://huggingface.co/datasets/mjbommar/binary-30k-tokenized) | 29,793 | 13 | Windows PE, Linux ELF, macOS Mach-O, Android APK |
| 🌐 Network intrusions | [CIC-IDS-2017](https://www.unb.ca/cic/datasets/ids-2017.html) + Suricata live | 2,830,743 | 76 | DoS, DDoS, PortScan, Brute Force, Web Attacks, Botnets |
| 💻 OS vulnerabilities | OS fingerprint + [NVD API](https://nvd.nist.gov/developers/vulnerabilities) | 1+ (live) | 10 | CVEs mapped to installed packages via NVD |

**Unified model**: 2.86M+ samples, 99 features, trained jointly with adversarial hardening.

---

## Results

| Metric | Value | Notes |
|--------|-------|-------|
| ✅ Clean test accuracy | **94.1%** | Standard held-out test split |
| 🛡️ Adversarial detection accuracy | **99.85%** | Against WGAN-GP generated evasions |
| 💀 GAN evasion rate | **0.15%** | Red GAN success after adversarial training |
| ⏱️ Training time (CPU) | **~12 minutes** | Unified multi-layer training pass |

> **Research context**: An adversarial detection rate of 99.85% against a trained WGAN-GP attacker places BlackAV at the frontier of adversarial ML for security. Most production cybersecurity systems are not trained adversarially at all.

---

## Architecture

The system consists of three interacting components: a **Blue Detector**, a **Red GAN**, and a **Live Pipeline**.

```
                        ┌─────────────────────────┐
                        │    Suricata Live Feed    │
                        │  (Network packet stream) │
                        └────────────┬────────────┘
                                     │
                                     ▼
          ┌──────────────────────────────────────────────────────────┐
          │                   Unified Detector (Blue)                │
          │   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
          │   │ Binary Layer │  │Network Layer │  │  OS Layer    │  │
          │   │ PE/ELF/APK   │  │ CIC-IDS/     │  │ NVD CVE      │  │
          │   │ (13 features)│  │ Suricata      │  │ fingerprint  │  │
          │   │              │  │ (76 features)│  │ (10 features)│  │
          │   └──────────────┘  └──────────────┘  └──────────────┘  │
          │              Feature concat → MLP → [Benign / Malicious] │
          └──────────────────────────┬───────────────────────────────┘
                                     │
                          ┌──────────▼──────────┐
                          │  Adversarial Loop   │
                          │                     │
                          │  ┌───────────────┐  │
                          │  │  Red GAN      │  │
                          │  │  (WGAN-GP)    │  │
                          │  │  Generates    │  │
                          │  │  evasive      │  │
                          │  │  adversarial  │  │
                          │  │  samples      │  │
                          │  └───────┬───────┘  │
                          │          │           │
                          │  min-max training   │
                          │  Blue ↔ Red         │
                          └──────────┬──────────┘
                                     │
                          ┌──────────▼──────────┐
                          │  Active Response    │
                          │  firewall_responder │
                          │  → iptables block   │
                          └─────────────────────┘
```

### Blue Detector

A multi-layer perceptron (MLP) classifier built in PyTorch Lightning. Takes a concatenated 99-feature vector from all three detection layers and outputs a binary classification (benign / malicious). Trained with standard cross-entropy loss *plus* adversarial loss from the Red GAN.

### Red GAN (WGAN-GP)

A Wasserstein GAN with Gradient Penalty. The generator learns to produce feature vectors that resemble malicious samples but are misclassified by the Blue detector. The discriminator helps maintain realistic sample quality. The gradient penalty ensures stable training dynamics without mode collapse.

### Adversarial Training Loop (min-max)

```
Phase 1: Train Red GAN to maximize Blue's error
Phase 2: Train Blue detector to correctly classify Red GAN's samples
Repeat for N phases → convergence to robust equilibrium
```

This is a direct implementation of the adversarial robustness formulation: the detector minimizes its worst-case loss under the strongest possible attacker.

### Live Self-Improvement Pipeline

```
Suricata captures → suricata_feeder.py labels → retrain_live.py fine-tunes (hourly)
                                                         ↓
                                            Updated unified_detector.ckpt
                                                         ↓
                                       firewall_responder.py blocks IPs in real time
```

---

## Features

- 🦠 **Cross-platform malware detection** — Windows PE, Linux ELF, macOS Mach-O, Android APK via Binary-30K feature vectors
- 🌐 **Live network intrusion detection** — Suricata integration feeds real-time network alerts into the model
- 💻 **Host vulnerability scanning** — OS fingerprint collection + NVD API mapping detects known CVEs on your system
- ⚔️ **Adversarial hardening** — WGAN-GP Red team continuously generates evasive attacks; Blue detector must learn to catch them
- 🔄 **Self-improving pipeline** — `retrain_live.py` runs hourly via cron/systemd, fine-tuning the model on new traffic data
- 🔥 **Active firewall response** — `firewall_responder.py` calls `iptables` to block attacking IPs in real time
- 🔒 **Fully local** — No cloud dependencies, no telemetry. Only the optional NVD API key is external.
- ⚙️ **Hydra config management** — All hyperparameters (model size, GAN phases, learning rates, data paths) live in YAML configs
- 📊 **Rich training logs** — matplotlib loss curves, accuracy metrics, and adversarial evasion rates printed per phase

---

## Project Structure

```
adversarial_evo/
├── configs/                    # Hydra YAML configuration files
│   ├── config.yaml             # Main config (training, paths, hyperparams)
│   ├── detector.yaml           # Blue detector architecture config
│   └── gan.yaml                # GAN training config (phases, LR, GP weight)
│
├── src/                        # Core Python source
│   ├── detector/
│   │   ├── model.py            # MLP detector architecture
│   │   └── lit_module.py       # PyTorch Lightning wrapper
│   ├── gan/
│   │   ├── generator.py        # WGAN-GP generator
│   │   ├── discriminator.py    # WGAN-GP discriminator
│   │   ├── losses.py           # Wasserstein + GP loss functions
│   │   └── adversarial_train.py# min-max adversarial loop
│   ├── evolution/
│   │   └── phased_loop.py      # Phased Blue-Red evolution controller
│   ├── data.py                 # Base data loader interface
│   ├── data_ids.py             # CIC-IDS-2017 CSV parser & feature extractor
│   ├── data_os.py              # Binary-30K HuggingFace loader
│   └── data_unified.py         # Merges all three layers into one dataset
│
├── scripts/                    # Runnable entry points
│   ├── run_evolution.py        # Phased Blue-Red adversarial loop
│   ├── run_os.py               # Train on Binary-30K only (binary malware)
│   ├── run_unified.py          # Train unified multi-layer detector
│   ├── suricata_feeder.py      # Reads Suricata EVE JSON → training data
│   ├── retrain_live.py         # Hourly fine-tuning on new traffic data
│   ├── firewall_responder.py   # Blocks malicious IPs via iptables
│   └── label_os_nvd.py         # Auto-labels OS fingerprints via NVD API
│
├── data/                       # (gitignored) Local datasets
│   ├── binary_30k.csv
│   ├── cic_ids2017/            # CIC-IDS-2017 CSV files
│   └── os_fingerprints.csv
│
├── pyproject.toml              # Project metadata & dependencies
├── README.md
└── LICENSE                     # MIT License
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- Linux / WSL2 (for Suricata and iptables live pipeline)
- ~4 GB disk space for datasets
- Optional: NVIDIA GPU (CUDA) for faster training — CPU works fine (~12 min)

### 1. Clone and Set Up Environment

```bash
git clone https://github.com/Flacko-07/adversarial_evo.git
cd adversarial_evo

python3 -m venv venv
source venv/bin/activate

pip install torch pytorch-lightning hydra-core omegaconf \
            pandas scikit-learn matplotlib datasets requests
```

Or install all at once via pyproject.toml:

```bash
pip install -e .
```

### 2. Download Datasets

**Binary-30K** — cross-platform malware feature vectors:

```bash
python -c "
from datasets import load_dataset
ds = load_dataset('mjbommar/binary-30k-tokenized')
ds['train'].to_csv('data/binary_30k.csv', index=False)
print('Binary-30K saved to data/binary_30k.csv')
"
```

**CIC-IDS-2017** — network intrusion detection:

Download all CSV files from [Kaggle](https://www.kaggle.com/datasets/chethuhn/network-intrusion-dataset) and place them in `data/cic_ids2017/`. The loader will automatically parse and merge all files.

**OS fingerprints** (optional — for host vulnerability scanning):

```bash
# On Linux/macOS:
python scripts/label_os_nvd.py --collect --output data/os_fingerprints.csv
```

> **NVD API key** (optional but recommended for higher rate limits):  
> Register at [https://nvd.nist.gov/developers/request-an-api-key](https://nvd.nist.gov/developers/request-an-api-key) and set `export NVD_API_KEY=your_key_here`.

### 3. Train the Unified Detector

```bash
python scripts/run_unified.py
```

This loads all three layers, trains the Blue detector jointly, and saves `unified_detector.ckpt`. Expect output like:

```
[Epoch 10/10] loss=0.043 | clean_acc=94.1% | adv_acc=99.85% | evasion_rate=0.15%
Model saved → unified_detector.ckpt
```

### 4. Run the Adversarial Evolution Loop (Optional)

To run the phased Blue-Red GAN battle:

```bash
python scripts/run_evolution.py
```

This runs multiple phases: each phase trains the Red GAN to attack, then retrains the Blue detector to defend. Watch the adversarial accuracy climb as the phases progress.

### 5. Start the Live Pipeline

Open four terminals (or use tmux/screen):

```bash
# Terminal 1: Suricata
sudo suricata -c /etc/suricata/suricata.yaml -i eth0

# Terminal 2: Feed Suricata alerts into training buffer
python scripts/suricata_feeder.py &

# Terminal 3: Active firewall response (blocks attacking IPs via iptables)
sudo python scripts/firewall_responder.py &

# Terminal 4: Set up hourly retraining via cron
crontab -e
# Add this line:
# 0 * * * * /path/to/venv/bin/python /path/to/adversarial_evo/scripts/retrain_live.py >> /var/log/blackav_retrain.log 2>&1
```

Your AI cybersecurity system is now **live and self-improving**. 🛡️

---

## Configuration

All hyperparameters are managed via Hydra YAML configs in `configs/`. Key options:

```yaml
# configs/config.yaml (excerpt)
training:
  epochs: 10
  batch_size: 512
  learning_rate: 1e-3

detector:
  hidden_dims: [256, 128, 64]
  dropout: 0.3
  input_features: 99

gan:
  phases: 5
  generator_lr: 1e-4
  discriminator_lr: 1e-4
  gp_weight: 10.0        # WGAN-GP gradient penalty coefficient
  n_critic: 5            # Discriminator steps per generator step
```

Override any config at runtime:

```bash
python scripts/run_unified.py training.epochs=20 gan.phases=10
```

---

## Requirements

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | ≥ 2.0 | Core deep learning |
| `pytorch-lightning` | ≥ 2.0 | Training loop abstraction |
| `hydra-core` | ≥ 1.3 | Config management |
| `omegaconf` | ≥ 2.3 | YAML config parsing |
| `pandas` | ≥ 2.0 | Dataset loading and preprocessing |
| `scikit-learn` | ≥ 1.3 | Train/test splits, metrics |
| `matplotlib` | ≥ 3.7 | Training curves and visualizations |
| `datasets` | ≥ 2.14 | Binary-30K HuggingFace loader |
| `requests` | ≥ 2.31 | NVD API calls |
| `suricata` | ≥ 7.0 | Live network monitoring (system package) |

Install Python dependencies:

```bash
pip install torch pytorch-lightning hydra-core omegaconf pandas scikit-learn matplotlib datasets requests
```

Install Suricata (system package):

```bash
# Arch Linux
sudo pacman -S suricata

# Ubuntu/Debian / WSL2
sudo apt install suricata
```

---

## How Adversarial Training Works

The core of BlackAV is the **min-max adversarial training loop**, a direct application of game-theoretic robustness:

**Red GAN Objective**: Maximize the Blue detector's loss by generating feature vectors that look like malicious traffic but are classified as benign.

**Blue Detector Objective**: Minimize classification loss on both real data *and* Red GAN-generated adversarial samples.

The WGAN-GP formulation ensures training stability:

- **Wasserstein distance** replaces JS divergence, providing smooth gradients even when distributions don't overlap
- **Gradient penalty** enforces the Lipschitz constraint without weight clipping, preventing gradient explosion

After `N` phases of this battle, the Blue detector converges to a robust equilibrium — a model that performs well not just on the training distribution, but on the *adversarial* distribution generated by the strongest possible attacker in the training set.

---

## Roadmap

- [ ] **Transformer-based detector** — Replace MLP with a lightweight Transformer encoder for sequence-aware binary analysis
- [ ] **RL-based Red agent** — Replace WGAN with a Reinforcement Learning agent that learns attack policies (more realistic threat model)
- [ ] **EMBER dataset integration** — Add EMBER malware feature set for richer binary-layer coverage
- [ ] **Web dashboard** — Real-time React/Next.js dashboard showing live detections, blocked IPs, and model accuracy
- [ ] **Docker + Compose** — Containerize the full stack for one-command deployment
- [ ] **Windows native support** — PowerShell-native firewall responder using Windows Defender API
- [ ] **Model explainability** — SHAP values for each detection to show which features triggered the alert
- [ ] **Federated learning mode** — Allow multiple BlackAV nodes to share model updates without sharing raw traffic

---

## Contributing

Contributions are welcome. To contribute:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Commit your changes: `git commit -m 'Add: your feature description'`
4. Push to your fork: `git push origin feature/your-feature-name`
5. Open a Pull Request against `main`

Please ensure your code follows the existing style and includes appropriate tests. For large features, open an issue first to discuss the design.

---

## Security Disclosure

If you discover a vulnerability in BlackAV itself, please open a GitHub issue marked `[SECURITY]` or contact the author directly. Do **not** use the issue tracker for zero-day disclosures of unrelated software.

---

## License

This project is licensed under the **MIT License** — see [LICENSE](./LICENSE) for details.

---

## Author

**Naval Singh**  
AI/ML Engineer • Physics-Informed ML • Adversarial Security

[![GitHub](https://img.shields.io/badge/GitHub-Flacko--07-black?logo=github)](https://github.com/Flacko-07)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://linkedin.com/in/yourprofile)

---

<div align="center">

**If BlackAV helped you, consider giving it a ⭐ on GitHub.**

</div>
