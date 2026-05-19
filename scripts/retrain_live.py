#!/usr/bin/env python3
"""Fine‑tune the unified detector on new live Suricata data."""
import sys, os, torch, pytorch_lightning as pl
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_unified import load_unified_dataset
from src.detector.model import DetectorMLP
from src.gan.models import Generator, Critic
from src.gan.adversarial_training_lit import AdversarialTrainingLit
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

MODEL_CHECKPOINT = "unified_detector.ckpt"   # save after first full training
LIVE_CSV = "data/live_suricata.csv"
ORIG_BINARY = "data/binary_30k.csv"
ORIG_IDS = "data/cic_ids2017"
ORIG_OS = "data/labeled_os_fingerprints.csv"

def main():
    # Load original data (small fraction to avoid catastrophic forgetting)
    X_orig, y_orig, input_dim, _ = load_unified_dataset(
        binary_csv=ORIG_BINARY,
        ids_csv_dir=ORIG_IDS,
        os_csv=ORIG_OS
    )
    # Load live data
    X_live, y_live, _, _ = load_unified_dataset(
        binary_csv=None,
        ids_csv_dir=None,
        os_csv=LIVE_CSV
    )
    # Combine
    import numpy as np
    X = np.vstack([X_orig, X_live])
    y = np.concatenate([y_orig, y_live])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                             torch.tensor(y_train, dtype=torch.float32).view(-1,1))
    test_ds  = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                             torch.tensor(y_test, dtype=torch.float32).view(-1,1))
    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True, num_workers=4)
    test_loader  = DataLoader(test_ds, batch_size=512, num_workers=4)
    real_attack = torch.tensor(X_train[y_train==1], dtype=torch.float32)

    # Build model
    detector = DetectorMLP(input_dim=input_dim, hidden_dims=[512,256,128], dropout=0.4)
    if os.path.exists(MODEL_CHECKPOINT):
        print("Loading existing checkpoint …")
        detector.load_state_dict(torch.load(MODEL_CHECKPOINT))
    generator = Generator(noise_dim=16, cond_dim=1, output_dim=input_dim, hidden_dims=[256,128])
    critic = Critic(input_dim=input_dim, cond_dim=1, hidden_dims=[256,128])

    adv_module = AdversarialTrainingLit(
        detector=detector, generator=generator, critic=critic,
        real_attack_data=real_attack,
        lr_det=0.0001, lr_gen=0.00005, lr_crit=0.00005,   # lower lr for fine‑tuning
        n_critic=3, lambda_gp=10, anti_weight=0.05,
        noise_dim=16, cond_dim=1, batch_size=512, adv_ratio=0.5
    )

    trainer = pl.Trainer(max_epochs=5, accelerator='cpu', devices=1, enable_progress_bar=True)
    trainer.fit(adv_module, train_loader)

    # Save updated model
    torch.save(detector.state_dict(), MODEL_CHECKPOINT)
    print(f"Model saved to {MODEL_CHECKPOINT}")

if __name__ == "__main__":
    main()