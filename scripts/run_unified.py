import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import pytorch_lightning as pl
from src.data_unified import load_unified_dataset
from src.detector.model import DetectorMLP
from src.gan.models import Generator, Critic
from src.gan.adversarial_training_lit import AdversarialTrainingLit
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

def main():
    X, y, input_dim, feature_info = load_unified_dataset(
        binary_csv='./data/binary_30k.csv',
        ids_csv_dir='./data/cic_ids2017',
        os_csv='./data/labeled_os_fingerprints.csv'
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                             torch.tensor(y_train, dtype=torch.float32).view(-1,1))
    test_ds  = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                             torch.tensor(y_test, dtype=torch.float32).view(-1,1))
    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True, num_workers=4)
    test_loader  = DataLoader(test_ds, batch_size=512, num_workers=4)

    real_attack = torch.tensor(X_train[y_train==1], dtype=torch.float32)

    detector = DetectorMLP(input_dim=input_dim, hidden_dims=[512,256,128], dropout=0.4)
    generator = Generator(noise_dim=16, cond_dim=1, output_dim=input_dim, hidden_dims=[256,128])
    critic = Critic(input_dim=input_dim, cond_dim=1, hidden_dims=[256,128])

    adv_module = AdversarialTrainingLit(
        detector=detector,
        generator=generator,
        critic=critic,
        real_attack_data=real_attack,
        lr_det=0.0005,
        lr_gen=0.0001,
        lr_crit=0.0001,
        n_critic=3,
        lambda_gp=10,
        anti_weight=0.05,
        noise_dim=16,
        cond_dim=1,
        batch_size=512,
        adv_ratio=0.5
    )

    trainer = pl.Trainer(max_epochs=30, accelerator='cpu', devices=1, enable_progress_bar=True)
    trainer.fit(adv_module, train_loader)
    # Save trained detector
    torch.save(adv_module.detector.state_dict(), "unified_detector.ckpt")
    print("✅ Saved unified detector to unified_detector.ckpt")

    adv_module.eval()
    with torch.no_grad():
        correct, total = 0, 0
        for xb, yb in test_loader:
            logits = adv_module.detector(xb)
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == yb).sum().item()
            total += yb.size(0)
        clean_acc = correct / total

        noise = torch.randn(2000, 16)
        cond = torch.ones(2000, 1)
        fake = adv_module.generator(noise, cond)
        fake_logits = adv_module.detector(fake)
        fake_preds = (torch.sigmoid(fake_logits) > 0.5).float()
        adv_acc = fake_preds.mean().item()

    print(f"\nUnified detector – Clean accuracy: {clean_acc:.4f}")
    print(f"Unified detector – Adversarial accuracy: {adv_acc:.4f} (evasion: {1-adv_acc:.4f})")

if __name__ == "__main__":
    main()