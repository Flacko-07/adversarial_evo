import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from src.data_ids import load_ids_data
import os as _os
from src.data_os import load_binary_30k

def generate_synthetic_data(n_samples=5000, n_features=10, test_size=0.2, seed=42):
    np.random.seed(seed)
    X_norm = np.random.randn(n_samples // 2, n_features)
    X_att = np.random.randn(n_samples // 2, n_features) * 1.2 + 1.0
    X = np.vstack([X_norm, X_att])
    y = np.hstack([np.zeros(n_samples // 2), np.ones(n_samples // 2)]).astype(np.float32)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
    return (X_train, y_train), (X_test, y_test), n_features

def get_dataloaders(X_train, y_train, X_test, y_test, batch_size=64, num_workers=0, pin_memory=False):
    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                             torch.tensor(y_train, dtype=torch.float32).view(-1, 1))
    test_ds  = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                             torch.tensor(y_test, dtype=torch.float32).view(-1, 1))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, test_loader

def real_attack_tensor(X_train, y_train):
    X_att = X_train[y_train == 1]
    return torch.tensor(X_att, dtype=torch.float32)

def load_data(cfg, seed=42):
    """Unified loader: tries OS data, then IDS, then synthetic."""
    # 1. Try OS vulnerability dataset (Binary-30K)
    os_csv = cfg.data.get("os_csv", None)
    if os_csv and _os.path.exists(os_csv):
        print(f"[*] Loading OS vulnerability data from {os_csv}")
        X, y, input_dim = load_binary_30k(os_csv, seed=seed)
        # Split and create loaders (same as before)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed
        )
        train_loader, test_loader = get_dataloaders(
            X_train, y_train, X_test, y_test,
            batch_size=cfg.data.batch_size,
            num_workers=cfg.data.num_workers,
            pin_memory=cfg.data.get("pin_memory", False)
        )
        real_attack = real_attack_tensor(X_train, y_train)
        return train_loader, test_loader, real_attack, input_dim

    # 2. Otherwise fall back to IDS or synthetic (your existing code)
    try:
        X, y = load_ids_data(
            source=cfg.data.get("source", "auto"),
            csv_dir=cfg.data.get("csv_dir"),
            eve_path=cfg.data.get("eve_path"),
        )
        print(f"[✓] Using real IDS data: {len(X)} total samples")
    except FileNotFoundError:
        n_samples = cfg.data.get("n_samples", 5000)
        n_features = cfg.data.get("n_features", 10)
        print(f"[!] No real IDS data found – using synthetic data ({n_samples} samples, {n_features} features)")
        (X_train, y_train), (X_test, y_test), input_dim = generate_synthetic_data(
            n_samples=n_samples, n_features=n_features, test_size=0.2, seed=seed
        )
        train_loader, test_loader = get_dataloaders(
            X_train, y_train, X_test, y_test,
            batch_size=cfg.data.batch_size,
            num_workers=cfg.data.num_workers,
            pin_memory=cfg.data.get("pin_memory", False),
        )
        real_attack = real_attack_tensor(X_train, y_train)
        return train_loader, test_loader, real_attack, input_dim

    # Real data split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    input_dim = X.shape[1]
    train_loader, test_loader = get_dataloaders(
        X_train, y_train, X_test, y_test,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.get("pin_memory", False),
    )
    real_attack = real_attack_tensor(X_train, y_train)
    return train_loader, test_loader, real_attack, input_dim