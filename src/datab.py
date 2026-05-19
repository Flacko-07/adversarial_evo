import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

def generate_synthetic_data(n_samples=5000, n_features=10, test_size=0.2, seed=42):
    np.random.seed(seed)
    # Normal (class 0)
    X_norm = np.random.randn(n_samples // 2, n_features)
    # Attack (class 1) – shifted
    X_att = np.random.randn(n_samples // 2, n_features) * 1.2 + 1.0
    X = np.vstack([X_norm, X_att])
    y = np.hstack([np.zeros(n_samples // 2), np.ones(n_samples // 2)]).astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
    return (X_train, y_train), (X_test, y_test)

def get_dataloaders(X_train, y_train, X_test, y_test, batch_size=64, num_workers=0):
    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                             torch.tensor(y_train, dtype=torch.float32).view(-1,1))
    test_ds  = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                             torch.tensor(y_test, dtype=torch.float32).view(-1,1))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader

def real_attack_tensor(X_train, y_train):
    """Return tensor of only attack samples for GAN training."""
    X_att = X_train[y_train == 1]
    return torch.tensor(X_att, dtype=torch.float32)
