# -*- coding: utf-8 -*-
"""
Train a simple deep autoencoder on normal (non-fraud) transactions only.
Reconstruction error can later be used as an anomaly score for fraud.
"""
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from prepare_data import load_data, get_feature_matrix


BASE = Path(__file__).resolve().parent.parent


class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out


def train_autoencoder(epochs: int = 10, batch_size: int = 256, lr: float = 1e-3):
    df = load_data()
    X_train, X_test, y_train, y_test = get_feature_matrix(df)

    # Use only normal transactions (Class=0) for training
    normal_idx = np.where(y_train == 0)[0]
    X_train_norm = X_train.iloc[normal_idx].values.astype(np.float32)
    X_test_all = X_test.values.astype(np.float32)
    y_test_all = y_test.values

    input_dim = X_train_norm.shape[1]
    model = Autoencoder(input_dim=input_dim, latent_dim=16)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train_norm)), batch_size=batch_size, shuffle=True
    )

    optim = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for (batch,) in train_loader:
            batch = batch.to(device)
            optim.zero_grad()
            recon = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optim.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{epochs} - loss: {epoch_loss:.6f}")

    # Compute reconstruction errors on test set and store them
    model.eval()
    with torch.no_grad():
        x_t = torch.from_numpy(X_test_all).to(device)
        recon = model(x_t).cpu().numpy()
    errors = np.mean((X_test_all - recon) ** 2, axis=1)

    # Save model and errors + labels so evaluation can pick thresholds
    models_dir = BASE / "models"
    models_dir.mkdir(exist_ok=True)
    torch.save(model.state_dict(), models_dir / "autoencoder.pt")
    print("Saved models/autoencoder.pt")

    reports_dir = BASE / "reports"
    reports_dir.mkdir(exist_ok=True)
    np.save(reports_dir / "autoencoder_errors.npy", errors)
    np.save(reports_dir / "autoencoder_labels.npy", y_test_all)
    print("Saved reconstruction errors and labels to reports/")

    return model, errors, y_test_all


if __name__ == "__main__":
    train_autoencoder()

