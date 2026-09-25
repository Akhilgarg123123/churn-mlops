"""
A small PyTorch MLP wrapped with a sklearn-like fit / predict_proba
interface, so it can be evaluated with the exact same metrics code as
the sklearn models, and combined with the shared ColumnTransformer in
one joblib-savable object.
"""

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin


class ChurnMLP(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x)  # raw logits -- loss fn applies sigmoid


def _to_dense_float32(X):
    """preprocess.py now forces dense output (sparse_output=False), but this
    stays defensive: if X ever arrives as a scipy.sparse matrix anyway,
    np.asarray(X, dtype=np.float32) silently produces a broken array
    instead of raising, and the real error only shows up later inside
    torch.tensor(). Better to fail fast / convert explicitly here."""
    if sp.issparse(X):
        X = X.toarray()
    return np.asarray(X, dtype=np.float32)


class TorchMLPClassifier(BaseEstimator, ClassifierMixin):
    """sklearn-compatible wrapper so this can sit in the same Pipeline
    pattern and be scored with the same sklearn metrics as the other
    models."""

    def __init__(self, epochs=30, lr=1e-3, batch_size=64, pos_weight=1.0):
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.pos_weight = pos_weight
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, X, y):
        X = _to_dense_float32(X)
        y = np.asarray(y, dtype=np.float32).reshape(-1, 1)

        self.model_ = ChurnMLP(X.shape[1]).to(self.device)
        opt = torch.optim.Adam(self.model_.parameters(), lr=self.lr)
        # pos_weight up-weights the minority (churn) class in the loss,
        # the PyTorch equivalent of class_weight="balanced"
        loss_fn = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([self.pos_weight], device=self.device)
        )

        X_t = torch.tensor(X, device=self.device)
        y_t = torch.tensor(y, device=self.device)

        self.model_.train()
        n = X_t.shape[0]
        for _ in range(self.epochs):
            perm = torch.randperm(n)
            for i in range(0, n, self.batch_size):
                idx = perm[i : i + self.batch_size]
                opt.zero_grad()
                logits = self.model_(X_t[idx])
                loss = loss_fn(logits, y_t[idx])
                loss.backward()
                opt.step()
        return self

    def predict_proba(self, X):
        X = _to_dense_float32(X)
        self.model_.eval()
        with torch.no_grad():
            logits = self.model_(torch.tensor(X, device=self.device))
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
        return np.column_stack([1 - probs, probs])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)