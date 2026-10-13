"""
Forecasting models for Stage 3. All predict Δ (change of the log-scale target
from the decision CCE) for every (horizon, target) pair.

Statistical baselines (no fitting): persistence (Δ = 0), rolling mean of the
last 5 CCEs, and window mean (all 20 CCEs).
Classical ML (scikit-learn): ridge regression and histogram gradient-boosted
trees on tabular lag features, one model per (horizon, target).
Neural (PyTorch): LSTM and Transformer encoder, one multi-output model with a
masked loss over the available (horizon, target) training pairs.
"""

import math
import random
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from src.data.windows import TARGETS


# ---------------------------------------------------------------- statistical

def stat_baseline(w, kind):
    """Δ predictions (N, H, T) for persistence / rolling-mean baselines."""
    N, H, T = w["Y"].shape
    if kind == "persistence":
        return np.zeros((N, H, T), np.float32)
    n = {"rolling5": 5, "window_mean": w["X"].shape[1]}[kind]
    idx = [w["features"].index(t) for t in TARGETS]
    mean = w["X"][:, -n:, idx].mean(axis=1)                       # (N, T)
    return np.repeat((mean - w["last"])[:, None, :], H, axis=1).astype(np.float32)


# ---------------------------------------------------------------- scikit-learn

def fit_sklearn(kind, params, Xtr, ytr, seed=0):
    if kind == "ridge":
        m = Ridge(alpha=params["alpha"])
    else:
        m = HistGradientBoostingRegressor(
            learning_rate=params["learning_rate"], max_iter=params["max_iter"],
            max_leaf_nodes=params["max_leaf_nodes"], min_samples_leaf=params.get("min_samples_leaf", 40),
            l2_regularization=params.get("l2_regularization", 1.0),
            early_stopping=False, random_state=seed)
    return m.fit(Xtr, ytr)


# ---------------------------------------------------------------- neural

class TransformerForecaster(nn.Module):
    """Input projection + learned positional embedding + pre-norm encoder;
    readout = [last token ‖ mean over tokens] -> (H*T) Δ predictions."""

    def __init__(self, n_features, n_out, window, d_model=64, nhead=4, num_layers=2,
                 dim_ff=128, dropout=0.1, positional=True):
        super().__init__()
        self.inp = nn.Linear(n_features, d_model)
        self.pos = nn.Parameter(torch.zeros(1, window, d_model)) if positional else None
        if positional:
            nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_ff, dropout, batch_first=True,
                                           activation="gelu", norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Linear(2 * d_model, d_model), nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(d_model, n_out))

    def forward(self, x):
        h = self.inp(x)
        if self.pos is not None:
            h = h + self.pos
        h = self.norm(self.encoder(h))
        return self.head(torch.cat([h[:, -1], h.mean(dim=1)], dim=1))


class LSTMForecaster(nn.Module):
    def __init__(self, n_features, n_out, window, hidden=64, num_layers=1, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, n_out))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1])


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_net(kind, n_features, n_out, window, params):
    p = {k: v for k, v in params.items() if k not in ("lr", "weight_decay", "batch_size")}
    if kind == "transformer":
        return TransformerForecaster(n_features, n_out, window, **p)
    return LSTMForecaster(n_features, n_out, window, **p)


def train_net(kind, params, X, Yd, mask_tr, mask_val, seed, max_epochs=80, patience=10,
              log=None):
    """
    X: (N, L, F) scaled inputs. Yd: (N, H, T) scaled Δ targets (NaN = missing).
    mask_tr / mask_val: (N, H) bool pair masks. Early stopping on masked val MSE.
    Returns (model, history).
    """
    set_seed(seed)
    N, H, T = Yd.shape
    model = build_net(kind, X.shape[2], H * T, X.shape[1], params)
    opt = torch.optim.AdamW(model.parameters(), lr=params["lr"],
                            weight_decay=params.get("weight_decay", 1e-4))
    bs = params.get("batch_size", 256)

    def masks(m):
        full = m[:, :, None] & np.isfinite(Yd)
        return full

    Mtr, Mval = masks(mask_tr), masks(mask_val)
    rows_tr = np.flatnonzero(Mtr.any(axis=(1, 2)))
    rows_val = np.flatnonzero(Mval.any(axis=(1, 2)))
    Xt = torch.from_numpy(X)
    Yt = torch.from_numpy(np.nan_to_num(Yd).reshape(N, H * T))
    Mt_tr = torch.from_numpy(Mtr.reshape(N, H * T).astype(np.float32))
    Mt_val = torch.from_numpy(Mval.reshape(N, H * T).astype(np.float32))

    def masked_loss(idx, M):
        pred = model(Xt[idx])
        m = M[idx]
        return ((pred - Yt[idx]) ** 2 * m).sum() / m.sum().clamp(min=1)

    best, best_state, bad, history = math.inf, None, 0, []
    g = torch.Generator().manual_seed(seed)
    for epoch in range(max_epochs):
        t0 = time.time()
        model.train()
        perm = rows_tr[torch.randperm(len(rows_tr), generator=g).numpy()]
        tot, n = 0.0, 0
        for i in range(0, len(perm), bs):
            idx = torch.from_numpy(perm[i:i + bs])
            opt.zero_grad()
            loss = masked_loss(idx, Mt_tr)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(idx)
            n += len(idx)
        model.eval()
        with torch.no_grad():
            vl, vn = 0.0, 0.0
            for i in range(0, len(rows_val), 2048):
                idx = torch.from_numpy(rows_val[i:i + 2048])
                pred = model(Xt[idx])
                m = Mt_val[idx]
                vl += float(((pred - Yt[idx]) ** 2 * m).sum())
                vn += float(m.sum())
            val = vl / vn
        history.append(dict(epoch=epoch + 1, train=tot / n, val=val, sec=time.time() - t0))
        if log:
            log(f"    epoch {epoch + 1:3d} train {tot / n:.4f} val {val:.4f} ({time.time() - t0:.1f}s)")
        if val < best - 1e-5:
            best, bad = val, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, history


def predict_net(model, X, H, T):
    with torch.no_grad():
        out = [model(torch.from_numpy(X[i:i + 4096])).numpy() for i in range(0, len(X), 4096)]
    return np.concatenate(out).reshape(len(X), H, T)
