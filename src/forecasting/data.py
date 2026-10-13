"""
Stage 3 data layer: windows, tabular features, and train-only scaling.

Every learned model predicts the *change* in the log-scale target relative to
its value at the decision CCE (Δ = y_{n+k} − y_n). Persistence is therefore the
zero-change model, and a learned model has to beat it by forecasting changes.
"""

import numpy as np
import pandas as pd

from src.data.windows import (ENV_FEATURES, HORIZONS, STATE_FEATURES, TARGETS, add_features,
                              check_windows, make_windows)

WINDOW = 20
CCE_PATH = "data/processed/drift_v3/cce.parquet"
ROLES = ["dev", "heldout_qubit", "heldout_device"]
EVAL_SETS = {  # name -> (role, split code)
    "val": ("dev", 1),
    "test_primary": ("dev", 2),
    "test_unseen_qubit": ("heldout_qubit", 2),
    "test_unseen_device": ("heldout_device", 2),
}
FEATURE_SETS = {
    "targets_only": TARGETS,
    "state": STATE_FEATURES,
    "state_env": STATE_FEATURES + ENV_FEATURES,
}
N_LAGS = 4


def load_windows(feature_set="state", window=WINDOW):
    c = add_features(pd.read_parquet(CCE_PATH))
    w = make_windows(c, window=window, features=FEATURE_SETS[feature_set], roles=ROLES)
    check_windows(w, window)
    w["feature_set"] = feature_set
    w["target_idx"] = [w["features"].index(t) for t in TARGETS] if all(
        t in w["features"] for t in TARGETS) else None
    return w


def pair_mask(w, role, split_code):
    """(N, H) boolean: sample belongs to role, and its horizon-k target is in split."""
    r = (w["meta"]["role"].to_numpy() == role)[:, None]
    return r & (w["split"] == split_code)


def tabular(w):
    """Flattened features for the classical models: last-step features, plus for each
    target variable its lags 1..N_LAGS and its window mean and std."""
    X = w["X"]
    last = X[:, -1, :]
    parts = [last]
    for t in TARGETS:
        j = w["features"].index(t)
        series = X[:, :, j]
        parts.append(series[:, -1 - N_LAGS:-1][:, ::-1])          # lags 1..N_LAGS
        parts.append(series.mean(axis=1, keepdims=True))
        parts.append(series.std(axis=1, keepdims=True))
    return np.concatenate(parts, axis=1).astype(np.float32)


class Standardizer:
    """Per-column mean/std fitted on training rows only."""

    def fit(self, A):
        self.mean = np.nanmean(A, axis=0)
        self.std = np.nanstd(A, axis=0)
        self.std[~np.isfinite(self.std) | (self.std == 0)] = 1.0
        self.mean[~np.isfinite(self.mean)] = 0.0
        return self

    def transform(self, A):
        return ((A - self.mean) / self.std).astype(np.float32)


def delta_targets(w):
    """Δ targets (N, H, T): future log value minus value at the decision CCE."""
    return w["Y"] - w["last"][:, None, :]


def train_pairs(w):
    """Rows with at least one horizon in the training split (development qubits)."""
    m = pair_mask(w, "dev", 0)
    return m.any(axis=1), m
