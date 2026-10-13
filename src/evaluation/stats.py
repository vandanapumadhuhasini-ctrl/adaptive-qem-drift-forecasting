"""
Statistics for the pre-registered protocol (docs/PROTOCOL.md §3–§5).

Per-series MAE: each qubit contributes the mean |error| of its samples, and the
aggregate MAE is the unweighted mean over series. CIs come from a cluster
bootstrap that resamples series with replacement within each backend.
"""

import numpy as np
import pandas as pd


def series_table(abs_err, series_id, backend):
    """Sum and count of |error| per series (one row per series)."""
    df = pd.DataFrame({"s": series_id, "b": backend, "e": abs_err})
    g = df.groupby("s", sort=True)
    return g["e"].sum().to_numpy(), g["e"].size().to_numpy(), g["b"].first().to_numpy()


def bootstrap_weights(backends, B=10_000, seed=0):
    """(B, n_series) multinomial counts, resampling series within each backend."""
    rng = np.random.default_rng(seed)
    n = len(backends)
    W = np.zeros((B, n), np.float32)
    for b in np.unique(backends):
        idx = np.flatnonzero(backends == b)
        draws = rng.integers(0, len(idx), size=(B, len(idx)))
        np.add.at(W, (np.repeat(np.arange(B), len(idx)), idx[draws.ravel()]), 1.0)
    return W


def per_series_mae(sums, counts, W=None):
    """Unweighted mean over series of per-series MAE; with W, one value per bootstrap draw."""
    mae_s = sums / counts
    if W is None:
        return float(mae_s.mean())
    return (W @ mae_s) / W.sum(axis=1)


def relative_improvement(mae_m, mae_b):
    return 1.0 - mae_m / mae_b


def holm(pvals, alpha=0.05):
    """Holm–Bonferroni: returns a boolean 'reject' array in input order."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    reject = np.zeros(len(p), bool)
    for rank, i in enumerate(order):
        if p[i] <= alpha / (len(p) - rank):
            reject[i] = True
        else:
            break
    return reject
