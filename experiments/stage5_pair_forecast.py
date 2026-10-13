"""
Stage 5a: pair-level CZ-error forecasting (decision-relevant quantity).

    python3 -m experiments.stage5_pair_forecast run       # tune on val, final models, save predictions
    python3 -m experiments.stage5_pair_forecast evaluate --sets val|test   (test needs the Stage 5 freeze)

Models: persistence, window mean (20 epochs), ridge, gradient-boosted trees,
Transformer (5-seed ensemble). Targets: Δ log CZ error at k = 1, 4, 7 epochs.
"""

import argparse
import itertools
import json
import os
import time

import numpy as np
import pandas as pd
import torch

from src.evaluation.stats import bootstrap_weights, per_series_mae, series_table
from src.forecasting.data import Standardizer
from src.forecasting.models import fit_sklearn, predict_net, train_net
from src.forecasting.pairs import HORIZONS, make_pair_windows

OUT = "results/stage5/pair_forecast"
SEEDS = [0, 1, 2, 3, 4]
RIDGE = [dict(alpha=a) for a in [0.1, 1, 10, 100, 1000, 10000]]
HGB = [dict(learning_rate=lr, max_iter=it, max_leaf_nodes=nl, min_samples_leaf=40, l2_regularization=1.0)
       for lr in [0.03, 0.1] for it in [200, 500] for nl in [15, 31]]
TF = [dict(d_model=d, num_layers=nl, dropout=0.1, lr=lr, nhead=4, dim_ff=2 * d, weight_decay=1e-4, batch_size=256)
      for d in [32, 64] for nl in [1, 2] for lr in [1e-3, 3e-4]]
torch.set_num_threads(8)


def log(msg):
    os.makedirs(OUT, exist_ok=True)
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(f"{OUT}/log.txt", "a") as f:
        f.write(line + "\n")


def load():
    t = pd.read_parquet("data/processed/drift_v3/pair_epochs.parquet")
    w = make_pair_windows(t)
    m = w["meta"]
    w["sid"] = (m.backend + ":" + m.a.astype(str) + "-" + m.b.astype(str)).to_numpy()
    return w


def masks(w, code):
    return (w["meta"].role.values == "dev")[:, None] & (w["split"] == code)


def val_mae(w, pred):
    """Mean over horizons of per-series val MAE (pairs weighted equally)."""
    D = w["Y"] - w["last"][:, None, :]
    mv = masks(w, 1)
    out = []
    for h in range(len(HORIZONS)):
        m = mv[:, h] & np.isfinite(D[:, h, 0])
        s = pd.DataFrame({"s": w["sid"][m], "e": np.abs(pred[m, h, 0] - D[m, h, 0])}).groupby("s").e.mean()
        out.append(s.mean())
    return float(np.mean(out))


def tabular(w):
    X = w["X"]
    j = w["features"].index("log_cz")
    s = X[:, :, j]
    return np.concatenate([X[:, -1, :], s[:, -5:-1][:, ::-1], s.mean(1, keepdims=True), s.std(1, keepdims=True)], 1)


def run():
    w = load()
    np.savez_compressed(f"{OUT}/windows_arrays.npz", Y=w["Y"], last=w["last"], split=w["split"])
    w["meta"].to_parquet(f"{OUT}/windows_meta.parquet")
    N, H, _ = w["Y"].shape
    D = w["Y"] - w["last"][:, None, :]
    mtr, mval = masks(w, 0), masks(w, 1)
    rows = mtr.any(1)
    preds, sel = {}, {}

    preds["persistence"] = np.zeros((N, H, 1), np.float32)
    j = w["features"].index("log_cz")
    preds["window_mean"] = np.repeat((w["X"][:, :, j].mean(1) - w["last"][:, 0])[:, None, None], H, 1)
    for k in ["persistence", "window_mean"]:
        log(f"{k:12s} val MAE {val_mae(w, preds[k]):.4f}")

    tab = tabular(w)
    st = Standardizer().fit(tab[rows])
    tab = st.transform(tab)
    for kind, grid in [("ridge", RIDGE), ("hgb", HGB)]:
        p = np.zeros((N, H, 1), np.float32)
        sel[kind] = {}
        for h in range(H):
            tr = mtr[:, h] & np.isfinite(D[:, h, 0])
            va = mval[:, h] & np.isfinite(D[:, h, 0])
            best = None
            for params in grid:
                m = fit_sklearn(kind, params, tab[tr], D[tr, h, 0], seed=0)
                e = pd.DataFrame({"s": w["sid"][va], "e": np.abs(m.predict(tab[va]) - D[va, h, 0])}).groupby("s").e.mean().mean()
                if best is None or e < best[0]:
                    best = (e, params, m)
            p[:, h, 0] = best[2].predict(tab)
            sel[kind][f"k{HORIZONS[h]}"] = best[1]
        preds[kind] = p
        log(f"{kind:12s} val MAE {val_mae(w, p):.4f}")

    L, F = w["X"].shape[1:]
    sx = Standardizer().fit(w["X"][rows].reshape(-1, F))
    X = sx.transform(w["X"].reshape(-1, F)).reshape(N, L, F)
    sy = Standardizer().fit(np.where(mtr[:, :, None], D, np.nan).reshape(N, -1))
    Yd = sy.transform(D.reshape(N, -1)).reshape(D.shape)
    unscale = lambda p: (p.reshape(N, -1) * sy.std + sy.mean).reshape(N, H, 1)
    results = []
    for params in TF:
        model, hist = train_net("transformer", params, X, Yd, mtr, mval, 0, 80, 10)
        e = val_mae(w, unscale(predict_net(model, X, H, 1)))
        results.append((e, params))
        log(f"transformer  {params} val MAE {e:.4f} ({len(hist)} epochs)")
    best = min(results, key=lambda r: r[0])[1]
    sel["transformer"] = best
    per_seed = []
    os.makedirs("checkpoints/stage5_pair", exist_ok=True)
    for s in SEEDS:
        model, _ = train_net("transformer", best, X, Yd, mtr, mval, s, 80, 10)
        per_seed.append(unscale(predict_net(model, X, H, 1)))
        torch.save(dict(state_dict=model.state_dict(), params=best, seed=s, x_mean=sx.mean, x_std=sx.std,
                        y_mean=sy.mean, y_std=sy.std), f"checkpoints/stage5_pair/transformer_seed{s}.pt")
    preds["transformer"] = np.mean(per_seed, 0)
    log(f"transformer  ensemble val MAE {val_mae(w, preds['transformer']):.4f}")

    os.makedirs(f"{OUT}/predictions", exist_ok=True)
    for k, p in preds.items():
        np.savez_compressed(f"{OUT}/predictions/{k}.npz", pred=p.astype(np.float32))
    with open(f"{OUT}/selected.json", "w") as f:
        json.dump(sel, f, indent=2)


def evaluate(which):
    if which == "test":
        from experiments.stage5_freeze import check_frozen
        check_frozen()
    w = load()
    D = w["Y"] - w["last"][:, None, :]
    sets = {"val": [("val", "dev", 1)],
            "test": [("test_primary", "dev", 2), ("test_unseen_qubit", "heldout_qubit", 2),
                     ("test_unseen_device", "heldout_device", 2)]}[which]
    models = ["persistence", "window_mean", "ridge", "hgb", "transformer"]
    P = {m: np.load(f"{OUT}/predictions/{m}.npz")["pred"] for m in models}
    rows = []
    for name, role, code in sets:
        for h, k in enumerate(HORIZONS):
            m = (w["meta"].role.values == role) & (w["split"][:, h] == code) & np.isfinite(D[:, h, 0])
            _, _, bk = series_table(np.zeros(m.sum()), w["sid"][m], w["meta"].backend.values[m])
            W = bootstrap_weights(bk, 10_000, 0)
            st = {}
            for mod in models:
                e = np.abs(P[mod][m, h, 0] - D[m, h, 0])
                s_sum, s_cnt, _ = series_table(e, w["sid"][m], w["meta"].backend.values[m])
                st[mod] = (per_series_mae(s_sum, s_cnt), per_series_mae(s_sum, s_cnt, W))
            for mod in models:
                R = 1 - st[mod][1] / st["persistence"][1]
                rows.append(dict(eval_set=name, k=k, model=mod, n_pairs=int(m.sum()), mae_log=st[mod][0],
                                 R_vs_persistence=1 - st[mod][0] / st["persistence"][0],
                                 R_ci_lo=float(np.percentile(R, 2.5)), R_ci_hi=float(np.percentile(R, 97.5))))
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/metrics_{which}.csv", index=False)
    print(df.pivot_table(index=["eval_set", "k"], columns="model", values="mae_log").round(4).to_string())
    print(df.pivot_table(index=["eval_set", "k"], columns="model", values="R_vs_persistence").round(3).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "evaluate"])
    ap.add_argument("--sets", choices=["val", "test"], default="val")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    run() if a.cmd == "run" else evaluate(a.sets)
