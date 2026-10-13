"""
Stage 3: multi-horizon forecasting of real IBM calibration drift.

    python3 -m experiments.stage3_forecasting tune      # grid search on validation
    python3 -m experiments.stage3_forecasting final     # final models (5 seeds) + ablations
    python3 -m experiments.stage3_forecasting evaluate --sets val
    python3 -m experiments.stage3_forecasting freeze    # record hashes before any test look
    python3 -m experiments.stage3_forecasting evaluate --sets test   # needs configs/frozen_stage3.yaml

Outputs go to results/stage3/ (predictions/, metrics_*.csv, h1_*.json, logs).
"""

import argparse
import hashlib
import itertools
import json
import os
import platform
import sys
import time

import numpy as np
import pandas as pd
import sklearn
import torch
import yaml

from src.data.windows import TARGETS
from src.evaluation.stats import bootstrap_weights, holm, per_series_mae, series_table
from src.forecasting.data import (EVAL_SETS, Standardizer, delta_targets, load_windows,
                                  pair_mask, tabular, train_pairs)
from src.forecasting.models import fit_sklearn, predict_net, stat_baseline, train_net

CFG_PATH = "configs/stage3.yaml"
OUT = "results/stage3"
PRED = f"{OUT}/predictions"
FROZEN = "configs/frozen_stage3.yaml"
BASELINES = ["persistence", "rolling5", "window_mean", "ridge", "hgb", "lstm"]

torch.set_num_threads(8)


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(f"{OUT}/stage3.log", "a") as f:
        f.write(line + "\n")


def cfg():
    with open(CFG_PATH) as f:
        return yaml.safe_load(f)


# ------------------------------------------------------------------ data prep

def prepare(feature_set, window):
    w = load_windows(feature_set, window)
    rows, mtr = train_pairs(w)
    mval = pair_mask(w, "dev", 1)
    N, L, F = w["X"].shape
    sx = Standardizer().fit(w["X"][rows].reshape(-1, F))
    X = sx.transform(w["X"].reshape(-1, F)).reshape(N, L, F)
    D = delta_targets(w)
    sy = Standardizer().fit(np.where(mtr[:, :, None], D, np.nan).reshape(N, -1))
    Yd = sy.transform(D.reshape(N, -1)).reshape(D.shape)
    return dict(w=w, X=X, D=D, Yd=Yd, sx=sx, sy=sy, mtr=mtr, mval=mval, rows=rows)


def unscale(p, pred_scaled):
    N, H, T = pred_scaled.shape
    return (pred_scaled.reshape(N, -1) * p["sy"].std + p["sy"].mean).reshape(N, H, T)


def series_ids(w):
    m = w["meta"]
    return (m["backend"] + ":" + m["qubit"].astype(str)).to_numpy(), m["backend"].to_numpy()


def val_ratio(p, pred):
    """Selection score: mean over (k, target) of per-series val MAE / persistence MAE."""
    sid, _ = series_ids(p["w"])
    ratios = []
    for h, t in itertools.product(range(p["D"].shape[1]), range(p["D"].shape[2])):
        m = p["mval"][:, h] & np.isfinite(p["D"][:, h, t])
        err = np.abs(pred[m, h, t] - p["D"][m, h, t])
        ref = np.abs(p["D"][m, h, t])
        s = pd.DataFrame({"s": sid[m], "e": err, "r": ref}).groupby("s").mean()
        ratios.append(s["e"].mean() / s["r"].mean())
    return float(np.mean(ratios))


def save_pred(name, pred, extra=None):
    os.makedirs(PRED, exist_ok=True)
    np.savez_compressed(f"{PRED}/{name}.npz", pred=pred.astype(np.float32), **(extra or {}))


# ------------------------------------------------------------------ tune

def tune():
    c = cfg()
    os.makedirs(OUT, exist_ok=True)
    log("=== tune (validation only) ===")
    p = prepare(c["feature_set"], c["window"])
    w, D = p["w"], p["D"]
    N, H, T = D.shape
    np.savez_compressed(f"{OUT}/windows_arrays.npz", Y=w["Y"], last=w["last"], split=w["split"])
    w["meta"].to_parquet(f"{OUT}/windows_meta.parquet")
    selected = {}

    for kind in ["persistence", "rolling5", "window_mean"]:
        pred = stat_baseline(w, kind)
        save_pred(kind, pred)
        log(f"{kind:12s} val ratio {val_ratio(p, pred):.4f}")

    # scikit-learn: separate model per (horizon, target), params chosen per cell on val
    tab = tabular(w)
    st = Standardizer().fit(tab[p["rows"]])
    tab = st.transform(tab)
    sid, _ = series_ids(w)
    for kind in ["ridge", "hgb"]:
        grid = [dict(zip(c["grids"][kind], v)) for v in itertools.product(*c["grids"][kind].values())]
        pred = np.zeros((N, H, T), np.float32)
        choice = {}
        for h, t in itertools.product(range(H), range(T)):
            tr = p["mtr"][:, h] & np.isfinite(D[:, h, t])
            va = p["mval"][:, h] & np.isfinite(D[:, h, t])
            best = (np.inf, None, None)
            for params in grid:
                m = fit_sklearn(kind, params, tab[tr], D[tr, h, t], seed=c["tune_seed"])
                yv = m.predict(tab[va])
                mae = pd.DataFrame({"s": sid[va], "e": np.abs(yv - D[va, h, t])}).groupby("s")["e"].mean().mean()
                if mae < best[0]:
                    best = (mae, params, m)
            pred[:, h, t] = best[2].predict(tab)
            choice[f"k{c['horizons'][h]}/{TARGETS[t]}"] = best[1]
        save_pred(kind, pred)
        selected[kind] = choice
        log(f"{kind:12s} val ratio {val_ratio(p, pred):.4f}")

    # neural: one config for all cells, chosen by val ratio with the tuning seed
    tr_cfg = c["training"]
    for kind in ["lstm", "transformer"]:
        g = dict(c["grids"][kind])
        mult = g.pop("dim_ff_mult", None)
        results = []
        for v in itertools.product(*g.values()):
            params = dict(zip(g, v), weight_decay=tr_cfg["weight_decay"], batch_size=tr_cfg["batch_size"])
            if kind == "transformer":
                params["dim_ff"] = params["d_model"] * mult[0]
            t0 = time.time()
            model, hist = train_net(kind, params, p["X"], p["Yd"], p["mtr"], p["mval"], c["tune_seed"],
                                    tr_cfg["max_epochs"], tr_cfg["patience"])
            pred = unscale(p, predict_net(model, p["X"], H, T))
            score = val_ratio(p, pred)
            best_ep = int(np.argmin([x["val"] for x in hist]) + 1)
            results.append(dict(params=params, val_ratio=score, best_epoch=best_ep, epochs=len(hist)))
            log(f"{kind:12s} {params} -> val ratio {score:.4f} (best epoch {best_ep}, {time.time() - t0:.0f}s)")
        best = min(results, key=lambda r: r["val_ratio"])
        selected[kind] = best["params"]
        with open(f"{OUT}/tune_{kind}.json", "w") as f:
            json.dump(results, f, indent=2)

    with open(f"{OUT}/selected.yaml", "w") as f:
        yaml.safe_dump(selected, f, sort_keys=False)
    log("selected configs written")


# ------------------------------------------------------------------ final

def final():
    c = cfg()
    with open(f"{OUT}/selected.yaml") as f:
        sel = yaml.safe_load(f)
    tr_cfg = c["training"]
    log("=== final neural models (all seeds) + ablations ===")

    runs = [("lstm", "lstm", c["feature_set"], {}), ("transformer", "transformer", c["feature_set"], {})]
    for name, spec in c["ablations"].items():
        runs.append((name, "transformer", spec.get("feature_set", c["feature_set"]),
                     {k: v for k, v in spec.items() if k != "feature_set"}))

    ref_meta = pd.read_parquet(f"{OUT}/windows_meta.parquet")
    cache = {}
    for name, kind, fs, override in runs:
        if fs not in cache:
            cache[fs] = prepare(fs, c["window"])
            assert cache[fs]["w"]["meta"][["backend", "qubit", "decision_time"]].equals(
                ref_meta[["backend", "qubit", "decision_time"]]), "sample set differs between feature sets"
        p = cache[fs]
        N, H, T = p["D"].shape
        params = dict(sel[kind], **override)
        preds, hists = [], []
        for seed in c["seeds"]:
            t0 = time.time()
            model, hist = train_net(kind, params, p["X"], p["Yd"], p["mtr"], p["mval"], seed,
                                    tr_cfg["max_epochs"], tr_cfg["patience"])
            preds.append(unscale(p, predict_net(model, p["X"], H, T)))
            hists.append(hist)
            os.makedirs("checkpoints/stage3", exist_ok=True)
            torch.save(dict(state_dict=model.state_dict(), kind=kind, params=params, feature_set=fs,
                            seed=seed, x_mean=p["sx"].mean, x_std=p["sx"].std,
                            y_mean=p["sy"].mean, y_std=p["sy"].std),
                       f"checkpoints/stage3/{name}_seed{seed}.pt")
            log(f"{name:18s} seed {seed}: val ratio {val_ratio(p, preds[-1]):.4f}, "
                f"{len(hist)} epochs, {time.time() - t0:.0f}s")
        ens = np.mean(preds, axis=0)
        save_pred(name, ens, dict(per_seed=np.asarray(preds, np.float32)))
        with open(f"{OUT}/history_{name}.json", "w") as f:
            json.dump(hists, f)
        log(f"{name:18s} ensemble val ratio {val_ratio(p, ens):.4f}")

    env = dict(python=sys.version.split()[0], platform=platform.platform(), numpy=np.__version__,
               pandas=pd.__version__, torch=torch.__version__, sklearn=sklearn.__version__,
               threads=torch.get_num_threads())
    with open(f"{OUT}/environment.json", "w") as f:
        json.dump(env, f, indent=2)


# ------------------------------------------------------------------ freeze

def sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def freeze():
    files = sorted(os.listdir(PRED))
    frozen = dict(
        frozen_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        config_sha256=sha(CFG_PATH), selected_sha256=sha(f"{OUT}/selected.yaml"),
        data_sha256=sha("data/processed/drift_v3/cce.parquet"),
        predictions={f: sha(f"{PRED}/{f}") for f in files},
        note="Created before any test-set metric was computed (PROTOCOL §6).",
    )
    with open(FROZEN, "w") as f:
        yaml.safe_dump(frozen, f, sort_keys=False)
    log(f"frozen {len(files)} prediction files -> {FROZEN}")


def check_frozen():
    if not os.path.exists(FROZEN):
        raise SystemExit(f"refusing to evaluate test sets: {FROZEN} not found (run `freeze` first)")
    with open(FROZEN) as f:
        fr = yaml.safe_load(f)
    for name, h in fr["predictions"].items():
        if sha(f"{PRED}/{name}") != h:
            raise SystemExit(f"prediction file changed after freezing: {name}")
    if sha("data/processed/drift_v3/cce.parquet") != fr["data_sha256"]:
        raise SystemExit("data changed after freezing")


# ------------------------------------------------------------------ evaluate

def evaluate(which):
    c = cfg()
    sets = ["val"] if which == "val" else [s for s in EVAL_SETS if s != "val"]
    if which == "test":
        check_frozen()
        log("=== TEST evaluation (frozen) ===")
    arr = np.load(f"{OUT}/windows_arrays.npz")
    meta = pd.read_parquet(f"{OUT}/windows_meta.parquet")
    Y, last, split = arr["Y"], arr["last"], arr["split"]
    D = Y - last[:, None, :]
    role = meta["role"].to_numpy()
    sid = (meta["backend"] + ":" + meta["qubit"].astype(str)).to_numpy()
    backend = meta["backend"].to_numpy()
    models = [f[:-4] for f in sorted(os.listdir(PRED))]
    preds = {m: np.load(f"{PRED}/{m}.npz")["pred"] for m in models}
    B, bseed = c["bootstrap"]["B"], c["bootstrap"]["seed"]
    thr = c["large_change_log_threshold"]

    rows, h1 = [], []
    for set_name in sets:
        r_name, code = EVAL_SETS[set_name]
        for hi, k in enumerate(c["horizons"]):
            for ti, tgt in enumerate(TARGETS):
                m = (role == r_name) & (split[:, hi] == code) & np.isfinite(D[:, hi, ti])
                if not m.any():
                    continue
                y_true, base = D[m, hi, ti], last[m, ti]
                _, _, bk = series_table(np.zeros(m.sum()), sid[m], backend[m])
                W = bootstrap_weights(bk, B, bseed)
                stats = {}
                for name in models:
                    e = np.abs(preds[name][m, hi, ti] - y_true)
                    s_sum, s_cnt, _ = series_table(e, sid[m], backend[m])
                    mae = per_series_mae(s_sum, s_cnt)
                    boot = per_series_mae(s_sum, s_cnt, W)
                    natural = np.abs(np.exp(base + preds[name][m, hi, ti]) - np.exp(base + y_true))
                    n_sum, n_cnt, _ = series_table(natural, sid[m], backend[m])
                    big = np.abs(y_true) > thr
                    stats[name] = dict(mae=mae, boot=boot)
                    rows.append(dict(
                        eval_set=set_name, k=k, target=tgt, model=name, n_pairs=int(m.sum()),
                        n_series=len(s_cnt), mae_log=mae,
                        mae_ci_lo=float(np.percentile(boot, 2.5)), mae_ci_hi=float(np.percentile(boot, 97.5)),
                        rmse_log=float(np.sqrt(np.mean(e ** 2))),
                        mae_natural=per_series_mae(n_sum, n_cnt),
                        large_change_frac=float(big.mean()),
                        large_change_mae=float(e[big].mean()) if big.any() else np.nan))
                for name in models:
                    base_stats = stats["persistence"]
                    R = 1 - stats[name]["boot"] / base_stats["boot"]
                    rows[-len(models) + models.index(name)].update(
                        R_vs_persistence=1 - stats[name]["mae"] / base_stats["mae"],
                        R_ci_lo=float(np.percentile(R, 2.5)), R_ci_hi=float(np.percentile(R, 97.5)))
                if "transformer" in stats:
                    entry = dict(eval_set=set_name, k=k, target=tgt, comparisons={})
                    ps = []
                    for b in [b for b in BASELINES if b in stats]:
                        R = 1 - stats["transformer"]["boot"] / stats[b]["boot"]
                        R0 = 1 - stats["transformer"]["mae"] / stats[b]["mae"]
                        pval = float(max((R <= 0).mean(), 1.0 / B))
                        entry["comparisons"][b] = dict(R=R0, ci=[float(np.percentile(R, 2.5)),
                                                                 float(np.percentile(R, 97.5))], p=pval)
                        ps.append(pval)
                    entry["p_iut"] = max(ps)
                    entry["practical_pass"] = all(v["R"] >= 0.05 for v in entry["comparisons"].values())
                    h1.append(entry)
        log(f"evaluated {set_name}")

    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/metrics_{which}.csv", index=False)
    for set_name in sets:
        cells = [e for e in h1 if e["eval_set"] == set_name and e["k"] >= 2]
        rej = holm([e["p_iut"] for e in cells])
        for e, r in zip(cells, rej):
            e["holm_reject"] = bool(r)
            e["H1_success"] = bool(r and e["practical_pass"])
    with open(f"{OUT}/h1_{which}.json", "w") as f:
        json.dump(h1, f, indent=2)
    log(f"wrote metrics_{which}.csv and h1_{which}.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["tune", "final", "freeze", "evaluate"])
    ap.add_argument("--sets", choices=["val", "test"], default="val")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    {"tune": tune, "final": final, "freeze": freeze}.get(a.cmd, lambda: evaluate(a.sets))()
