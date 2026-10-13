"""
Stage 3 figures, generated only from saved results:
results/stage3/metrics_{val,test}.csv, predictions/*.npz, windows_*.

    python3 -m experiments.plot_stage3 --which val
    python3 -m experiments.plot_stage3 --which test

Writes results/stage3/figures/<which>/*.png (300 dpi) and *.pdf, plus the
data table behind each figure as CSV next to it.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

from src.data.windows import TARGETS
from src.evaluation.plotting import (INK, INK_2, MODEL_COLORS, MODEL_LABELS, SET_LABELS,
                                     TARGET_LABELS, plt, save, setup)

OUT = "results/stage3"
MAIN_MODELS = ["persistence", "rolling5", "window_mean", "ridge", "hgb", "lstm", "transformer"]
ABLATIONS = ["transformer", "tf_targets_only", "tf_state_env", "tf_no_positional"]
MARKERS = {"persistence": "o", "rolling5": "v", "window_mean": "s", "ridge": "D",
           "hgb": "^", "lstm": "P", "transformer": "*"}


def fig_mae_vs_horizon(df, set_name, fdir):
    d = df[(df.eval_set == set_name) & df.model.isin(MAIN_MODELS)]
    fig, axes = plt.subplots(1, 5, figsize=(10.5, 2.6))
    for ax, tgt in zip(axes, TARGETS):
        dt = d[d.target == tgt]
        for m in MAIN_MODELS:
            x = dt[dt.model == m].sort_values("k")
            ax.errorbar(x.k, x.mae_log, yerr=[x.mae_log - x.mae_ci_lo, x.mae_ci_hi - x.mae_log],
                        color=MODEL_COLORS[m], marker=MARKERS[m], capsize=2, lw=1.3,
                        label=MODEL_LABELS[m], ms=5 if m == "transformer" else 3.5)
        ax.set_title(TARGET_LABELS[tgt])
        ax.set_xticks([1, 2, 4, 7])
        ax.set_xlabel("horizon k (calibration cycles)")
    axes[0].set_ylabel("MAE of log value\n(≈ relative error)")
    axes[-1].legend(frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.suptitle(f"Forecast error vs horizon — {SET_LABELS[set_name]} (95 % cluster-bootstrap CI)",
                 x=0.01, ha="left", color=INK, fontsize=9)
    fig.tight_layout()
    save(fig, f"{fdir}/mae_vs_horizon_{set_name}")
    d.to_csv(f"{fdir}/mae_vs_horizon_{set_name}.csv", index=False)


def fig_skill(df, set_name, fdir):
    models = [m for m in MAIN_MODELS if m != "persistence"]
    d = df[(df.eval_set == set_name) & df.model.isin(models)]
    fig, axes = plt.subplots(1, 4, figsize=(10.5, 2.8), sharey=True)
    for ax, k in zip(axes, [1, 2, 4, 7]):
        dk = d[d.k == k]
        for i, m in enumerate(models):
            x = dk[dk.model == m].set_index("target").loc[TARGETS]
            y = np.arange(len(TARGETS)) + (i - len(models) / 2) * 0.12
            ax.errorbar(100 * x.R_vs_persistence, y,
                        xerr=[100 * (x.R_vs_persistence - x.R_ci_lo), 100 * (x.R_ci_hi - x.R_vs_persistence)],
                        fmt=MARKERS[m], color=MODEL_COLORS[m], ms=4 if m == "transformer" else 3,
                        capsize=1.5, lw=1, label=MODEL_LABELS[m])
        ax.axvline(0, color=INK_2, lw=0.8)
        ax.set_yticks(range(len(TARGETS)))
        ax.set_yticklabels([TARGET_LABELS[t] for t in TARGETS])
        ax.set_title(f"k = {k}")
        ax.set_xlabel("MAE reduction vs persistence (%)")
    axes[-1].legend(frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.suptitle(f"Skill relative to persistence — {SET_LABELS[set_name]}", x=0.01, ha="left",
                 color=INK, fontsize=9)
    fig.tight_layout()
    save(fig, f"{fdir}/skill_vs_persistence_{set_name}")
    d.to_csv(f"{fdir}/skill_vs_persistence_{set_name}.csv", index=False)


def fig_eval_sets(df, fdir):
    sets = [s for s in ["test_primary", "test_unseen_qubit", "test_unseen_device"] if s in set(df.eval_set)]
    models = ["window_mean", "ridge", "hgb", "lstm", "transformer"]
    d = df[df.eval_set.isin(sets) & df.model.isin(models) & df.k.isin([1, 4])]
    fig, axes = plt.subplots(2, 5, figsize=(10.5, 4.4), sharey="row")
    for r, k in enumerate([1, 4]):
        for ax, tgt in zip(axes[r], TARGETS):
            x0 = np.arange(len(sets))
            for i, m in enumerate(models):
                x = d[(d.k == k) & (d.target == tgt) & (d.model == m)].set_index("eval_set").loc[sets]
                xs = x0 + (i - (len(models) - 1) / 2) * 0.15
                ax.bar(xs, 100 * x.R_vs_persistence, width=0.13, color=MODEL_COLORS[m],
                       label=MODEL_LABELS[m])
                ax.errorbar(xs, 100 * x.R_vs_persistence,
                            yerr=[100 * (x.R_vs_persistence - x.R_ci_lo), 100 * (x.R_ci_hi - x.R_vs_persistence)],
                            fmt="none", ecolor=INK_2, lw=0.7, capsize=1)
            ax.axhline(0, color=INK_2, lw=0.8)
            ax.set_xticks(x0)
            ax.set_xticklabels(["dev qubits", "unseen qubits", "unseen device"][:len(sets)], rotation=20)
            ax.set_title(f"{TARGET_LABELS[tgt]}, k = {k}")
            ax.grid(axis="x", visible=False)
        axes[r][0].set_ylabel("MAE reduction vs\npersistence (%)")
    axes[0][-1].legend(frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.suptitle("Generalisation to unseen qubits and an unseen device (test, 95 % CI)",
                 x=0.01, ha="left", color=INK, fontsize=9)
    fig.tight_layout()
    save(fig, f"{fdir}/generalisation_test")
    d.to_csv(f"{fdir}/generalisation_test.csv", index=False)


def fig_ablation(df, set_name, fdir):
    d = df[(df.eval_set == set_name) & df.model.isin(ABLATIONS)]
    fig, axes = plt.subplots(1, 5, figsize=(10.5, 2.5), sharey=True)
    for ax, tgt in zip(axes, TARGETS):
        dt = d[d.target == tgt]
        ref = dt[dt.model == "transformer"].set_index("k").mae_log
        for m in ABLATIONS:
            x = dt[dt.model == m].set_index("k").sort_index()
            ax.plot(x.index, 100 * (x.mae_log / ref.loc[x.index] - 1), marker="o",
                    color=MODEL_COLORS[m], label=MODEL_LABELS[m])
        ax.axhline(0, color=INK_2, lw=0.8)
        ax.set_xticks([1, 2, 4, 7])
        ax.set_title(TARGET_LABELS[tgt])
        ax.set_xlabel("horizon k")
    axes[0].set_ylabel("MAE change vs full\nTransformer (%)")
    axes[-1].legend(frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.suptitle(f"Transformer ablations — {SET_LABELS[set_name]} (positive = worse)",
                 x=0.01, ha="left", color=INK, fontsize=9)
    fig.tight_layout()
    save(fig, f"{fdir}/ablation_{set_name}")
    d.to_csv(f"{fdir}/ablation_{set_name}.csv", index=False)


def load_preds():
    arr = np.load(f"{OUT}/windows_arrays.npz")
    meta = pd.read_parquet(f"{OUT}/windows_meta.parquet")
    preds = {m: np.load(f"{OUT}/predictions/{m}.npz")["pred"] for m in MAIN_MODELS}
    return arr, meta, preds


def fig_examples(set_name, code, role, fdir, arr, meta, preds, k_idx=0, tgt="log_t1", n_qubits=3):
    ti = TARGETS.index(tgt)
    Y, last, split = arr["Y"], arr["last"], arr["split"]
    m = (meta.role.values == role) & (split[:, k_idx] == code) & np.isfinite(Y[:, k_idx, ti])
    sub = meta[m].assign(i=np.flatnonzero(m))
    # deterministic choice: qubits at the 25th/50th/75th percentile of series variability
    var = sub.groupby(["backend", "qubit"]).i.apply(lambda ix: np.std(Y[ix.values, k_idx, ti]))
    picks = var.sort_values().index[[int(q * (len(var) - 1)) for q in (0.25, 0.5, 0.75)][:n_qubits]]
    fig, axes = plt.subplots(n_qubits, 1, figsize=(7.2, 1.9 * n_qubits), sharex=True)
    rows = []
    for ax, (b, q) in zip(np.atleast_1d(axes), picks):
        ix = sub[(sub.backend == b) & (sub.qubit == q)].sort_values("decision_time").i.values
        t = meta.decision_time.values[ix]
        actual = np.exp(Y[ix, k_idx, ti])
        ax.plot(t, actual, color=INK, lw=1.4, marker="o", ms=2.5, label="actual (next calibration)")
        for mname in ["persistence", "window_mean", "transformer"]:
            fc = np.exp(last[ix, ti] + preds[mname][ix, k_idx, ti])
            ax.plot(t, fc, color=MODEL_COLORS[mname], lw=1.1, label=MODEL_LABELS[mname],
                    ls="--" if mname == "persistence" else "-")
            rows += [dict(backend=b, qubit=q, time=tt, model=mname, forecast=f, actual=a)
                     for tt, f, a in zip(t, fc, actual)]
        ax.set_title(f"{b} qubit {q}", loc="left")
        ax.set_ylabel(f"{TARGET_LABELS[tgt]} (µs)")
    np.atleast_1d(axes)[0].legend(frameon=False, ncol=4, loc="lower left", bbox_to_anchor=(0, 1.12))
    fig.autofmt_xdate()
    fig.tight_layout()
    save(fig, f"{fdir}/examples_{tgt}_k1_{set_name}")
    pd.DataFrame(rows).to_csv(f"{fdir}/examples_{tgt}_k1_{set_name}.csv", index=False)


def fig_error_over_time(set_name, code, role, fdir, arr, meta, preds, k_idx=0):
    Y, last, split = arr["Y"], arr["last"], arr["split"]
    D = Y - last[:, None, :]
    fig, axes = plt.subplots(1, 2, figsize=(10, 2.8), sharey=False)
    rows = []
    for ax, tgt in zip(axes, ["log_t1", "log_cz_err_mean"]):
        ti = TARGETS.index(tgt)
        m = (meta.role.values == role) & (split[:, k_idx] == code) & np.isfinite(D[:, k_idx, ti])
        day = pd.to_datetime(meta.decision_time.values[m]).floor("D")
        for mname in ["persistence", "window_mean", "hgb", "transformer"]:
            e = np.abs(preds[mname][m, k_idx, ti] - D[m, k_idx, ti])
            s = pd.Series(e, index=day).groupby(level=0).mean()
            ax.plot(s.index, s.values, color=MODEL_COLORS[mname], label=MODEL_LABELS[mname], lw=1.2)
            rows += [dict(target=tgt, model=mname, day=d, mean_abs_err=v) for d, v in s.items()]
        ax.set_title(f"{TARGET_LABELS[tgt]}, k = 1")
        ax.set_ylabel("daily mean |error| (log)")
    axes[0].legend(frameon=False, fontsize=7)
    fig.autofmt_xdate()
    fig.suptitle(f"Forecast error over time — {SET_LABELS[set_name]}", x=0.01, ha="left", color=INK, fontsize=9)
    fig.tight_layout()
    save(fig, f"{fdir}/error_over_time_{set_name}")
    pd.DataFrame(rows).to_csv(f"{fdir}/error_over_time_{set_name}.csv", index=False)


def fig_residuals(set_name, code, role, fdir, arr, meta, preds, best_classical, k_idx=0, tgt="log_t1"):
    Y, last, split = arr["Y"], arr["last"], arr["split"]
    ti = TARGETS.index(tgt)
    D = Y[:, k_idx, ti] - last[:, ti]
    m = (meta.role.values == role) & (split[:, k_idx] == code) & np.isfinite(D)
    fig, axes = plt.subplots(1, 3, figsize=(10, 2.8))
    models = ["persistence", best_classical, "transformer"]
    bins = np.linspace(-1.5, 1.5, 61)
    for mname in models:
        r = D[m] - preds[mname][m, k_idx, ti]
        axes[0].hist(r, bins=bins, histtype="step", color=MODEL_COLORS[mname], lw=1.3,
                     label=f"{MODEL_LABELS[mname]} (sd {r.std():.3f})")
    axes[0].set_xlabel("residual (log T1)")
    axes[0].set_ylabel("count")
    axes[0].legend(frameon=False, fontsize=6, loc="upper left", bbox_to_anchor=(0, 1.32))
    for ax, mname in zip(axes[1:], [best_classical, "transformer"]):
        p = preds[mname][m, k_idx, ti]
        ax.scatter(p, D[m], s=2, alpha=0.25, color=MODEL_COLORS[mname], rasterized=True)
        lim = np.percentile(np.abs(D[m]), 99.5)
        ax.plot([-lim, lim], [-lim, lim], color=INK_2, lw=0.8, ls="--")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xlabel("predicted change (log T1)")
        ax.set_ylabel("actual change (log T1)")
        ax.set_title(MODEL_LABELS[mname])
    fig.suptitle(f"Residuals, T1, k = 1 — {SET_LABELS[set_name]}", x=0.01, ha="left", color=INK, fontsize=9)
    fig.tight_layout()
    save(fig, f"{fdir}/residuals_t1_k1_{set_name}")


def fig_training_curves(fdir):
    fig, axes = plt.subplots(1, 2, figsize=(8, 2.6), sharey=True)
    for ax, name in zip(axes, ["transformer", "lstm"]):
        with open(f"{OUT}/history_{name}.json") as f:
            hists = json.load(f)
        for s, h in enumerate(hists):
            ep = [x["epoch"] for x in h]
            ax.plot(ep, [x["train"] for x in h], color=MODEL_COLORS[name], lw=0.9, alpha=0.5,
                    label="train" if s == 0 else None)
            ax.plot(ep, [x["val"] for x in h], color=INK, lw=0.9, alpha=0.7,
                    label="validation" if s == 0 else None)
        ax.set_title(f"{MODEL_LABELS[name]} (5 seeds)")
        ax.set_xlabel("epoch")
    axes[0].set_ylabel("masked MSE (standardised Δ)")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    save(fig, f"{fdir}/training_curves")


def fig_per_qubit(set_name, code, role, fdir, arr, meta, preds, k_idx=0):
    """Distribution over qubits of per-qubit MAE ratio: Transformer / comparator."""
    Y, last, split = arr["Y"], arr["last"], arr["split"]
    D = Y - last[:, None, :]
    sid = (meta.backend + ":" + meta.qubit.astype(str)).to_numpy()
    fig, axes = plt.subplots(1, 5, figsize=(10.5, 2.4), sharey=True)
    rows = []
    for ax, tgt in zip(axes, TARGETS):
        ti = TARGETS.index(tgt)
        m = (meta.role.values == role) & (split[:, k_idx] == code) & np.isfinite(D[:, k_idx, ti])
        for comp in ["persistence", "window_mean"]:
            e = pd.DataFrame({"s": sid[m], "b": meta.backend.values[m],
                              "tf": np.abs(preds["transformer"][m, k_idx, ti] - D[m, k_idx, ti]),
                              "c": np.abs(preds[comp][m, k_idx, ti] - D[m, k_idx, ti])}).groupby(["s", "b"]).mean()
            r = np.log2(e.tf / e.c)
            ax.hist(r, bins=np.linspace(-1.5, 1.0, 41), histtype="step", lw=1.3, color=MODEL_COLORS[comp],
                    label=f"vs {MODEL_LABELS[comp]} (median {np.median(r):+.2f})")
            rows += [dict(target=tgt, comparator=comp, series=sv, backend=bv, log2_ratio=v)
                     for (sv, bv), v in r.items()]
        ax.axvline(0, color=INK_2, lw=0.8)
        ax.set_title(TARGET_LABELS[tgt])
        ax.set_xlabel("log2(MAE_TF / MAE_comp) per qubit")
        ax.legend(frameon=False, fontsize=5.5, loc="upper left")
    axes[0].set_ylabel("qubits")
    fig.suptitle(f"Per-qubit comparison, k = 1 — {SET_LABELS[set_name]} (left of 0 = Transformer better)",
                 x=0.01, ha="left", color=INK, fontsize=9)
    fig.tight_layout()
    save(fig, f"{fdir}/per_qubit_{set_name}")
    pd.DataFrame(rows).to_csv(f"{fdir}/per_qubit_{set_name}.csv", index=False)


def main(which):
    setup()
    df = pd.read_csv(f"{OUT}/metrics_{which}.csv")
    fdir = f"{OUT}/figures/{which}"
    arr, meta, preds = load_preds()
    sets = ["val"] if which == "val" else ["test_primary"]
    for s in sets:
        fig_mae_vs_horizon(df, s, fdir)
        fig_skill(df, s, fdir)
        fig_ablation(df, s, fdir)
    if which == "test":
        for s in ["test_unseen_qubit", "test_unseen_device"]:
            fig_mae_vs_horizon(df, s, fdir)
        fig_eval_sets(df, fdir)
    s, (role, code) = sets[0], {"val": ("dev", 1), "test_primary": ("dev", 2)}[sets[0]]
    # best classical model = lower mean val MAE over cells (chosen on validation, never on test)
    val = pd.read_csv(f"{OUT}/metrics_val.csv")
    best_classical = val[val.model.isin(["ridge", "hgb"])].groupby("model").mae_log.mean().idxmin()
    fig_examples(s, code, role, fdir, arr, meta, preds)
    fig_error_over_time(s, code, role, fdir, arr, meta, preds)
    fig_residuals(s, code, role, fdir, arr, meta, preds, best_classical)
    fig_per_qubit(s, code, role, fdir, arr, meta, preds)
    fig_training_curves(fdir)
    print("figures written to", fdir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["val", "test"], default="val")
    main(ap.parse_args().which)
