"""
Stage 5 figures from saved results (Matplotlib; 300-dpi PNG + vector PDF + CSV).

    python3 -m experiments.plot_stage5
"""

import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.evaluation.plotting import INK, INK_2, MODEL_COLORS, MODEL_LABELS, plt, save, setup
from src.evaluation.stats import bootstrap_weights

OUT = "results/stage5"
FIG = f"{OUT}/figures"
PCOL = {"adaptive_persistence": "#eda100", "adaptive_window_mean": "#1baf7a", "adaptive_ridge": "#e87ba4",
        "adaptive_transformer": "#2a78d6", "adaptive_oracle": "#0b0b0b", "best_fixed": INK_2,
        "cmab_probe": "#4a3aa7", "cmab_warm": "#9085e9"}
PLAB = {"adaptive_persistence": "Adaptive: persistence", "adaptive_window_mean": "Adaptive: window mean",
        "adaptive_ridge": "Adaptive: ridge", "adaptive_transformer": "Adaptive: Transformer",
        "adaptive_oracle": "Adaptive: oracle state*", "best_fixed": "Best fixed",
        "cmab_probe": "CMAB (probe reward)", "cmab_warm": "CMAB (model warm start)"}


def fig_hypotheses(h):
    cells = [("H2_primary_k1_dev", None)]
    fam = {"persistence": "H2_secondary", "best fixed": "H3a_vs_best_fixed", "CMAB-probe": "H4_vs_cmab_probe",
           "window mean": "vs_window_mean", "ridge": "vs_ridge"}
    rows = []
    p = h["H2_primary_k1_dev"]
    rows.append(dict(label="vs persistence, k=1 (H2 primary)", pct=p["pct"], lo=100 * p["ci"][0] / (p["mean_diff"] / (p["pct"] / 100)),
                     hi=100 * p["ci"][1] / (p["mean_diff"] / (p["pct"] / 100)), sig=p["statistical"]))
    for name, f in fam.items():
        for r in h[f]:
            base = r["mean_diff"] / (r["pct"] / 100) if r["pct"] else np.nan
            tag = "" if r["role"] == "dev" else ", unseen device"
            rows.append(dict(label=f"vs {name}, k={r['k']}{tag}", pct=r["pct"], lo=100 * r["ci"][0] / base,
                             hi=100 * r["ci"][1] / base, sig=r["holm_reject"]))
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    y = np.arange(len(df))[::-1]
    for yi, r in zip(y, df.itertuples()):
        col = "#2a78d6" if r.sig else INK_2
        ax.errorbar(r.pct, yi, xerr=[[r.pct - r.lo], [r.hi - r.pct]], fmt="o", color=col, capsize=2, ms=4)
    ax.axvline(0, color=INK_2, lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(df.label)
    ax.set_xlabel("Transformer-driven adaptation: error difference (%)  ← better")
    ax.set_title("Test results, development chains, real drift (R1+R2); blue = significant after Holm",
                 loc="left", fontsize=8)
    fig.tight_layout()
    save(fig, f"{FIG}/hypotheses_test")
    df.to_csv(f"{FIG}/hypotheses_test.csv", index=False)


def fig_horizon(pol):
    d = pol[pol.regime.isin(["R1_incoherent", "R2_coherent"]) & (pol.role == "dev")]
    pols = ["best_fixed", "cmab_probe", "adaptive_persistence", "adaptive_window_mean", "adaptive_ridge",
            "adaptive_transformer", "adaptive_oracle"]
    rows = []
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for p in pols:
        means, los, his = [], [], []
        for k in (1, 4, 7):
            x = d[(d.k == k) & (d.policy == p)]
            cl = x.backend + ":" + x.chain.astype(str)
            u, inv = np.unique(cl, return_inverse=True)
            W = bootstrap_weights(np.array([c.split(":")[0] for c in u]), 2000, 0)
            s = np.bincount(inv, weights=x.error, minlength=len(u))
            n = np.bincount(inv, minlength=len(u))
            b = (W @ s) / (W @ n)
            means.append(x.error.mean())
            los.append(np.percentile(b, 2.5))
            his.append(np.percentile(b, 97.5))
            rows.append(dict(policy=p, k=k, mean=means[-1], ci_lo=los[-1], ci_hi=his[-1], n=len(x)))
        off = (pols.index(p) - 3) * 0.08
        ax.errorbar(np.array([1, 4, 7]) + off, means, yerr=[np.array(means) - los, np.array(his) - means],
                    marker="o", color=PCOL[p], label=PLAB[p], capsize=2, lw=1.3, ms=3.5)
    ax.set_xticks([1, 4, 7])
    ax.set_xlabel("decision horizon k (days between decision and execution)")
    ax.set_ylabel("mean |error| per observable")
    ax.legend(frameon=False, fontsize=6.5, loc="center left", bbox_to_anchor=(1.01, 0.5))
    ax.set_title("Test, dev chains, R1+R2. Absolute levels (95 % cluster CI, between-chain spread); decision\nsets differ by k; paired policy comparisons: hypotheses_test figure. * not realisable", loc="left", fontsize=7)
    fig.tight_layout()
    save(fig, f"{FIG}/error_by_horizon_test")
    pd.DataFrame(rows).to_csv(f"{FIG}/error_by_horizon_test.csv", index=False)


def fig_pair_forecast():
    m = pd.read_csv(f"{OUT}/pair_forecast/metrics_test.csv")
    models = ["window_mean", "ridge", "hgb", "transformer"]
    sets = ["test_primary", "test_unseen_qubit", "test_unseen_device"]
    fig, axes = plt.subplots(1, 3, figsize=(9, 2.8), sharey=True)
    for ax, s in zip(axes, sets):
        for i, mod in enumerate(models):
            x = m[(m.eval_set == s) & (m.model == mod)].sort_values("k")
            ax.errorbar(x.k + (i - 1.5) * 0.15, 100 * x.R_vs_persistence,
                        yerr=[100 * (x.R_vs_persistence - x.R_ci_lo), 100 * (x.R_ci_hi - x.R_vs_persistence)],
                        marker="o", color=MODEL_COLORS[mod], label=MODEL_LABELS[mod], capsize=2, ms=3.5, lw=1.2)
        ax.axhline(0, color=INK_2, lw=0.8)
        ax.set_xticks([1, 4, 7])
        ax.set_xlabel("horizon k (epochs)")
        ax.set_title({"test_primary": "dev pairs", "test_unseen_qubit": "unseen-qubit pairs",
                      "test_unseen_device": "unseen device"}[s])
    axes[0].set_ylabel("pair CZ-error MAE reduction\nvs persistence (%)")
    axes[-1].legend(frameon=False, fontsize=6.5, loc="lower right")
    fig.suptitle("Pair-level CZ-error forecasting skill (test, 95 % cluster CI)", x=0.01, ha="left", color=INK, fontsize=9)
    fig.tight_layout()
    save(fig, f"{FIG}/pair_forecast_skill_test")


def fig_probe_quality():
    idx = pd.read_parquet(f"{OUT}/sim_index.parquet")
    arr = np.load(f"{OUT}/sim_arrays.npy")
    t = idx[idx.quantity.isin(["err_true", "probe_draw"]) & (idx.delta == 0.03)]
    piv = t.pivot_table(index=["backend", "chain", "epoch", "circuit"], columns="quantity", values="row")
    rows = []
    for (b, ch, e, c), r in piv.iterrows():
        rho = spearmanr(arr[int(r.err_true)], arr[int(r.probe_draw)]).statistic
        rows.append(dict(circuit=c, rho=rho, best_match=int(np.argmin(arr[int(r.err_true)]) == np.argmin(arr[int(r.probe_draw)]))))
    df = pd.DataFrame(rows)
    circs = sorted(df.circuit.unique())
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    ax.boxplot([df[df.circuit == c].rho.dropna() for c in circs], tick_labels=circs, showfliers=False)
    ax.axhline(0, color=INK_2, lw=0.8)
    ax.set_ylabel("Spearman ρ (probe reward vs\ntrue error ranking of actions)")
    ax.set_title(f"Bandit reward quality: median ρ {df.rho.median():.2f}; probe picks the true best action "
                 f"in {100 * df.best_match.mean():.0f} % of states", loc="left", fontsize=8)
    fig.tight_layout()
    save(fig, f"{FIG}/probe_reward_quality")
    df.groupby("circuit").agg(median_rho=("rho", "median"), best_match=("best_match", "mean")).to_csv(
        f"{FIG}/probe_reward_quality.csv")


def main():
    setup()
    h = json.load(open(f"{OUT}/hypotheses_test.json"))
    pol = pd.read_parquet(f"{OUT}/policy_test.parquet")
    fig_hypotheses(h)
    fig_horizon(pol)
    fig_pair_forecast()
    fig_probe_quality()
    print("figures written to", FIG)


if __name__ == "__main__":
    main()
