"""
Stage 4 figures from saved policy results (results/stage4/policy_{split}.csv,
summary_{split}.json). Matplotlib only; PNG at 300 dpi plus vector PDF.

    python3 -m experiments.plot_stage4 --split val
"""

import argparse
import json

import numpy as np
import pandas as pd

from src.evaluation.plotting import INK, INK_2, plt, save, setup

OUT = "results/stage4"
POLICIES = ["none", "best_fixed", "best_fixed_per_circuit", "cmab_probe", "cmab_warm", "cmab_oracle",
            "adaptive_persistence", "adaptive_window_mean", "adaptive_ridge", "adaptive_transformer",
            "adaptive_oracle"]
LABELS = {"none": "No mitigation", "best_fixed": "Best fixed", "best_fixed_per_circuit": "Best fixed / circuit",
          "cmab_probe": "CMAB (probe reward)", "cmab_warm": "CMAB (model warm start)",
          "cmab_oracle": "CMAB (oracle reward)*", "adaptive_persistence": "Adaptive: persistence",
          "adaptive_window_mean": "Adaptive: window mean", "adaptive_ridge": "Adaptive: ridge",
          "adaptive_transformer": "Adaptive: Transformer", "adaptive_oracle": "Adaptive: oracle state*"}
COLORS = {"none": "#eb6834", "best_fixed": INK_2, "best_fixed_per_circuit": "#8a8985",
          "cmab_probe": "#4a3aa7", "cmab_warm": "#9085e9", "cmab_oracle": "#c3c2b7",
          "adaptive_persistence": "#eda100", "adaptive_window_mean": "#1baf7a",
          "adaptive_ridge": "#e87ba4", "adaptive_transformer": "#2a78d6", "adaptive_oracle": "#0b0b0b"}
REGIMES = ["R1_incoherent", "R2_coherent", "R3_mismatch"]
RLABEL = {"R1_incoherent": "R1 incoherent drift", "R2_coherent": "R2 coherent drift",
          "R3_mismatch": "R3 model mismatch", "R4_stationary": "R4 stationary"}


def boot_ci(df, B=2000, seed=0):
    """Cluster bootstrap (chains within backend) of the mean error."""
    rng = np.random.default_rng(seed)
    g = df.groupby(["backend", "chain"])["error"].agg(["sum", "size"]).reset_index()
    means = []
    for _ in range(B):
        parts = [d.sample(len(d), replace=True, random_state=int(rng.integers(1 << 31))) for _, d in g.groupby("backend")]
        s = pd.concat(parts)
        means.append(s["sum"].sum() / s["size"].sum())
    return np.percentile(means, [2.5, 97.5])


def fig_errors(df, split, B=2000):
    """Paired difference of each policy from the best fixed action (same decisions),
    in % of best-fixed error, with a paired cluster-bootstrap CI (chains within backend)."""
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6), sharey=True)
    rows = []
    for ax, r in zip(axes, REGIMES):
        d = df[(df.regime == r) & (df.role == "dev")]
        piv = d.pivot_table(index=["backend", "chain", "dec", "circuit"], columns="policy", values="error")
        ref = piv["best_fixed"]
        clusters = piv.index.droplevel([2, 3])
        pols = [p for p in POLICIES if p in piv and p not in ("none", "best_fixed")]
        keys = clusters.unique()
        by_b = {}
        for k in keys:
            by_b.setdefault(k[0], []).append(k)
        for i, p in enumerate(pols):
            diff = (piv[p] - ref)
            g = pd.DataFrame({"d": diff.values, "r": ref.values}, index=clusters).groupby(level=[0, 1]).sum()
            point = 100 * g.d.sum() / g.r.sum()
            boots = []
            for _ in range(B):
                idx = [k for b, ks in by_b.items() for k in [ks[j] for j in rng.integers(0, len(ks), len(ks))]]
                gg = g.loc[idx]
                boots.append(100 * gg.d.sum() / gg.r.sum())
            lo, hi = np.percentile(boots, [2.5, 97.5])
            ax.errorbar(point, i, xerr=[[point - lo], [hi - point]], fmt="o", color=COLORS[p], capsize=2, ms=4)
            rows.append(dict(regime=r, policy=p, pct_vs_best_fixed=point, ci_lo=lo, ci_hi=hi))
        ax.axvline(0, color=INK_2, lw=0.8)
        ax.set_yticks(range(len(pols)))
        ax.set_yticklabels([LABELS[p] for p in pols])
        ax.set_title(RLABEL[r])
        ax.set_xlabel("error vs best fixed (%)  ← better")
        ax.invert_yaxis()
    fig.suptitle(f"Paired comparison with the best fixed action — {split} period, development chains "
                 "(95 % paired cluster-bootstrap CI; * = not realisable)", x=0.01, ha="left", color=INK, fontsize=9)
    fig.tight_layout()
    save(fig, f"{OUT}/figures/{split}/policy_vs_best_fixed")
    pd.DataFrame(rows).to_csv(f"{OUT}/figures/{split}/policy_vs_best_fixed.csv", index=False)


def fig_value_of_information(summary, split):
    pols = ["cmab_probe", "cmab_warm", "adaptive_persistence", "adaptive_window_mean", "adaptive_ridge",
            "adaptive_transformer", "adaptive_oracle"]
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    width = 0.11
    rows = []
    for j, p in enumerate(pols):
        vals = [summary[f"{r}/dev"]["captured_fraction"].get(p, np.nan) for r in REGIMES]
        x = np.arange(len(REGIMES)) + (j - len(pols) / 2) * width
        ax.bar(x, 100 * np.array(vals), width=width * 0.9, color=COLORS[p], label=LABELS[p])
        rows += [dict(regime=r, policy=p, captured_pct=100 * v) for r, v in zip(REGIMES, vals)]
    ax.axhline(0, color=INK_2, lw=0.8)
    ax.axhline(100, color=INK_2, lw=0.8, ls="--")
    ax.set_xticks(range(len(REGIMES)))
    ax.set_xticklabels([RLABEL[r] for r in REGIMES])
    ax.set_ylabel("share of oracle gain over\nbest fixed captured (%)")
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, fontsize=6.5, loc="center left", bbox_to_anchor=(1.01, 0.5))
    gains = ", ".join(f"{RLABEL[r].split()[0]} {100 * summary[f'{r}/dev']['oracle_gain']['mean']:.3f}" for r in REGIMES)
    ax.set_title(f"Value of information — oracle gain over best fixed (×100): {gains}", loc="left", fontsize=8)
    fig.tight_layout()
    save(fig, f"{OUT}/figures/{split}/value_of_information")
    pd.DataFrame(rows).to_csv(f"{OUT}/figures/{split}/value_of_information.csv", index=False)


def fig_actions(df, split):
    pols = ["adaptive_persistence", "adaptive_transformer", "adaptive_oracle", "cmab_probe"]
    d = df[(df.regime == "R2_coherent") & (df.role == "dev") & df.policy.isin(pols)]
    tab = pd.crosstab(d.action, d.policy, normalize="columns")[pols]
    tab = tab.loc[tab.max(axis=1).sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    im = ax.imshow(100 * tab.values, aspect="auto", cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks(range(len(pols)))
    ax.set_xticklabels([LABELS[p] for p in pols], rotation=15)
    ax.set_yticks(range(len(tab)))
    ax.set_yticklabels(tab.index, fontsize=6.5)
    ax.grid(False)
    for (i, j), v in np.ndenumerate(100 * tab.values):
        if v >= 1:
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=6,
                    color="white" if v > 50 else INK)
    fig.colorbar(im, ax=ax, label="% of decisions")
    ax.set_title("Chosen actions, R2 coherent drift (development chains)", loc="left")
    fig.tight_layout()
    save(fig, f"{OUT}/figures/{split}/action_choices_R2")
    (100 * tab).to_csv(f"{OUT}/figures/{split}/action_choices_R2.csv")


def main(split):
    setup()
    df = pd.read_csv(f"{OUT}/policy_{split}.csv")
    with open(f"{OUT}/summary_{split}.json") as f:
        summary = json.load(f)
    fig_errors(df, split)
    fig_value_of_information(summary, split)
    fig_actions(df, split)
    print("figures written")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    main(ap.parse_args().split)
