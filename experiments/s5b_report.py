"""
Stage 5b validation report for the decision-focused controller (validation only).

    python3 -m experiments.s5b_report
"""

import json

import numpy as np
import pandas as pd

from src.evaluation.stats import bootstrap_weights

OUT = "results/stage5b"


def paired(df, a, b, B=5000):
    cl = (df.backend + ":" + df.chain.astype(str)).to_numpy()
    u, inv = np.unique(cl, return_inverse=True)
    W = bootstrap_weights(np.array([x.split(":")[0] for x in u]), B, 0)
    d = (df[a] - df[b]).to_numpy()
    s, n = np.bincount(inv, weights=d, minlength=len(u)), np.bincount(inv, minlength=len(u))
    bt = (W @ s) / (W @ n)
    return dict(diff=float(d.mean()), pct=float(100 * d.mean() / df[b].mean()),
                ci=[float(np.percentile(bt, 2.5)), float(np.percentile(bt, 97.5))],
                ci_pct=[float(100 * np.percentile(bt, 2.5) / df[b].mean()), float(100 * np.percentile(bt, 97.5) / df[b].mean())])


def main():
    df = pd.read_parquet(f"{OUT}/dft_results.parquet")
    v = df[df.split == "val"].copy()
    # existing forecast-driven controller and validation-chosen best fixed (Stage 5, same decisions)
    pv = pd.read_parquet("results/stage5/policy_val.parquet")
    pv = pv[pv.policy.isin(["adaptive_transformer", "best_fixed"]) & pv.regime.isin(["R1_incoherent", "R2_coherent"])]
    pv = pv.pivot_table(index=["k", "regime", "circuit", "backend", "chain", "epoch"], columns="policy", values="error").reset_index()
    pv = pv.rename(columns={"adaptive_transformer": "forecast_transformer", "best_fixed": "best_fixed_val"})
    v = v.merge(pv, on=["k", "regime", "circuit", "backend", "chain", "epoch"], how="inner")
    pols = ["best_fixed_train", "best_fixed_val", "persistence", "forecast_transformer", "dft_no_history",
            "dft_mlp_summary", "dft_transformer", "oracle"]
    rep = {"n_units": int(len(v)), "n_chains": int(v.groupby(["backend", "chain"]).ngroups)}
    rep["mean_error_by_k"] = {int(k): {p: float(g[p].mean()) for p in pols} for k, g in v.groupby("k")}
    rep["mean_error_all"] = {p: float(v[p].mean()) for p in pols}
    comps = [("dft_transformer", "persistence"), ("dft_transformer", "dft_no_history"),
             ("dft_transformer", "dft_mlp_summary"), ("dft_transformer", "forecast_transformer"),
             ("dft_transformer", "best_fixed_train"), ("dft_no_history", "persistence"),
             ("dft_mlp_summary", "persistence"), ("oracle", "persistence")]
    rep["paired"] = {}
    for k, g in [("all", v)] + [(str(k), g) for k, g in v.groupby("k")]:
        rep["paired"][k] = {f"{a} - {b}": paired(g, a, b) for a, b in comps}
    h6 = rep["paired"]["1"]["dft_transformer - persistence"]
    rep["H6_validation"] = dict(cell="k=1, R1+R2, dev, validation", **h6,
                                meets_criteria=bool(h6["ci"][1] < 0 and h6["pct"] <= -1.0))
    rep["per_circuit_pct_vs_persistence"] = {
        c: {m: float(100 * (g[m].mean() / g.persistence.mean() - 1)) for m in ["dft_transformer", "dft_no_history", "oracle"]}
        for c, g in v.groupby("circuit")}
    json.dump(rep, open(f"{OUT}/dft_validation_report.json", "w"), indent=2)
    print(json.dumps({k: rep[k] for k in ["n_units", "n_chains", "H6_validation"]}, indent=2))
    print("\nmean error (validation, dev, R1+R2):")
    print(pd.DataFrame(rep["mean_error_by_k"]).T.round(5).to_string())
    for k in ["all", "1", "4", "7"]:
        print(f"\npaired, k={k}:")
        for c, x in rep["paired"][k].items():
            print(f"   {c:42s} {x['pct']:+6.2f}%  CI [{x['ci_pct'][0]:+.2f}%, {x['ci_pct'][1]:+.2f}%]")
    print("\nper circuit (% vs persistence):")
    print(pd.DataFrame(rep["per_circuit_pct_vs_persistence"]).T.round(2).to_string())


if __name__ == "__main__":
    main()
