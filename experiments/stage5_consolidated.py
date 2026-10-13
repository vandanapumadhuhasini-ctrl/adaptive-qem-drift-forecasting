"""
Consolidated comparison table (descriptive, post hoc): every policy against persistence-driven
adaptation on the Stage 5 test results, development chains, real drift R1 + R2 pooled.

    python3 -m experiments.stage5_consolidated

Uses the same paired cluster bootstrap as the pre-registered tests (experiments.stage5_policies.paired).
Percentages are (policy − persistence) / persistence. p is two-sided; Holm adjustment is applied
within each horizon across the listed policies. This table summarises saved test results; it selects
nothing and is not a new hypothesis test.
Needs results/stage5/policy_test.parquet. Writes results/stage5/consolidated_test.csv.
"""

import numpy as np
import pandas as pd

from experiments.stage5_policies import paired

POLICIES = [  # (policy, label, shots per observable per decision)
    ("none", "No mitigation", "8,192"),
    ("best_fixed", "Best fixed (global)", "8,192"),
    ("best_fixed_per_circuit", "Best fixed (per circuit)", "8,192"),
    ("cmab_probe", "Contextual bandit, probe reward", "7,168 + 1,024 probe"),
    ("cmab_warm", "Contextual bandit, model warm start", "7,168 + 1,024 probe"),
    ("adaptive_window_mean", "Adaptive: window-mean forecast", "8,192"),
    ("adaptive_ridge", "Adaptive: ridge forecast", "8,192"),
    ("adaptive_transformer", "Adaptive: Transformer forecast", "8,192"),
    ("adaptive_oracle", "Adaptive: true state (oracle, not realisable)", "8,192"),
]


def holm_adjust(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = running
    return adj


def main():
    df = pd.read_parquet("results/stage5/policy_test.parquet")
    df = df[df.regime.isin(["R1_incoherent", "R2_coherent"]) & (df.role == "dev")]
    rows = []
    for k, cell in df.groupby("k"):
        piv = cell.pivot_table(index=["backend", "chain", "epoch", "circuit", "regime"], columns="policy", values="error")
        base = piv["adaptive_persistence"].mean()
        block = [dict(k=k, policy="adaptive_persistence", label="Adaptive: latest calibration (persistence)",
                      shots="8,192", mean_error=base, pct_vs_persistence=0.0, ci_lo_pct=np.nan, ci_hi_pct=np.nan,
                      p_two_sided=np.nan, n_units=len(piv), n_chains=np.nan)]
        for pol, label, shots in POLICIES:
            r = paired(cell, pol, "adaptive_persistence")
            p2 = min(1.0, 2 * min(r["p_one_sided"], 1 - r["p_one_sided"] + 1e-4))
            block.append(dict(k=k, policy=pol, label=label, shots=shots, mean_error=piv[pol].mean(),
                              pct_vs_persistence=r["pct"], ci_lo_pct=100 * r["ci"][0] / base,
                              ci_hi_pct=100 * r["ci"][1] / base, p_two_sided=p2,
                              n_units=r["n_units"], n_chains=r["n_chains"]))
        tested = [b for b in block if not np.isnan(b["p_two_sided"])]
        for b, a in zip(tested, holm_adjust([b["p_two_sided"] for b in tested])):
            b["p_holm"] = a
        rows += block
    out = pd.DataFrame(rows)
    out.to_csv("results/stage5/consolidated_test.csv", index=False)
    show = out.assign(ci=out.apply(lambda r: "" if np.isnan(r.ci_lo_pct) else f"[{r.ci_lo_pct:+.2f}, {r.ci_hi_pct:+.2f}]", axis=1))
    print(show[["k", "label", "mean_error", "pct_vs_persistence", "ci", "p_holm", "n_units", "n_chains"]]
          .round(4).to_string(index=False))


if __name__ == "__main__":
    main()
