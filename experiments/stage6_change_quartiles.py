"""
Stage 6 (exploratory, test period): decision value by size of the device change.

    python3 -m experiments.stage6_change_quartiles

Development chains, regimes R1 + R2. Test decisions are binned into quartiles of the CZ
change between the published and the execution-time calibration (max |log ratio| over
the three chain pairs). Per horizon k and quartile: Transformer and oracle state versus
persistence, persistence versus best fixed, and how often the Transformer CZ forecast is
closer to the execution-time value than the published one.
Needs results/stage5/policy_test.parquet (written by experiments.stage5_policies).
Writes results/stage6/h5_change_quartiles.csv. Exploratory: not a pre-registered test.
"""

import os

import numpy as np
import pandas as pd

OUT = "results/stage6"


def main():
    os.makedirs(OUT, exist_ok=True)
    pol = pd.read_parquet("results/stage5/policy_test.parquet")
    dec = pd.read_parquet("results/stage5/decisions.parquet")
    em = pd.read_parquet("results/stage5/epoch_states.parquet")
    S = np.load("results/stage5/epoch_states.npy")
    F = np.load("results/stage5/forecast_states.npy")
    sidx = pd.Series(np.arange(len(em)), index=pd.MultiIndex.from_frame(em[["backend", "chain", "epoch"]]))
    dec["frow"] = np.arange(len(dec))
    d = dec[(dec.split == "test") & (dec.role == "dev")].copy()
    pub = np.stack([S[sidx[(r.backend, r.chain, r.epoch)]] for r in d.itertuples()])
    tru = np.stack([S[sidx[(r.backend, r.chain, r.exec_epoch)]] for r in d.itertuples()])
    tf = F[d.frow.values, 2]  # Transformer forecast state
    d["cz_change"] = np.abs(np.log(tru[:, 12:15] / pub[:, 12:15])).max(1)
    d["tf_cz_err"] = np.abs(np.log(tf[:, 12:15] / tru[:, 12:15])).max(1)
    d["pub_cz_err"] = d.cz_change
    piv = pol[pol.regime.isin(["R1_incoherent", "R2_coherent"]) & (pol.role == "dev")].pivot_table(
        index=["k", "backend", "chain", "epoch", "circuit", "regime"], columns="policy", values="error").reset_index()
    m = piv.merge(d[["k", "backend", "chain", "epoch", "cz_change", "tf_cz_err", "pub_cz_err"]],
                  on=["k", "backend", "chain", "epoch"])
    m["bin"] = pd.qcut(m.cz_change, 4, labels=["Q1 small", "Q2", "Q3", "Q4 large"])
    rows = []
    for (k, b), x in m.groupby(["k", "bin"], observed=True):
        rows.append(dict(
            k=k, change_quartile=str(b), median_cz_change=x.cz_change.median(), n=len(x),
            tf_minus_persist_pct=100 * (x.adaptive_transformer - x.adaptive_persistence).mean() / x.adaptive_persistence.mean(),
            oracle_minus_persist_pct=100 * (x.adaptive_oracle - x.adaptive_persistence).mean() / x.adaptive_persistence.mean(),
            persist_minus_bestfixed_pct=100 * (x.adaptive_persistence - x.best_fixed).mean() / x.best_fixed.mean(),
            tf_closer_than_pub=float((x.tf_cz_err < x.pub_cz_err).mean())))
    r = pd.DataFrame(rows)
    r.to_csv(f"{OUT}/h5_change_quartiles.csv", index=False)
    print(r.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
