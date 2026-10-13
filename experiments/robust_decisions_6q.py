"""
R-B: decision table for 6-qubit chains at k = 1 (same rules as experiments/stage5_decisions.py).
Writes results/robustness/rb/{epoch_states.parquet,epoch_states.npy,decisions.parquet,forecast_states.npy}.

    python3 -m experiments.robust_decisions_6q
"""

import os

import numpy as np
import pandas as pd

from src.data.windows import TARGETS
from src.qem.device import build_epochs, coupling_pairs, pair_states, qubit_epoch_rows
from src.qem.nq import select_chains_n

OUT = "results/robustness/rb"
N, K = 6, 1
FORECASTERS = ["window_mean", "ridge", "transformer"]
N_CHAINS = {"ibm_fez": 9, "ibm_kingston": 10}
T_VAL, T_TEST = pd.Timestamp("2026-08-01"), pd.Timestamp("2026-09-01")


def main():
    os.makedirs(OUT, exist_ok=True)
    cce = pd.read_parquet("data/processed/drift_v3/cce.parquet")
    epochs = build_epochs(cce)
    rows = qubit_epoch_rows(cce, epochs)
    m3 = pd.read_parquet("results/stage3/windows_meta.parquet").reset_index(names="win_row")
    rows = rows.merge(m3[["backend", "qubit", "decision_time", "win_row"]], on=["backend", "qubit", "decision_time"],
                      how="left").set_index(["backend", "qubit", "epoch"]).sort_index()
    last3 = np.load("results/stage3/windows_arrays.npz")["last"]
    p3 = {f: np.load(f"results/stage3/predictions/{f}.npz")["pred"][:, 0, :] for f in FORECASTERS}
    pm = pd.read_parquet("results/stage5/pair_forecast/windows_meta.parquet").reset_index(names="pw_row")
    pm = pm.set_index(["backend", "a", "b", "epoch"]).sort_index()
    plast = np.load("results/stage5/pair_forecast/windows_arrays.npz")["last"][:, 0]
    pp = {f: np.load(f"results/stage5/pair_forecast/predictions/{f}.npz")["pred"][:, 0, 0] for f in FORECASTERS}
    ti = {t: TARGETS.index(t) for t in TARGETS}
    cz_events, pairs = coupling_pairs()
    ep_states, ep_meta, decisions, fstates = [], [], [], []
    for b, n_ch in N_CHAINS.items():
        allowed = set(cce.loc[(cce.backend == b) & (cce.role == "dev"), "qubit"])
        chains = select_chains_n(pairs, b, allowed, n_ch, N, seed=0)
        ep = epochs[epochs.backend == b].set_index("epoch")
        pst = pair_states(cz_events, b, [(c[i], c[i + 1]) for c in chains for i in range(N - 1)], ep["epoch_time"].values)
        for ci, ch in enumerate(chains):
            S, info = {}, {}
            for e in ep.index:
                try:
                    r = [rows.loc[(b, q, e)] for q in ch]
                except KeyError:
                    continue
                cz = [pst[(ch[i], ch[i + 1])][ep.loc[e, "epoch_time"]] for i in range(N - 1)]
                v = np.concatenate([[x["t1"] for x in r], [x["t2"] for x in r], [x["sx_err"] for x in r], cz])
                if np.all(np.isfinite(v)):
                    S[e], info[e] = v, r
                    ep_states.append(v)
                    ep_meta.append(dict(backend=b, chain=ci, qubits=",".join(map(str, ch)), epoch=int(e),
                                        epoch_time=ep.loc[e, "epoch_time"]))
            for e in S:
                if e + K not in S:
                    continue
                r1, r2 = info[e], info[e + K]
                if any(x2["cce_index"] != x1["cce_index"] + K or pd.isna(x1["win_row"]) or x2["gap_h"] > 168
                       for x1, x2 in zip(r1, r2)):
                    continue
                w = [int(x["win_row"]) for x in r1]
                try:
                    pr = [int(pm.loc[(b, min(ch[i], ch[i + 1]), max(ch[i], ch[i + 1]), e), "pw_row"]) for i in range(N - 1)]
                except KeyError:
                    continue
                fc = []
                for f in FORECASTERS:
                    d = p3[f][w]
                    fc.append(np.concatenate([np.exp(last3[w, ti["log_t1"]] + d[:, ti["log_t1"]]),
                                              np.exp(last3[w, ti["log_t2"]] + d[:, ti["log_t2"]]),
                                              np.exp(last3[w, ti["log_sx_err"]] + d[:, ti["log_sx_err"]]),
                                              np.exp(plast[pr] + pp[f][pr])]))
                if not np.all(np.isfinite(fc)):
                    continue
                t2 = ep.loc[e + K, "epoch_time"]
                decisions.append(dict(backend=b, chain=ci, epoch=int(e), k=K, exec_epoch=int(e + K),
                                      decision_time=ep.loc[e, "epoch_time"], exec_time=t2, role="dev",
                                      split="train" if t2 < T_VAL else ("val" if t2 < T_TEST else "test")))
                fstates.append(np.stack(fc))
    pd.DataFrame(ep_meta).to_parquet(f"{OUT}/epoch_states.parquet")
    np.save(f"{OUT}/epoch_states.npy", np.array(ep_states))
    dec = pd.DataFrame(decisions)
    dec.to_parquet(f"{OUT}/decisions.parquet")
    np.save(f"{OUT}/forecast_states.npy", np.array(fstates))
    print(dec.groupby(["backend", "split"]).size().to_string())


if __name__ == "__main__":
    main()
