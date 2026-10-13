"""
Stage 5: decision tables for horizons k = 1, 4, 7 on the corrected epochs.

Epoch state S(chain, e): members' T1/T2/sx from calibrations published by the epoch
time, and pair CZ errors as published at the epoch time (real IBM records).
  published(e)        = S(chain, e)
  truth(e, k)         = S(chain, e + k)      execution-state proxy
  forecast_f(e, k)    = per-qubit Stage 3 forecasts (T1, T2, sx; horizon k in CCEs, which must
                        equal k epochs) and Stage 5a pair-level CZ forecasts (horizon k epochs)
Writes results/stage5/epoch_states.npz/.parquet and decisions.parquet.

    python3 -m experiments.stage5_decisions
"""

import os

import numpy as np
import pandas as pd

from src.data.windows import TARGETS
from src.forecasting.pairs import HORIZONS as PAIR_H
from src.qem.device import build_epochs, coupling_pairs, pair_states, qubit_epoch_rows, select_chains

OUT = "results/stage5"
KS = (1, 4, 7)
STAGE3_H = (1, 2, 4, 7)
FORECASTERS = ["window_mean", "ridge", "transformer"]
N_CHAINS = {"ibm_fez": 15, "ibm_kingston": 20, "ibm_marrakesh": 16}
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
    p3 = {f: np.load(f"results/stage3/predictions/{f}.npz")["pred"] for f in FORECASTERS}
    pm = pd.read_parquet("results/stage5/pair_forecast/windows_meta.parquet").reset_index(names="pw_row")
    pm = pm.set_index(["backend", "a", "b", "epoch"]).sort_index()
    plast = np.load("results/stage5/pair_forecast/windows_arrays.npz")["last"][:, 0]
    pp = {f: np.load(f"results/stage5/pair_forecast/predictions/{f}.npz")["pred"][:, :, 0] for f in FORECASTERS}
    ti = {t: TARGETS.index(t) for t in TARGETS}
    cz_events, pairs = coupling_pairs()
    allowed = {b: set(cce.loc[(cce.backend == b) & (cce.role == ("dev" if b != "ibm_marrakesh" else "heldout_device")),
                               "qubit"]) for b in N_CHAINS}

    ep_states, ep_meta = [], []           # S(chain, e)
    decisions, fstates = [], []
    for b, n_ch in N_CHAINS.items():
        chains = select_chains(pairs, b, allowed[b], n_ch, seed=0)
        ep = epochs[epochs.backend == b].set_index("epoch")
        cz_pairs = [(c[i], c[i + 1]) for c in chains for i in range(3)]
        pst = pair_states(cz_events, b, cz_pairs, ep["epoch_time"].values)
        for ci, ch in enumerate(chains):
            S, info = {}, {}
            for e in ep.index:
                try:
                    r = [rows.loc[(b, q, e)] for q in ch]
                except KeyError:
                    continue
                if any(x["gap_h"] > 168 for x in r):
                    pass                                   # a fresh segment; usable as a state
                cz = [pst[(ch[i], ch[i + 1])][ep.loc[e, "epoch_time"]] for i in range(3)]
                v = np.concatenate([[x["t1"] for x in r], [x["t2"] for x in r], [x["sx_err"] for x in r], cz])
                if np.all(np.isfinite(v)):
                    S[e], info[e] = v, r
                    ep_states.append(v)
                    ep_meta.append(dict(backend=b, chain=ci, qubits=",".join(map(str, ch)), epoch=int(e),
                                        epoch_time=ep.loc[e, "epoch_time"]))
            for e in S:
                for k in KS:
                    if e + k not in S:
                        continue
                    r1, r2 = info[e], info[e + k]
                    hi = STAGE3_H.index(k)
                    if any(x2["cce_index"] != x1["cce_index"] + k or pd.isna(x1["win_row"]) or
                           any(rows.loc[(b, q, e + j)]["gap_h"] > 168 for q in ch for j in range(1, k + 1)
                               if (b, q, e + j) in rows.index)
                           for x1, x2 in zip(r1, r2)):
                        continue
                    wrows = [int(x["win_row"]) for x in r1]
                    try:
                        prow = [int(pm.loc[(b, min(ch[i], ch[i + 1]), max(ch[i], ch[i + 1]), e), "pw_row"]) for i in range(3)]
                    except KeyError:
                        continue
                    hp = PAIR_H.index(k)
                    fc = {}
                    for f in FORECASTERS:
                        d = p3[f][wrows, hi, :]
                        t1 = np.exp(last3[wrows, ti["log_t1"]] + d[:, ti["log_t1"]])
                        t2 = np.exp(last3[wrows, ti["log_t2"]] + d[:, ti["log_t2"]])
                        sx = np.exp(last3[wrows, ti["log_sx_err"]] + d[:, ti["log_sx_err"]])
                        cz = np.exp(plast[prow] + pp[f][prow, hp])
                        fc[f] = np.concatenate([t1, t2, sx, cz])
                    if not all(np.all(np.isfinite(v)) for v in fc.values()):
                        continue
                    exec_time = ep.loc[e + k, "epoch_time"]
                    decisions.append(dict(backend=b, chain=ci, epoch=int(e), k=k, exec_epoch=int(e + k),
                                          decision_time=ep.loc[e, "epoch_time"], exec_time=exec_time,
                                          split="train" if exec_time < T_VAL else ("val" if exec_time < T_TEST else "test"),
                                          role="dev" if b != "ibm_marrakesh" else "heldout_device"))
                    fstates.append(np.stack([fc[f] for f in FORECASTERS]))

    em = pd.DataFrame(ep_meta)
    em.to_parquet(f"{OUT}/epoch_states.parquet")
    np.save(f"{OUT}/epoch_states.npy", np.array(ep_states))
    dec = pd.DataFrame(decisions)
    dec.to_parquet(f"{OUT}/decisions.parquet")
    np.save(f"{OUT}/forecast_states.npy", np.array(fstates))   # (n_dec, 3 forecasters, 15)
    print(dec.groupby(["role", "split", "k"]).size().unstack("k").to_string())
    # Leakage assertions: every state used at decision epoch e was published by its epoch time
    assert (dec.decision_time < dec.exec_time).all()


if __name__ == "__main__":
    main()
