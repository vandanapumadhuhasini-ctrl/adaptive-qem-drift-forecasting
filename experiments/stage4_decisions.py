"""
Stage 4: build the decision table (real calibration states; no simulation yet).

For each chain c and epoch e with a next epoch e+1 (horizon k = 1):
  published  : the state published at e (what persistence uses)
  truth      : the state at e+1 (execution-state proxy)
  forecasts  : window_mean / ridge / transformer, from frozen Stage 3 predictions
  stationary : chain's median training-period state (R4 control)

    python3 -m experiments.stage4_decisions
Writes results/stage4/decisions.parquet and results/stage4/states.npz.
"""

import json
import os

import numpy as np
import pandas as pd

from src.data.windows import TARGETS
from src.qem.device import build_epochs, coupling_pairs, pair_states, qubit_epoch_rows, select_chains

OUT = "results/stage4"
FORECASTERS = ["window_mean", "ridge", "transformer"]
N_CHAINS = {"ibm_fez": 15, "ibm_kingston": 20, "ibm_marrakesh": 16}
T_VAL, T_TEST = pd.Timestamp("2026-08-01"), pd.Timestamp("2026-09-01")
K = 1


def main():
    os.makedirs(OUT, exist_ok=True)
    cce = pd.read_parquet("data/processed/drift_v3/cce.parquet")
    meta = pd.read_parquet("results/stage3/windows_meta.parquet").reset_index(names="win_row")
    arr = np.load("results/stage3/windows_arrays.npz")
    last = arr["last"]
    preds = {f: np.load(f"results/stage3/predictions/{f}.npz")["pred"][:, K - 1, :] for f in FORECASTERS}
    ti = {t: TARGETS.index(t) for t in TARGETS}

    epochs = build_epochs(cce)
    rows = qubit_epoch_rows(cce, epochs)
    rows = rows.merge(meta[["backend", "qubit", "decision_time", "win_row"]],
                      on=["backend", "qubit", "decision_time"], how="left")
    rows = rows.set_index(["backend", "qubit", "epoch"])
    cz_events, pairs = coupling_pairs()

    allowed = {b: set(cce.loc[(cce.backend == b) & (cce.role == ("dev" if b != "ibm_marrakesh" else "heldout_device")), "qubit"])
               for b in N_CHAINS}
    records, states = [], {s: [] for s in ["published", "truth", "stationary"] + FORECASTERS}

    for b, n_ch in N_CHAINS.items():
        chains = select_chains(pairs, b, allowed[b], n_ch, seed=0)
        ep = epochs[epochs.backend == b].set_index("epoch")
        cz_pairs = [(c[i], c[i + 1]) for c in chains for i in range(3)]
        pst = pair_states(cz_events, b, cz_pairs, ep["epoch_time"].values)

        for ci, chain in enumerate(chains):
            # stationary state: median of the chain's training-period published states
            stat = []
            for e in ep.index:
                if ep.loc[e, "day"] >= T_VAL:
                    continue
                try:
                    r = [rows.loc[(b, q, e)] for q in chain]
                except KeyError:
                    continue
                cz = [pst[(chain[i], chain[i + 1])][ep.loc[e, "epoch_time"]] for i in range(3)]
                stat.append(np.concatenate([[x["t1"] for x in r], [x["t2"] for x in r], [x["sx_err"] for x in r], cz]))
            stat = np.nanmedian(np.array(stat), axis=0)

            for e in ep.index[:-K]:
                e2 = e + K
                try:
                    r1 = [rows.loc[(b, q, e)] for q in chain]
                    r2 = [rows.loc[(b, q, e2)] for q in chain]
                except KeyError:
                    continue
                # the next epoch must be each qubit's next CCE (matches the Stage 3 horizon)
                # and must not cross a segment break (gap > 168 h, Stage 2 rule)
                if any(x2["cce_index"] != x1["cce_index"] + K or x2["gap_h"] > 168
                       for x1, x2 in zip(r1, r2)):
                    continue
                if any(pd.isna(x["win_row"]) for x in r1):
                    continue
                t_e, t_e2 = ep.loc[e, "epoch_time"], ep.loc[e2, "epoch_time"]
                cz_pub = np.array([pst[(chain[i], chain[i + 1])][t_e] for i in range(3)])
                cz_true = np.array([pst[(chain[i], chain[i + 1])][t_e2] for i in range(3)])
                pub = np.concatenate([[x["t1"] for x in r1], [x["t2"] for x in r1], [x["sx_err"] for x in r1], cz_pub])
                tru = np.concatenate([[x["t1"] for x in r2], [x["t2"] for x in r2], [x["sx_err"] for x in r2], cz_true])
                if not (np.all(np.isfinite(pub)) and np.all(np.isfinite(tru))):
                    continue
                win = [int(x["win_row"]) for x in r1]
                fc = {}
                for f in FORECASTERS:
                    d = preds[f][win]                                          # (4, T) predicted Δ log
                    t1 = np.exp(last[win, ti["log_t1"]] + d[:, ti["log_t1"]])
                    t2 = np.exp(last[win, ti["log_t2"]] + d[:, ti["log_t2"]])
                    sx = np.exp(last[win, ti["log_sx_err"]] + d[:, ti["log_sx_err"]])
                    dcz = d[:, ti["log_cz_err_mean"]]
                    cz = cz_pub * np.exp([(dcz[i] + dcz[i + 1]) / 2 for i in range(3)])
                    fc[f] = np.concatenate([t1, t2, sx, cz])
                day2 = ep.loc[e2, "day"]
                split = "train" if day2 < T_VAL else ("val" if day2 < T_TEST else "test")
                records.append(dict(backend=b, chain=ci, qubits=",".join(map(str, chain)), epoch=int(e),
                                    decision_time=t_e, exec_time=t_e2, split=split,
                                    role="dev" if b != "ibm_marrakesh" else "heldout_device"))
                states["published"].append(pub)
                states["truth"].append(tru)
                states["stationary"].append(stat)
                for f in FORECASTERS:
                    states[f].append(fc[f])

    dec = pd.DataFrame(records)
    dec.to_parquet(f"{OUT}/decisions.parquet")
    np.savez_compressed(f"{OUT}/states.npz", **{k: np.array(v) for k, v in states.items()})
    summary = dec.groupby(["role", "split"]).size().to_dict()
    with open(f"{OUT}/decisions_summary.json", "w") as fh:
        json.dump({f"{a}/{b}": int(n) for (a, b), n in summary.items()}, fh, indent=2)
    print(dec.groupby(["backend", "split"]).size().to_string())
    # Sanity: forecast states only use information published at e (built from Stage 3 rows at e)
    S = np.array(states["truth"]); P = np.array(states["published"])
    print("median |log(truth/published)| T1:", np.median(np.abs(np.log(S[:, :4] / P[:, :4]))).round(3))


if __name__ == "__main__":
    main()
