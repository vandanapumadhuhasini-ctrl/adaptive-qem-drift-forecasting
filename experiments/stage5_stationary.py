"""
Stage 5 H3b: stationary control (PROTOCOL §5). Each chain is frozen at the median of its
training-period epoch states (δ = 0). Every state source sees the same state, so every
adaptive policy makes the same choice. Non-inferiority of Transformer-driven adaptation
versus the best fixed action is tested with the protocol's +2 % margin on the upper CI bound.
Uses only frozen inputs (epoch states, selections); no test-period quantity is selected.

    python3 -m experiments.stage5_stationary
"""

import json
from multiprocessing import get_context

import numpy as np
import pandas as pd

from experiments.stage5_freeze import check_frozen
from experiments.stage5_simulate import run_unit
from src.evaluation.stats import bootstrap_weights
from src.qem.mitigation import ACTION_NAMES

OUT = "results/stage5"


def main():
    check_frozen()
    from src.qem.benchmarks import all_benchmarks
    em = pd.read_parquet(f"{OUT}/epoch_states.parquet")
    S = np.load(f"{OUT}/epoch_states.npy")
    em["row"] = np.arange(len(em))
    train = em[em.epoch_time < pd.Timestamp("2026-08-01")]
    tasks = []
    for (b, ch), d in train.groupby(["backend", "chain"]):
        st = np.median(S[d.row.to_numpy()], axis=0)
        for n in [x["name"] for x in all_benchmarks()]:
            tasks.append((f"stat|{b}|{ch}", n, b, st, True, True, [(1, {"transformer": st})]))
    with get_context("spawn").Pool(12) as pool:
        res = pool.map(run_unit, tasks)
    sel = json.load(open(f"{OUT}/selections.json"))["k1/R1_incoherent"]
    bf = ACTION_NAMES.index(sel["best_fixed"])
    rows = []
    for (key, circ, out), t in zip(res, tasks):
        E = out[("err_true", 0.0)]
        ch_tf = int(np.argmin(out[("pred_transformer_k1", 0.0)]))
        ch_p = int(np.argmin(out[("pred_published", 0.0)]))
        assert ch_tf == ch_p, "identical states must give identical choices"
        rows.append(dict(backend=t[2], chain=key, circuit=circ, err_tf=E[ch_tf], err_bf=E[bf],
                         err_bf_circuit=E[ACTION_NAMES.index(sel["best_fixed_per_circuit"][circ])],
                         err_oracle=E.min()))
    df = pd.DataFrame(rows)
    dev = df[df.backend != "ibm_marrakesh"]
    cl, inv = np.unique(dev.chain, return_inverse=True)
    W = bootstrap_weights(np.array([c.split("|")[1] for c in cl]), 10_000, 0)
    s_tf = np.bincount(inv, weights=dev.err_tf, minlength=len(cl))
    s_bf = np.bincount(inv, weights=dev.err_bf, minlength=len(cl))
    ratio = (W @ s_tf) / (W @ s_bf) - 1
    out = dict(n_units=len(dev), n_chains=len(cl), mean_tf=float(dev.err_tf.mean()), mean_bf=float(dev.err_bf.mean()),
               mean_bf_per_circuit=float(dev.err_bf_circuit.mean()), mean_oracle=float(dev.err_oracle.mean()),
               rel_diff=float(dev.err_tf.sum() / dev.err_bf.sum() - 1),
               ci=[float(np.percentile(ratio, 2.5)), float(np.percentile(ratio, 97.5))])
    out["H3b_noninferior"] = out["ci"][1] < 0.02
    df.to_csv(f"{OUT}/stationary_test.csv", index=False)
    with open(f"{OUT}/h3b_stationary.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
