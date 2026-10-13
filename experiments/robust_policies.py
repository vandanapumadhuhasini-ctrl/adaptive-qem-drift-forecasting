"""
Robustness evaluation (docs/ROBUSTNESS_PLAN.md), k = 1.

    python3 -m experiments.robust_policies ra|rb --split val    # selects best fixed actions
    python3 -m experiments.robust_freeze                        # before any robustness test metric
    python3 -m experiments.robust_policies ra|rb --split test

Reuses the Stage 5 statistics and bandit code unchanged; bandit hyperparameters are
the Stage 5 validation selections (k = 1), not re-tuned.
"""

import argparse
import json

import numpy as np
import pandas as pd

from experiments.robust_sim import CIRCUITS, OUT, SRC
from experiments.stage5_policies import bandit_stream, context, paired
from src.qem.mitigation import ACTION_NAMES
from src.qem.nq import benchmark_n

REGIMES = {"ra": {"R1_incoherent": (0.0, 0.0), "R2_coherent": (0.03, 0.03), "R3_mismatch": (0.03, 0.0)},
           "rb": {"R1_incoherent": (0.0, 0.0), "R2_coherent": (0.03, 0.03)}}
SOURCES = ["persistence", "window_mean", "ridge", "transformer", "oracle"]


class Cache:
    def __init__(self, d):
        t = pd.read_parquet(f"{d}/sim_index.parquet")
        self.idx = dict(zip(zip(t.backend, t.chain, t.epoch, t.circuit, t.quantity, t.delta), t.row))
        self.arr = np.load(f"{d}/sim_arrays.npy")
        self.exec_keys = set(zip(*[t.loc[t.quantity == "err_true", c] for c in ["backend", "chain", "epoch"]]))
        self.dec_keys = set(zip(*[t.loc[t.quantity == "pred_published", c] for c in ["backend", "chain", "epoch"]]))

    def get(self, b, ch, e, circ, q, d):
        return self.arr[self.idx[(b, ch, e, circ, q, d)]]


def main(which, split):
    if split == "test":
        from experiments.robust_freeze import check_frozen
        check_frozen()
    out, src = OUT[which], SRC[which]
    cache = Cache(out)
    circuits = [benchmark_n(*s)["name"] for s in CIRCUITS[which]]
    dec = pd.read_parquet(f"{src}/decisions.parquet")
    dec = dec[dec.k == 1]
    em = pd.read_parquet(f"{src}/epoch_states.parquet")
    S = np.load(f"{src}/epoch_states.npy")
    sidx = pd.Series(np.arange(len(em)), index=pd.MultiIndex.from_frame(em[["backend", "chain", "epoch"]]))
    keep = [(b, c, e) in cache.dec_keys and (b, c, x) in cache.exec_keys
            for b, c, e, x in zip(dec.backend, dec.chain, dec.epoch, dec.exec_epoch)]
    dec = dec[keep]
    sel_path = f"{out}/selections.json"
    sel = json.load(open(sel_path)) if split == "test" else {}
    s5 = json.load(open("results/stage5/selections.json"))
    rows = []
    for regime, (dt, dm) in REGIMES[which].items():
        ev = dec[dec.split == split]
        U = pd.DataFrame([dict(backend=r.backend, chain=r.chain, epoch=r.epoch, exec_epoch=r.exec_epoch, split=r.split,
                               role=r.role, circuit=c, X=context(S[sidx[(r.backend, r.chain, r.epoch)]]))
                          for r in dec[dec.split.isin(["train", "val", split])].itertuples() for c in circuits])
        evu = U[U.split == split]
        E = np.stack([cache.get(r.backend, r.chain, r.exec_epoch, r.circuit, "err_true", dt) for r in evu.itertuples()])
        pol = {"none": E[:, ACTION_NAMES.index("none")]}
        dev = (evu.role == "dev").to_numpy()
        if split == "val":
            sel[regime] = dict(best_fixed=ACTION_NAMES[int(np.argmin(E[dev].mean(0)))],
                               best_fixed_per_circuit={c: ACTION_NAMES[int(np.argmin(E[dev & (evu.circuit == c).to_numpy()].mean(0)))]
                                                       for c in circuits})
        bf = ACTION_NAMES.index(sel[regime]["best_fixed"])
        pol["best_fixed"] = E[:, bf]
        pol["best_fixed_per_circuit"] = np.array([E[i, ACTION_NAMES.index(sel[regime]["best_fixed_per_circuit"][c])]
                                                  for i, c in enumerate(evu.circuit)])
        for s in SOURCES:
            q = "pred_published" if s == "persistence" else f"pred_{s}_k1"
            P = np.stack([cache.get(r.backend, r.chain, r.epoch, r.circuit, q, dm) for r in evu.itertuples()])
            pol[f"adaptive_{s}"] = E[np.arange(len(E)), P.argmin(1)]
        if which == "ra":
            for variant in ["probe", "warm", "oracle"]:
                p = s5[f"k1/{regime}"][f"cmab_{variant}"]
                res = {}
                for _, st in U.groupby(["backend", "chain", "circuit"]):
                    res.update(bandit_stream(st.sort_values("epoch"), cache, dt, dm, p, variant, 1))
                pol[f"cmab_{variant}"] = np.array([res[i][1] for i in evu.index])
        for name, err in pol.items():
            for j, r in enumerate(evu.itertuples()):
                rows.append(dict(k=1, regime=regime, policy=name, backend=r.backend, chain=r.chain, epoch=r.epoch,
                                 circuit=r.circuit, role=r.role, error=float(err[j])))
    df = pd.DataFrame(rows)
    df.to_parquet(f"{out}/policy_{split}.parquet")
    if split == "val":
        json.dump(sel, open(sel_path, "w"), indent=2)
    real = df[df.regime.isin(["R1_incoherent", "R2_coherent"]) & (df.role == "dev")]
    comps = [("adaptive_transformer", b) for b in ["adaptive_persistence", "best_fixed", "best_fixed_per_circuit",
                                                   "adaptive_window_mean", "adaptive_ridge"]]
    if which == "ra":
        comps.append(("adaptive_transformer", "cmab_probe"))
    comps.append(("adaptive_oracle", "adaptive_persistence"))
    summary = {f"{a} - {b}": paired(real, a, b) for a, b in comps}
    summary["mean_error"] = {f"{r}/{p}": float(v) for (r, p), v in df[df.role == "dev"].groupby(["regime", "policy"]).error.mean().items()}
    json.dump(summary, open(f"{out}/summary_{split}.json", "w"), indent=2)
    for k, v in summary.items():
        if k != "mean_error":
            print(f"{which} {split}  {k:52s} {v['pct']:+6.2f}%  CI[{v['ci'][0]:+.5f},{v['ci'][1]:+.5f}]  chains={v['n_chains']} units={v['n_units']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["ra", "rb"])
    ap.add_argument("--split", choices=["val", "test"], default="val")
    a = ap.parse_args()
    main(a.which, a.split)
