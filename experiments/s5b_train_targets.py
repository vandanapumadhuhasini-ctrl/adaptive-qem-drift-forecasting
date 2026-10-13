"""
Stage 5b (decision-focused Transformer controller): simulate training targets.

For every training-period decision (k = 1, 4, 7; all chains) we need, at the decision
epoch, the model-based predicted error of every action from the published state, and at
the execution epoch, the true expected error of every action. Uses the frozen Stage 5
simulator (experiments/stage5_simulate.run_unit) unchanged; skips keys already present in
the Stage 5 cache. Validation and test keys are not simulated here (already cached).

    python3 -m experiments.s5b_train_targets
"""

import time
from multiprocessing import get_context

import numpy as np
import pandas as pd

from experiments.stage5_simulate import run_unit

OUT = "results/stage5b"


def main(workers=12):
    import os
    os.makedirs(OUT, exist_ok=True)
    from src.qem.benchmarks import all_benchmarks
    names = [b["name"] for b in all_benchmarks()]
    em = pd.read_parquet("results/stage5/epoch_states.parquet")
    S = np.load("results/stage5/epoch_states.npy")
    sidx = pd.Series(np.arange(len(em)), index=pd.MultiIndex.from_frame(em[["backend", "chain", "epoch"]]))
    dec = pd.read_parquet("results/stage5/decisions.parquet")
    tr = dec[dec.split == "train"]
    have = pd.read_parquet("results/stage5/sim_index.parquet")
    have_exec = set(zip(*[have.loc[have.quantity == "err_true", c] for c in ["backend", "chain", "epoch"]]))
    have_pub = set(zip(*[have.loc[have.quantity == "pred_published", c] for c in ["backend", "chain", "epoch"]]))
    exec_keys = set(zip(tr.backend, tr.chain, tr.exec_epoch)) - have_exec
    pub_keys = set(zip(tr.backend, tr.chain, tr.epoch)) - have_pub
    tasks = [(f"{b}|{c}|{e}", n, b, S[sidx[(b, c, e)]], (b, c, e) in exec_keys, (b, c, e) in pub_keys, [])
             for (b, c, e) in sorted(exec_keys | pub_keys) for n in names]
    print(f"{len(tasks)} units", flush=True)
    t0, res = time.time(), []
    with get_context("spawn").Pool(workers) as pool:
        for i, r in enumerate(pool.imap_unordered(run_unit, tasks, chunksize=8), 1):
            res.append(r)
            if i % 3000 == 0:
                print(f"  {i}/{len(tasks)} {time.time() - t0:.0f}s", flush=True)
    rows, arrays = [], []
    for key, circ, out in res:
        b, c, e = key.split("|")
        for (q, d), v in out.items():
            if q in ("err_true", "pred_published"):
                rows.append(dict(backend=b, chain=int(c), epoch=int(e), circuit=circ, quantity=q, delta=d, row=len(arrays)))
                arrays.append(v)
    pd.DataFrame(rows).to_parquet(f"{OUT}/train_index.parquet")
    np.save(f"{OUT}/train_arrays.npy", np.asarray(arrays, np.float32))
    print(f"wrote {len(arrays)} arrays in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
