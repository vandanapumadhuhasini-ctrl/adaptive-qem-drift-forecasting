"""
Stage 5 simulation cache (confirmatory; corrected epochs).

Unit = (backend, chain, epoch e, circuit). Per unit, as needed:
  exec arrays   at S(e): err_true, err78, probe_draw, probe_var        (if e is an execution epoch)
  pred[published] at S(e)                                             (if e is a decision epoch)
  pred[src, k]  for src in window_mean/ridge/transformer (forecast states) and oracle = S(e+k)
Seeds: evaluation arrays depend only on (unit, δ); controller predictions use one seed
per (unit, δ) shared by every source and horizon (common random numbers).

    python3 -m experiments.stage5_simulate [--workers 12] [--history 60]
"""

import argparse
import hashlib
import time
from multiprocessing import get_context

import numpy as np
import pandas as pd

OUT = "results/stage5"
DELTAS = (0.0, 0.03)
FORECASTERS = ["window_mean", "ridge", "transformer"]


def _seed(*parts):
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:12], 16)


def _state(v):
    return dict(t1=v[0:4], t2=v[4:8], sx=v[8:12], cz=v[12:15])


_C = {}


def _circ(name):
    from experiments.stage4_simulate import _circuits
    if name not in _C:
        _C[name] = _circuits(name)
    return _C[name]


def run_unit(task):
    from src.qem.device import noise_model
    from src.qem.mitigation import PROBE_FRACTION, SHOTS, exact_values, expected_errors
    key, circ, backend, S_e, need_exec, need_pub, kjobs = task
    b, main, ach, probe, ach_p, probe_obs = _circ(circ)
    out = {}
    for d in DELTAS:
        nm = noise_model(_state(S_e), backend, d)
        vals = exact_values(main, nm, b["observables"], threads=1)
        rng_c = np.random.default_rng(_seed(key, circ, d, "controller"))
        ctrl_state = rng_c.bit_generator.state
        if need_exec:
            rng = np.random.default_rng(_seed(key, circ, d, "eval"))
            out[("err_true", d)] = expected_errors(vals, ach, b["ideal"], rng)
            out[("err78", d)] = expected_errors(vals, ach, b["ideal"], rng, shots=int(SHOTS * (1 - PROBE_FRACTION)))
            pv = exact_values(probe, nm, b["observables"], threads=1)
            rng_p = np.random.default_rng(_seed(key, circ, d, "probe"))
            out[("probe_draw", d)] = expected_errors(pv, ach_p, b["probe_ideal"], rng_p, shots=int(SHOTS * PROBE_FRACTION),
                                                     obs_idx=probe_obs, single_draw=True)
            _, out[("probe_var", d)] = expected_errors(pv, ach_p, b["probe_ideal"], rng_p,
                                                       shots=int(SHOTS * PROBE_FRACTION), obs_idx=probe_obs,
                                                       trials=50, return_var=True)
        if need_pub:
            rng_c.bit_generator.state = ctrl_state
            out[("pred_published", d)] = expected_errors(vals, ach, b["ideal"], rng_c)
        for k, states in kjobs:
            for src, v in states.items():
                vv = exact_values(main, noise_model(_state(v), backend, d), b["observables"], threads=1)
                rng_c.bit_generator.state = ctrl_state          # same random numbers for every source
                out[(f"pred_{src}_k{k}", d)] = expected_errors(vv, ach, b["ideal"], rng_c)
    return key, circ, out


def build_tasks(history):
    from src.qem.benchmarks import all_benchmarks
    em = pd.read_parquet(f"{OUT}/epoch_states.parquet")
    S = np.load(f"{OUT}/epoch_states.npy")
    em["row"] = np.arange(len(em))
    sidx = em.set_index(["backend", "chain", "epoch"])["row"]
    dec = pd.read_parquet(f"{OUT}/decisions.parquet")
    F = np.load(f"{OUT}/forecast_states.npy")
    dec["frow"] = np.arange(len(dec))

    # bandit history: training decisions whose execution lies in the last `history` training epochs
    tr = dec[dec.split == "train"]
    last_tr = tr.groupby(["backend", "chain"]).exec_epoch.transform("max")
    hist = tr[tr.exec_epoch > last_tr - history]
    ev = dec[dec.split.isin(["val", "test"])]
    use = pd.concat([hist, ev])

    exec_keys = set(zip(use.backend, use.chain, use.exec_epoch))
    pub_keys = set(zip(use.backend, use.chain, use.epoch))
    kjobs = {}
    for r in ev.itertuples():
        st = {f: F[r.frow, i] for i, f in enumerate(FORECASTERS)}
        st["oracle"] = S[sidx[(r.backend, r.chain, r.exec_epoch)]]
        kjobs.setdefault((r.backend, r.chain, r.epoch), []).append((r.k, st))
    keys = exec_keys | pub_keys
    names = [b["name"] for b in all_benchmarks()]
    tasks = []
    for (b, ch, e) in sorted(keys):
        S_e = S[sidx[(b, ch, e)]]
        for n in names:
            tasks.append((f"{b}|{ch}|{e}", n, b, S_e, (b, ch, e) in exec_keys, (b, ch, e) in pub_keys,
                          kjobs.get((b, ch, e), [])))
    return tasks


def main(workers, history):
    tasks = build_tasks(history)
    n_sims = sum(2 * (int(t[4]) * 2 + 1 + sum(len(s) for _, s in t[6])) for t in tasks)
    print(f"{len(tasks)} units, ~{n_sims} density-matrix runs, {workers} workers", flush=True)
    t0 = time.time()
    res = []
    with get_context("spawn").Pool(workers) as pool:
        for i, r in enumerate(pool.imap_unordered(run_unit, tasks, chunksize=4), 1):
            res.append(r)
            if i % 1000 == 0:
                print(f"  {i}/{len(tasks)} units, {time.time() - t0:.0f}s", flush=True)
    rows, arrays = [], []
    for key, circ, out in res:
        b, ch, e = key.split("|")
        for (name, d), v in out.items():
            rows.append(dict(backend=b, chain=int(ch), epoch=int(e), circuit=circ, quantity=name, delta=d,
                             row=len(arrays)))
            arrays.append(v)
    idx = pd.DataFrame(rows)
    idx.to_parquet(f"{OUT}/sim_index.parquet")
    np.save(f"{OUT}/sim_arrays.npy", np.asarray(arrays, np.float32))
    print(f"wrote {len(arrays)} arrays in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--history", type=int, default=60)
    a = ap.parse_args()
    main(a.workers, a.history)
