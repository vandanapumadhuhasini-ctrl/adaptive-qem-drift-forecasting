"""
Robustness simulation caches (docs/ROBUSTNESS_PLAN.md), same format as results/stage5.

    python3 -m experiments.robust_sim ra     # 8 new 4-qubit circuit instances, k = 1, with bandit inputs
    python3 -m experiments.robust_sim rb     # 6-qubit chains (HEA d4, TFIM d3), k = 1, no bandit
"""

import argparse
import hashlib
import os
import time
from multiprocessing import get_context

import numpy as np
import pandas as pd

DELTAS = (0.0, 0.03)
FORECASTERS = ["window_mean", "ridge", "transformer"]
CIRCUITS = {"ra": [(f, d, 4, s) for f, ds in [("hea", (4, 8)), ("qaoa", (2, 4))] for d in ds for s in (12, 13)],
            "rb": [("hea", 4, 6, 11), ("tfim", 3, 6, 11)]}
SRC = {"ra": "results/stage5", "rb": "results/robustness/rb"}
OUT = {"ra": "results/robustness/ra", "rb": "results/robustness/rb"}


def _seed(*parts):
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:12], 16)


_C = {}


def _circ(spec):
    from src.qem.mitigation import circuit_set
    from src.qem.nq import benchmark_n
    if spec not in _C:
        b = benchmark_n(*spec)
        main, ach = circuit_set(b["native"], b["observables"])
        probe, ach_p = circuit_set(b["probe"], b["observables"])
        probe_obs = [i for i, v in enumerate(b["probe_ideal"]) if abs(abs(v) - 1) < 1e-9]
        _C[spec] = (b, main, ach, probe, ach_p, probe_obs)
    return _C[spec]


def _state(v, n):
    return dict(t1=v[0:n], t2=v[n:2 * n], sx=v[2 * n:3 * n], cz=v[3 * n:4 * n - 1])


def run_unit(task):
    from src.qem.mitigation import PROBE_FRACTION, SHOTS, exact_values, expected_errors
    from src.qem.nq import noise_model_n
    key, spec, backend, S_e, need_exec, need_pub, kjobs, with_probe = task
    n = spec[2]
    b, main, ach, probe, ach_p, probe_obs = _circ(spec)
    name, out = b["name"], {}
    for d in DELTAS:
        nm = noise_model_n(_state(S_e, n), backend, d)
        vals = exact_values(main, nm, b["observables"], threads=1)
        rng_c = np.random.default_rng(_seed(key, name, d, "controller"))
        cst = rng_c.bit_generator.state
        if need_exec:
            rng = np.random.default_rng(_seed(key, name, d, "eval"))
            out[("err_true", d)] = expected_errors(vals, ach, b["ideal"], rng)
            if with_probe:
                out[("err78", d)] = expected_errors(vals, ach, b["ideal"], rng, shots=int(SHOTS * (1 - PROBE_FRACTION)))
                pv = exact_values(probe, nm, b["observables"], threads=1)
                rng_p = np.random.default_rng(_seed(key, name, d, "probe"))
                out[("probe_draw", d)] = expected_errors(pv, ach_p, b["probe_ideal"], rng_p,
                                                         shots=int(SHOTS * PROBE_FRACTION), obs_idx=probe_obs,
                                                         single_draw=True)
                _, out[("probe_var", d)] = expected_errors(pv, ach_p, b["probe_ideal"], rng_p,
                                                           shots=int(SHOTS * PROBE_FRACTION), obs_idx=probe_obs,
                                                           trials=50, return_var=True)
        if need_pub:
            rng_c.bit_generator.state = cst
            out[("pred_published", d)] = expected_errors(vals, ach, b["ideal"], rng_c)
        for k, states in kjobs:
            for src, v in states.items():
                vv = exact_values(main, noise_model_n(_state(v, n), backend, d), b["observables"], threads=1)
                rng_c.bit_generator.state = cst
                out[(f"pred_{src}_k{k}", d)] = expected_errors(vv, ach, b["ideal"], rng_c)
    return key, name, out


def build_tasks(which, history=60):
    src = SRC[which]
    em = pd.read_parquet(f"{src}/epoch_states.parquet")
    S = np.load(f"{src}/epoch_states.npy")
    sidx = pd.Series(np.arange(len(em)), index=pd.MultiIndex.from_frame(em[["backend", "chain", "epoch"]]))
    dec = pd.read_parquet(f"{src}/decisions.parquet")
    F = np.load(f"{src}/forecast_states.npy")
    dec["frow"] = np.arange(len(dec))
    dec = dec[dec.k == 1]
    ev = dec[dec.split.isin(["val", "test"])]
    use = ev
    if which == "ra":
        tr = dec[dec.split == "train"]
        last = tr.groupby(["backend", "chain"]).exec_epoch.transform("max")
        use = pd.concat([tr[tr.exec_epoch > last - history], ev])
    exec_keys = set(zip(use.backend, use.chain, use.exec_epoch))
    pub_keys = set(zip(use.backend, use.chain, use.epoch))
    kjobs = {}
    for r in ev.itertuples():
        st = {f: F[r.frow, i] for i, f in enumerate(FORECASTERS)}
        st["oracle"] = S[sidx[(r.backend, r.chain, r.exec_epoch)]]
        kjobs.setdefault((r.backend, r.chain, r.epoch), []).append((1, st))
    tasks = []
    for (b, ch, e) in sorted(exec_keys | pub_keys):
        for spec in CIRCUITS[which]:
            tasks.append((f"{b}|{ch}|{e}", spec, b, S[sidx[(b, ch, e)]], (b, ch, e) in exec_keys,
                          (b, ch, e) in pub_keys, kjobs.get((b, ch, e), []), which == "ra"))
    return tasks


def main(which, workers):
    os.makedirs(OUT[which], exist_ok=True)
    tasks = build_tasks(which)
    print(f"{which}: {len(tasks)} units, {workers} workers", flush=True)
    t0, res = time.time(), []
    with get_context("spawn").Pool(workers) as pool:
        for i, r in enumerate(pool.imap_unordered(run_unit, tasks, chunksize=4), 1):
            res.append(r)
            if i % 2000 == 0:
                print(f"  {i}/{len(tasks)} units, {time.time() - t0:.0f}s", flush=True)
    rows, arrays = [], []
    for key, circ, out in res:
        b, ch, e = key.split("|")
        for (q, d), v in out.items():
            rows.append(dict(backend=b, chain=int(ch), epoch=int(e), circuit=circ, quantity=q, delta=d, row=len(arrays)))
            arrays.append(v)
    pd.DataFrame(rows).to_parquet(f"{OUT[which]}/sim_index.parquet")
    np.save(f"{OUT[which]}/sim_arrays.npy", np.asarray(arrays, np.float32))
    print(f"wrote {len(arrays)} arrays in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["ra", "rb"])
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    main(a.which, a.workers)
