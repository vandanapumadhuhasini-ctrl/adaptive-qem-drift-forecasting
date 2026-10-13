"""
Stage 4: exact-simulation cache for every (decision, circuit) unit.

For each unit and each coherent-phase setting δ ∈ {0, 0.03}:
  err_true   expected |error| of every action at the TRUE state, full budget S
  err78      same with 7/8 S (what the bandit has left after probing)
  probe_draw one shot-noise realisation of the probe-circuit error, S/8 shots
             (bandit feedback; informative probe observables only)
  probe_var  variance of the probe estimate (the base paper's Var term)
  pred[src]  the controller's predicted expected error of every action, from the
             noise model built on state source src (published / forecasts / truth)
Common random numbers: all policies are later scored on the same err_true arrays.

    python3 -m experiments.stage4_simulate --split val [--workers 12]
"""

import argparse
import hashlib
import os
import time
from multiprocessing import get_context

import numpy as np
import pandas as pd

OUT = "results/stage4"
DELTAS = (0.0, 0.03)
SOURCES = ["published", "window_mean", "ridge", "transformer", "truth"]


def _seed(*parts):
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:12], 16)


def _state(vec):
    return dict(t1=vec[0:4], t2=vec[4:8], sx=vec[8:12], cz=vec[12:15])


_CACHE = {}


def _circuits(bench_name):
    from src.qem.benchmarks import all_benchmarks
    from src.qem.mitigation import circuit_set
    if bench_name not in _CACHE:
        b = next(x for x in all_benchmarks() if x["name"] == bench_name)
        main, ach = circuit_set(b["native"], b["observables"])
        probe, ach_p = circuit_set(b["probe"], b["observables"])
        probe_obs = [i for i, v in enumerate(b["probe_ideal"]) if abs(abs(v) - 1) < 1e-9]
        _CACHE[bench_name] = (b, main, ach, probe, ach_p, probe_obs)
    return _CACHE[bench_name]


def run_unit(task):
    from src.qem.device import noise_model
    from src.qem.mitigation import PROBE_FRACTION, SHOTS, exact_values, expected_errors
    uid, bench_name, backend, states = task[:4]
    sources = task[4] if len(task) > 4 else SOURCES
    b, main, ach, probe, ach_p, probe_obs = _circuits(bench_name)
    out = {}
    for d in DELTAS:
        truth_vals = exact_values(main, noise_model(_state(states["truth"]), backend, d), b["observables"], threads=1)
        rng = np.random.default_rng(_seed(uid, bench_name, d, "eval"))
        out[f"err_true_{d}"] = expected_errors(truth_vals, ach, b["ideal"], rng)
        out[f"err78_{d}"] = expected_errors(truth_vals, ach, b["ideal"], rng,
                                            shots=int(SHOTS * (1 - PROBE_FRACTION)))
        pv = exact_values(probe, noise_model(_state(states["truth"]), backend, d), b["observables"], threads=1)
        rng_p = np.random.default_rng(_seed(uid, bench_name, d, "probe"))
        out[f"probe_draw_{d}"] = expected_errors(pv, ach_p, b["probe_ideal"], rng_p,
                                                 shots=int(SHOTS * PROBE_FRACTION), obs_idx=probe_obs,
                                                 single_draw=True)
        _, out[f"probe_var_{d}"] = expected_errors(pv, ach_p, b["probe_ideal"], rng_p,
                                                   shots=int(SHOTS * PROBE_FRACTION), obs_idx=probe_obs,
                                                   trials=50, return_var=True)
        for src in sources:
            vals = truth_vals if src == "truth" else exact_values(
                main, noise_model(_state(states[src]), backend, d), b["observables"], threads=1)
            # common random numbers across state sources: choices differ only through the states
            rng_c = np.random.default_rng(_seed(uid, bench_name, d, "controller"))
            out[f"pred_{src}_{d}"] = expected_errors(vals, ach, b["ideal"], rng_c)
    return uid, bench_name, out


def main(split, workers, stationary, bandit_only=False):
    from src.qem.benchmarks import all_benchmarks
    from src.qem.mitigation import ACTION_NAMES
    dec = pd.read_parquet(f"{OUT}/decisions.parquet")
    st = np.load(f"{OUT}/states.npz")
    names = [b["name"] for b in all_benchmarks()]

    if stationary:   # one unit per (chain, circuit): device frozen at the training median state
        first = dec[dec.split == split].drop_duplicates(["backend", "chain"]).index
        tasks = [(f"stat_{dec.backend[i]}_{dec.chain[i]}", n, dec.backend[i],
                  {s: st["stationary"][i] for s in SOURCES}) for i in first for n in names]
        tag = f"stationary_{split}"
    else:
        idx = dec.index[dec.split == split]
        tasks = [(int(i), n, dec.backend[i],
                  {"truth": st["truth"][i], "published": st["published"][i], "window_mean": st["window_mean"][i],
                   "ridge": st["ridge"][i], "transformer": st["transformer"][i]}) for i in idx for n in names]
        tag = split
        if bandit_only:   # truth + probe arrays only (bandit online history), plus the published state
            tasks = [t + (["published"],) for t in tasks]
            tag = f"{split}_bandit"
    print(f"{len(tasks)} units ({tag}), {workers} workers", flush=True)
    t0 = time.time()
    results = []
    with get_context("spawn").Pool(workers) as pool:
        for k, r in enumerate(pool.imap_unordered(run_unit, tasks, chunksize=4), 1):
            results.append(r)
            if k % 300 == 0:
                print(f"  {k}/{len(tasks)} units, {time.time() - t0:.0f}s", flush=True)
    results.sort(key=lambda r: (str(r[0]), r[1]))
    keys = list(results[0][2])
    np.savez_compressed(
        f"{OUT}/sim_{tag}.npz", unit=np.array([str(r[0]) for r in results]),
        circuit=np.array([r[1] for r in results]), actions=np.array(ACTION_NAMES),
        **{k: np.stack([r[2][k] for r in results]) for k in keys})
    print(f"wrote {OUT}/sim_{tag}.npz in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--stationary", action="store_true")
    ap.add_argument("--bandit-only", action="store_true")
    a = ap.parse_args()
    main(a.split, a.workers, a.stationary, a.bandit_only)
