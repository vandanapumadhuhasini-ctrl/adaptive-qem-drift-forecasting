"""
Stage 5: policy evaluation and pre-registered hypothesis tests (PROTOCOL.md §4–§5).

    python3 -m experiments.stage5_policies --split val      # tuning / selection (bandit, best fixed)
    python3 -m experiments.stage5_policies --split test     # requires configs/frozen_stage5.yaml

Selections made on validation (best fixed actions, bandit hyperparameters) are saved
to results/stage5/selections.json and *reused unchanged* for the test split.
"""

import argparse
import itertools
import json
import os

import numpy as np
import pandas as pd

from src.evaluation.stats import bootstrap_weights, holm
from src.qem.mitigation import ACTION_NAMES, ACTIONS, N_TWIRL
from src.qem.policies import COST, LinUCB, REWARD_SCALE

OUT = "results/stage5"
KS = (1, 4, 7)
REGIMES = {"R1_incoherent": (0.0, 0.0), "R2_coherent": (0.03, 0.03), "R3_mismatch": (0.03, 0.0)}
SOURCES = ["persistence", "window_mean", "ridge", "transformer", "oracle"]
GRID = dict(ctx=["full", "intercept"], eta=[0.0, 0.1, 0.3], rho=[0.9, 0.97, 1.0], lam_c=[0.0, 0.02], beta=[0.0, 1.0])
N_CIRC = np.array([len(a.scales) * (N_TWIRL if a.twirl else 1) for a in ACTIONS])
A = len(ACTIONS)


class Cache:
    def __init__(self):
        t = pd.read_parquet(f"{OUT}/sim_index.parquet")
        self.exec_keys = set(zip(*[t.loc[t.quantity == "err_true", c] for c in ["backend", "chain", "epoch"]]))
        self.dec_keys = set(zip(*[t.loc[t.quantity == "pred_published", c] for c in ["backend", "chain", "epoch"]]))
        self.idx = dict(zip(zip(t.backend, t.chain, t.epoch, t.circuit, t.quantity, t.delta), t.row))
        self.arr = np.load(f"{OUT}/sim_arrays.npy")

    def get(self, b, ch, e, circ, q, d):
        return self.arr[self.idx[(b, ch, e, circ, q, d)]]


def context(S):
    return np.array([1.0, np.log(S[0:4]).mean(), np.log(S[4:8]).mean(), np.log(S[8:12]).mean(),
                     np.log(S[12:15]).mean()])


def bandit_stream(units, cache, d_true, d_model, params, variant, k):
    """units: DataFrame of one (backend, chain, circuit) stream, sorted by decision epoch,
    with context X. Feedback for a decision arrives once its execution epoch is reached."""
    ucb = LinUCB(A, 5 if params["ctx"] == "full" else 1, eta=params["eta"], rho=params["rho"], lam_c=params["lam_c"])
    pending, rewards, out = [], [], {}
    init_buf = []
    for t, u in enumerate(units.itertuples()):
        x = u.X if params["ctx"] == "full" else u.X[:1]
        # deliver matured feedback
        ready = [p for p in pending if p[0] <= u.epoch]
        pending = [p for p in pending if p[0] > u.epoch]
        for _, kk, xx, r in ready:
            rewards.append(r)
            if len(rewards) <= A and variant != "warm":
                init_buf.append((kk, xx, r))
                if len(rewards) == A:
                    c0 = np.mean(rewards)
                    for a_, x_, r_ in init_buf:
                        ucb.update(a_, x_, r_ - c0)
            else:
                ucb.update(kk, xx, r - np.mean(rewards[:-1]) if len(rewards) > 1 else 0.0)
        if variant == "warm" and t < 10:
            w = -REWARD_SCALE * cache.get(u.backend, u.chain, u.epoch, u.circuit, "pred_published", d_model)
            ucb.pseudo_update_all(x, w - w.mean())
        arm = t if (variant != "warm" and t < A) else ucb.select(x)
        e_exec = u.exec_epoch
        if variant == "oracle":
            r = -REWARD_SCALE * cache.get(u.backend, u.chain, e_exec, u.circuit, "err_true", d_true)[arm]
            err = cache.get(u.backend, u.chain, e_exec, u.circuit, "err_true", d_true)[arm]
        else:
            r = -REWARD_SCALE * (cache.get(u.backend, u.chain, e_exec, u.circuit, "probe_draw", d_true)[arm]
                                 + params["beta"] * cache.get(u.backend, u.chain, e_exec, u.circuit, "probe_var", d_true)[arm])
            err = cache.get(u.backend, u.chain, e_exec, u.circuit, "err78", d_true)[arm]
        pending.append((e_exec, arm, x, r))
        out[u.Index] = (arm, err)
    return out


def units_for(dec, split, k, circuits, S, sidx):
    d = dec[(dec.k == k) & dec.split.isin(["train", split] if split == "val" else ["train", "val", split])]
    rows = []
    for r in d.itertuples():
        for c in circuits:
            rows.append(dict(backend=r.backend, chain=r.chain, epoch=r.epoch, exec_epoch=r.exec_epoch, k=k,
                             split=r.split, role=r.role, circuit=c, X=context(S[sidx[(r.backend, r.chain, r.epoch)]])))
    return pd.DataFrame(rows)


def evaluate(split):
    from src.qem.benchmarks import all_benchmarks
    circuits = [b["name"] for b in all_benchmarks()]
    cache = Cache()
    dec = pd.read_parquet(f"{OUT}/decisions.parquet")
    em = pd.read_parquet(f"{OUT}/epoch_states.parquet")
    S = np.load(f"{OUT}/epoch_states.npy")
    sidx = pd.Series(np.arange(len(em)), index=pd.MultiIndex.from_frame(em[["backend", "chain", "epoch"]]))
    # restrict the bandit history to training decisions that were simulated (last 60 epochs)

    sel_path = f"{OUT}/selections.json"
    selections = json.load(open(sel_path)) if split == "test" else {}
    rows = []
    for k in KS:
        U = units_for(dec, split, k, circuits, S, sidx)
        # keep units whose decision epoch has controller predictions and whose execution
        # epoch has execution arrays (bandit history is limited to the simulated window)
        U = U[[(b, c, e) in cache.dec_keys and (b, c, x) in cache.exec_keys
               for b, c, e, x in zip(U.backend, U.chain, U.epoch, U.exec_epoch)]]
        U = U.reset_index(drop=True)
        ev = U[U.split == split]
        if ev.empty:
            continue
        for regime, (dt, dm) in REGIMES.items():
            key = f"k{k}/{regime}"
            E = np.stack([cache.get(r.backend, r.chain, r.exec_epoch, r.circuit, "err_true", dt) for r in ev.itertuples()])
            pol, arms = {"none": E[:, ACTION_NAMES.index("none")]}, {}
            if split == "val":
                dev = (ev.role == "dev").to_numpy()
                bf = int(np.argmin(E[dev].mean(0)))
                bfc = {c: int(np.argmin(E[dev & (ev.circuit == c).to_numpy()].mean(0))) for c in circuits}
                selections[key] = dict(best_fixed=ACTION_NAMES[bf], best_fixed_per_circuit={c: ACTION_NAMES[a] for c, a in bfc.items()})
            bf = ACTION_NAMES.index(selections[key]["best_fixed"])
            bfc = {c: ACTION_NAMES.index(a) for c, a in selections[key]["best_fixed_per_circuit"].items()}
            pol["best_fixed"], arms["best_fixed"] = E[:, bf], np.full(len(E), bf)
            a_c = np.array([bfc[c] for c in ev.circuit])
            pol["best_fixed_per_circuit"], arms["best_fixed_per_circuit"] = E[np.arange(len(E)), a_c], a_c
            for src in SOURCES:
                q = "pred_published" if src == "persistence" else f"pred_{src}_k{k}"
                P = np.stack([cache.get(r.backend, r.chain, r.epoch, r.circuit, q, dm) for r in ev.itertuples()])
                ch = P.argmin(1)
                pol[f"adaptive_{src}"], arms[f"adaptive_{src}"] = E[np.arange(len(E)), ch], ch
            for variant in ["probe", "warm", "oracle"]:
                if split == "val":
                    best = None
                    for vals in itertools.product(*GRID.values()):
                        p = dict(zip(GRID, vals))
                        if variant == "oracle" and p["beta"]:
                            continue
                        res = {}
                        for _, s in U.groupby(["backend", "chain", "circuit"]):
                            res.update(bandit_stream(s.sort_values("epoch"), cache, dt, dm, p, variant, k))
                        err = np.array([res[i][1] for i in ev.index])
                        score = err[(ev.role == "dev").to_numpy()].mean()
                        if best is None or score < best[0]:
                            best = (score, p)
                    selections[key][f"cmab_{variant}"] = best[1]
                p = selections[key][f"cmab_{variant}"]
                res = {}
                for _, s in U.groupby(["backend", "chain", "circuit"]):
                    res.update(bandit_stream(s.sort_values("epoch"), cache, dt, dm, p, variant, k))
                pol[f"cmab_{variant}"] = np.array([res[i][1] for i in ev.index])
                arms[f"cmab_{variant}"] = np.array([res[i][0] for i in ev.index])
            for name, err in pol.items():
                a = arms.get(name)
                for j, r in enumerate(ev.itertuples()):
                    rows.append(dict(k=k, regime=regime, policy=name, backend=r.backend, chain=r.chain, epoch=r.epoch,
                                     circuit=r.circuit, role=r.role, error=float(err[j]),
                                     action=None if a is None else ACTION_NAMES[int(a[j])],
                                     cost=np.nan if a is None else COST[int(a[j])],
                                     n_circuits=np.nan if a is None else N_CIRC[int(a[j])]))
            print(f"done {split} {key}: {len(ev)} units", flush=True)
    df = pd.DataFrame(rows)
    df.to_parquet(f"{OUT}/policy_{split}.parquet")
    if split == "val":
        with open(sel_path, "w") as f:
            json.dump(selections, f, indent=2)
    return df


def paired(df_cell, a, b, B=10_000, seed=0):
    piv = df_cell.pivot_table(index=["backend", "chain", "epoch", "circuit", "regime"], columns="policy", values="error")
    cl = piv.index.get_level_values(0) + ":" + piv.index.get_level_values(1).astype(str)
    clusters, inv = np.unique(cl, return_inverse=True)
    W = bootstrap_weights(np.array([c.split(":")[0] for c in clusters]), B, seed)
    cnt = np.bincount(inv, minlength=len(clusters))

    def boot(v):
        s = np.bincount(inv, weights=v, minlength=len(clusters))
        return v.mean(), (W @ s) / (W @ cnt)

    m, bt = boot((piv[a] - piv[b]).to_numpy())
    out = dict(mean_diff=float(m), ci=[float(np.percentile(bt, 2.5)), float(np.percentile(bt, 97.5))],
               p_one_sided=float(max((bt >= 0).mean(), 1 / B)), pct=float(100 * m / piv[b].mean()),
               n_units=int(len(piv)), n_chains=int(len(clusters)))
    if {"best_fixed", "adaptive_oracle"} <= set(piv):
        g, gb = boot((piv["best_fixed"] - piv["adaptive_oracle"]).to_numpy())
        ca = (piv["best_fixed"] - piv[a]).to_numpy()
        cb = (piv["best_fixed"] - piv[b]).to_numpy()
        _, cab = boot(ca)
        _, cbb = boot(cb)
        out["oracle_gain"] = dict(mean=float(g), ci=[float(np.percentile(gb, 2.5)), float(np.percentile(gb, 97.5))])
        out["captured_a"], out["captured_b"] = float(ca.mean() / g), float(cb.mean() / g)
        cdiff = (cab - cbb) / gb
        out["captured_diff"] = float(ca.mean() / g - cb.mean() / g)
        out["captured_diff_ci"] = [float(np.nanpercentile(cdiff, 2.5)), float(np.nanpercentile(cdiff, 97.5))]
    return out


def hypotheses(df, split):
    res = {}
    real = df[df.regime.isin(["R1_incoherent", "R2_coherent"])]
    # H2 primary: k = 1, real drift pooled, development chains
    c = real[(real.k == 1) & (real.role == "dev")]
    h2 = paired(c, "adaptive_transformer", "adaptive_persistence")
    g_ok = h2["oracle_gain"]["ci"][0] > 0
    h2["statistical"] = h2["ci"][1] < 0
    h2["practical"] = (h2["captured_diff"] >= 0.10) if g_ok else None
    h2["evaluable"] = g_ok
    h2["success"] = bool(h2["statistical"] and g_ok and h2["practical"])
    res["H2_primary_k1_dev"] = h2
    # secondary families (Holm)
    fams = {"H2_secondary": [("adaptive_transformer", "adaptive_persistence", k, "dev") for k in (4, 7)]
                            + [("adaptive_transformer", "adaptive_persistence", 1, "heldout_device")],
            "H3a_vs_best_fixed": [("adaptive_transformer", "best_fixed", k, "dev") for k in KS],
            "H4_vs_cmab_probe": [("adaptive_transformer", "cmab_probe", k, "dev") for k in KS],
            "vs_window_mean": [("adaptive_transformer", "adaptive_window_mean", k, "dev") for k in KS],
            "vs_ridge": [("adaptive_transformer", "adaptive_ridge", k, "dev") for k in KS],
            "forecast_vs_persistence_any": [(f"adaptive_{s}", "adaptive_persistence", k, "dev")
                                            for s in ["window_mean", "ridge"] for k in KS]}
    for fam, cells in fams.items():
        out = []
        for a, b, k, role in cells:
            c = real[(real.k == k) & (real.role == role)]
            if c.empty or a not in set(c.policy) or b not in set(c.policy):
                continue
            r = paired(c, a, b)
            r.update(a=a, b=b, k=k, role=role)
            out.append(r)
        rej = holm([r["p_one_sided"] for r in out]) if out else []
        for r, x in zip(out, rej):
            r["holm_reject"] = bool(x)
        res[fam] = out
    # per-regime table
    res["by_regime"] = {}
    for (k, regime, role), c in df.groupby(["k", "regime", "role"]):
        res["by_regime"][f"k{k}/{regime}/{role}"] = {p: float(v) for p, v in c.groupby("policy").error.mean().items()}
    with open(f"{OUT}/hypotheses_{split}.json", "w") as f:
        json.dump(res, f, indent=2)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "test"], default="val")
    a = ap.parse_args()
    if a.split == "test":
        from experiments.stage5_freeze import check_frozen
        check_frozen()
    df = evaluate(a.split)
    r = hypotheses(df, a.split)
    h = r["H2_primary_k1_dev"]
    print(f"H2 primary ({a.split}): diff {h['mean_diff']:+.6f} ({h['pct']:+.2f}%) CI {h['ci']} "
          f"captured diff {h.get('captured_diff')} evaluable {h['evaluable']} success {h['success']}")
