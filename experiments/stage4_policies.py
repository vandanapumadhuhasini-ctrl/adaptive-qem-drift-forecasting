"""
Stage 4: evaluate all policies on the simulation cache (validation period only).

    python3 -m experiments.stage4_policies --split val

Regimes: R1 incoherent (truth δ=0, model δ=0), R2 coherent (0.03, 0.03),
R3 mismatch (truth 0.03, model 0), R4 stationary control (δ=0, frozen state).
Writes results/stage4/policy_{split}.csv (per unit × policy), summary_{split}.json
and cmab_tuning.json.
"""

import argparse
import itertools
import json

import numpy as np
import pandas as pd

from src.evaluation.stats import bootstrap_weights
from src.qem.mitigation import ACTION_NAMES, ACTIONS, N_TWIRL
from src.qem.policies import COST, controller_choice, run_bandit_stream

OUT = "results/stage4"
REGIMES = {"R1_incoherent": (0.0, 0.0), "R2_coherent": (0.03, 0.03), "R3_mismatch": (0.03, 0.0)}
SOURCES = {"persistence": "published", "window_mean": "window_mean", "ridge": "ridge",
           "transformer": "transformer", "oracle": "truth"}
# LinUCB grid (validation only). Context: full = [1, z(log T1, T2, sx, CZ)]; intercept = [1].
# All variants use play-each-arm-once initialisation and causal reward centring (src/qem/policies.py).
CMAB_GRID = dict(ctx=["full", "intercept"], eta=[0.0, 0.1, 0.3, 1.0], rho=[0.9, 0.97, 1.0],
                 lam_c=[0.0, 0.02], beta=[0.0, 1.0])
N_CIRCUITS = np.array([len(a.scales) * (N_TWIRL if a.twirl else 1) for a in ACTIONS])


def load(tag):
    z = np.load(f"{OUT}/sim_{tag}.npz")
    return {k: z[k] for k in z.files}


def contexts(dec, states, train_mask):
    """Bandit context from the PUBLISHED state: [1, z(mean log T1, T2, sx, CZ)]."""
    P = states["published"]
    feats = np.stack([np.log(P[:, 0:4]).mean(1), np.log(P[:, 4:8]).mean(1),
                      np.log(P[:, 8:12]).mean(1), np.log(P[:, 12:15]).mean(1)], axis=1)
    mu, sd = feats[train_mask].mean(0), feats[train_mask].std(0)
    return np.hstack([np.ones((len(P), 1)), (feats - mu) / sd])


def bandit_streams(dec, X, sims, d_true, d_model, params, variant):
    """Run every (backend, chain, circuit) stream over train + split epochs; return per-unit
    (decision index, circuit) -> (arm, error)."""
    out = {}
    units = pd.DataFrame({"dec": sims["dec"], "circuit": sims["circuit"], "row": np.arange(len(sims["dec"]))})
    units = units.merge(dec[["backend", "chain", "epoch"]], left_on="dec", right_index=True)
    for (b, ch, circ), u in units.sort_values("epoch").groupby(["backend", "chain", "circuit"]):
        r = u["row"].to_numpy()
        Xs = X[u["dec"].to_numpy()]
        if params.get("ctx", "full") == "intercept":
            Xs = Xs[:, :1]
        arms, errs = run_bandit_stream(
            Xs, sims[f"probe_draw_{d_true}"][r], sims[f"probe_var_{d_true}"][r],
            sims[f"err_true_{d_true}"][r] if variant == "oracle" else sims[f"err78_{d_true}"][r],
            params, variant=variant, oracle_err=sims[f"err_true_{d_true}"][r],
            warm_pred=sims[f"pred_published_{d_model}"][r])
        for dd, c, a, e in zip(u["dec"], [circ] * len(u), arms, errs):
            out[(int(dd), c)] = (int(a), float(e))
    return out


def merge_sims(*parts):
    keys = set.intersection(*[set(p) for p in parts])
    return {k: np.concatenate([p[k] for p in parts]) for k in keys}


def main(split):
    dec = pd.read_parquet(f"{OUT}/decisions.parquet")
    states = np.load(f"{OUT}/states.npz")
    X = contexts(dec, states, (dec.split == "train").to_numpy())

    ev = load(split)
    ev["dec"] = ev["unit"].astype(int)
    hist = load("train_bandit")
    hist["dec"] = hist["unit"].astype(int)
    stream = merge_sims(hist, ev)                       # bandit sees train then split epochs
    ev_dec = dec.loc[ev["dec"]].reset_index(drop=True)
    in_dev = (ev_dec.role == "dev").to_numpy()

    rows, tuning = [], {}
    for regime, (dt, dm) in REGIMES.items():
        E = ev[f"err_true_{dt}"]
        pol = {"none": E[:, ACTION_NAMES.index("none")]}
        # best fixed: global and per circuit, both chosen on this (validation) split
        bf = int(np.argmin(E[in_dev].mean(0)))
        pol["best_fixed"] = E[:, bf]
        per_c = {c: int(np.argmin(E[in_dev & (ev["circuit"] == c)].mean(0))) for c in np.unique(ev["circuit"])}
        pol["best_fixed_per_circuit"] = np.array([E[i, per_c[c]] for i, c in enumerate(ev["circuit"])])
        choice = {}
        for name, src in SOURCES.items():
            ch = controller_choice(ev[f"pred_{src}_{dm}"])
            choice[name] = ch
            pol[f"adaptive_{name}"] = E[np.arange(len(E)), ch]

        # CMAB: tune on this split (mean over dev units), per variant
        res = {}
        for variant in ["probe", "warm", "oracle"]:
            best = None
            for vals in itertools.product(*CMAB_GRID.values()):
                p = dict(zip(CMAB_GRID, vals))
                if variant == "oracle" and p["beta"] != 0.0:
                    continue
                out = bandit_streams(dec, X, stream, dt, dm, p, variant)
                e = np.array([out[(d, c)][1] for d, c in zip(ev["dec"], ev["circuit"])])
                score = e[in_dev].mean()
                if best is None or score < best[0]:
                    arms = np.array([out[(d, c)][0] for d, c in zip(ev["dec"], ev["circuit"])])
                    best = (score, p, e, arms)
            res[variant] = best
            pol[f"cmab_{variant}"] = best[2]
            choice[f"cmab_{variant}"] = best[3]
        tuning[regime] = {v: dict(params=res[v][1], val_mean=float(res[v][0])) for v in res}

        for name, err in pol.items():
            arm = choice.get(name.replace("adaptive_", ""), None)
            if name in ("best_fixed",):
                arm = np.full(len(E), bf)
            for i in range(len(E)):
                a = None if arm is None else int(arm[i])
                rows.append(dict(regime=regime, policy=name, dec=int(ev["dec"][i]), circuit=ev["circuit"][i],
                                 backend=ev_dec.backend[i], chain=int(ev_dec.chain[i]), role=ev_dec.role[i],
                                 error=float(err[i]), action=None if a is None else ACTION_NAMES[a],
                                 cost=np.nan if a is None else COST[a],
                                 n_circuits=np.nan if a is None else N_CIRCUITS[a]))

    # R4 stationary control (no drift by construction)
    st = load(f"stationary_{split}")
    E = st["err_true_0.0"]
    for name, src in SOURCES.items():
        ch = controller_choice(st[f"pred_{src}_0.0"])
        for i in range(len(E)):
            rows.append(dict(regime="R4_stationary", policy=f"adaptive_{name}", dec=-1, circuit=st["circuit"][i],
                             backend=st["unit"][i].split("_")[1] + "_" + st["unit"][i].split("_")[2],
                             chain=int(st["unit"][i].split("_")[-1]), role="dev", error=float(E[i, ch[i]]),
                             action=ACTION_NAMES[ch[i]], cost=COST[ch[i]], n_circuits=N_CIRCUITS[ch[i]]))
    bf = int(np.argmin(E.mean(0)))
    for i in range(len(E)):
        rows.append(dict(regime="R4_stationary", policy="best_fixed", dec=-1, circuit=st["circuit"][i],
                         backend=st["unit"][i].split("_")[1] + "_" + st["unit"][i].split("_")[2],
                         chain=int(st["unit"][i].split("_")[-1]), role="dev", error=float(E[i, bf]),
                         action=ACTION_NAMES[bf], cost=COST[bf], n_circuits=N_CIRCUITS[bf]))

    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/policy_{split}.csv", index=False)
    with open(f"{OUT}/cmab_tuning.json", "w") as f:
        json.dump(tuning, f, indent=2)
    summarise(df, split)


def summarise(df, split, B=10_000):
    out = {}
    comps = [("adaptive_transformer", "adaptive_persistence"), ("adaptive_transformer", "adaptive_window_mean"),
             ("adaptive_transformer", "adaptive_ridge"), ("adaptive_transformer", "best_fixed"),
             ("adaptive_transformer", "cmab_probe"), ("adaptive_transformer", "cmab_warm"),
             ("adaptive_window_mean", "adaptive_persistence"), ("adaptive_persistence", "best_fixed"),
             ("adaptive_oracle", "best_fixed"), ("cmab_probe", "best_fixed")]
    for (regime, role), d in df.groupby(["regime", "role"]):
        piv = d.pivot_table(index=["backend", "chain", "dec", "circuit"], columns="policy", values="error")
        cl = piv.index.get_level_values(0) + ":" + piv.index.get_level_values(1).astype(str)
        clusters, inv = np.unique(cl, return_inverse=True)
        backends = np.array([c.split(":")[0] for c in clusters])
        W = bootstrap_weights(backends, B, 0)
        cnt = np.bincount(inv, minlength=len(clusters))

        def boot_mean(v):
            s = np.bincount(inv, weights=v, minlength=len(clusters))
            return float(v.mean()), (W @ s) / (W @ cnt)

        entry = dict(n_units=len(piv), n_chains=len(clusters), mean_error={}, comparisons={})
        for p in piv.columns:
            m, _ = boot_mean(piv[p].to_numpy())
            entry["mean_error"][p] = m
        for a, b in comps:
            if a in piv and b in piv:
                m, bt = boot_mean((piv[a] - piv[b]).to_numpy())
                entry["comparisons"][f"{a} - {b}"] = dict(
                    mean_diff=m, ci=[float(np.percentile(bt, 2.5)), float(np.percentile(bt, 97.5))],
                    pct=100 * m / piv[b].mean())
        if {"best_fixed", "adaptive_oracle"} <= set(piv):
            G = (piv["best_fixed"] - piv["adaptive_oracle"]).to_numpy()
            gm, gb = boot_mean(G)
            entry["oracle_gain"] = dict(mean=gm, ci=[float(np.percentile(gb, 2.5)), float(np.percentile(gb, 97.5))])
            entry["captured_fraction"] = {p: float((piv["best_fixed"] - piv[p]).mean() / G.mean())
                                          for p in piv.columns if G.mean() != 0}
        out[f"{regime}/{role}"] = entry
    costs = df.groupby(["regime", "policy"])[["cost", "n_circuits"]].mean().round(3)
    out["cost"] = {f"{r}/{p}": v for (r, p), v in costs.to_dict(orient="index").items()}
    with open(f"{OUT}/summary_{split}.json", "w") as f:
        json.dump(out, f, indent=2, default=float)
    for key, e in out.items():
        if key == "cost":
            continue
        print(f"\n[{key}] units={e['n_units']} chains={e['n_chains']}")
        for p, v in sorted(e["mean_error"].items(), key=lambda kv: kv[1]):
            print(f"   {p:28s} {v:.5f}")
        for k, c in e["comparisons"].items():
            print(f"   {k:52s} {c['mean_diff']:+.5f} [{c['ci'][0]:+.5f},{c['ci'][1]:+.5f}] ({c['pct']:+.1f}%)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    main(ap.parse_args().split)
