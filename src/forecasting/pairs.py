"""
Stage 5a: pair-level CZ-error series at decision-epoch resolution (real IBM records).

For each undirected coupled pair (a < b) and each chip-wide epoch e, the published
CZ error is the latest valid (non-ceiling) value of either direction with
publication time ≤ epoch time; this is the same definition the device model uses
(src/qem/device.pair_states). Windows of L past epochs predict the change in log
CZ error at epochs e + k. Split labels come from the target epoch time.
"""

import numpy as np
import pandas as pd

from src.data.windows import PROB_FLOOR
from src.qem.device import build_epochs, coupling_pairs, qubit_epoch_rows

HORIZONS = (1, 4, 7)
WINDOW = 20
FEATURES = ["log_cz", "log1p_cz_age_h", "log1p_epoch_gap_h", "pair_log_t1", "pair_log_t2", "pair_log_sx",
            "pair_ceiling_recent"]
T_VAL, T_TEST = np.datetime64("2026-08-01"), np.datetime64("2026-09-01")


def build_pair_table():
    cce = pd.read_parquet("data/processed/drift_v3/cce.parquet")
    epochs = build_epochs(cce)
    qrows = qubit_epoch_rows(cce, epochs)[["backend", "qubit", "epoch", "t1", "t2", "sx_err", "role"]]
    cz, pairs = coupling_pairs()
    cz = cz.assign(a=np.minimum(cz.qubit_a, cz.qubit_b), b=np.maximum(cz.qubit_a, cz.qubit_b))
    und = pairs.assign(a=np.minimum(pairs.qubit_a, pairs.qubit_b),
                       b=np.maximum(pairs.qubit_a, pairs.qubit_b))[["backend", "a", "b"]].drop_duplicates()
    out = []
    for bk in ["ibm_fez", "ibm_kingston", "ibm_marrakesh"]:
        ep = epochs[epochs.backend == bk].sort_values("epoch_time")
        grid = ep[["epoch", "epoch_time"]].assign(_k=1).merge(und[und.backend == bk].assign(_k=1), on="_k")
        grid = grid.drop(columns="_k").sort_values("epoch_time")
        c = cz[cz.backend == bk].sort_values("observed_time")
        valid = c[~c.is_failure_ceiling][["a", "b", "observed_time", "calibrated_time", "value"]]
        x = pd.merge_asof(grid, valid, left_on="epoch_time", right_on="observed_time", by=["a", "b"])
        ceil = c[c.is_failure_ceiling][["a", "b", "observed_time"]].rename(columns={"observed_time": "ceil_time"})
        x = pd.merge_asof(x.sort_values("epoch_time"), ceil.sort_values("ceil_time"), left_on="epoch_time",
                          right_on="ceil_time", by=["a", "b"])
        x["pair_ceiling_recent"] = ((x.epoch_time - x.ceil_time).dt.total_seconds() < 7 * 86400).astype(float)
        x["cz"] = x["value"]
        x["cz_age_h"] = (x.epoch_time - x.calibrated_time).dt.total_seconds() / 3600
        q = qrows[qrows.backend == bk]
        for side in ["a", "b"]:
            x = x.merge(q.rename(columns={"qubit": side, "t1": f"t1_{side}", "t2": f"t2_{side}",
                                          "sx_err": f"sx_{side}", "role": f"role_{side}"})
                        .drop(columns="backend"), on=[side, "epoch"], how="left")
        x["backend"] = bk
        out.append(x)
    t = pd.concat(out, ignore_index=True).sort_values(["backend", "a", "b", "epoch"]).reset_index(drop=True)
    t["log_cz"] = np.log(t["cz"].clip(lower=PROB_FLOOR))
    t["log1p_cz_age_h"] = np.log1p(t["cz_age_h"].clip(lower=0))
    g = t.groupby(["backend", "a", "b"], sort=False)
    t["epoch_gap_h"] = g["epoch_time"].diff().dt.total_seconds() / 3600
    t["log1p_epoch_gap_h"] = np.log1p(t["epoch_gap_h"].fillna(24.0))
    t["pair_log_t1"] = np.log(t[["t1_a", "t1_b"]]).mean(axis=1)
    t["pair_log_t2"] = np.log(t[["t2_a", "t2_b"]]).mean(axis=1)
    t["pair_log_sx"] = np.log(t[["sx_a", "sx_b"]].clip(lower=PROB_FLOOR)).mean(axis=1)
    t["segment"] = g["epoch_gap_h"].transform(lambda s: (s > 168).cumsum())
    # pair role: dev only if both qubits are development qubits
    role_a, role_b = g["role_a"].transform("first"), g["role_b"].transform("first")
    t["role"] = np.where(t.backend == "ibm_marrakesh", "heldout_device",
                         np.where((role_a == "dev") & (role_b == "dev"), "dev", "heldout_qubit"))
    t["target_log_cz"] = t["log_cz"].where(t["cz"].notna())
    t[FEATURES] = t.groupby(["backend", "a", "b"], sort=False)[FEATURES].ffill().fillna(0.0)
    return t


def make_pair_windows(t, window=WINDOW, horizons=HORIZONS):
    X, Y, S, LAST, meta = [], [], [], [], []
    for (bk, a, b, seg), d in t.groupby(["backend", "a", "b", "segment"], sort=False):
        F = d[FEATURES].to_numpy(np.float32)
        y = d["target_log_cz"].to_numpy(np.float32)
        times = d["epoch_time"].to_numpy("datetime64[ns]")
        ep = d["epoch"].to_numpy()
        n = len(d)
        for end in range(window - 1, n - 1):
            if not np.isfinite(y[end]):
                continue
            yy = np.full((len(horizons), 1), np.nan, np.float32)
            ss = np.full(len(horizons), -1, np.int8)
            for i, k in enumerate(horizons):
                if end + k < n and ep[end + k] == ep[end] + k:      # exactly k epochs later
                    yy[i, 0] = y[end + k]
                    ss[i] = 0 if times[end + k] < T_VAL else (1 if times[end + k] < T_TEST else 2)
            X.append(F[end - window + 1:end + 1])
            Y.append(yy)
            S.append(ss)
            LAST.append([y[end]])
            meta.append((bk, int(a), int(b), d["role"].iloc[0], int(ep[end]), times[end]))
    w = dict(X=np.asarray(X, np.float32), Y=np.asarray(Y, np.float32), split=np.asarray(S, np.int8),
             last=np.asarray(LAST, np.float32), horizons=tuple(horizons), features=list(FEATURES),
             meta=pd.DataFrame(meta, columns=["backend", "a", "b", "role", "epoch", "decision_time"]))
    # leakage checks: a training pair's target precedes T_val, so its decision does too
    for i in range(len(horizons)):
        assert np.all(w["meta"]["decision_time"].to_numpy("datetime64[ns]")[w["split"][:, i] == 0] < T_VAL)
    assert not np.isnan(w["X"]).any()
    return w
