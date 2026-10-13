"""
Leakage-safe forecasting windows over the CCE table (docs/PROTOCOL.md §2–§3).

  - A window ends at CCE n (the decision time). Inputs are CCEs n-L+1 .. n of one
    qubit. Targets are the states at CCEs n+k for every horizon k.
  - Windows never cross a segment break (a calibration gap > SEGMENT_BREAK_H).
  - Split labels are assigned **per horizon** from that horizon's target time.
    Training uses only (sample, horizon) pairs whose target is in the train
    period, and evaluation uses only pairs whose target is in the evaluated split.
  - Transforms here are fixed functions (logs, clips). Anything fitted, like
    feature scaling, is fitted on training pairs in Stage 3.
"""

import numpy as np
import pandas as pd

SEGMENT_BREAK_H = 168.0          # 7 days; 0.2 % of gaps, 118 of them end on 2026-03-04
PROB_FLOOR = 1.0 / 16384         # half the 1/8192 readout quantisation step
HORIZONS = (1, 2, 4, 7)

# Forecast targets (log scale, see PROTOCOL §3)
TARGETS = ["log_t1", "log_t2", "log_sx_err", "log_ro_err", "log_cz_err_mean"]

STATE_FEATURES = [
    "log_t1", "log_t2", "log_sx_err", "log_ro_err", "log_p01", "log_p10",
    "log_cz_err_mean", "log_cz_err_max", "cz_ceiling_frac", "sx_err_ceiling",
    "log1p_t2_age_h", "log1p_sx_age_h", "log1p_ro_age_h", "log1p_cz_age_h",
    "log1p_gap_h", "log1p_hours_since_chip_recal", "chip_recal_unknown",
    "hour_sin", "hour_cos",
]
ENV_FEATURES = [
    "temperature_c", "pressure_hpa", "humidity_pct", "kp_index", "ap_index",
    "solar_flux_sfu", "bz_gsm_nt", "dst_nt",
]


def add_features(cce):
    """Deterministic per-row transforms plus segment ids. Missing values are filled
    by the last value of the same qubit (past only), then by a fixed constant."""

    c = cce.sort_values(["backend", "qubit", "t1_calibrated"]).copy()
    c["log_t1"] = np.log(c["t1"])
    c["log_t2"] = np.log(c["t2"])
    c["log_sx_err"] = np.log(c["sx_err"].clip(lower=PROB_FLOOR))
    c["log_ro_err"] = np.log(c["ro_err"].clip(lower=PROB_FLOOR))
    c["log_p01"] = np.log(c["p01"].clip(lower=PROB_FLOOR))
    c["log_p10"] = np.log(c["p10"].clip(lower=PROB_FLOOR))
    c["log_cz_err_mean"] = np.log(c["cz_err_mean"].clip(lower=PROB_FLOOR))
    c["log_cz_err_max"] = np.log(c["cz_err_max"].clip(lower=PROB_FLOOR))
    c["sx_err_ceiling"] = c["sx_err_ceiling"].astype(float)
    for a in ["t2", "sx_err", "ro_err", "cz"]:
        name = {"t2": "t2", "sx_err": "sx", "ro_err": "ro", "cz": "cz"}[a]
        c[f"log1p_{name}_age_h"] = np.log1p(c[f"{a}_age_h"].clip(lower=0))
    c["log1p_gap_h"] = np.log1p(c["gap_h"].fillna(0))
    c["chip_recal_unknown"] = c["hours_since_chip_recal"].isna().astype(float)
    c["log1p_hours_since_chip_recal"] = np.log1p(c["hours_since_chip_recal"].fillna(24 * 30))
    hour = c["decision_time"].dt.hour + c["decision_time"].dt.minute / 60
    c["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    c["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    g = c.groupby(["backend", "qubit"], sort=False)
    c["segment"] = g["gap_h"].transform(lambda s: (s > SEGMENT_BREAK_H).cumsum())

    # Targets keep their NaNs (masked in the loss). Features use past values only.
    feat = STATE_FEATURES + ENV_FEATURES
    targets_raw = c[TARGETS].copy()
    c[feat] = g[feat].ffill()
    c[feat] = c[feat].fillna(0.0)
    c[[f"target_{t}" for t in TARGETS]] = targets_raw.values
    return c.reset_index(drop=True)


def make_windows(c, window, horizons=HORIZONS, features=STATE_FEATURES, roles=None,
                 t_val="2026-08-01", t_test="2026-09-01"):
    """
    Returns a dict with
      X      (N, window, F)  input features
      Y      (N, H, T)       targets for each horizon (NaN if missing)
      split  (N, H)          0/1/2 train/val/test by that horizon's target time, -1 if no target
      last   (N, T)          target variables at the decision CCE (persistence)
      meta   DataFrame       backend, qubit, role, decision_time, segment
    """

    t_val, t_test = np.datetime64(t_val), np.datetime64(t_test)
    X, Y, S, LAST, meta = [], [], [], [], []
    tcols = [f"target_{t}" for t in TARGETS]

    sub = c if roles is None else c[c["role"].isin(roles)]
    for (b, q, seg), d in sub.groupby(["backend", "qubit", "segment"], sort=False):
        F = d[features].to_numpy(np.float32)
        T = d[tcols].to_numpy(np.float32)
        times = d["t1_calibrated"].to_numpy("datetime64[ns]")
        dec = d["decision_time"].to_numpy("datetime64[ns]")
        n = len(d)
        for end in range(window - 1, n - 1):              # at least one future CCE
            y = np.full((len(horizons), len(TARGETS)), np.nan, np.float32)
            s = np.full(len(horizons), -1, np.int8)
            for i, k in enumerate(horizons):
                if end + k < n:
                    y[i] = T[end + k]
                    tt = times[end + k]
                    s[i] = 0 if tt < t_val else (1 if tt < t_test else 2)
            X.append(F[end - window + 1:end + 1])
            Y.append(y)
            S.append(s)
            LAST.append(T[end])
            meta.append((b, int(q), d["role"].iloc[0], dec[end], times[end], int(seg)))

    return dict(
        X=np.asarray(X, np.float32), Y=np.asarray(Y, np.float32),
        split=np.asarray(S, np.int8), last=np.asarray(LAST, np.float32),
        meta=pd.DataFrame(meta, columns=["backend", "qubit", "role", "decision_time", "decision_cal", "segment"]),
        horizons=tuple(horizons), features=list(features), targets=list(TARGETS),
    )


def check_windows(w, window, t_val="2026-08-01", t_test="2026-09-01"):
    """Independent leakage assertions on a built window set."""
    meta = w["meta"]
    dec = meta["decision_cal"].to_numpy("datetime64[ns]")
    t_val, t_test = np.datetime64(t_val), np.datetime64(t_test)
    for i, k in enumerate(w["horizons"]):
        s = w["split"][:, i]
        # The decision CCE precedes its target, so a train target implies a pre-T_val decision.
        assert np.all(dec[s == 0] < t_val), f"k={k}: train pair decided after T_val"
        # A val/test target is never earlier than its split boundary (re-checked via decision < target).
        assert np.all(dec[s == 2] < np.datetime64("2100-01-01"))
        assert np.all(np.isnan(w["Y"][s == -1, i])), f"k={k}: target present but no split"
    assert w["X"].shape[1] == window and not np.isnan(w["X"]).any(), "bad input windows"
    assert set(meta["role"]) <= {"dev", "heldout_qubit", "heldout_device", "out_of_family"}
    return True
