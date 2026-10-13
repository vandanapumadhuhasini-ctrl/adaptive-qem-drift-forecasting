"""
Stage 2: per-qubit calibration-state table from the full IBM calibration-drift
dataset (real hardware calibration records). See docs/PROTOCOL.md §1–§2.

Pipeline
  1. Read the revision-pinned parquet (only the needed property families).
  2. Harmonise units: T1/T2 rows with NULL unit are seconds -> microseconds.
  3. Keep genuine measurements only: an event counts if its value differs
     from the previous event of the same (backend, property, qubit_a, qubit_b)
     series. Re-timestamped unchanged values are not new observations.
  4. One row per coherence-calibration event (CCE = a T1 measurement). Every
     other property is attached *as published at the decision time*
     (as-of join on observed_time), so no future information enters a row.
  5. Assign qubit / backend roles and write the table plus a count report.

Run:  python3 -m src.data.drift_dataset
"""

import json
import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

CONFIG_PATH = "configs/drift_data.yaml"

STATE_PROPS = {
    # family -> output column prefix
    "T2": "t2",
    "sx_error": "sx_err",
    "readout_error": "ro_err",
    "prob_meas0_prep1": "p01",
    "prob_meas1_prep0": "p10",
}

RAW_COLUMNS = [
    "backend", "property_family", "qubit_a", "qubit_b", "value", "unit",
    "is_failure_ceiling", "observed_time", "calibrated_time", "chipwide_recal_event_id",
]


def load_config(path=CONFIG_PATH):
    with open(path) as f:
        return yaml.safe_load(f)


# ------------------------------------------------------------------ steps 1–3

def read_events(cfg):
    fams = cfg["families"]
    wanted = fams["coherence"] + fams["qubit_error"] + fams["readout"] + fams["pair_error"]
    env = cfg["environment"]

    table = pq.read_table(
        cfg["source"]["parquet"],
        columns=RAW_COLUMNS + env,
        filters=[("property_family", "in", wanted)],
    )
    df = table.to_pandas()
    for c in ["backend", "property_family", "unit"]:
        df[c] = df[c].astype("object")
    for c in ["observed_time", "calibrated_time"]:
        df[c] = df[c].dt.tz_convert(None)          # UTC, tz-naive

    # Clock skew: a few events carry a calibrated_time up to ~6 min after the
    # poll that published them. Treat a value as available at the later of the
    # two timestamps so the as-of joins can never see it early.
    df["clock_skew"] = df["calibrated_time"] > df["observed_time"]
    df["observed_time"] = df[["observed_time", "calibrated_time"]].max(axis=1)

    coherence = df["property_family"].isin(fams["coherence"])
    in_seconds = coherence & df["unit"].isna()
    df.loc[in_seconds, "value"] = df.loc[in_seconds, "value"] * 1e6
    df["qubit_b"] = df["qubit_b"].fillna(-1).astype("int64")
    return df


def genuine_events(df):
    key = ["backend", "property_family", "qubit_a", "qubit_b"]
    df = df.sort_values(key + ["calibrated_time"], kind="stable")
    changed = df.groupby(key, sort=False)["value"].diff().ne(0)
    return df[changed].reset_index(drop=True)


# ------------------------------------------------------------------ step 4

def _asof(left, right, by, prefix):
    """Latest `right` event published (observed_time) at or before left.decision_time."""

    right = right.sort_values("observed_time")[
        by + ["observed_time", "calibrated_time", "value", "is_failure_ceiling"]
    ].rename(columns={
        "observed_time": f"{prefix}_published",
        "calibrated_time": f"{prefix}_calibrated",
        "value": prefix,
        "is_failure_ceiling": f"{prefix}_ceiling",
    })
    out = pd.merge_asof(
        left.sort_values("decision_time"), right,
        left_on="decision_time", right_on=f"{prefix}_published",
        by=by, direction="backward", allow_exact_matches=True,
    )
    out[f"{prefix}_ceiling"] = out[f"{prefix}_ceiling"].astype("boolean").fillna(False).astype(bool)
    out.loc[out[f"{prefix}_ceiling"], prefix] = np.nan
    out[f"{prefix}_age_h"] = (out["decision_time"] - out[f"{prefix}_calibrated"]).dt.total_seconds() / 3600
    return out.drop(columns=[f"{prefix}_published", f"{prefix}_calibrated"])


def build_cce_table(ev, raw_env_cols):
    t1 = ev[ev["property_family"] == "T1"]
    base = t1[["backend", "qubit_a", "calibrated_time", "observed_time", "value"] + raw_env_cols].rename(
        columns={"qubit_a": "qubit", "calibrated_time": "t1_calibrated",
                 "observed_time": "decision_time", "value": "t1"})

    cce = base.rename(columns={"qubit": "qubit_a"})
    for fam, prefix in STATE_PROPS.items():
        cce = _asof(cce, ev[ev["property_family"] == fam], ["backend", "qubit_a"], prefix)

    # CZ error over the qubit's couplings (pairs stored once per direction)
    cz = ev[ev["property_family"] == "cz_error"]
    pairs = cz[["backend", "qubit_a", "qubit_b"]].drop_duplicates()
    expanded = cce[["backend", "qubit_a", "decision_time", "t1_calibrated"]].merge(
        pairs, on=["backend", "qubit_a"])
    expanded = _asof(expanded, cz, ["backend", "qubit_a", "qubit_b"], "cz")
    agg = expanded.groupby(["backend", "qubit_a", "t1_calibrated"]).agg(
        cz_err_mean=("cz", "mean"),
        cz_err_max=("cz", "max"),
        cz_n_couplings=("qubit_b", "nunique"),
        cz_ceiling_frac=("cz_ceiling", "mean"),
        cz_age_h=("cz_age_h", "max"),
    ).reset_index()
    cce = cce.merge(agg, on=["backend", "qubit_a", "t1_calibrated"], how="left")

    # Hours since the last chip-wide recalibration, as published by decision time
    recal = (ev.dropna(subset=["chipwide_recal_event_id"])
               .groupby(["backend", "chipwide_recal_event_id"])["observed_time"].min()
               .reset_index().rename(columns={"observed_time": "recal_time"})
               .sort_values("recal_time"))
    recal["backend"] = recal["backend"].astype(cce["backend"].dtype)
    cce = pd.merge_asof(cce.sort_values("decision_time"), recal[["backend", "recal_time"]],
                        left_on="decision_time", right_on="recal_time", by="backend",
                        direction="backward")
    cce["hours_since_chip_recal"] = (cce["decision_time"] - cce["recal_time"]).dt.total_seconds() / 3600
    cce = cce.drop(columns="recal_time").rename(columns={"qubit_a": "qubit"})

    cce = cce.sort_values(["backend", "qubit", "t1_calibrated"]).reset_index(drop=True)
    g = cce.groupby(["backend", "qubit"])
    cce["cce_index"] = g.cumcount()
    cce["gap_h"] = g["t1_calibrated"].diff().dt.total_seconds() / 3600
    return cce


# ------------------------------------------------------------------ step 5

def assign_roles(cce, cfg):
    sp = cfg["splits"]
    rng = np.random.default_rng(sp["heldout_qubit_seed"])
    heldout = {}
    for b in sp["dev_backends"]:
        qubits = np.sort(cce.loc[cce["backend"] == b, "qubit"].unique())
        n = int(round(sp["heldout_qubit_fraction"] * len(qubits)))
        heldout[b] = sorted(int(q) for q in rng.choice(qubits, size=n, replace=False))

    role = pd.Series("unused", index=cce.index, dtype="object")
    for b in sp["dev_backends"]:
        m = cce["backend"] == b
        role[m] = np.where(cce.loc[m, "qubit"].isin(heldout[b]), "heldout_qubit", "dev")
    role[cce["backend"] == sp["heldout_device"]] = "heldout_device"
    role[cce["backend"] == sp["out_of_family"]] = "out_of_family"
    cce["role"] = role
    return cce, heldout


def time_split(target_time, cfg):
    """0 = train, 1 = val, 2 = test, by target time (PROTOCOL §2)."""
    t_val = np.datetime64(cfg["splits"]["t_val"])
    t_test = np.datetime64(cfg["splits"]["t_test"])
    t = np.asarray(target_time, dtype="datetime64[ns]")
    return np.where(t < t_val, 0, np.where(t < t_test, 1, 2))


def check_leakage(cce):
    """Assertions run on every build."""
    for prefix in list(STATE_PROPS.values()) + ["cz"]:
        age = cce[f"{prefix}_age_h"]
        assert (age.dropna() >= -1e-9).all(), f"{prefix}: value calibrated after decision time"
    assert (cce["decision_time"] >= cce["t1_calibrated"]).all(), "T1 published before it was measured"
    assert cce.groupby(["backend", "qubit"])["t1_calibrated"].apply(
        lambda s: s.is_monotonic_increasing).all(), "CCEs out of order"
    dev = set(map(tuple, cce.loc[cce["role"] == "dev", ["backend", "qubit"]].drop_duplicates().values))
    held = set(map(tuple, cce.loc[cce["role"] == "heldout_qubit", ["backend", "qubit"]].drop_duplicates().values))
    assert not dev & held, "qubit in both dev and held-out sets"


# ------------------------------------------------------------------ main

def build(cfg=None):
    cfg = cfg or load_config()
    out_dir = cfg["output_dir"]
    os.makedirs(out_dir, exist_ok=True)
    report = {"source": cfg["source"]}

    report["total_rows_in_file"] = pq.ParquetFile(cfg["source"]["parquet"]).metadata.num_rows
    raw = read_events(cfg)
    report["rows_selected_families"] = raw.groupby("property_family").size().to_dict()

    ev = genuine_events(raw)
    report["genuine_events"] = ev.groupby("property_family").size().to_dict()
    report["clock_skew_events_adjusted"] = int(ev["clock_skew"].sum())
    report["failure_ceilings_in_genuine"] = ev.groupby("property_family")["is_failure_ceiling"].sum().astype(int).to_dict()

    cce = build_cce_table(ev, cfg["environment"])
    cce, heldout = assign_roles(cce, cfg)
    check_leakage(cce)

    cce["time_split"] = time_split(cce["t1_calibrated"], cfg)

    report["cce_rows"] = len(cce)
    report["cce_by_backend"] = cce.groupby("backend").size().to_dict()
    report["qubits_by_backend"] = cce.groupby("backend")["qubit"].nunique().to_dict()
    report["cce_by_role_and_split"] = {
        f"{r}/{['train', 'val', 'test'][s]}": int(n)
        for (r, s), n in cce.groupby(["role", "time_split"]).size().items()
    }
    report["time_span_by_backend"] = {
        b: [str(d["t1_calibrated"].min()), str(d["t1_calibrated"].max())]
        for b, d in cce.groupby("backend")}
    report["missing_fraction"] = cce[[p for p in STATE_PROPS.values()] + ["cz_err_mean", "hours_since_chip_recal"]
                                     + cfg["environment"]].isna().mean().round(4).to_dict()
    report["gap_h_quantiles"] = cce["gap_h"].quantile([0.5, 0.9, 0.99, 1.0]).round(1).to_dict()
    report["heldout_qubits"] = heldout

    cce.to_parquet(os.path.join(out_dir, "cce.parquet"), index=False)
    with open(os.path.join(out_dir, "splits.json"), "w") as f:
        json.dump({"t_val": cfg["splits"]["t_val"], "t_test": cfg["splits"]["t_test"],
                   "dev_backends": cfg["splits"]["dev_backends"],
                   "heldout_device": cfg["splits"]["heldout_device"],
                   "out_of_family": cfg["splits"]["out_of_family"],
                   "heldout_qubits": heldout}, f, indent=2)
    with open(os.path.join(out_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)
    return cce, report


if __name__ == "__main__":
    _, rep = build()
    print(json.dumps({k: rep[k] for k in [
        "total_rows_in_file", "genuine_events", "cce_rows", "qubits_by_backend",
        "cce_by_role_and_split", "missing_fraction", "gap_h_quantiles"]}, indent=2, default=str))
