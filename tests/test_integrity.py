"""
Integrity of the saved results: leakage rules, frozen inputs, and that the headline
numbers quoted in README.md / the report are the ones in the result files.
Uses only tracked files, except the freeze checks, which need the regenerable
(git-ignored) caches and are skipped when those are absent.
"""

import json
import os
import re

import numpy as np
import pandas as pd
import pytest

T_VAL, T_TEST = pd.Timestamp("2026-08-01"), pd.Timestamp("2026-09-01")


def _pct_ci(r):
    base = r["mean_diff"] / (r["pct"] / 100)
    return r["pct"], 100 * r["ci"][0] / base, 100 * r["ci"][1] / base


def test_stage5_decisions_respect_the_time_splits():
    d = pd.read_parquet("results/stage5/decisions.parquet")
    assert (d.decision_time < d.exec_time).all()
    assert (d.exec_epoch - d.epoch == d.k).all()
    assert (d.loc[d.split == "train", "exec_time"] < T_VAL).all()
    v = d[d.split == "val"]
    assert ((v.exec_time >= T_VAL) & (v.exec_time < T_TEST)).all()
    assert (d.loc[d.split == "test", "exec_time"] >= T_TEST).all()
    assert set(d.role) == {"dev", "heldout_device"}


def test_heldout_qubits_and_device_never_in_training_windows():
    meta = pd.read_parquet("results/stage3/windows_meta.parquet")
    splits = json.load(open("data/processed/drift_v3/splits.json"))
    held = {(b, q) for b, qs in splits["heldout_qubits"].items() for q in qs}
    dev = meta[meta.role == "dev"]
    assert not any((b, q) in held for b, q in zip(dev.backend, dev.qubit))
    assert set(meta.loc[meta.role == "heldout_device", "backend"]) <= {splits["heldout_device"]}


def test_dataset_counts_match_the_documentation():
    r = json.load(open("data/processed/drift_v3/report.json"))
    assert r["total_rows_in_file"] == 16_140_252
    assert sum(r["genuine_events"].values()) == 2_024_027
    assert r["cce_rows"] == 101_891 == len(pd.read_parquet("data/processed/drift_v3/cce.parquet"))
    assert sum(r["qubits_by_backend"].values()) == 600


@pytest.mark.parametrize("module", ["experiments.stage3_forecasting", "experiments.stage5_freeze",
                                    "experiments.robust_freeze"])
def test_frozen_inputs_unchanged(module):
    import importlib
    needed = {"experiments.stage3_forecasting": "results/stage3/predictions",
              "experiments.stage5_freeze": "results/stage5/sim_arrays.npy",
              "experiments.robust_freeze": "results/robustness/ra/sim_arrays.npy"}[module]
    if not os.path.exists(needed):
        pytest.skip(f"{needed} not present (regenerate with reproduce.py)")
    importlib.import_module(module).check_frozen()   # raises SystemExit if anything changed


def test_headline_numbers_trace_to_result_files():
    H = json.load(open("results/stage5/hypotheses_test.json"))
    pct, lo, hi = _pct_ci(H["H2_primary_k1_dev"])
    assert round(pct, 2) == 0.02 and round(lo, 2) == -0.29 and round(hi, 2) == 0.54
    assert H["H2_primary_k1_dev"]["success"] is False
    assert [round(r["pct"], 1) for r in H["H3a_vs_best_fixed"]] == [-5.6, -4.2, -0.3]
    assert [round(r["pct"], 1) for r in H["H4_vs_cmab_probe"]] == [-15.0, -18.7, -14.5]
    s = json.load(open("results/stage5/h3b_stationary.json"))
    assert round(100 * s["rel_diff"], 1) == -4.9 and s["H3b_noninferior"]
    ra = json.load(open("results/robustness/ra/summary_test.json"))
    rb = json.load(open("results/robustness/rb/summary_test.json"))
    assert round(ra["adaptive_transformer - adaptive_persistence"]["pct"], 2) == -0.42
    assert round(ra["adaptive_transformer - cmab_probe"]["pct"], 1) == -17.8
    assert round(rb["adaptive_transformer - adaptive_persistence"]["pct"], 2) == -0.35
    assert round(rb["adaptive_transformer - best_fixed"]["pct"], 1) == -3.6
    h1 = json.load(open("results/stage3/h1_test.json"))
    r = [x["comparisons"]["persistence"]["R"] for x in h1 if x["eval_set"] == "test_primary"]
    assert round(100 * min(r), 1) == 9.7 and round(100 * max(r), 1) == 25.7
    assert not any(x["practical_pass"] for x in h1)


def test_report_error_table_matches_results():
    br = json.load(open("results/stage5/hypotheses_test.json"))["by_regime"]
    cols = ["none", "best_fixed", "cmab_probe", "adaptive_persistence", "adaptive_window_mean",
            "adaptive_ridge", "adaptive_transformer", "adaptive_oracle"]
    tex = open("report/QEM_Project_Report.tex").read()
    for k, lab in [(1, "1 day"), (4, "4 days"), (7, "7 days")]:
        for reg, rl in [("R1_incoherent", "R1"), ("R2_coherent", "R2")]:
            row = f"{lab} & {rl} & " + " & ".join(f"{br[f'k{k}/{reg}/dev'][c]:.4f}" for c in cols)
            assert row in tex, row


def test_stage5b_used_no_test_data():
    df = pd.read_parquet("results/stage5b/dft_results.parquet")
    if "split" in df.columns:
        assert set(df.split) <= {"train", "val"}
    rep = json.dumps(json.load(open("results/stage5b/dft_validation_report.json")))
    assert not re.search(r'"test', rep)
