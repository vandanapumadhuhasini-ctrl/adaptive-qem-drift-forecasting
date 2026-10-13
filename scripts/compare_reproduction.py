"""
Compare a from-scratch reproduction with the project's results.

    python3 scripts/compare_reproduction.py <reproduction_dir>

Byte-identical files are reported as IDENTICAL. Otherwise numeric content is compared
(max absolute difference), so harmless serialisation differences are separated from real
numerical differences.
"""

import hashlib
import json
import sys

import numpy as np
import pandas as pd

KEY_FILES = [
    "data/processed/drift_v3/cce.parquet", "data/processed/drift_v3/pair_epochs.parquet",
    "results/stage3/metrics_val.csv", "results/stage3/metrics_test.csv", "results/stage3/h1_test.json",
    "results/stage5/pair_forecast/metrics_val.csv", "results/stage5/pair_forecast/metrics_test.csv",
    "results/stage5/decisions.parquet", "results/stage5/forecast_states.npy", "results/stage5/selections.json",
    "results/stage5/hypotheses_val.json", "results/stage5/hypotheses_test.json", "results/stage5/h3b_stationary.json",
    "results/robustness/ra/summary_test.json", "results/robustness/rb/summary_test.json",
]


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def numeric_diff(a, b):
    if a.endswith(".json"):
        fa, fb = pd.json_normalize(json.load(open(a)), sep="/"), pd.json_normalize(json.load(open(b)), sep="/")
        x = pd.concat([fa, fb]).T
    elif a.endswith(".csv"):
        x = pd.concat([pd.read_csv(a).select_dtypes("number"), pd.read_csv(b).select_dtypes("number")], axis=1)
        h = x.shape[1] // 2
        return float(np.nanmax(np.abs(x.iloc[:, :h].to_numpy() - x.iloc[:, h:].to_numpy())))
    elif a.endswith(".npy"):
        return float(np.nanmax(np.abs(np.load(a) - np.load(b))))
    else:
        da, db = pd.read_parquet(a).select_dtypes("number"), pd.read_parquet(b).select_dtypes("number")
        return float(np.nanmax(np.abs(da.to_numpy(float) - db.to_numpy(float))))
    vals = x.apply(pd.to_numeric, errors="coerce").dropna()
    return float((vals.iloc[:, 0] - vals.iloc[:, 1]).abs().max()) if len(vals) else 0.0


def main(rep):
    ok = True
    for f in KEY_FILES:
        try:
            if sha(f) == sha(f"{rep}/{f}"):
                print(f"IDENTICAL      {f}")
            else:
                d = numeric_diff(f, f"{rep}/{f}")
                print(f"DIFFERS (max |Δ| = {d:.3g})  {f}")
                ok = ok and d < 1e-9
        except FileNotFoundError as e:
            print(f"MISSING        {f}  ({e.filename})")
            ok = False
    print("\nREPRODUCED" if ok else "\nNOT FULLY REPRODUCED (see above)")


if __name__ == "__main__":
    main(sys.argv[1])
