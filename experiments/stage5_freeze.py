"""
Stage 5 freeze: record SHA-256 of every input to the test evaluation (PROTOCOL §6).

    python3 -m experiments.stage5_freeze            # after validation selection, before any test metric
Test scripts call check_frozen() and refuse to run if anything changed.
"""

import glob
import hashlib
import time

import yaml

from src.evaluation.cache_digest import unordered_ok

FROZEN = "configs/frozen_stage5.yaml"
FILES = [
    "results/stage5/decisions.parquet", "results/stage5/epoch_states.parquet", "results/stage5/epoch_states.npy",
    "results/stage5/forecast_states.npy", "results/stage5/sim_index.parquet", "results/stage5/sim_arrays.npy",
    "results/stage5/selections.json", "data/processed/drift_v3/cce.parquet", "data/processed/drift_v3/pair_epochs.parquet",
    "src/qem/device.py", "src/qem/mitigation.py", "src/qem/policies.py", "src/qem/benchmarks.py",
    "experiments/stage5_policies.py", "experiments/stage5_simulate.py", "experiments/stage5_decisions.py",
    "src/evaluation/stats.py", "docs/PROTOCOL.md",
] + sorted(glob.glob("results/stage5/pair_forecast/predictions/*.npz"))


def sha(p):
    with open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def freeze():
    rec = dict(frozen_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               note="Created before any Stage 5 test-period metric was computed.",
               files={p: sha(p) for p in FILES})
    with open(FROZEN, "w") as f:
        yaml.safe_dump(rec, f, sort_keys=False)
    print(f"frozen {len(FILES)} files -> {FROZEN}")


def check_frozen():
    try:
        rec = yaml.safe_load(open(FROZEN))
    except FileNotFoundError:
        raise SystemExit(f"refusing test evaluation: {FROZEN} missing (run experiments.stage5_freeze)")
    changed = unordered_ok([p for p, h in rec["files"].items() if sha(p) != h])
    if changed:
        raise SystemExit(f"refusing test evaluation: changed after freeze: {changed}")


if __name__ == "__main__":
    freeze()
