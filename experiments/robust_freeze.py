"""Freeze robustness inputs before any robustness test metric (docs/ROBUSTNESS_PLAN.md)."""

import hashlib
import time

import yaml

from src.evaluation.cache_digest import unordered_ok

FROZEN = "configs/frozen_robustness.yaml"
FILES = ["results/robustness/ra/sim_arrays.npy", "results/robustness/ra/sim_index.parquet",
         "results/robustness/ra/selections.json", "results/robustness/rb/sim_arrays.npy",
         "results/robustness/rb/sim_index.parquet", "results/robustness/rb/selections.json",
         "results/robustness/rb/decisions.parquet", "results/robustness/rb/epoch_states.npy",
         "results/robustness/rb/forecast_states.npy", "results/stage5/selections.json",
         "experiments/robust_policies.py", "experiments/robust_sim.py", "src/qem/nq.py",
         "docs/ROBUSTNESS_PLAN.md"]


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def check_frozen():
    try:
        rec = yaml.safe_load(open(FROZEN))
    except FileNotFoundError:
        raise SystemExit(f"refusing robustness test evaluation: {FROZEN} missing")
    bad = unordered_ok([p for p, h in rec["files"].items() if sha(p) != h])
    if bad:
        raise SystemExit(f"refusing: changed after freeze: {bad}")


if __name__ == "__main__":
    yaml.safe_dump(dict(frozen_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        files={p: sha(p) for p in FILES}), open(FROZEN, "w"), sort_keys=False)
    print(f"frozen {len(FILES)} files -> {FROZEN}")
