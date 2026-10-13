"""
Order-insensitive digest of a simulation cache (sim_index.parquet + sim_arrays.npy).

The simulators collect results with Pool.imap_unordered, so the row order of a cache
depends on which worker finishes first; the content is deterministic. The freeze files
(configs/frozen_*.yaml) hash raw bytes. This digest lets a reproduction whose cache has
the same content in a different row order pass the freeze check.

    python3 -m src.evaluation.cache_digest     # record digests of the frozen caches

The digests in configs/frozen_cache_content.yaml were recorded from the original caches
after confirming they still matched their raw frozen hashes (docs/PROTOCOL_ADDENDUM.md).
"""

import hashlib
import os

import numpy as np
import pandas as pd
import yaml

RECORD = "configs/frozen_cache_content.yaml"
CACHES = ["results/stage5", "results/robustness/ra", "results/robustness/rb"]


def digest(d):
    idx = pd.read_parquet(f"{d}/sim_index.parquet")
    arr = np.load(f"{d}/sim_arrays.npy")
    keys = [c for c in idx.columns if c != "row"]
    idx = idx.sort_values(keys, kind="stable").reset_index(drop=True)
    h = hashlib.sha256()
    h.update(idx[keys].to_csv(index=False).encode())
    h.update(str(arr.dtype).encode())
    h.update(np.ascontiguousarray(arr[idx["row"].to_numpy()]).tobytes())
    return h.hexdigest()


def unordered_ok(changed):
    """Drop sim-cache files from `changed` when their cache has the recorded content digest."""
    if not os.path.exists(RECORD):
        return changed
    rec = yaml.safe_load(open(RECORD))["caches"]
    ok = {d for d in {os.path.dirname(p) for p in changed}
          if d in rec and os.path.exists(f"{d}/sim_index.parquet") and digest(d) == rec[d]}
    return [p for p in changed if not (os.path.dirname(p) in ok and os.path.basename(p) in
                                         ("sim_index.parquet", "sim_arrays.npy"))]


if __name__ == "__main__":
    yaml.safe_dump(dict(note="Order-insensitive content digests of the frozen simulation caches.",
                        caches={d: digest(d) for d in CACHES}), open(RECORD, "w"), sort_keys=False)
    print(f"recorded {len(CACHES)} cache digests -> {RECORD}")
