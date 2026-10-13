"""
Download the IBM calibration-drift dataset at the pinned revision and verify it.

    python3 scripts/download_ibm_data.py

Writes data/raw/ibm_calibration_drift/ (parquet, dataset card, PROVENANCE.json).
The revision and checksum come from configs/drift_data.yaml; the download fails
if the checksum does not match.
"""

import datetime
import hashlib
import json
import os

import yaml
from huggingface_hub import HfApi, hf_hub_download

REPO = "phanerozoic/qiskit-calibration-drift"
OUT = "data/raw/ibm_calibration_drift"
PARQUET = "data/train-00000-of-00001.parquet"


def main():
    with open("configs/drift_data.yaml") as f:
        src = yaml.safe_load(f)["source"]
    rev = src["revision"]

    for name in [PARQUET, "README.md", "2026-05-20-report.md"]:
        hf_hub_download(REPO, name, repo_type="dataset", revision=rev, local_dir=OUT)

    path = os.path.join(OUT, PARQUET)
    with open(path, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    if digest != src["sha256"]:
        raise SystemExit(f"checksum mismatch: {digest} != {src['sha256']}")

    info = HfApi().dataset_info(REPO, revision=rev)
    prov = dict(
        dataset=REPO, url=f"https://huggingface.co/datasets/{REPO}", revision=rev,
        last_modified=str(info.last_modified), file=PARQUET,
        bytes=os.path.getsize(path), sha256=digest,
        license="CC-BY-4.0 (SN column CC-BY-NC-4.0 per WDC-SILSO; excluded from modelling)",
        upstream_collector="https://github.com/CharlesCNorton/qiskit-calibration-drift",
        retrieved_utc=datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        retrieved_with="huggingface_hub.hf_hub_download (revision-pinned)",
    )
    with open(os.path.join(OUT, "PROVENANCE.json"), "w") as f:
        json.dump(prov, f, indent=2)
    print("OK", digest)


if __name__ == "__main__":
    main()
