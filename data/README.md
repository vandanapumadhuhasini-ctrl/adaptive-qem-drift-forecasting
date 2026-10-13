# Data

## Source and licence

| Field | Value |
|---|---|
| Dataset | `phanerozoic/qiskit-calibration-drift` (Hugging Face), collected by [CharlesCNorton/qiskit-calibration-drift](https://github.com/CharlesCNorton/qiskit-calibration-drift) |
| Content | Calibration records of IBM Quantum Heron processors (`backend.properties()` polled every 30 min), joined to weather and space-weather covariates |
| Revision used | `539bc9994d664a7b559dee6edf12d95976f58af2` (pinned) |
| File | `train-00000-of-00001.parquet`, 70,498,179 bytes, SHA-256 `6c9feb0f2b526b708857d5a4a2aa94fd03770783b0612c6d0f19f7507a752f0b` |
| Licence | CC-BY-4.0. The sunspot-number column (`SN`, WDC-SILSO) is CC-BY-NC-4.0 and is **not used** anywhere in this project |

Full provenance: `raw/ibm_calibration_drift/PROVENANCE.json`. The dataset card is kept unchanged in `raw/ibm_calibration_drift/README.md`, and the statistical report distributed in the same Hugging Face repository (2026-05-20 version of the data) is in `raw/ibm_calibration_drift/2026-05-20-report.md`. Both are downloaded by the script below.

**Attribution:** if you use these data, cite the Hugging Face dataset above (reference `[driftdata]` in `report/QEM_Project_Report.tex`).

## What is in the repository

| Path | Tracked? | Content |
|---|---|---|
| `raw/ibm_calibration_drift/PROVENANCE.json`, `README.md`, `2026-05-20-report.md` | yes | provenance, dataset card, report distributed with the dataset |
| `raw/ibm_calibration_drift/data/train-00000-of-00001.parquet` | **no** (70.5 MB, git-ignored) | the raw dataset; download it with the command below |
| `processed/drift_v3/cce.parquet` | yes | 101,891 coherence-calibration events (CCEs): one row per qubit and T1/T2 calibration, with the qubit's other properties as of that time |
| `processed/drift_v3/pair_epochs.parquet` | yes | pair-level CZ error per chip-wide daily epoch (Stage 5a) |
| `processed/drift_v3/report.json` | yes | row counts, genuine-event counts, clock-skew fixes, splits by role |
| `processed/drift_v3/splits.json` | yes | split dates, device roles and the held-out qubit lists |

## Download and rebuild

```bash
python3 scripts/download_ibm_data.py      # pinned revision; fails if the SHA-256 differs
python3 -m src.data.drift_dataset         # rebuilds processed/drift_v3/ (about 1 min)
```

## Splits (fixed in `docs/PROTOCOL.md` before any model was trained)

- **Time:** train before 2026-08-01, validation 2026-08-01 to 2026-08-31, test from 2026-09-01 (UTC) to the end of the data (2026-10-12).
- **Development devices:** `ibm_fez` and `ibm_kingston`. 20 % of their qubits (seed 2026) are held out entirely ("unseen qubits").
- **Unseen device:** `ibm_marrakesh`, never used for training.
- **Out of family:** `ibm_torino` (Heron r1), used only for descriptive statistics.

## Important distinction

The calibration histories are **real** IBM measurements. All quantum-circuit outcomes in this project are **simulated** (Qiskit Aer density-matrix simulation with noise models built from those calibrations). No circuit was run on quantum hardware.
