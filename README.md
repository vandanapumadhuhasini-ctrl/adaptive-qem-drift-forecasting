# Transformer-Based Real-Time Adaptive Quantum Error Mitigation for Non-Stationary Noise

This project asks whether **forecasting real quantum-hardware calibration drift** (with a Transformer and simpler models) can make **adaptive quantum error mitigation** (ZNE extrapolation, ZNE noise-scaling and Pauli twirling) better than fixed mitigation, adaptation to the latest calibration ("persistence"), and a contextual-bandit adaptive ZNE (Rahman & Nguyen, *IEEE JSAC* 44, 2026, doi:10.1109/JSAC.2026.3721807). All comparisons use real IBM calibration histories.

**Evidence type:**
- Device states are **real IBM Heron calibration records** (4 backends, 600 qubits, Jan–Oct 2026).
- Circuit execution is **simulated** in Qiskit Aer, with noise models built from those records.
- No quantum hardware was used.

## Main results (pre-registered, test set evaluated once after freezing)

| Question | Answer | Evidence |
|---|---|---|
| Does a Transformer forecast calibrations better than simple baselines? | Better than persistence (−10 % to −26 % error), **not** better than a window mean, ridge or gradient-boosted trees (H1 not supported) | `docs/experiment_reports/STAGE3_FORECASTING_REPORT.md` |
| Does forecast-driven adaptation beat adapting to the latest calibration? | **No**, at 1, 4 or 7 days ahead, on an unseen device, for new circuit instances, or at 6 qubits (H2 not supported) | `docs/experiment_reports/STAGE5_RESULTS.md`, `docs/experiment_reports/STAGE6_CONTRIBUTION.md` |
| Does calibration-informed adaptive ZNE + twirling beat fixed mitigation? | **Yes:** −5.6 % (1 day) and −4.2 % (4 days); non-inferior under stationary noise (H3) | same |
| Does it beat the contextual-bandit adaptive ZNE of the base paper? | **Yes:** −15 % to −19 % at an equal shot budget (H4); the bandit's Clifford-probe reward ranks actions poorly | same |
| How much could forecasting add at most? | A perfect forecast (oracle) beats persistence by 1.4–1.7 % at 1 day (2.5–2.6 % in the robustness checks), 3.0–3.7 % at 4 days and 5.2–7.1 % at 7 days, concentrated in abrupt device changes. Current forecasters capture almost none of it | same |

**Interpretation:** calibration-informed, model-based adaptation is what helps. Better forecast accuracy did not translate into better mitigation decisions. A decision-focused Transformer controller was explored afterwards as exploratory work (`docs/experiment_reports/STAGE5B_DFT.md`).

## Repository structure

```text
src/
  data/          drift_dataset.py (real calibration data → per-qubit states), windows.py (leakage-safe windows)
  forecasting/   models (persistence, rolling/window means, ridge, gradient-boosted trees, LSTM, Transformer), pairs.py
  qem/           device.py (epochs, chains, noise model), benchmarks.py (circuits), mitigation.py (ZNE, twirling),
                 policies.py (model-based controller, LinUCB bandit), nq.py (n-qubit generalisation), zne.py, noise.py
  evaluation/    stats.py (cluster bootstrap, Holm), plotting.py, cache_digest.py (freeze checks)
experiments/     stage3_*, stage5_*, stage6_*, robust_* (confirmatory pipeline); stage4_* (development stage,
                 validation only); s5b_* (exploratory); plot_stage*.py
scripts/         download_ibm_data.py (pinned, checksum-verified), compare_reproduction.py
configs/         data / forecasting configs and frozen-input hashes (frozen_*.yaml)
data/            raw/ (IBM parquet, git-ignored; provenance) and processed/ — see data/README.md
checkpoints/     trained forecasting models (Stage 3 per-qubit, Stage 5a pair-level)
results/         metrics (CSV/JSON) and figures (PNG 300 dpi + PDF + CSV) per stage
tests/           pytest: unit, leakage, freeze and number-traceability tests
docs/            FINAL_TECHNICAL_DOCUMENT.md (start here), PROTOCOL.md (+ addendum), ROBUSTNESS_PLAN.md,
                 research/, experiment_reports/, audit/  — index in docs/README.md
report/          IEEE-format project report (LaTeX) and slides/ (Beamer)
reproduce.py     single entry point
```

## Setup and reproduction

Developed and verified with the system Python 3.14.5 on macOS (Apple M4 Pro, 14 cores). The tests and the saved Transformer models were also checked under Python 3.11 (bit-identical forecasts).

```bash
python3 -m pip install -r requirements.txt
```

```bash
python3 scripts/download_ibm_data.py
```

```bash
python3 reproduce.py --list
```

```bash
python3 reproduce.py all
```

```bash
python3 -m pytest -q tests
```

- `scripts/download_ibm_data.py` fetches the dataset at the pinned revision and verifies its SHA-256.
- `reproduce.py all` runs the confirmatory pipeline (about 4–5 h with 12 workers). `python3 reproduce.py stage4 s5b` adds the validation-only development stage and the exploratory experiment. Stages can also be run individually.
- The confirmatory pipeline was reproduced from scratch on 2026-10-13: all key result files were byte-identical (`docs/audit/AUDIT_REPORT.md` §5).
- Test evaluations refuse to run if any frozen input changed (`configs/frozen_*.yaml`).
- Simulation caches are seeded and deterministic. Large regenerable caches are git-ignored.
- Commit hashes cited in `docs/` (e.g. the freeze commits) refer to the local development history, which is not published. The freeze files themselves carry UTC timestamps and SHA-256 hashes of every frozen input.

## Data and licences

- Calibration data: `phanerozoic/qiskit-calibration-drift` (Hugging Face), revision `539bc99…`, CC-BY-4.0. The CC-BY-NC Sunspot-Number column is excluded. Provenance: `data/raw/ibm_calibration_drift/PROVENANCE.json`.
- Data quality and splits: `docs/experiment_reports/STAGE2_DATA_REPORT.md`.

## Methodology documents

`docs/FINAL_TECHNICAL_DOCUMENT.md` (complete description) · `docs/audit/AUDIT_REPORT.md` (audit and IEEE suitability) · `docs/PROTOCOL.md` (pre-registration, frozen) · `docs/PROTOCOL_ADDENDUM.md` · `docs/research/BASE_PAPER_REVIEW.md` · `docs/research/STAGE1_RESEARCH_PROPOSAL.md` · `docs/research/STAGE4_DESIGN.md` · `docs/ROBUSTNESS_PLAN.md` · `docs/research/PROJECT_PLAN.md`

## Limitations

- **Simulated execution.**
  - Readout and idle errors are not modelled.
  - Coherent error enters only as a fixed CZ phase.
  - Chains of 4 qubits (6-qubit robustness check).
- **The execution-time device state is proxied by the next published calibration.**
- **The model-based controller assumes a calibrated noise model;** under model mismatch its advantage shrinks.
- **The 7-day horizon has few decisions** (19 chains).
