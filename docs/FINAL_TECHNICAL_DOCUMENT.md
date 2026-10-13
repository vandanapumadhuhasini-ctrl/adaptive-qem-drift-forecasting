# Transformer-Based Real-Time Adaptive Quantum Error Mitigation for Non-Stationary Noise
## Final Technical Document

**Status:** complete, 2026-10-13.

Every number in this document comes from a file in `results/` or `data/processed/`, re-checked during the audit (`docs/audit/AUDIT_REPORT.md` §3).

**Evidence types:**
- **Real:** IBM Heron calibration records.
- **Simulated:** all quantum-circuit outcomes (Qiskit Aer). No quantum hardware was used.
- **Confirmatory:** pre-registered tests, each evaluated once on a frozen test period.
- **Validation only / exploratory:** not confirmatory evidence, and labelled as such wherever it appears.

---

## 1. Project title and scope

**Title:** *Transformer-Based Real-Time Adaptive Quantum Error Mitigation for Non-Stationary Noise.* The title was kept at the owner's request.

**Scope:**
- Use real calibration histories of IBM superconducting processors (T1, T2, gate and readout errors, which change from day to day) to adapt quantum error mitigation (QEM).
- The mitigation choices are the zero-noise-extrapolation (ZNE) scale factors, the extrapolator and Pauli twirling.
- Ask whether *forecasting* the drift with a Transformer gives better mitigation decisions than simpler alternatives.

**What the title promises, against what the evidence supports:**

| Element of the title | Supported? | Evidence |
|---|---|---|
| *Adaptive* QEM | **Yes** | Calibration-informed adaptation beats the best fixed setting by 5.6 % (1 day) and 4.2 % (4 days) of mitigated error, and a contextual-bandit adaptive ZNE by 15–19 % (§12.3) |
| *Non-stationary noise* | **Yes** | Real day-to-day drift: 51 % of next-day T1 values differ by more than 25 % (§12.1) |
| *Transformer-based* | **No** | The Transformer forecasts drift better than persistence but not better than simple models (H1). Forecast-driven decisions are not better than decisions from the latest calibration (H2). A decision-focused Transformer also failed (§13) |
| *Real-time* | **Not tested** | Decisions are made once per daily calibration, in simulation. No latency-constrained or hardware loop was built |

---

## 2. Abstract

**Problem.** Quantum error mitigation methods such as ZNE and Pauli twirling are usually configured once. The noise of superconducting processors, however, drifts between calibrations.

**Approach.**
- **Data:** 2,024,027 genuine calibration measurements from four IBM Heron processors (600 qubits, January–October 2026), from a public, revision-pinned dataset.
- **Forecasting:** we trained Transformer, LSTM, ridge, gradient-boosted-tree and simple statistical forecasters for per-qubit and pair-level calibration properties.
- **Controller:** a model-based controller selects one of 14 mitigation actions for each decision. It works by simulating every action under a noise model built from a state estimate.
- **Comparison:** controllers driven by different state estimates (latest calibration, forecasts, true future state) are compared with the best fixed action and with a faithful reimplementation of the base paper's contextual-bandit adaptive ZNE.
- **Setting:** equal shot budget, on 4-qubit chains, with pre-registered hypotheses and a once-only test period.

**Findings.**
- Calibration-informed adaptation lowered the simulated mitigated error by 5.6 % (1 day ahead) and 4.2 % (4 days) against the best fixed action. It lowered it by 15.0–18.7 % against the bandit, and was not worse under stationary noise.
- Forecasting did not help. The Transformer was 9.7–25.7 % more accurate than persistence but not more accurate than simple models. Its forecasts changed the decision error by +0.02 % (95 % CI −0.29 to +0.54 %) relative to the latest calibration.
- A perfect forecast would have been worth only 1.4–1.7 % at 1 day and 5.2–7.1 % at 7 days, mostly during abrupt device changes.
- A decision-focused Transformer controller was also no better (validation only).

**Limitations.** Simulated execution, small circuits, a proxy for the true execution-time state, and one six-week test period.

---

## 3. Introduction and problem statement

Superconducting qubits are noisy:
- Energy relaxation (T1), dephasing (T2), imperfect single- and two-qubit gates and readout errors bias the expectation values that near-term algorithms estimate.
- These parameters are **not constant**. Two-level-system defects, recalibrations and the environment make them drift from day to day and sometimes jump abruptly (Klimov et al. 2018; Proctor et al. 2020).

QEM reduces this bias without error correction:
- **ZNE** runs a circuit at amplified noise levels and extrapolates back to zero noise.
- **Pauli twirling** randomises coherent errors into stochastic Pauli noise, which ZNE handles better.

Each technique has settings: which scale factors, which fit, whether to twirl. The best setting depends on the current noise. A setting that is good on one day can be poor a few days later.

**Problem addressed.** Given the calibration history of a device up to the decision time, choose the mitigation setting for a circuit that will run *k* days later (k = 1, 4, 7), so as to minimise the error of the mitigated expectation values at a fixed shot budget.

---

## 4. Research motivation and gap

**Limitations of existing approaches:**
- Fixed mitigation ignores drift.
- The base paper (Rahman & Nguyen 2026) adapts the ZNE folding with a contextual bandit, but:
  - it is **reactive**: it learns from feedback after the fact and does not forecast;
  - it was evaluated on a **prescribed synthetic drift schedule**, not on real calibration records;
  - it adapts only the fold set, not the extrapolator or twirling;
  - it spends shots on exploration.
- Learning-based QEM (Clifford data regression, deep-learning mitigators) learns a correction for a *static* device.

**The gap identified:** use *forecasts of real calibration telemetry* to make adaptive ZNE + twirling *proactive* (no exploration shots), and test whether this beats fixed, reactive and bandit adaptation under real drift.

**The gap actually addressed.** The implementation tests exactly this, in simulation with real calibration states. It found that:
- the proactive *forecasting* part adds nothing measurable;
- the *calibration-informed, model-based* part, which needs only the latest calibration, gives the whole benefit.

So the project closes the gap with a largely negative answer about forecasting and a positive answer about calibration-informed adaptation.

---

## 5. Base paper review

**Citation (verified via Crossref, 2026-10-13):** R. Rahman and D. C. Nguyen, "Quantum Noise Mitigation With Adaptive Zero-Noise Extrapolation: A Contextual Multi-Armed Bandits Approach," *IEEE Journal on Selected Areas in Communications*, vol. 44, pp. 5600–5614, 2026, doi: 10.1109/JSAC.2026.3721807. The preprint is arXiv:2608.06426.

The details below are from `docs/research/BASE_PAPER_REVIEW.md`, which was based on about 78 % of the arXiv full text.

- **Problem:** fixed ZNE settings become suboptimal when hardware noise drifts over time.
- **Method:**
  - a cost-aware, discounted LinUCB contextual bandit;
  - arms are fold-scale sets {1,3}, {1,3,5} and {1,3,5,7}, each with a fixed extrapolator;
  - context: training epoch, loss and accuracy at λ = 1, variance across scales, batch size;
  - reward: −cross-entropy − β·variance − depth penalty.
- **Setting:**
  - hybrid variational classifiers (CIFAR-10, EuroSAT, 4–32 qubits);
  - Lindblad noise with a *prescribed* time-varying drift profile (PennyLane simulation);
  - inference on Rigetti hardware for 30 samples.
- **Baselines:** fixed ZNE and 11 reimplemented methods.
- **Contributions:**
  - fewer circuit executions than fixed folding: about 39–41 % fewer round trips;
  - slightly higher accuracy than the best baseline: 85.58 % vs 85.24 % on CIFAR-10.
- **Limitations:**
  - reactive, with no forecasting;
  - synthetic drift;
  - exploration cost;
  - twirling only as a baseline;
  - a task-level metric (accuracy) rather than the error of a physical observable;
  - limited hardware validation.

**How this project differs (implemented and tested):**

| Aspect | Base paper | This project |
|---|---|---|
| Drift | Prescribed synthetic schedule | Real IBM calibration histories, 4 devices, 600 qubits |
| Information used | Training feedback | Published calibration (and forecasts of it) |
| Adapted settings | Fold-scale set | Fold-scale set + extrapolator + twirling (14 actions) |
| Decision rule | Online bandit | Model-based controller, no exploration shots. The base paper's bandit is reimplemented as a baseline |
| Metric | Classification accuracy, cost | Error of mitigated Pauli expectation values at an equal shot budget |
| Statistics | Point estimates | Pre-registered tests, cluster-bootstrap CIs, Holm correction |
| Hardware | Rigetti, inference only | None (simulation) |

---

## 6. Objectives, research questions and hypotheses

**Central question** (`docs/PROTOCOL.md`, frozen 2026-10-12): *can Transformer-based forecasting of real quantum hardware calibration drift improve QEM decisions compared with fixed mitigation, persistence-based adaptation and contextual-bandit adaptation, under a fair execution and shot budget?*

| Hypothesis (pre-registered) | Success criterion | Outcome |
|---|---|---|
| **H1** Transformer forecasts beat every baseline (k ≥ 2) | CI < 0 against every baseline **and** ≥ 5 % reduction | **Not supported**: 0 of 15 cells |
| **H2** (primary) Transformer-driven adaptation beats persistence-driven adaptation (k = 1, real drift, development chains) | CI < 0 **and** ≥ 10 percentage points more of the oracle gain captured | **Not supported** (+0.02 %, CI includes 0) |
| **H3a** beats the best fixed action under drift | CI < 0 | **Supported at k = 1 and 4**, not at k = 7 |
| **H3b** non-inferior under stationary noise | upper CI bound < +2 % | **Supported** (−4.9 %, CI [−7.4, −1.0] %) |
| **H4** beats the contextual bandit (probe reward) at equal shots | CI < 0 | **Supported** at k = 1, 4, 7 |
| **H5** behaviour under abrupt changes and model mismatch | exploratory | Described (§12.5) |
| **H6** (added after Stage 5, before any training) decision-focused Transformer beats persistence | CI < 0 and ≥ 1 % | **Not met** (validation only; §13) |

---

## 7. Dataset

| Item | Value |
|---|---|
| Source | `phanerozoic/qiskit-calibration-drift` (Hugging Face), collected by a GitHub Action polling IBM `backend.properties()` every 30 min |
| Revision, file | `539bc9994d664a7b559dee6edf12d95976f58af2`; 70,498,179 bytes; SHA-256 `6c9feb0f…752f0b` |
| Licence | CC-BY-4.0. The CC-BY-NC sunspot-number column is excluded |
| Rows in file | 16,140,252 |
| Genuine measurements (value changed) | 2,024,027 |
| Calibration states used for modelling (CCEs) | 101,891 |
| Backends | ibm_fez (156 qubits), ibm_kingston (155), ibm_marrakesh (156), ibm_torino (133) |
| Period | fez, kingston, marrakesh: Jan/Feb → 12 Oct 2026. Torino (retired): Nov 2025 → Apr 2026 |
| Cadence | median 24.8 h between calibrations of a qubit |

Source files: `data/processed/drift_v3/report.json` and `docs/experiment_reports/STAGE2_DATA_REPORT.md`.

**Preprocessing:**
1. **Unit harmonisation.** T1 and T2 were stored in seconds before a May 2026 schema change and in microseconds after it. All are converted to µs.
2. **Genuine events.** The dataset has no flag for a new measurement, so we use a heuristic: an event counts only if its value differs from the previous event of the same property and qubit (or pair). For example, 89 % of single-qubit gate-error events are re-timestamps of unchanged values.
3. **Failure ceilings** (invalid measurements) are excluded as targets and masked in features.
4. **Clock skew:** 22 events had a calibration time after their publication. They are treated as available at the later time.
5. **Calibration state (CCE).** One row per qubit per T1 calibration. Other properties are attached by an as-of join on *publication* time, so a state never contains information that was not yet public.
6. **Segments.** Gaps over 7 days start a new segment, and forecasting windows never cross one.

**Splits** (fixed before training): see `data/README.md`.
- **Time:** train before 2026-08-01, validation in August, test from 2026-09-01 to 2026-10-12.
- **Development backends:** fez and kingston, with 20 % of their qubits held out (seed 2026).
- **Unseen device:** marrakesh.
- **Descriptive only:** torino.

**Real versus simulated.** The calibration states are real. Every circuit outcome, error value and policy comparison comes from simulation driven by those states.

---

## 8. Methodology

### 8.1 Data acquisition and preprocessing
- `scripts/download_ibm_data.py` downloads the pinned revision and verifies the SHA-256.
- `src/data/drift_dataset.py` performs steps 1–6 of §7 and writes `cce.parquet` and `report.json`.

### 8.2 Features and time-series windows
`src/data/windows.py`:
- **Input:** the last 20 calibration states of a qubit, each with 19 features:
  - log T1, T2, sx error, readout error, P(0|1), P(1|0);
  - log mean and max CZ error over the qubit's couplings, and the fraction of couplings at the failure ceiling;
  - sx failure flag;
  - ages of each property;
  - gap since the previous calibration;
  - hours since the last chip-wide recalibration, with an unknown flag;
  - hour of day (sin/cos).
- **Targets:** the change Δ in log T1, log T2, log sx error, log readout error and log mean CZ error, k = 1, 2, 4, 7 calibrations ahead. Persistence (Δ = 0) is therefore the natural baseline.
- **Split rule:** a sample belongs to the split of its *target* time. Scalers are fitted on training samples only.
- **Leakage checks:** assertions run in the pipeline (`check_windows`).

### 8.3 Forecasting models and training
`src/forecasting/models.py` and `experiments/stage3_forecasting.py`:
- **Baselines:** persistence; mean of the last 5 values; mean of the window of 20 values.
- **Ridge regression and histogram gradient-boosted trees** on 49 lag features, one model per target and horizon.
- **LSTM:** hidden 32, 1 layer, dropout 0.3.
- **Transformer encoder.** Selected on validation from a grid of 16 configurations:
  - learned positional embedding;
  - pre-norm, 1 layer, model width 32, 4 heads, feed-forward 64, dropout 0.1;
  - learning rate 1e-3, weight decay 1e-4, batch 256;
  - early stopping with patience 10 (at most 80 epochs);
  - a single multi-horizon, multi-target head with a masked MSE loss.
- **Seeds:** neural models are 5-seed ensembles (seeds 0–4).
- **Transformer ablations:** targets-only inputs, inputs plus environmental covariates, and no positional encoding.
- **Pair-level CZ forecasting** (Stage 5a; `src/forecasting/pairs.py`, `experiments/stage5_pair_forecast.py`):
  - the same model families, on the CZ error of each coupled pair per daily epoch, at k = 1, 4, 7 days;
  - selected Transformer: width 64, 1 layer, 4 heads, feed-forward 128, learning rate 3e-4.

### 8.4 Decision setting and quantum-circuit simulation
`src/qem/device.py`, `experiments/stage5_decisions.py`, `experiments/stage5_simulate.py`.
- **Epochs.** IBM recalibrates T1 chip-wide once a day, so one decision is made per backend-day. The decision time is the moment by which 90 % of that day's calibrations had been published. Calibrations published later are unavailable at that epoch; this is the leakage fix, logged in `PROTOCOL.md` §9.
- **Chains.** Disjoint linear chains of 4 physically coupled qubits from the real coupling map:
  - 30 development chains (fez 12, kingston 18);
  - 15 on the unseen device.
- **Execution state.** The calibration published at epoch e + k is the proxy for the true device state when the circuit runs.
- **Noise model** (Qiskit Aer, density matrix, native gates {CZ, sx, rz, x}):
  - sx and x: thermal relaxation over the real gate duration (24, 32 or 36 ns by backend);
  - CZ: relaxation on both qubits over 68 ns;
  - each gate gets a depolarising channel so that its average gate infidelity equals the reported error (checked to within 0.07 %; unit-tested);
  - rz: noiseless (virtual);
  - readout and idle errors: not modelled.
- **Regimes:**
  - **R1:** incoherent drift.
  - **R2:** plus a coherent controlled-phase error of 0.03 rad on every CZ, known to the controller.
  - **R3 (mismatch):** the device has the phase error but the controller's model does not.
  - **Stationary control:** each chain frozen at its training-period median state.
- **Circuits** (`src/qem/benchmarks.py`): six 4-qubit circuits on the native gate set:

  | Circuit | CZ gates | Observables |
  |---|---:|---:|
  | Hardware-efficient ansatz, depth 4 | 12 | 8 |
  | Hardware-efficient ansatz, depth 8 | 24 | 10 |
  | TFIM Trotter, 3 steps | 18 | 15 |
  | TFIM Trotter, 6 steps | 36 | 16 |
  | QAOA, p = 2 | 12 | 11 |
  | QAOA, p = 4 | 24 | 10 |

  The observables are single-qubit and nearest-neighbour Z/X Pauli strings plus ZZZZ, kept when the noiseless value exceeds 0.2 in magnitude.

### 8.5 Zero-noise extrapolation
`src/qem/mitigation.py`, `src/qem/zne.py`:
- **Noise scaling:** local CZ folding, with scale factors from {1, 2, 3, 5, 7}.
- **Extrapolators:** linear, Richardson (polynomial through all points), exponential (log-linear fit, falling back to linear), and a quadratic polynomial for {1,3,5,7}.
- **Clipping:** estimates are clipped to [−1, 1].

### 8.6 Pauli twirling
- Each CZ is conjugated by random Paulis that leave the ideal circuit unchanged. Eight seeded instances are used per action, and the shots are split evenly among them.
- The twirl Paulis are absorbed into neighbouring single-qubit gates, so they add no noise.
- Noiseless folded and twirled circuits were checked to equal the original exactly.

### 8.7 Action set and shot budget
- **14 actions:**
  - none;
  - twirl only;
  - 6 ZNE settings, without and with twirling: {1,3} linear; {1,2,3} linear; {1,3,5} Richardson; {1,3,5} exponential; {1,3,5,7} quadratic; {1,3,5,7} exponential.
- The base paper's arms {1,3}, {1,3,5} and {1,3,5,7} are included.
- **Budget:** every policy uses 8,192 shots per observable per decision, split over scale factors.

### 8.8 Adaptive controller and policy selection
`src/qem/policies.py`, `experiments/stage5_policies.py`.

**Model-based controller:**
1. Build the noise model from a *state estimate* of the chain at execution time.
2. Simulate all 14 actions exactly.
3. Estimate each action's expected absolute error including shot noise (200 Monte-Carlo trials).
4. Pick the action with the lowest predicted error.

The policies differ only in the state estimate:

| Policy | State estimate | Notes |
|---|---|---|
| Persistence | latest published calibration | reactive |
| Window mean, ridge, Transformer | per-qubit forecasts (T1, T2, sx) + pair-level CZ forecasts of the same family | forecast-driven |
| Oracle | true execution state | upper bound, not realisable |
| None | — | no mitigation |
| Best fixed | one action for all decisions | selected on the validation period; also a per-circuit variant |

*Disclosure:* `PROTOCOL.md` §4 says "best fixed chosen on Train". The implementation selects it on validation, which is documented in `PROTOCOL_ADDENDUM.md`.

### 8.9 Contextual multi-armed bandit baseline
Our reimplementation of the base paper's LinUCB (Eqs. 17–19). It is not the authors' code, and the deviations are listed below:
- **Context:** intercept plus mean log T1, T2, sx and CZ of the chain.
- **Arms:** the same 14 actions.
- **Reward:** the negative error on a **Clifford probe**, i.e. the same circuit with angles snapped to multiples of π/2, so its ideal value is known. It is computed from 1/8 of the shots (1,024), leaving 7,168 for the main estimate. Variance and cost penalties follow the paper.
- **Delayed feedback:** a decision's reward arrives only after its execution epoch (k days later).
- **Online learning:** the bandit learns online from the last 60 training epochs onward.
- **Tuning:** hyperparameters (context, η, ρ, λ_c, β) were tuned on validation per regime. For example, at k = 1 / R1 the selection was full context, η = 0.1, ρ = 1, λ_c = 0.02, β = 1.
- **Implementation choices where the paper is silent:**
  - reward scaling;
  - causal reward centring;
  - play-each-arm-once initialisation.

  Without the last two, even a bandit rewarded with the true error was worse than a fixed action.
- **Reference variants (not realisable):**
  - a warm start from the controller's model;
  - an oracle reward equal to the true error, with no probe shots.

### 8.10 Decision-focused Transformer (exploratory)
See §13.

### 8.11 Evaluation metrics, seeds and statistics
- **Forecast metric:** mean absolute error of the log value, averaged per qubit (each qubit counts once). The relative improvement is R = 1 − MAE_model / MAE_baseline.
- **Decision metric:** the mean over observables of E|Ê − E_ideal|, with 200 Monte-Carlo shot trials using **common random numbers** for all policies.
- **Comparisons:** paired differences, with 95 % CIs from a **cluster bootstrap** (10,000 resamples of qubits or chains within each backend).
- **Testing:**
  - one-sided p-values;
  - **Holm–Bonferroni** within each hypothesis family;
  - an intersection–union test for H1 ("beats every baseline").
- **Seeds:**
  - held-out qubits: 2026;
  - neural models: 0–4;
  - transpiler: 0;
  - twirling: 1234;
  - bootstrap: 0;
  - Monte-Carlo: one seed per decision, shared across policies.

---

## 9. System architecture and workflow

```text
Real IBM calibration records (Hugging Face, pinned)
   │  download + SHA-256 check
   ▼
Genuine events → calibration states (CCE) with as-of joins on publication time
   │                                   │
   ▼                                   ▼
Per-qubit windows → forecasters       Pair-level CZ series → pair forecasters
 (Stage 3, H1)                         (Stage 5a)
   │                                   │
   └──────────────┬────────────────────┘
                  ▼
Daily decision epochs × 4-qubit chains × horizons k = 1, 4, 7
   state estimates: latest calibration | forecasts | true future state
                  │
                  ▼
Noise model (Qiskit Aer) per state estimate ── 6 circuits × 14 actions, exact simulation
                  │
                  ├─► model-based controller: argmin predicted error   (persistence, forecasts, oracle)
                  ├─► best fixed (validation-selected)                 (global / per circuit)
                  └─► LinUCB bandit with Clifford-probe reward, delayed feedback
                  │
                  ▼
Evaluation on the true execution state: error per decision (shared shot-noise draws)
                  │
                  ▼
Paired cluster bootstrap, Holm → H2, H3a, H3b, H4; robustness R-A, R-B; exploratory analyses
```

---

## 10. Implementation details

**Entry point:** `reproduce.py`. The command `python3 reproduce.py --list` shows every step.

| Stage (`reproduce.py` name) | Modules | Output |
|---|---|---|
| `data` | `src/data/drift_dataset.py` | `data/processed/drift_v3/` |
| `forecast` | `experiments/stage3_forecasting.py`, `plot_stage3.py` | `results/stage3/`, `checkpoints/stage3/` |
| `pair_forecast` | `src/forecasting/pairs.py`, `experiments/stage5_pair_forecast.py` | `results/stage5/pair_forecast/`, `checkpoints/stage5_pair/` |
| `qem` | `experiments/stage5_{decisions,simulate,policies,stationary}.py`, `plot_stage5.py` | `results/stage5/` |
| `stage6` | `experiments/stage6_change_quartiles.py` | `results/stage6/` |
| `robustness` | `experiments/robust_*.py`, `src/qem/nq.py` | `results/robustness/` |
| `stage4` (optional) | `experiments/stage4_*.py` | `results/stage4/` (validation only) |
| `s5b` (optional) | `experiments/s5b_*.py` | `results/stage5b/` (validation only) |

**Test guards.** Test evaluations refuse to run unless the frozen inputs match `configs/frozen_stage3.yaml`, `frozen_stage5.yaml` or `frozen_robustness.yaml`.

**Software:** system Python 3.14.5 with no virtual environment. Pinned packages (`requirements.txt`, all verified installed):
- numpy 2.5.1, pandas 3.0.5, pyarrow 25.0.1, PyYAML 6.0.3;
- SciPy 1.18.0, scikit-learn 1.9.1, PyTorch 2.13.0 (CPU), Matplotlib 3.11.1;
- Qiskit 2.5.1, Qiskit Aer 0.17.2;
- huggingface_hub 1.28.0 (download only);
- pytest 9.1.1 (tests only).

**Hardware used:** Apple M4 Pro, 14 cores. Observed runtimes with 12 worker processes:

| Step | Runtime |
|---|---|
| Data | ~1 min |
| Stage 3 | ~45 min |
| Pair forecasting | ~10 min |
| Stage 5 simulation | 64 min (3,817 s) |
| Robustness simulations | 16 + 58 min |
| Stage 4 (optional) | ~65 min |
| Stage 5b (optional) | ~75 min |
| **Total** | **about 4–5 h for `all`** |

---

## 11. Experimental setup

**Separation of training, validation and test:**
- Forecasters are fitted on training data, and hyperparameters are selected on validation.
- **Selected on validation only:**
  - the best fixed actions;
  - bandit hyperparameters;
  - the forecaster used by each policy.
- **Test sets were evaluated once, after freezing:**
  - Stage 3 test (frozen 2026-10-12 15:32 UTC);
  - Stage 5 test (frozen before any Stage 5 test metric);
  - robustness test (separate freeze).
- **Freeze contents:** the freeze files hold SHA-256 hashes of every input, including code and the protocol. Stage 5 freezes 23 files.

**Information parity.** Every policy sees only data published up to its decision epoch.

**Robustness checks** (pre-registered in `docs/ROBUSTNESS_PLAN.md`):
- **R-A:** eight new random instances of the circuits (new seeds).
- **R-B:** 6-qubit chains (16 chains).
- Both at k = 1, development chains, R1 + R2.

**Leakage safeguards:**
- as-of joins on publication time;
- the epoch-time fix;
- split by target or execution time;
- held-out qubits and device excluded from all fitting and selection;
- assertions in code;
- tests in `tests/test_integrity.py`.

**Limits to the independence of the final evaluation:**
- There is one test period (6 weeks), and it has now been used:
  - the exploratory Stage 6 analysis looked at test results;
  - the Stage 5b idea was motivated by them (disclosed).
- The 7-day horizon has only 19 chains in test, and the unseen device has no 7-day test decisions.
- Any new method must be confirmed on calibration data published after 2026-10-12.

---

## 12. Results and analysis

### 12.1 Forecasting (Stage 3, test, development qubits; confirmatory for H1)

Per-qubit MAE of the log value, 1 / 7 calibrations ahead:

| Property | Persistence | Window mean | Ridge | Transformer |
|---|---|---|---|---|
| T1 | 0.306 / 0.311 | 0.229 / 0.234 | 0.233 / 0.235 | 0.229 / 0.232 |
| T2 | 0.236 / 0.241 | 0.178 / 0.182 | 0.181 / 0.184 | 0.180 / 0.183 |
| sx error | 0.241 / 0.326 | 0.242 / 0.251 | 0.222 / 0.245 | 0.216 / 0.242 |
| Readout error | 0.371 / 0.389 | 0.292 / 0.309 | 0.287 / 0.303 | 0.294 / 0.314 |
| CZ error (mean) | 0.191 / 0.246 | 0.191 / 0.207 | 0.183 / 0.201 | 0.173 / 0.200 |

Source: `results/stage3/metrics_test.csv`.

- **Against persistence:** the Transformer is better in every cell (9.7–25.7 %), and so are all learned models (4.6–25.7 %).
- **H1 (beats every baseline, k ≥ 2): 0 of 15 cells.** The window mean is as good for T1 and T2, and ridge is better for readout. The same holds for unseen qubits (Transformer vs persistence 8.0–28.3 %) and the unseen device (7.2–25.6 %).
- **Why:** day-to-day changes are large and mostly noise around a slowly moving level. 51 % of next-calibration T1 values differ by more than 25 %, so averaging captures most of what is predictable.

**Pair-level CZ forecasts** (test, development pairs, improvement over persistence; `results/stage5/pair_forecast/metrics_test.csv`):

| Model | 1 day | 4 days | 7 days |
|---|---:|---:|---:|
| Window mean | 3.5 % | 14.9 % | 14.2 % |
| Ridge | 11.0 % | 20.0 % | 19.2 % |
| Gradient-boosted trees | 12.2 % | 19.2 % | 18.0 % |
| Transformer | 11.5 % | 18.7 % | 18.1 % |

### 12.2 Primary decision result (H2; test; confirmatory)

| Comparison (Transformer-driven minus …) | Horizon / set | Relative difference | 95 % CI | Verdict |
|---|---|---:|---|---|
| Persistence (**primary**) | k = 1, development, 29 chains, 8,028 units | **+0.02 %** | [−0.29, +0.54] % | not supported |
| Persistence | k = 4 | −0.41 % | [−0.77, +0.07] % | n.s. after Holm |
| Persistence | k = 7 | −0.05 % | [−1.03, +1.49] % | n.s. |
| Persistence | unseen device, k = 1 | −0.29 % | [−0.95, +0.28] % | n.s. |
| Window mean / ridge | k = 1, 4, 7 | −1.12 % … +0.93 % | all include 0 | no difference |

Source: `results/stage5/hypotheses_test.json`.

- The share of the oracle gain captured is 0.789 (Transformer) versus 0.792 (persistence).
- No forecaster beat persistence at any horizon.

### 12.3 Adaptation versus fixed and bandit (H3, H4; test; confirmatory)

| Comparison | k = 1 | k = 4 | k = 7 |
|---|---|---|---|
| Transformer-driven − best fixed (H3a) | **−5.6 %** [−11.7, −2.0] | **−4.2 %** [−9.4, −0.9] | −0.3 % [−4.4, +6.7], n.s. |
| Transformer-driven − bandit, probe reward (H4) | **−15.0 %** [−17.9, −12.8] | **−18.7 %** [−24.1, −14.4] | **−14.5 %** [−17.8, −10.9] |

- **Stationary control (H3b):** −4.9 % [−7.4, −1.0] % against best fixed (30 chains, 180 units). Non-inferior, and in fact better.
- **The credit belongs to calibration-informed adaptation.** Persistence-driven adaptation achieves the same, because the Transformer adds nothing (§12.2).

**All policies on one scale** (test, k = 1, development chains, R1 + R2; 8,028 units, 29 chains; `results/stage5/consolidated_test.csv`, `experiments/stage5_consolidated.py`):
- Percentages are (policy − persistence) / persistence, so positive means a higher error than persistence.
- p-values are two-sided and Holm-adjusted over the nine comparisons.
- This is a descriptive summary computed after the pre-registered tests; it selects nothing.

| Policy | Mean error | vs persistence | 95 % CI | Holm p (two-sided) | Shots per observable |
|---|---:|---:|---|---:|---|
| Adaptive: latest calibration (persistence) | 0.0238 | — | — | — | 8,192 |
| No mitigation | 0.0462 | +94.43 % | [+64.14, +132.80] % | < 0.01 | 8,192 |
| Best fixed (global) | 0.0252 | +5.96 % | [+1.91, +12.61] % | < 0.01 | 8,192 |
| Best fixed (per circuit) | 0.0248 | +4.29 % | [+0.12, +10.97] % | 0.14 | 8,192 |
| Contextual bandit, probe reward | 0.0280 | +17.73 % | [+14.87, +21.46] % | < 0.01 | 7,168 + 1,024 probe |
| Contextual bandit, model warm start | 0.0267 | +12.24 % | [+8.73, +16.68] % | < 0.01 | 7,168 + 1,024 probe |
| Adaptive: window-mean forecast | 0.0241 | +1.15 % | [-0.51, +4.00] % | 1.00 | 8,192 |
| Adaptive: ridge forecast | 0.0238 | +0.07 % | [-0.22, +0.53] % | 1.00 | 8,192 |
| Adaptive: Transformer forecast | 0.0238 | +0.02 % | [-0.29, +0.54] % | 1.00 | 8,192 |
| Adaptive: true state (oracle, not realisable) | 0.0234 | -1.57 % | [-2.82, -0.70] % | < 0.01 | 8,192 |

The hypothesis table above reports each comparison relative to the *comparator's* error. For example, the Transformer-driven error is 15.0 % below the bandit's. Measured against persistence-driven adaptation, the bandit's error is 17.7 % higher. Both statements describe the same gap. All bandit results concern our reimplementation under our simulated conditions, not the published method in general.

Mean error per observable (test, development chains; source `hypotheses_test.json` → `by_regime`):

| Horizon, regime | None | Best fixed | Bandit | Persistence | Window mean | Ridge | Transformer | Oracle* |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 d, R1 | 0.0458 | 0.0246 | 0.0277 | 0.0239 | 0.0241 | 0.0240 | 0.0239 | 0.0235 |
| 1 d, R2 | 0.0467 | 0.0258 | 0.0283 | 0.0236 | 0.0240 | 0.0236 | 0.0237 | 0.0233 |
| 4 d, R1 | 0.0489 | 0.0263 | 0.0309 | 0.0259 | 0.0260 | 0.0259 | 0.0258 | 0.0251 |
| 4 d, R2 | 0.0496 | 0.0275 | 0.0324 | 0.0258 | 0.0259 | 0.0258 | 0.0257 | 0.0248 |
| 7 d, R1 | 0.0507 | 0.0267 | 0.0316 | 0.0272 | 0.0270 | 0.0271 | 0.0270 | 0.0258 |
| 7 d, R2 | 0.0514 | 0.0280 | 0.0323 | 0.0274 | 0.0271 | 0.0273 | 0.0275 | 0.0255 |

\* not realisable.

**Per-regime consistency** (k = 1, development chains; Transformer-driven adaptation compared with each alternative):

| Regime | vs persistence | vs best fixed | vs best fixed per circuit | vs bandit |
|---|---:|---:|---:|---:|
| R1 | −0.04 % | −2.9 % | −2.9 % | −13.5 % |
| R2 | +0.08 % | −8.2 % | −5.3 % | −16.5 % |
| R3 (mismatch) | −0.36 % | −1.2 % | +1.9 % | −10.2 % |

Each cell is a relative difference of mean errors from the `by_regime` table; the hypothesis CIs above use R1 + R2 pooled.

**Why the bandit underperforms** (test; `results/stage5/figures/probe_reward_quality.csv`):
- The Clifford-probe reward ranks the 14 actions poorly: median Spearman correlation with the true ranking is 0.03 (`qaoa_d2`) to 0.56 (`tfim_d6`).
- The probe finds the truly best action in only 6–14 % of states.
- The probes cost 12.5 % of the shots.

### 12.4 Value of information (test)

The oracle (perfect knowledge of the execution state) beats persistence by:
- 1.4–1.7 % at 1 day;
- 3.0–3.7 % at 4 days;
- 5.2–7.1 % at 7 days.

These figures are for development chains, R1/R2. In the robustness checks the 1-day ceiling is 2.5 % (R-A) and 2.6 % (R-B).

**Exploratory analysis by size of the device change** (post hoc; `results/stage6/h5_change_quartiles.csv`):
- Quartiles are of the CZ change between the published and the execution-time calibration.
- In the top quartile (median change ×2.3), the oracle beats persistence by 6.4 / 9.3 / 14.1 % (k = 1 / 4 / 7), and persistence is *worse* than best fixed (+1.0 / +3.0 / +5.9 %).
- In that quartile, the Transformer's CZ forecast is closer to the truth than persistence 90–97 % of the time. Even so, it recovers only 1.2–1.4 % (k = 1, 4) and nothing at k = 7, and it is slightly worse in the small-change quartiles.
- **Interpretation:** forecasts improve accuracy mainly by shrinking towards recent levels, which rarely changes the controller's ranking of actions. Forecast accuracy and decision value diverge.

### 12.5 Robustness (test; pre-registered)

| Transformer-driven adaptation vs | Main study | R-A: 8 new circuit instances | R-B: 6-qubit chains |
|---|---|---|---|
| Persistence | +0.02 % (n.s.) | −0.42 % [−0.98, +0.33] (n.s.) | −0.35 % [−0.95, +0.16] (n.s.) |
| Best fixed | −5.6 % (sig.) | −5.7 % [−11.1, −2.6] (sig.) | −3.6 % [−8.7, +0.04] (CI includes 0) |
| Best fixed per circuit | — | −4.4 % (sig.) | −3.6 % (CI includes 0) |
| Bandit (probe reward) | −15.0 % (sig.) | −17.8 % [−21.4, −15.0] (sig.) | not run (by design) |
| Oracle − persistence | 1.4–1.7 % | 2.5 % | 2.6 % |

Sources: `results/robustness/{ra,rb}/summary_test.json`.

- **Model mismatch (R3; exploratory H5):** when the controller's model omits the coherent phase, the advantage over the global best fixed action shrinks to 1.2 % (k = 1). Against the best fixed action per circuit it is lost (+1.9 %). This is the main practical risk of model-based control.

### 12.6 Validation-stage evidence (Stage 4; validation only; not confirmatory)
- On validation, persistence-driven adaptation beat best fixed by 3.8 % (R1) and 8.4 % (R2).
- Calibration-informed controllers captured 61–87 % of the oracle's gain over best fixed (45–87 % when regenerated with the current code). Bandit variants were below zero.
- These Stage 4 numbers were produced **before** the epoch-timing leakage fix, and they are superseded by Stage 5. See Appendix C.

### 12.7 Figures

| Figure | File | What it shows |
|---|---|---|
| Data coverage | `results/stage2/data_coverage.pdf` | calibrations per day and median T1 per backend, with split dates |
| Forecast error vs horizon | `results/stage3/figures/test/mae_vs_horizon_test_primary.pdf` | MAE by model and horizon, with CIs |
| Skill vs persistence | `results/stage3/figures/test/skill_vs_persistence_test_primary.pdf` | relative improvement over persistence |
| Example forecasts | `results/stage3/figures/test/examples_log_t1_k1_test_primary.pdf` | T1 series with forecasts |
| Generalisation | `results/stage3/figures/test/generalisation_test.pdf` | development vs unseen qubits vs unseen device |
| Pair forecast skill | `results/stage5/figures/pair_forecast_skill_test.pdf` | pair-level CZ forecasters |
| Hypotheses | `results/stage5/figures/hypotheses_test.pdf` | paired differences with CIs (forest plot) |
| Error by horizon | `results/stage5/figures/error_by_horizon_test.pdf` | mean error per policy and horizon |
| Probe reward quality | `results/stage5/figures/probe_reward_quality.pdf` | probe ranking vs true ranking |
| Value of information (validation) | `results/stage4/figures/val/value_of_information.pdf` | captured share of the oracle gain (Stage 4) |

All figures are Matplotlib output: PNG at 300 dpi plus PDF, with the plotted data as CSV.

---

## 13. Decision-focused Transformer experiment (exploratory; validation only)

**Why it was attempted.**
- Stage 5 showed that more accurate forecasts do not change decisions.
- A decision-focused model might succeed where accuracy-focused ones fail. It does this by learning directly to correct the controller's predicted error of each action from the chain's calibration history.
- It was pre-registered as H6 in `PROTOCOL_ADDENDUM.md` before any model was trained. The motivation, which included test-period findings, was disclosed.

**Method** (`experiments/s5b_train_targets.py`, `s5b_dft.py`, `s5b_report.py`):
- **Inputs:**
  - 20 epochs of chain calibration history;
  - the model-based predicted error of each action from the published state;
  - circuit, horizon and regime.
- **Target:** the simulated true error of each action at the execution epoch.
- **Data:** training-period decisions on development chains (k = 1, 4, 7; R1, R2; 6 circuits). 8,814 new simulation units were run for the targets (70 min).
- **Early stopping:** on the last 20 % of the training period. Validation decisions were never used for fitting or stopping.
- **Models, trained identically with 3 seeds each:**
  - a Transformer over the history;
  - an MLP on summary statistics;
  - a **no-history control**.
- **Training time:** 64 s for all 9 models.

**Results** (validation; mean error per observable; `results/stage5b/dft_validation_report.json`):

| Policy | k = 1 | k = 4 | k = 7 |
|---|---:|---:|---:|
| Persistence (model-based) | 0.02407 | 0.02553 | 0.02696 |
| DFT: Transformer | 0.02414 | 0.02552 | 0.02692 |
| DFT: MLP summary | 0.02423 | 0.02568 | 0.02698 |
| DFT: no-history control | 0.02402 | 0.02543 | 0.02696 |
| Oracle state* | 0.02359 | 0.02484 | 0.02626 |

\* not realisable.

- **H6 at k = 1:** DFT-Transformer − persistence = **+0.28 %**, CI [+0.14, +0.44] %. It is significantly worse, so the pre-registered criterion is **not met**.
- **At k = 4 and 7:** −0.05 % and −0.14 %, not significant.

**Artifact check.**
- The no-history control was slightly better than persistence (−0.22 % at k = 1). It cannot see drift, so its gain is consistent with a *static* correction to the simulator-based predictions (for example, systematic Monte-Carlo bias) rather than drift information. The experiment does not identify the mechanism.
- Adding history made the model worse (+0.51 % at k = 1, Transformer vs no-history), consistent with over-fitting.

**Conclusion.**
- The control results are consistent with the model learning a simulator-specific correction. The experiment does not establish that the Transformer learned decision-relevant drift information.
- The experiment was stopped as agreed, is reported as exploratory, and was **not** evaluated on test data.
- A confirmatory evaluation would need calibration data published after 2026-10-12.

---

## 14. Discussion

**What worked:**
- Using the published calibration to configure mitigation through a physics-based model:
  - against the best fixed configuration: 4–6 % lower error at 1–4 days, at zero extra shots;
  - against a reactive bandit: 15–19 % lower.
- The bandit fails here for a measurable reason: its feedback signal (a Clifford probe) carries little information about which action is best for the actual circuit.

**What did not work:**
- Forecasting the calibration, with a Transformer or with simpler models, did not improve decisions. This is despite measurably better forecasts (10–26 % lower error than persistence).
- **The benefit of calibration-informed adaptation must not be confused with a benefit of forecasting.** The former is real in this setting; the latter was not found.

**Why this matters:**
- For practitioners, the result points to a cheap, effective step: re-select the mitigation settings from each new calibration with a noise-model simulation.
- For research on learned mitigation, it shows that forecast accuracy is the wrong objective. The available value of foresight is small (about 1–7 %), concentrated in abrupt changes, and invisible to average-error metrics.

**Justified conclusions:** the four claims in `docs/audit/AUDIT_REPORT.md` §6. In short:
1. a benchmark on real drift;
2. the gains of calibration-informed adaptation;
3. the value-of-information bound;
4. the bandit diagnosis.

---

## 15. Limitations and threats to validity

1. **Simulation-to-hardware gap:**
   - no circuit ran on hardware;
   - readout, idle, leakage and crosstalk errors are not modelled;
   - coherent error is a single fixed CZ phase;
   - the noise model matches reported gate errors by construction, which real devices may not.
2. **Proxy ground truth:** the next published calibration stands in for the device state during execution.
3. **Dataset limitations:**
   - four devices of one vendor and one processor family;
   - one daily calibration cadence;
   - environmental covariates are 22–73 % missing;
   - CZ-error histories are pair-level and sometimes stale.
4. **Limited independent test evidence:** one test period of six weeks, now used. 7-day results rest on 19 chains, and the unseen device has no 7-day test decisions.
5. **Baselines:**
   - The best fixed action was chosen on one validation month. It was selected on validation, not on training as the protocol text says (disclosed).
   - The bandit reimplementation follows the paper where it is specific. Some choices (reward centring, initialisation, scaling) were ours, made to strengthen it.
6. **Model complexity and data:** the selected Transformer is the smallest in its grid. With about 190 calibrations per qubit, larger models over-fit.
7. **Circuits:** 4-qubit chains (6 in R-B) and six circuit types with 12–36 CZ gates (eight more instances in R-A).
8. **Runtime and cost:** the controller needs an exact simulation per decision, about 0.3 s for 4 qubits. This grows exponentially and needs a surrogate beyond roughly 10–12 qubits. The full pipeline takes 4–5 h on a 14-core laptop.
9. **Stage 4 results** predate the epoch-timing fix and are not regenerated identically by the current code (validation only; Appendix C).
10. **Title claims:** "Transformer-based" and "real-time" are not supported (§1).

---

## 16. Conclusion and future work

On real IBM calibration drift, in simulation:
- calibration-informed, model-based selection of ZNE and twirling settings reduced mitigated error by 4–6 % against the best fixed setting and by 15–19 % against the base paper's contextual bandit, at an equal shot budget;
- Transformer forecasting of the drift, and simpler forecasting, did not improve these decisions;
- the value of perfect foresight is small and concentrated in abrupt device changes.

**Future work** (not done):
1. Confirm on hardware with a small set of circuits on an IBM backend.
2. Confirm on calibration data published after 2026-10-12.
3. Train forecasters with decision-aware objectives (regret or ranking losses).
4. Gate forecasts so they are used only when an abrupt change is predicted.
5. Add readout and idle noise to the controller's model, and study robustness to model mismatch.
6. Replace exact simulation with a surrogate for larger circuits.

---

## 17. Reproducibility instructions

```bash
python3 -m pip install -r requirements.txt
python3 scripts/download_ibm_data.py          # pinned revision; fails on checksum mismatch
python3 reproduce.py --list                   # show all steps
python3 reproduce.py all                      # confirmatory pipeline, about 4–5 h with 12 workers
python3 reproduce.py stage4 s5b               # optional: validation-only development and exploratory stages
python3 -m pytest -q tests                    # 18 tests, a few seconds
python3 scripts/compare_reproduction.py <dir> # compare another reproduction with this one
```

**Expected outputs:** `results/stage3/`, `results/stage5/`, `results/stage6/` and `results/robustness/`. They should be identical to the committed files; this was verified byte-for-byte on 2026-10-13 for all 15 key files.

**Troubleshooting:**

| Problem | Cause and fix |
|---|---|
| "refusing test evaluation: changed after freeze" | An input differs from the frozen hash. Do not edit frozen files; re-run the earlier stages unchanged. A simulation cache that differs only in row order is accepted automatically |
| Download checksum error | The script pins the revision. A mismatch means a corrupted download; delete `data/raw/ibm_calibration_drift/data/` and retry |
| Runtime | Pass `--workers` to the simulation steps to match your CPU. Keep the machine awake: sleep pauses the run but does not change results |
| Memory | Density-matrix simulation of 6-qubit chains (robustness R-B) is the heaviest step. Reduce workers if memory is short |

---

## 18. References

All verified on 2026-10-13: 17 via Crossref DOI, 2 via arXiv. Full LaTeX entries are in `report/QEM_Project_Report.tex`.

1. J. Preskill, "Quantum computing in the NISQ era and beyond," *Quantum*, vol. 2, p. 79, 2018. doi:10.22331/q-2018-08-06-79
2. Z. Cai *et al.*, "Quantum error mitigation," *Rev. Mod. Phys.*, vol. 95, p. 045005, 2023. doi:10.1103/RevModPhys.95.045005
3. K. Temme, S. Bravyi, J. M. Gambetta, "Error mitigation for short-depth quantum circuits," *Phys. Rev. Lett.*, vol. 119, p. 180509, 2017. doi:10.1103/PhysRevLett.119.180509
4. Y. Li, S. C. Benjamin, "Efficient variational quantum simulator incorporating active error minimization," *Phys. Rev. X*, vol. 7, p. 021050, 2017. doi:10.1103/PhysRevX.7.021050
5. A. Kandala *et al.*, "Error mitigation extends the computational reach of a noisy quantum processor," *Nature*, vol. 567, pp. 491–495, 2019. doi:10.1038/s41586-019-1040-7
6. T. Giurgica-Tiron *et al.*, "Digital zero noise extrapolation for quantum error mitigation," *Proc. IEEE QCE*, 2020, pp. 306–316. doi:10.1109/QCE49297.2020.00045
7. J. J. Wallman, J. Emerson, "Noise tailoring for scalable quantum computation via randomized compiling," *Phys. Rev. A*, vol. 94, p. 052325, 2016. doi:10.1103/PhysRevA.94.052325
8. Y. Kim *et al.*, "Evidence for the utility of quantum computing before fault tolerance," *Nature*, vol. 618, pp. 500–505, 2023. doi:10.1038/s41586-023-06096-3
9. P. V. Klimov *et al.*, "Fluctuations of energy-relaxation times in superconducting qubits," *Phys. Rev. Lett.*, vol. 121, p. 090502, 2018. doi:10.1103/PhysRevLett.121.090502
10. T. Proctor *et al.*, "Detecting and tracking drift in quantum information processors," *Nat. Commun.*, vol. 11, p. 5396, 2020. doi:10.1038/s41467-020-19074-4
11. **R. Rahman, D. C. Nguyen, "Quantum noise mitigation with adaptive zero-noise extrapolation: A contextual multi-armed bandits approach," *IEEE J. Sel. Areas Commun.*, vol. 44, pp. 5600–5614, 2026. doi:10.1109/JSAC.2026.3721807** (base paper)
12. L. Hour, M. Go, Y. Han, "Improving zero-noise extrapolation for quantum-gate error mitigation using a noise-aware folding method," *IEEE Access*, vol. 14, pp. 37362–37372, 2026. doi:10.1109/ACCESS.2026.3672056
13. P. Czarnik *et al.*, "Error mitigation with Clifford quantum-circuit data," *Quantum*, vol. 5, p. 592, 2021. doi:10.22331/q-2021-11-26-592
14. J. Pérez-Guijarro, A. Pagès-Zamora, J. R. Fonollosa, "Extension of Clifford data regression methods for quantum error mitigation," *IEEE Trans. Quantum Eng.*, vol. 7, pp. 1–22, 2026. doi:10.1109/TQE.2026.3677541
15. M. Prodius *et al.*, "Robust design under uncertainty in quantum error mitigation," *IEEE Trans. Quantum Eng.*, vol. 7, pp. 1–13, 2026. doi:10.1109/TQE.2026.3680641
16. H. Liao *et al.*, "Machine learning for practical quantum error mitigation," *Nat. Mach. Intell.*, vol. 6, pp. 1478–1486, 2024. doi:10.1038/s42256-024-00927-2
17. L. Placidi *et al.*, "Deep learning approaches to quantum error mitigation," arXiv:2601.14226, 2026 (preprint).
18. L. Li, W. Chu, J. Langford, R. E. Schapire, "A contextual-bandit approach to personalized news article recommendation," *Proc. WWW*, 2010, pp. 661–670. doi:10.1145/1772690.1772758
19. A. Vaswani *et al.*, "Attention is all you need," *Advances in Neural Information Processing Systems*, vol. 30, 2017.
20. A. Javadi-Abhari *et al.*, "Quantum computing with Qiskit," arXiv:2405.08810, 2024.
21. "qiskit-calibration-drift," Hugging Face dataset `phanerozoic/qiskit-calibration-drift`, revision 539bc99, 2026. https://huggingface.co/datasets/phanerozoic/qiskit-calibration-drift

---

## Appendix A. Configurations

| Item | Value | File |
|---|---|---|
| Forecasting grids, seeds, training | — | `configs/stage3.yaml` |
| Selected forecasters (per-qubit) | — | `results/stage3/selected.yaml` |
| Selected forecasters (pair-level) | — | `results/stage5/pair_forecast/selected.json` |
| Best fixed actions and bandit hyperparameters per (k, regime) | — | `results/stage5/selections.json` |
| Example, k = 1 / R1 | best fixed `zne_exp_135`; bandit full context, η 0.1, ρ 1.0, λ_c 0.02, β 1.0 | same |
| Shots per observable per decision | 8,192 (bandit 7,168 main + 1,024 probe) | `src/qem/mitigation.py`, `src/qem/policies.py` |
| Scale factors / twirl instances / Monte-Carlo trials | {1, 2, 3, 5, 7} / 8 / 200 | `src/qem/mitigation.py` |
| Gate durations | sx 24 / 32 / 36 ns (fez / kingston / marrakesh), CZ 68 ns | `src/qem/device.py` |
| Coherent phase (R2, R3) | 0.03 rad per CZ | `experiments/stage5_policies.py` |
| Frozen inputs | — | `configs/frozen_*.yaml` |

## Appendix B. Protocol deviations and corrections

All are listed in `docs/PROTOCOL_ADDENDUM.md`:
- the epoch-time leakage fix (before any test);
- the best fixed action selected on validation;
- the order-insensitive cache digest;
- the per-stage freeze file names;
- the audit's document corrections (rounding, chain counts, the "every learned model" range);
- the Stage 5b pre-registration and its outcome.

## Appendix C. Reproduction record (2026-10-13)

**Confirmatory pipeline**, reproduced from scratch on a clean copy (code, configs and raw data only):
- These were byte-identical to the committed results:
  - `cce.parquet`, `pair_epochs.parquet`;
  - Stage 3 metrics and H1 (validation and test);
  - pair metrics (validation and test);
  - Stage 5 decisions, forecast states, selections, hypotheses (validation and test) and H3b;
  - R-A and R-B test summaries.
- The Stage 5 simulation cache came out in a different row order (parallel workers) with identical content.
- **Stage 6 analysis:** regenerated byte-identically.

**Stage 5b (exploratory):** regenerated byte-identically, including neural-network training (`dft_results.parquet`, `dft_validation_report.json`).

**Stage 4 (validation only):** the saved results predate the epoch-timing leakage fix, so the current code produces different decisions. The decision count is the same (2,970 development units per regime); decision times shift. The conclusions are unchanged:

| Validation, development chains | Saved (pre-fix) | Regenerated (current code) |
|---|---|---|
| Persistence-driven − best fixed, R1 / R2 / R3 | −3.8 / −8.4 / −2.0 % | −3.7 / −8.4 / −2.1 % |
| Transformer-driven − persistence-driven, R1 / R2 / R3 | +0.24 / +0.02 / +0.32 % | +0.07 / −0.06 / +0.39 % |
| Transformer-driven − bandit (probe), R1 / R2 / R3 | −12.0 / −17.1 / −11.1 % | −12.6 / −16.2 / −10.1 % |
| Bandit (probe) − best fixed, R1 / R2 / R3 | +9.5 / +10.6 / +10.6 % | +10.3 / +9.3 / +9.3 % |
| Share of oracle gain captured, adaptive policies | 61–87 % | 45–87 % (lowest: window mean under mismatch) |
| Share captured, bandit variants | below zero | below zero |

The committed `results/stage4/` files are the original (pre-fix) run, which the Stage 4 report describes. Running `python3 reproduce.py stage4` produces the regenerated values.

## Appendix D. Code listings and circuit diagrams

The report's appendix (`report/QEM_Project_Report.tex`) lists the core of the controller and the noise model. To draw circuit diagrams, run:

```python
from src.qem.benchmarks import all_benchmarks
for b in all_benchmarks():
    print(b["name"]); print(b["native"].draw())
```

This prints all six native circuits.
