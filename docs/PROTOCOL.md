# Pre-registered Evaluation Protocol

**Frozen on 2026-10-12, before any expanded-dataset model was trained or any test data was examined.**
Approved basis: `STAGE1_RESEARCH_PROPOSAL.md` (H1–H5), approved by the project owner on 2026-10-12.

Changes after this date are allowed only for reasons unrelated to test outcomes (for example, a bug fix). Each change must be listed in §9 with its date and reason **before** the test set is evaluated. Thresholds in §5 cannot be changed after test results are seen.

**Central question.** Can Transformer-based forecasting of real quantum hardware calibration drift improve quantum error mitigation decisions compared with fixed mitigation, persistence-based adaptation and contextual-bandit adaptation, under a fair execution and shot budget?

---

## 1. Data (real hardware calibration records)

- **Source:** `phanerozoic/qiskit-calibration-drift`, revision `539bc9994d664a7b559dee6edf12d95976f58af2`. Parquet SHA-256 `6c9feb0f…752f0b` (see `data/raw/ibm_calibration_drift/PROVENANCE.json`). CC-BY-4.0. The `SN` column (CC-BY-NC) is never used.
- **Unit of time:** a *coherence-calibration event* (CCE) of a qubit = one T1 calibration event. T2 is attached by as-of lookup; it falls within 1 min of T1 in 91 % of cases.
- **Genuine measurement:** an event whose value differs from the previous event of the same `(backend, property, qubit_a, qubit_b)` series. Re-timestamped unchanged values are not new observations.
- **Unit harmonisation:** T1/T2 rows with NULL `unit` are seconds and are multiplied by 10⁶ (µs). Error and probability properties are dimensionless in both schemas.
- **Failure ceilings** (`is_failure_ceiling`) are invalid measurements. They are excluded as targets and masked in features (last valid value carried forward, plus a flag).

## 2. Splits (fixed now)

Time cut points are fixed calendar dates: **T_val = 2026-08-01 00:00 UTC** and **T_test = 2026-09-01 00:00 UTC**. On the development backends they leave 68.8 % / 12.1 % / 19.1 % of T1 events before T_val, between the cuts, and after T_test (measured 2026-10-12, before any model was fitted).

| Set | Backends | Qubits | Target time |
|---|---|---|---|
| Train | ibm_fez, ibm_kingston | development qubits (80 %) | < T_val |
| Validation | ibm_fez, ibm_kingston | development qubits | [T_val, T_test) |
| **Test-primary** | ibm_fez, ibm_kingston | development qubits | ≥ T_test |
| Test-unseen-qubit | ibm_fez, ibm_kingston | held-out qubits (20 %, drawn with seed 2026 per backend) | ≥ T_test |
| Test-unseen-device | ibm_marrakesh (all qubits) | all | ≥ T_test |
| Out-of-family (descriptive) | ibm_torino (Heron r1, retired) | all | all |

**Leakage rules:**
- A sample belongs to the set of its **target** time. Inputs may use any data published **before the decision time**, as in deployment.
- Scalers and hyperparameters are fitted on Train or Validation only.
- Held-out qubits and ibm_marrakesh are never used for training or for any selection.
- Automated assertions in the data pipeline enforce these rules.

## 3. Forecasting (H1)

- **Targets:** log T1, log T2, log 1-qubit gate error (sx), log readout error, and log mean CZ error over the qubit's couplings.
- **Horizons:** k ∈ {1, 2, 4, 7} CCEs ahead. The decision time is the time of the last CCE in the input window.
- **Primary metric:** per-series MAE on the log scale (≈ relative error).
  - For model m and series s: MAE_{m,s} = mean |ŷ − y| over the test samples of s.
  - Aggregate: MAE_m = mean over series of MAE_{m,s}, so each qubit counts equally.
- **Relative improvement** of m over baseline b: R_{m,b} = 1 − MAE_m / MAE_b.
- **CI:** cluster bootstrap with B = 10,000. Series are resampled with replacement within backend, and R is recomputed on each resample. 95 % percentile interval.
- **Baselines:** persistence (last value), rolling mean of the last 5 CCEs, ridge-AR on lags, gradient-boosted trees on lags, LSTM. Baseline hyperparameters are tuned on Validation exactly like the Transformer's.

## 4. Decisions (H2–H5)

- **Decision:** at decision time τ (just after a CCE), for a qubit group G (connected physical qubits) and circuit C, a policy picks one action from the shared action set 𝒜. The circuit then executes on the device state at the target CCE (k cycles ahead). The *published* calibration at that CCE is the proxy for the true device state (a declared threat to validity).
- **Error of a decision:** err_π(d) = (1/|O|) Σ_{o∈O} E_shots |Ê_o − E_o^ideal|.
  - The expectation over shot noise uses M = 200 Monte-Carlo trials with **common random numbers**: the same seeds are used for every policy on the same decision.
  - The total shot budget per decision is S, identical for all policies. For bandits, probe or exploration shots are deducted from S.
- **Paired difference:** Δ_{π,π'} = mean_d [err_π(d) − err_π'(d)].
- **Oracle gain:** G = mean_d [err_BF(d) − err_OR(d)], where BF = best fixed (chosen on Train) and OR = adaptation using the true state.
- **Captured fraction:** C_π = mean_d [err_BF(d) − err_π(d)] / G.
- **CI:** cluster bootstrap with B = 10,000, resampling qubit groups within backend. All policies are recomputed on the same resample, which keeps them paired.

## 5. Success criteria (thresholds are pre-declared targets, not expected outcomes)

| Hyp. | Cell(s) | Statistical criterion | Practical criterion |
|---|---|---|---|
| H1 | each (target, k ≥ 2) on Test-primary | lower 95 % CI bound of R_{TF,b} > 0 for **every** baseline b | point estimate R_{TF,b} ≥ 0.05 for every b |
| **H2 (primary)** | real-drift regime, **k = 1**, all circuits pooled, Test-primary | upper 95 % CI bound of Δ_{TF,P} < 0 | C_TF − C_P ≥ 0.10 (point estimate). *Not evaluable* if the 95 % CI of G includes 0, because there would be no headroom to capture |
| H3a | real-drift, each k | upper CI bound of Δ_{TF,BF} < 0 | — |
| H3b | stationary control | non-inferiority: upper CI bound of (mean err_TF / mean err_BF − 1) < +0.02 | — |
| H4 | real-drift, each k | upper CI bound of Δ_{TF,CMAB-probe} < 0 | — |
| H5 | event windows, model mismatch | descriptive only | — |

**How the thresholds are interpreted:**
- "Beats every baseline" (H1) is an intersection–union test. It is valid at α = 0.05 without adjustment within a cell.
- **Multiplicity:** Holm–Bonferroni at family-wise α = 0.05 within each hypothesis family across its cells (H1: targets × horizons; H3a/H4: horizons). H2 has a single pre-declared primary cell, so no correction is applied. The other H2 cells (k ∈ {2, 4, 7}, the unseen-qubit and unseen-device sets) are secondary and Holm-corrected.
- **Why k = 1 is primary:** it is the real-time setting the project title refers to (a job runs before the next calibration). Longer horizons are secondary.
- The 5 % (H1) and 10-percentage-point (H2) thresholds are **practical-significance targets** on point estimates. The statistical criteria guard against chance. Both must hold for a "success" claim.
- The 2 % margin (H3b) is a non-inferiority margin on the **upper** confidence bound, so it is a statistical criterion.

## 6. Selection discipline

- Hyperparameters, architectures, action-set pruning and CMAB settings are chosen on **Validation only**.
- The test sets are evaluated **once**, by a script that refuses to run unless `configs/frozen.yaml` exists. That file records the config hash and date.
- Every test run is logged in §9.

## 7. Reporting

- Report all cells, including failures.
- Report effect sizes with CIs, not only p-values.
- Simulation-derived quantities are labelled *simulated*. Calibration data is labelled *real hardware calibration data*.

## 8. Pilot results (pre-protocol, for transparency)

These were obtained on the 1,000-record sample, a different split, and the T1-only state. They are not part of the confirmatory analysis:
- One-step T1 forecast: Transformer beat persistence (MAE 26.8 vs 28.6 µs).
- Decisions: Transformer-driven ≈ persistence-driven (−0.3 %, CI includes 0).

## 9. Change log

| Date | Change | Reason | Before test? |
|---|---|---|---|
| 2026-10-12 | Protocol frozen | — | yes |
| 2026-10-12 | §2 wording: the cut dates had been described as the 70 %/85 % quantiles; measured shares are 68.8 %/12.1 %/19.1 %. Dates unchanged | Factual correction found during split verification; no model trained, no test data examined | yes |
| 2026-10-12 | Data rules fixed during Stage 2 construction: (a) clock skew — 22 events with calibrated_time after their publishing poll (≤ 6 min) are treated as available at the later timestamp; (b) a gap > 168 h starts a new segment and windows never cross it (202 gaps, 0.2 %); (c) probability floor 1/16384 before log transforms; (d) T2 > 2·T1 (0.27 % of states) kept in data, clamped to 2·T1 only inside the noise simulator; (e) split labels are assigned per horizon from that horizon's target time | Data-construction decisions; no model fitted, no test data examined | yes |
| 2026-10-12 | Stage 3: the "window mean (20 CCEs)" forecaster was added to the H1 baseline set before tuning (the set became stricter, not looser) | Natural shrinkage baseline for a mean-reverting process | yes (added before any validation result) |
| 2026-10-12 | **Stage 3 test evaluated once** (freeze 2026-10-12T15:32:43Z, commit 1252ab0e). H1 not supported in any of the 15 cells | — | — (this is the test run) |
| 2026-10-12 | **Pre-declared for Stage 4, before any Stage 4 work:** besides the H2 primary comparison (Transformer- vs persistence-driven adaptation, unchanged), add *secondary* comparisons against window-mean-driven and ridge-driven adaptation, Holm-corrected, so that smoothing gains are not attributed to the Transformer. Thresholds unchanged | Stage 3 showed simple shrinkage forecasters match the Transformer; required for a fair Stage 4 | yes (Stage 4 not started) |
| 2026-10-12 | Stage 4 design fixed in `STAGE4_DESIGN.md`: chip-wide daily epochs, 4-qubit chains, multi-property noise model (sx/CZ errors matched to IBM reports), readout excluded (shared readout mitigation assumed), regimes R1–R4, 14-action set incl. base-paper arms {1,3},{1,3,5},{1,3,5,7} | Design | yes (no test data used) |
| 2026-10-12 | Bandit implementation fixes found on **validation**: (a) play-each-arm-once initialisation and causal reward centring — without them the zero ridge prior plus all-negative rewards makes unexplored arms look best (the true-error bandit was worse than a fixed action); (b) context choice {full, intercept} added to the tuning grid; (c) exploration η grid extended to include 0. Purpose: a fair, non-strawman baseline | Faithful baseline; documented as our choices where the paper is silent | yes |
| 2026-10-12 | Controller Monte-Carlo predictions now use common random numbers across state sources (they had differed by seed). Found because the R4 stationary control (identical states) showed spurious differences of ~0.1 %, the same scale as the effects under study | Variance reduction / correctness of paired comparison | yes |
| 2026-10-12 | Added "best fixed per circuit" as an additional, stronger static baseline (alongside the pre-declared global best fixed) | Stricter comparison | yes |
| 2026-10-12 | **Leakage bug fixed (Stage 4 epochs).** The epoch time was the *latest* publication of the day; late publications (initial snapshots, outages) pushed 53 of 628 epoch times forward by up to 54 days, so as-of lookups could include later-published values. New definition: epoch time = moment by which 90 % of the day's calibrations were published, the day is valid only if that is within 6 h of the median publication, and calibrations published after the epoch time are unavailable. Affected Stage 4 decisions: 0 validation, 250 training-period (bandit history), 73 test (never evaluated). All Stage 5 caches are rebuilt from the corrected epochs | Leakage correction | yes (test not evaluated) |
| 2026-10-13 | Stage 5 definitions (validation evidence only): (a) the CZ part of every forecaster's state is now a **pair-level CZ forecast** of the same model family (Stage 5a), replacing the Stage 4 mapping from per-qubit averages. H2's comparison (Transformer- vs persistence-driven adaptation) is unchanged in form; (b) horizons k = 4, 7 require each chain qubit's next k calibrations to fall in the next k epochs (Stage 3 horizon alignment); the unseen-device set is therefore evaluated at k = 1 only (7 test decisions at k = 4); (c) the bandit learns from the last 60 training epochs plus validation, with feedback delayed by k epochs; (d) noise model: float64 states, superoperator composition, unphysical forecast gate errors clipped at 0.5; (e) **H2 primary cell = k = 1, R1 + R2 pooled, development chains, test** — as pre-declared | Stage 5 design; no test data examined | yes |

