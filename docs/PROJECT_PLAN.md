# Project Plan — Forecast-Driven Adaptive QEM

**Repository:** developed locally on the system Python (no virtual environment); published as a single release commit.
**GitHub:** the old remote repository was deleted by the owner. No pushes and no new repository until Stage 7. The local remote was renamed `old-origin-deleted`. Local commits continue.

Last updated 2026-10-12. Status values: ✅ done and checked, 🟡 in progress or awaiting approval, ⬜ not started.

Every stage ends with a **"vs. base paper"** check: does this stage make our study stronger than Rahman & Nguyen (IEEE JSAC 2026), weaker, or neither?

| Stage | Status | Gate to proceed |
|---|---|---|
| 1 Base paper + research direction | ✅ approved 2026-10-12; protocol frozen in `PROTOCOL.md` | — |
| 2 Dataset expansion | ✅ acceptance criteria met (see `STAGE2_DATA_REPORT.md` §8) | — |
| 3 Forecasting | ✅ done; **H1 not supported** (see `STAGE3_FORECASTING_REPORT.md`) | — |
| 4 Controller + fair baselines | ✅ done (validation only; see `STAGE4_CONTROLLER_REPORT.md`) | — |
| 5 Broad experiments | ✅ done incl. robustness R-A/R-B (`STAGE5_RESULTS.md`, `STAGE6_CONTRIBUTION.md` §3) | — |
| 6 Contribution validation | 🟡 drafted (`STAGE6_CONTRIBUTION.md`); title / framing decision pending | Owner decision on framing |
| 7 Reproducibility + new GitHub repo | 🟡 migration done (see below); GitHub deferred to the end | Final verification; commit review |
| 8 IEEE submission | ⬜ | Stage 6 evidence |

---

## Stage 1 — Base paper and research direction

| Task | Status | Evidence |
|---|---|---|
| Verify citation, DOI, venue, status, official page | ✅ | Crossref record; IEEE Xplore doc 11644486; arXiv 2608.06426 |
| Read full paper | ✅ | arXiv HTML read end to end |
| Compare with our implementation and Placidi et al. | ✅ | Proposal §3 |
| Gap, RQ, hypotheses, metrics, success criteria | ✅ drafted | Proposal §4–§5 |
| Your approval | ✅ 2026-10-12 | Owner message; thresholds treated as pre-declared targets |
| Protocol with exact metric, CI and correction definitions | ✅ | `docs/PROTOCOL.md` (frozen before any expanded-data modelling) |

**Acceptance:** citation verified by ≥ 2 authoritative records ✅; hypotheses each have a metric, a test and a threshold ✅; approval pending.
**vs. base paper:** defines a study that is stricter (real drift, observable error, exploration shots counted). Whether the method is better is still open.

## Stage 2 — Dataset (improvement a)

**Depends on:** download permission.

**Tasks:**
1. ✅ Obtain `phanerozoic/qiskit-calibration-drift` (70.5 MB parquet, CC-BY-4.0, revision 539bc99, SHA-256 matches the HF LFS record) → `data/raw/ibm_calibration_drift/` (read-only, with `PROVENANCE.json`).
2. Profile the full schema: backends, qubits, properties, units, schema change around 2026-05-07 → 05-13, missing values, duplicates, `is_new_measurement`, `is_failure_ceiling`, chip-wide recalibration events.
3. **Count independent measurements.** De-duplicate on `(backend, property, qubit_a, qubit_b, calibrated_time)`; the 30-minute heartbeat replicas are not observations.
4. Harmonise units (T1/T2 s → µs), drop failure ceilings, and collapse the bit-identical sx/x/id/rx errors into one 1-qubit error. Drop rz_error (always zero) and the CC-BY-NC `SN` column.
5. Per-qubit state table on each property's own calibration clock. The model input is the latest value of each property *as known at decision time*: as-of joins only, never future values.
6. Splits:
   - **chronological:** train ≤ T₁ < val ≤ T₂ < test, with a gap of at least the window length so windows never straddle a boundary;
   - **whole-qubit hold-out:** a random 20 % of qubits per backend, never seen in training;
   - **whole-device hold-out:** one backend, e.g. `ibm_marrakesh`, with `ibm_torino` (Heron r1, retired) as an out-of-family check.
7. Data-quality report plus counts table. The data loader is versioned.

**Acceptance:**
- every count is reproducible from a script;
- zero leakage, checked by automated assertions (max train time < min val time, disjoint qubit IDs, scaler fitted on train only);
- the report lists coverage and limitations.

**vs. base paper:** stronger. Real multi-device drift instead of a prescribed proxy.

## Stage 3 — Forecasting (improvement b)

**Depends on:** Stage 2.

**Tasks:**
1. Targets: log T1, log T2, 1-qubit error, readout error, CZ error (pair-level).
2. Horizons: k ∈ {1, 2, 4, 7} calibration cycles of each property. Wall-clock equivalents are reported too.
3. Models: persistence; rolling/EWMA mean; ridge-AR on lags; gradient-boosted trees on lags; LSTM; **Transformer** (fix: positional encoding, smaller feed-forward width, horizon-conditioned multi-output head, optional quantile head).
4. Hyperparameter search on **validation only**, ≥ 5 seeds.
5. Metrics: MAE and RMSE per horizon and property; per-qubit and per-device breakdown; clustered bootstrap CIs; prediction-interval coverage if a quantile head is used.
6. **Decision relevance:** feed each forecaster into the same controller and measure the decision error (this links to H2).

**Acceptance:**
- every number is regenerated from a config;
- H1 is evaluated exactly as pre-registered;
- negative results are reported.

**Status (2026-10-12): ✅ complete.** Details in `STAGE3_FORECASTING_REPORT.md`.

| Task | Status / evidence |
|---|---|
| 1. Targets | ✅ 5 log targets; CZ averaged per qubit (pair-level forecasting deferred) |
| 2. Horizons | ✅ k = 1, 2, 4, 7 CCEs (median 24.8 h per CCE). Wall-clock equivalents not tabulated separately |
| 3. Models | ✅ persistence, rolling 5, window mean 20, ridge, gradient-boosted trees, LSTM, Transformer with positional encoding, small FF, multi-horizon Δ head. Quantile head **not** implemented (deferred) |
| 4. Validation-only tuning, ≥ 5 seeds | ✅ grids in `configs/stage3.yaml`; neural models are 5-seed ensembles |
| 5. Metrics | ✅ per-series MAE + 95 % cluster-bootstrap CI, RMSE, natural-unit MAE, large-change MAE, per-backend and per-qubit, 3 test sets |
| 6. Decision relevance | ✅ *preliminary* (pilot controller): forecasting cuts regret 28–40 % vs persistence; no Transformer-specific gain |
| H1 | ❌ **not supported** in any of the 15 pre-registered cells (window mean / ridge / trees match or beat the Transformer) |

**Remaining limitations:**
- the selected Transformer is at the grid boundary;
- no train + validation refit;
- no predictive-uncertainty head;
- natural-unit ranking differs for sx error;
- one unseen device.

**vs. base paper:** adds forecasting, which the base paper lacks. The evidence supports *forecasting* (shrinkage) over reacting to the latest state, but **not** the Transformer specifically.

## Stage 4 — Controller and fair baselines

**Depends on:** Stage 3 (validation results).

**Tasks:**
1. **Device noise model from a snapshot:** thermal relaxation (T1, T2, gate durations), depolarising error matched to the reported gate errors, readout assignment error, plus a regime-dependent coherent ZZ term. Unit tests check that the noisy simulation reproduces the reported error rates within a stated tolerance.
2. **Action set** (shared by every policy): {none} ∪ {{1,3}, {1,3,5}, {1,3,5,7}, {1,2,3}} × {linear, Richardson, exponential} × {twirl off, on}, pruned to valid combinations.
3. **Controller:** offline expected-error surface over the noise state, evaluated at the forecast state for the execution time. Documented decision cadence and information set.
4. **CMAB baseline,** faithful to Eqs. 16–20:
   - LinUCB with discount and cost term;
   - reward from Clifford probe circuits with probe shots deducted;
   - variants: cold-start, warm-start, oracle-reward;
   - hyperparameters tuned on validation and documented as *our choice*, since the paper does not specify them.
5. **Information-parity audit:** a table of what each policy sees at decision time.

**Acceptance:**
- every policy uses the same total shot budget per decision (verified by assertions);
- the CMAB reproduces the base paper's qualitative behaviour (converges to a {1,3}-type arm when noise is low) on a sanity scenario.

**vs. base paper:** stronger on fairness and action space. Weaker if the simulator model is mismatched to the device; H5 tests this.

**Stage 4 status (2026-10-12): ✅ complete on validation.**
- All 6 design acceptance checks pass.
- Model-based adaptation beats best fixed (R2 −8.4 %, significant) and the faithful bandit (−11 % to −17 %, significant).
- The Transformer does not beat persistence at k = 1 (≤ ±0.5 %); the value-of-information ceiling is 0.8–1.6 %.
- The bandit is limited by its reward signal: Clifford-probe ranking ρ ≈ 0.45.
- Details: `STAGE4_CONTROLLER_REPORT.md`.

**vs. base paper:** stronger, verifiably against the bandit approach, under real drift and equal budgets. The gain comes from model-based control, not from the Transformer.

## Stage 5 — Broad experiments (improvement d)

**Depends on:** Stage 4, with the analysis code frozen.

**Design:**
- 4 circuit families × 2–3 depths × 4–8 qubits;
- 3 regimes (stationary control, real drift, drift + model mismatch);
- multiple observables per circuit;
- ≥ 5 seeds;
- every policy evaluated on identical decisions.

**Ablations:**
- forecaster → persistence;
- horizon k;
- feature groups (T1 only → all calibration → + environment);
- ZNE only / twirling only / both;
- extrapolator fixed vs adapted;
- Transformer vs LSTM vs gradient-boosted trees.

**Cost reporting:** shots including probes, number of distinct circuits, classical compute time (forecast plus controller), simulator wall time.

**Acceptance:** test evaluated once; full tables and figures generated from the saved results by one command.

**vs. base paper:** broader circuit and regime coverage; smaller qubit counts.

**Stage 5 status (2026-10-13): core complete; test evaluated once after freeze (`1ecf9351`).**
- H2 not supported at any horizon or on the unseen device.
- H3a supported at k = 1, 4; H3b supported; H4 supported at every k.
- Value-of-information ceiling grows with horizon (1.4 → 7.1 %), but no forecaster captures it.
- Pending: extra circuit instances (seeds) and a 6-qubit robustness check.

## Stage 6 — Contribution validation

Answer with evidence:
1. Transformer vs persistence (H2).
2. vs best fixed (H3).
3. vs CMAB (H4).
4. Consistency across horizons, devices, circuits and regimes.
5. Effect size relative to overhead.
6. Behaviour under abrupt changes and inaccurate forecasts (H5).

If H2 fails: run the decision-boundary diagnosis, revise on *validation* evidence only, and report the original failure.

**Acceptance:** the contribution statement cites only verified numbers.

## Stage 7 — Reproducibility and engineering

**Migration (done 2026-10-12):**
- Every file was checksum-compared: identical apart from 3 docs written after the copy (now synced) and `.git/index`, a stat cache.
- `git fsck --full` passes; HEAD and history are identical.
- A full from-scratch rerun (retrain + QEM) in a clean copy reproduced every result file, figure and the checkpoint weights **bit-for-bit**.

**Tasks:**
- At the end: final verification, `.gitignore` review, publication-quality README, **new** GitHub repository, push. The old history contains `qem_env/` and a 244 MB CSV, so the new repository should start from a clean history; the exact approach is to be decided with the owner.
- One YAML config per experiment.
- `make`-style entry point.
- Seeds logged, environment pinned, data provenance and licence file.
- **GitHub:** the existing history contains `qem_env/` and a 244 MB CSV, so a new remote or a history rewrite must be chosen with you. Commit contents are reviewed before any push.

## Stage 8 — IEEE submission

Venue assessment only after Stage 6. Candidate venues (to be checked against scope and requirements): IEEE Transactions on Quantum Engineering (journal), IEEE QCE / Quantum Week (conference), IEEE JSAC special issues (base-paper venue; communications framing). Acceptance cannot be guaranteed.

---

## Change log

| Date | Change |
|---|---|
| 2026-10-12 | Stage 1 approved; `PROTOCOL.md` frozen; active repo moved to `~/Projects` (verified); old GitHub remote deleted by owner, local remote renamed; IBM drift parquet downloaded (rev 539bc99, SHA-256 verified) |
| 2026-10-12 | v2 pipeline (unit fix, per-qubit windows, chronological split); QEM simulation pilot; cleanup (`qem_env` deleted, old artifacts moved out of the project); verified copy to `~/Projects`; base paper verified; Stage 1 proposal and this plan written |
| 2026-10-12 | `archive/` removed from the project at the owner's request (identical copy remains in the Desktop backup). Policy: no backup/archive folders inside the project; version history = local git commits |
| 2026-10-12 | Stage 2 done: `src/data/drift_dataset.py`, `src/data/windows.py`, `configs/drift_data.yaml`; 101,891 CCEs from 600 qubits; leakage assertions passing; mid-May device event on fez/kingston verified as physical (not an artifact) |
| 2026-10-12 | Stage 3 done: scikit-learn 1.9.1; 7 forecasters + 3 Transformer ablations × 4 horizons × 5 properties × 4 evaluation sets; test evaluated once after freeze (1252ab0e). Transformer beats persistence by 9.7–25.7 % but not window mean / ridge / trees (H1 not supported). Decision preview: forecasting cuts pilot-controller regret 28–40 % vs persistence, with no Transformer-specific gain |
| 2026-10-12 | Stage 4 done (validation only): device/benchmark/mitigation/policy modules; all acceptance checks pass; bandit fairness fixes and common-random-number controller logged in PROTOCOL §9. Model-based adaptation > best fixed > CMAB; Transformer ≈ persistence at k = 1 |
| 2026-10-13 | Stage 5: epoch-time leakage fixed; pair CZ forecaster; k = 1/4/7 decisions; confirmatory cache; validation selection; freeze; single test evaluation. H2 not supported; H3a (k = 1, 4), H3b and H4 supported |
| 2026-10-13 | Robustness R-A (8 circuit instances) and R-B (6-qubit chains) pre-registered, frozen and tested once: conclusions unchanged. Stage 6 contribution document drafted; title change recommended |
