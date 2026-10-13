# Stage 1 — Research Proposal

*Transformer-Based Real-Time Adaptive Quantum Error Mitigation for Non-Stationary Noise*
Status: **draft for approval**, 2026-10-12. Everything in §1–§2 is verified. Everything from §3 on is **proposed** and has not been run.

---

## 1. Base paper: verified record

| Field | Value | How verified |
|---|---|---|
| Authors | Ratun Rahman, Dinh C. Nguyen (The University of Alabama in Huntsville; ORCIDs 0000-0001-6718-0112, 0000-0002-8092-6756) | Crossref |
| Title | Quantum Noise Mitigation With Adaptive Zero-Noise Extrapolation: A Contextual Multi-Armed Bandits Approach | Crossref + arXiv |
| Venue | *IEEE Journal on Selected Areas in Communications* (ISSN 0733-8716 / 1558-0008), **journal article** | Crossref `type: journal-article` |
| Volume / pages / year | vol. 44, pp. 5600–5614, 2026 (issue number not yet in Crossref) | Crossref |
| DOI | **10.1109/JSAC.2026.3721807** | Crossref |
| Official page | IEEE Xplore document **11644486** — https://ieeexplore.ieee.org/document/11644486/ | Crossref `resource.primary` |
| Status | Published (Crossref lists a version-of-record licence). Preprint: arXiv:2608.06426 v1, 5 Aug 2026, "Accepted at IEEE JSAC" | arXiv + Crossref |
| Code | https://github.com/Ratun11/cmab_zne (public, 7 commits; **no licence shown**) | GitHub page |

Not verified: the Xplore page text itself, because IEEE blocks automated access (HTTP 403). Open the DOI once in a browser before citing.

**IEEE citation:**
R. Rahman and D. C. Nguyen, "Quantum noise mitigation with adaptive zero-noise extrapolation: A contextual multi-armed bandits approach," *IEEE J. Sel. Areas Commun.*, vol. 44, pp. 5600–5614, 2026, doi: 10.1109/JSAC.2026.3721807.

## 2. Full review of the base paper

The arXiv full text was read in full: body, equations and tables, then references and author bios.

**Problem.** ZNE normally uses fixed or grid-searched fold scales, which assumes the noise is static. On NISQ hardware the noise drifts (thermal fluctuations, crosstalk, calibration cycles), so a fixed choice either wastes executions or misfits the extrapolation.

**Method (CMAB–ZNE).**
- **Arms:** fold-scale sets {1,3}, {1,3,5}, {1,3,5,7}. Folding is U(U†U)ⁿ, λ = 1+2n. The extrapolator is fixed per arm: Richardson or polynomial with degree ≤ 2, clipped to [−1, 1].
- **Algorithm:** cost-aware **LinUCB** with discount ρ < 1 for drift:
  - k_t = argmax_k (x_tᵀθ̂_k + η√(x_tᵀA_k⁻¹x_t) − λ_c C̃_k)
  - A ← ρA + xxᵀ, b ← ρb + Rx
- **Reward:** R = −CE − β·Var[Ê(0)] − α(d(Λ) − d₀)².
- **Context:** training epoch, loss and accuracy at λ = 1, variance across scales {1,3}, and batch size. **Hardware telemetry is not used** (it is listed as future work).
- **Cost:** C̃ is normalised round trips, bytes and runtime.
- **Stopping:** a budgeted stopping rule ends exploration early.
- **Theory:** a regret bound under a linear-reward assumption (proved for the non-discounted update).

**Setup.**
- Hybrid VQC classifiers (ResNet-18 or CNN features → AngleEmbedding + StronglyEntanglingLayers, 4–32 wires, 3–7 layers) on CIFAR-10 and EuroSAT.
- PennyLane `default.mixed` with depolarising, bit-flip, phase-flip, amplitude- and phase-damping channels, 1,024 shots.
- **Drift is a prescribed proxy** η̂ₜ that rises from about 0.25 to 1.0 over about 40 epochs. No formula is given and no measured calibration data is used.
- Hardware: Rigetti Cepheus-1-108Q via Amazon Braket, **frozen inference only**: 30 samples, 500 shots per scale, 240 tasks, about $369.

**Baselines.** Fixed {1,3,5}, grid-search ZNE, and 11 reimplemented methods: CDR, VD, PEC, RC, RL, HW-aware optimisation, standard ZNE, ZNE + twirling, noise-aware folding, global folding, Adaptive KIK.

**Results.** Against fixed {1,3,5}: −38.7 % / −41.2 % round trips, about −35 % bytes, about −30 % end-to-end cost, and +6.9 % / +6.4 % "estimator fidelity". Accuracy 85.58 % vs 85.24 % against the best baseline (CIFAR-10).

**Limitations and assumptions we observed.**
1. **Reactive.** No forecast; adaptation relies on recent feedback plus discounting.
2. **Synthetic drift.** No real calibration time series.
3. **Exploration cost.** Probe batches are spent learning.
4. **Twirling and the extrapolator are not adapted.** Twirling appears only as a baseline.
5. **Task metric.** Classification accuracy and cross-entropy, not the error of a physical observable.
6. **Small hardware test:** inference only, one device, 30 samples.
7. **Underspecified.** β, λ_c, ρ, η, the ridge λ₀, probe batch size and stopping thresholds are not given numerically. The definition of "estimator fidelity" is ambiguous (the abstract and Table V use different references). The method section describes circuit folding, while the simulation scales channel strength, p(λ) = λ·p_base.
8. **Linear-reward assumption.** The regret proof rests on it, and the authors note that polynomial fits may underfit strongly nonlinear noise.

## 3. Positioning against related work

| | Rahman & Nguyen (IEEE JSAC 2026) | Placidi et al. (arXiv 2601.14226, **preprint**) | Our project |
|---|---|---|---|
| Goal | Choose ZNE fold set online under drift | Learn noisy→ideal output-distribution map | Choose ZNE + extrapolator + twirling **ahead of time** from forecast drift |
| Uses real drift data | No (prescribed proxy) | Device data, but not as a time series | **Yes** (IBM calibration history) |
| Forecasting | No | No | **Yes** (Transformer, multi-horizon) |
| Learner's role | Decision policy (bandit) | Mitigation map itself (seq2seq Transformer) | Forecaster; the decision is model-based |
| Extra shots for learning | Probe and exploration shots | Large training sets of executed circuits | None online (offline simulator table) |
| Physics retained | ZNE | Black-box | ZNE + Pauli twirling (physics-based estimators) |
| Hardware | Rigetti, inference | IBM ≤ 5 qubits | None yet (simulation) |

Placidi et al. is **related work**. It shows Transformers can *be* the mitigator, whereas we use a Transformer to *forecast the device* and keep a physics-based estimator.

## 4. Research gap

The base paper adapts ZNE **reactively** to a **synthetic** drift signal using **exploration shots**, and **adapts only the fold set**. No verified 2026 IEEE work tests whether **forecasting real calibration drift** lets a controller choose the mitigation protocol **before** execution, at no extra shot cost, and whether that beats (i) the best fixed protocol, (ii) reacting to the latest calibration, and (iii) a contextual bandit.

This gap is narrow and testable. It is genuine only if the forecast changes decisions *for the better*. Our pilot (§7) shows that this is not yet established.

## 5. Primary research question and hypotheses

**RQ.** Can forecasting non-stationary quantum hardware noise with a Transformer improve the selection of quantum error mitigation strategies compared with fixed mitigation, persistence-based adaptation, and contextual-bandit adaptation under a fair execution budget?

Hypotheses are fixed **before** the expanded experiments. Statistical unit: a *decision* is one (device, qubit group, decision time, circuit). Inference uses a **block bootstrap clustered by qubit series**, because decisions on the same qubit are correlated, with Holm correction across horizons × regimes. Test-set results are looked at **once**, after all choices are frozen on validation.

| ID | Hypothesis | Primary metric | Success criterion (all must hold) |
|---|---|---|---|
| **H1** forecasting | For horizons k ≥ 2 calibration cycles, the Transformer's forecast error is lower than persistence, the rolling mean, and the best classical model | Per-series MAE (log-scale for T1/T2, absolute for error rates), per horizon | 95 % CI of the MAE difference < 0 against **every** baseline, and ≥ 5 % relative reduction |
| **H2** decision value *(primary)* | Transformer-driven adaptation gives lower mitigated-observable error than persistence-driven adaptation | Mean \|Ê − E_ideal\| per observable at equal total shots | CI < 0 after Holm correction in the non-stationary regime, and the Transformer captures ≥ 10 percentage points more of the oracle gain (oracle gain = error(best fixed) − error(oracle)) |
| **H3** vs fixed | Transformer adaptation beats the best fixed protocol (chosen on train) under non-stationary noise, and is non-inferior under stationary noise | Same | Non-stationary: CI < 0. Stationary: upper CI bound < +2 % (non-inferiority margin) |
| **H4** vs bandit | Transformer adaptation beats a faithful CMAB baseline at equal **total** shots, with the bandit's probe and exploration shots counted | Same, plus shots and circuits | CI < 0 against CMAB-probe. CMAB with an oracle reward is reported as an optimistic reference, not tested |
| **H5** robustness *(exploratory)* | Behaviour under abrupt changes (chip-wide recalibrations, TLS jumps) and under simulator–device model mismatch | Error conditioned on event windows | Descriptive only; no success claim. A bandit is *expected* to be more robust to model mismatch |

**Pre-declared failure handling.** If H2 fails, we diagnose with the following and report the negative result:
- *When does forecast accuracy translate into a different decision?* Decision-boundary analysis: how close the true T1 is to a configuration switch point.
- Forecast error by horizon.

**No claim of Transformer superiority will be made without H2.**

## 6. Proposed method (summary; details in the project plan)

1. **State** (real data). For each backend/qubit: T1, T2, sx/x error, readout error, prob_meas0_prep1 / prob_meas1_prep0, and CZ error on coupled pairs. Inputs also include calibration age and recalibration-event flags. Environmental covariates go in an ablation only.
2. **Forecast.** A multi-horizon Transformer (positional encoding, horizon-conditioned output, quantile or log-variance head) predicts each property k cycles ahead. Baselines: persistence, rolling or EWMA mean, a classical model (ridge-AR or gradient-boosted trees on lags), and an LSTM to test whether attention is necessary.
3. **Device simulation** (labelled *simulated*). An Aer noise model built from a calibration snapshot (thermal relaxation from T1/T2 with gate durations, depolarising terms matched to the reported gate errors, readout assignment errors), plus a stationary coherent ZZ term set by regime. Ground truth at decision time = the snapshot actually published at the execution time. This is a proxy, because calibration data cannot reveal the state between calibrations (threat to validity).
4. **Actions.** ZNE scale sets {1,3}, {1,3,5}, {1,3,5,7}, {1,2,3} (the base paper's arms plus one) × extrapolator {linear, Richardson, exponential} × twirling {off, on}, plus "no mitigation". Infeasible combinations are pruned.
5. **Controller.** An offline, simulator-derived expected-error surface over the predicted noise state gives argmin. It updates once per decision epoch, using only data published before the decision time, and is **predictive** because it acts on the forecast for the execution time k cycles ahead.
6. **Fair baselines.**
   - (a) No mitigation.
   - (b) Best fixed, chosen on the train period.
   - (c) Persistence-adaptive.
   - (d) Rolling-mean-adaptive.
   - (e) **CMAB-probe:** LinUCB as in Eqs. 16–20, same arms, context = the same calibration telemetry plus its own feedback. The reward comes from Clifford probe circuits with known ideal values, and probe shots are deducted from the same budget. Unspecified hyperparameters are tuned on validation.
   - (f) **CMAB warm-start** from the same simulator table, so it does not lose for lack of prior knowledge.
   - (g) CMAB with an oracle reward (optimistic reference).
   - (h) Oracle-T1/state adaptation (upper bound).
7. **Experiments.**
   - Circuit families: layered hardware-efficient, Trotterised Ising, QAOA-MaxCut, mirror/Clifford.
   - Depths: 2–4 settings. Qubits: 4–8 (density matrix).
   - Regimes: stationary control, real drift, real drift + model mismatch.
   - At least 5 seeds. Train / val / test by time × held-out qubits × held-out backend.

## 7. Pilot evidence already in hand (current code, verified)

- One-step T1 forecast, test split: Transformer MAE 26.8 µs vs persistence 28.6 and rolling mean 29.9.
- Decision level (coherent regime): Transformer-adaptive −9.5 % against best fixed (significant), but **not distinguishable from persistence** (−0.3 %, CI includes 0).
- **Implication:** H2 is a genuine risk. A one-step horizon, T1-only state, a single circuit, and 10 qubits are the most likely reasons. All are addressed by Stages 2–5.

## 8. Does our method beat the base paper? A candid pre-assessment

| Dimension | Likely stronger than base paper | Likely weaker than base paper |
|---|---|---|
| Realism of drift | Real multi-backend IBM calibration histories vs a prescribed proxy | — |
| Shot efficiency | No online exploration shots | Our controller relies on a simulator model; a bandit learns from reality |
| Action space | Adapts extrapolator and twirling too | — |
| Metric rigour | Observable error vs exact ideal values; paired, clustered CIs; pre-registered hypotheses | — |
| Theory | — | They prove a regret bound; we have none |
| Scale | — | Their circuits reach 32 qubits; density-matrix simulation limits us to about 8–10 |
| Hardware | — | They executed on Rigetti hardware; we are simulation-only unless IBM access is arranged |
| Venue fit | — | JSAC framing emphasises communication cost; our contribution fits quantum-engineering venues better |

**Bottom line (pre-results).** The method is *plausibly* a stronger study than the base paper on realism, fairness and rigour. It is *not yet* shown to give a better method. That is decided by H2 and H4. If both fail, the paper becomes a rigorous negative or benchmark result, which suits a conference better than a journal.

## 9. Contributions (conditional on results)

1. A leakage-safe benchmark of **real IBM Heron calibration drift** for QEM decision-making: multi-property, multi-backend, with whole-qubit and whole-device hold-outs. *(Unconditional.)*
2. A **forecast-driven, zero-exploration controller** that jointly selects ZNE scales, extrapolator and Pauli twirling. *(Unconditional as a method; its value depends on H2–H4.)*
3. A **fair head-to-head comparison against a faithful contextual-bandit adaptive ZNE**, with probe-shot accounting. *(Unconditional.)*
4. Evidence of **when forecasting helps or doesn't**, by horizon, property, regime and model mismatch. *(Unconditional; the direction depends on results.)*

## 10. Threats to validity (declared up front)

- The true noise between calibrations is unobservable, so the next published calibration is used as a proxy.
- Simulation-only device behaviour, and a noise model built from calibration summaries (no crosstalk or leakage unless modelled).
- The controller and the device share a simulator family. The model-mismatch regime tests this.
- Base-paper hyperparameters are unspecified, so CMAB tuning is done on validation and documented.
- The `SN` column is CC-BY-NC, so it is excluded from features. The dataset as a whole is CC-BY-4.0.
