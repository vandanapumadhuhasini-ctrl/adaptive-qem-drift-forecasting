# Stage 4 — Forecast-Driven Adaptive QEM Controller and Fair Baselines

Run 2026-10-12. **Validation period only** (target days 2026-08-01 → 2026-08-31); no test-period quantity has been computed.
Design: `docs/research/STAGE4_DESIGN.md`. Framing (approved): forecast-driven versus reactive versus bandit adaptation; the forecaster may be simple.

**Data and simulation:** device states are **real IBM calibration records**; circuit execution is **simulated** (Qiskit Aer density matrix).

## 1. What was built

| Component | File | Check |
|---|---|---|
| Decision epochs (chip-wide daily T1 recalibration), 4-qubit chains from the real coupling map, pair-level CZ states as published | `src/qem/device.py`, `experiments/stage4_decisions.py` | 30 development chains (fez 12, kingston 18; 15 and 20 were requested, but the coupling graph of eligible qubits allows fewer disjoint chains), 15 on unseen marrakesh (16 requested). Decisions: 495 validation and 757 test (development); 69 validation and 149 test (marrakesh) |
| Noise model from a calibration snapshot: relaxation (real durations: sx 24/32/36 ns, CZ 68 ns) plus depolarising top-up to IBM's reported sx/CZ errors; optional coherent CZ phase δ | `src/qem/device.py` | **Modelled gate infidelity within 0.07 % of reported** |
| 6 benchmark circuits (HEA, TFIM, QAOA × 2 depths; 12–36 CZ), observables with \|ideal\| > 0.2, Clifford probes | `src/qem/benchmarks.py` | Probes verified Clifford |
| CZ folding (scales 1, 2, 3, 5, 7), CZ Pauli twirling (8 instances), 14 shared actions, vectorised shot-noise estimators | `src/qem/mitigation.py` | **Noiseless folded and twirled circuits equal the ideal exactly**; vectorised estimators equal the reference to 1e-15 |
| Model-based predictive controller; LinUCB bandit (base paper Eqs. 17–19) in probe / warm / oracle-reward variants | `src/qem/policies.py`, `experiments/stage4_policies.py` | Bandit converges in a toy test (95–100 %); shot budget asserted |
| Simulation cache | `experiments/stage4_simulate.py` | 3,384 validation units, 252 stationary, 20,568 training-history units for the bandit |

## 2. Fairness provisions

| Item | How it is enforced |
|---|---|
| Same action set | All policies select from the same 14 actions, including the base paper's {1,3}, {1,3,5}, {1,3,5,7} |
| Same budget | 8,192 shots per observable per decision. The bandit spends 7,168 on the main estimate and 1,024 on probes (asserted in code) |
| Same information | Every policy sees only data published at the decision epoch. The bandit additionally gets its own past feedback and is allowed to learn online from the **first training epoch** (≈ 80 epochs per stream before validation) |
| Same scoring | Every policy is scored on the same simulated outcome arrays. The controller's predictions use common random numbers across state sources; the R4 control confirms identical choices when states are identical (difference exactly 0) |
| Bandit not weakened | Hyperparameters tuned on validation per regime (context full/intercept, η incl. 0, ρ, λ_c, β). Implementation choices where the paper is silent: reward scaling, causal reward centring, play-each-arm-once initialisation. Without the last two, even the true-error-rewarded bandit was worse than a fixed action (artifact found and fixed; `PROTOCOL.md` §9) |

## 3. Results (validation, development chains; mean |error| per observable)

| Policy | R1 incoherent | R2 coherent | R3 mismatch | R4 stationary |
|---|---:|---:|---:|---:|
| No mitigation | 0.04676 | 0.04746 | 0.04746 | — |
| Best fixed (global) | 0.02516 | 0.02610 | 0.02610 | 0.01806 |
| Best fixed (per circuit) | 0.02514 | 0.02526 | **0.02526** | — |
| CMAB, probe reward (realistic) | 0.02756 | 0.02886 | 0.02886 | — |
| CMAB, model warm start | 0.02664 | 0.02617 | 0.02819 | — |
| CMAB, oracle reward* | 0.02570 | 0.02550 | 0.02550 | — |
| Adaptive: persistence | **0.02420** | 0.02392 | 0.02557 | 0.01786 |
| Adaptive: window mean | 0.02437 | 0.02400 | 0.02566 | 0.01786 |
| Adaptive: ridge | 0.02422 | **0.02386** | 0.02556 | 0.01786 |
| Adaptive: Transformer | 0.02426 | 0.02392 | 0.02565 | 0.01786 |
| Adaptive: oracle state* | 0.02394 | 0.02354 | 0.02538 | 0.01786 |

\* not realisable; reference only.

Paired differences (95 % cluster bootstrap over 28 chains within backend; full table in `results/stage4/summary_val.json`):

| Comparison | R1 | R2 | R3 |
|---|---|---|---|
| Transformer − persistence (H2 analogue) | **+0.24 %** [+0.00001, +0.00012] | +0.02 % [−0.00005, +0.00007] | +0.32 % [−0.00003, +0.00020] |
| Transformer − window mean | −0.47 % (significant) | −0.34 % (significant) | −0.01 % |
| Transformer − ridge | +0.14 % | +0.24 % (significant) | +0.39 % |
| Adaptive (persistence) − best fixed | −3.8 % [−0.0030, +0.0001] | **−8.4 %** [−0.0047, −0.0009] | −2.0 % (n.s.) |
| Oracle state − best fixed (headroom) | −4.9 % | −9.8 % | −2.8 % (n.s.) |
| Transformer − CMAB-probe | **−12.0 %** | **−17.1 %** | **−11.1 %** |
| CMAB-probe − best fixed | **+9.5 %** | **+10.6 %** | **+10.6 %** |

Unseen device (ibm_marrakesh, 14 chains):
- Transformer − persistence: −1.9 % (R1), −1.3 % (R2), −0.5 % (R3), all with CIs that include 0.
- Captured share of the oracle gain: persistence 48 %, Transformer 63 %, ridge 71 % (R2).

Costs:
- **Shots:** identical by construction.
- **Execution cost (R2):** total folded length is similar for the adaptive policies (Σ scales ≈ 7.3–8.0) and lower than best fixed (9.0). Adaptive policies also execute fewer distinct circuits (5–6 against 24, because best fixed uses twirling). The probe-reward bandit averages Σ scales 9.8 and 9.9 circuits.
- **The bandit's probes cost 12.5 % of the shots.**
- **The controller's classical cost:** exact simulation of 45 circuits per state source (about 0.3 s for 4 qubits); it grows exponentially with qubit count.

## 4. Findings

1. **Calibration-informed, model-based adaptation beats fixed mitigation.**
   - In the coherent regime the gain is −8.4 % (significant). Incoherent: −3.8 % (borderline). Stationary control: −1.1 %.
   - Under model mismatch the gain shrinks to about 2 % and is not significant. A well-chosen fixed action per circuit then matches even the oracle-state controller.
2. **At a 1-day horizon, forecasting adds nothing measurable over persistence.**
   - Transformer, ridge and window mean are within ±0.5 % of persistence. The Transformer is slightly *worse* in R1 (+0.24 %, CI excludes 0).
   - The Transformer is better than the window mean but not better than ridge.
   - **Mechanism (diagnosed):**
     - circuit error is dominated by CZ error (median 2.5×10⁻³, against ≈ 2×10⁻⁴ from relaxation);
     - the published pair value is a strong predictor: its pair-level error against the execution state is 0.195 (log MAE), against 0.208 for the Transformer-based mapping and 0.246 for the window mean. *Correction (Stage 5 analysis): pair CZ values are genuinely recalibrated about daily (median age at decision 22.5 h; most hourly events are re-timestamps), not hourly. The forecasts did worse mainly because per-qubit CZ averages are a poor proxy for a specific pair, which motivates the pair-level forecaster in Stage 5;*
     - forecasting improves T1/T2 (0.23 vs 0.31) but those barely move the decision;
     - **value-of-information ceiling:** even the oracle state beats persistence by only 0.8–1.6 % of error (0.00019–0.00038).
3. **The faithful contextual bandit is the weakest adaptive approach: about 10 % worse than never adapting.**
   - Causes, measured:
     - (a) the Clifford-probe reward ranks actions poorly. The median per-decision Spearman correlation with the true error ranking is 0.44–0.45 (about 0 for `qaoa_d2`), and the probe identifies the true best action in only 11–12 % of decisions;
     - (b) the probes cost 12.5 % of the shots (≈ +0.001 error);
     - (c) with ≈ 90 epochs per stream and 14 arms, the contextual model cannot be learned. Tuning prefers intercept-only, η = 0.
   - Even with an unrealisable true-error reward, the bandit only reaches the best fixed action per circuit.
   - The base paper's setting (supervised classification) supplies labels that give a reward. Estimating observables on an arbitrary circuit has no such signal, which is precisely where model-based control is advantageous.
4. **On the unseen device the forecasters help more** (ridge / Transformer capture 63–71 % of the oracle gain, against persistence 48 % in R2), but the CIs include 0 with 14 chains.

**Answer to "does our method beat the base paper's approach?"** In this simulated setting with real calibration drift, **yes, decisively against the bandit** (−11 % to −17 % error, significant in every regime, with fewer shots used for the main estimate). The advantage comes from **calibration-informed model-based control**, not from the Transformer. The primary hypothesis H2 (Transformer beats persistence) is **not supported on validation** at k = 1, and the protocol does not allow claiming it.

## 5. Acceptance

| Criterion (design §7) | Status |
|---|---|
| 1. Noise model reproduces reported sx/CZ errors within 5 % | ✅ within 0.07 % |
| 2. Folded/twirled circuits noiseless-equivalent | ✅ exact |
| 3. Equal shot budget per decision, probes counted | ✅ asserted |
| 4. Information parity enforced | ✅ by construction; R4 identical choices (exact 0 difference) |
| 5. CMAB sanity convergence | ✅ 95–100 % |
| 6. Validation results for all policies, no test numbers | ✅ |

## 6. Limitations

- **Epoch-time leakage bug (found in Stage 5, fixed):** 250 training-period decisions (bandit history) and 73 test decisions had distorted epoch times. No validation decision was affected, so the validation conclusions above stand, but the bandit's training history was partly built from wrongly timed states. Stage 5 rebuilds everything from the corrected epochs (`PROTOCOL.md` §9).

- Simulation only: no hardware execution. Readout and idle errors are not modelled. 4-qubit chains (density-matrix limit).
- The execution-state proxy is the next published calibration.
- **Pair-level CZ forecasts are derived from per-qubit Stage 3 forecasts, not modelled directly.**
- Best fixed is selected and evaluated on the same validation data, which favours the baseline. In Stage 5 it will be selected on validation and evaluated on test.
- 28 development chains: CIs on absolute comparisons are wide (between-chain variance).

## 7. Recommendation for Stage 5 (for approval)

The approved framing holds, but the evidence points to where a forecasting contribution could still exist:

1. **Run the pre-declared longer horizons, k = 4 and 7 days** (planning ahead, or jobs queued for days). Persistence goes stale there, while gate-error forecasts stayed flatter in Stage 3. This is the main remaining test of "does forecasting help".
2. **Forecast the decision-relevant quantity.** Add a **pair-level CZ-error forecaster** (same Stage 3 protocol, validation-only selection), since CZ error drives the decisions.
3. **Report the value-of-information analysis as a contribution:** an upper bound on what any forecaster can add, measured on real drift.
4. **Broaden as planned:** test period (once, after a freeze), unseen device, more circuit instances/seeds, plus a 6-qubit variant to check that the conclusions hold.
5. **Treat the bandit's reward-signal problem as a central result** (probe-ranking diagnostics), not a footnote.
