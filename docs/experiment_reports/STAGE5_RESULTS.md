# Stage 5 — Confirmatory Experiments and Hypothesis Tests

Run 2026-10-12/13. The test period (executions 2026-09-01 → 2026-10-12) was evaluated **once**, after freezing all inputs (`configs/frozen_stage5.yaml`, commit `1ecf9351`).

**Data and simulation:** device states are **real IBM calibration records**; circuit execution is **simulated** (Qiskit Aer density matrix, 4-qubit chains). No hardware execution.

## 1. What changed relative to Stage 4 (all before the test)

1. **Leakage fix.** Epoch times were distorted by late publications (53/628 days). Fixed and logged (`PROTOCOL.md` §9). No validation decision had been affected.
2. **Pair-level CZ forecaster** (Stage 5a). CZ error drives the mitigation decision, so each forecaster's state now uses a pair-level CZ forecast from its own model family.
3. **Horizons k = 1, 4, 7 days** between decision and execution. Execution states are shared across horizons; the controller uses one common-random-number seed per decision across all sources and horizons.
4. **Bandit with feedback delayed by k**, online from the last 60 training epochs through validation into test. Hyperparameters tuned on validation.
5. **Noise-model numerics:** float64 states, superoperator composition, unphysical forecast gate errors clipped at 0.5. Real-state channels are unchanged (verified).

## 2. Pair-level CZ forecasting (test; MAE of log CZ error, reduction vs persistence)

| Set | k | Window mean | Ridge | Trees | Transformer |
|---|---|---:|---:|---:|---:|
| Development pairs | 1 | 3.5 % | 11.0 % | 12.2 % | 11.5 % |
| | 4 | 14.9 % | **20.0 %** | 19.2 % | 18.7 % |
| | 7 | 14.2 % | **19.2 %** | 18.0 % | 18.1 % |
| Unseen-qubit pairs | 1 / 4 / 7 | 3.2 / 13.7 / 14.3 % | 10.9 / **18.4** / **17.8** % | 11.7 / 16.3 / 15.4 % | 10.1 / 16.4 / 16.8 % |
| Unseen device | 1 / 4 / 7 | 8.5 / 20.4 / 21.8 % | 10.9 / **22.2** / **23.4** % | 12.0 / 20.6 / 20.6 % | **13.2** / 20.3 / 21.8 % |

Pair-level forecasts are 10–23 % more accurate than persistence. The Transformer matches but does not beat ridge or trees. (On validation the k = 1 gain was about 0–2 %; the test period has more day-to-day CZ volatility.)

## 3. Pre-registered hypotheses (test)

Development chains, real drift R1 + R2 pooled; 95 % cluster bootstrap (chains within backend); Holm correction within families.

| Hypothesis | Cell | Result | Verdict |
|---|---|---|---|
| **H1** forecasting | 15 cells (Stage 3) | Transformer never beats all baselines | **Not supported** (Stage 3) |
| **H2 primary** | k = 1, 29 chains, 8,028 units | Transformer − persistence = **+0.02 %**, CI [−0.29 %, +0.54 %]. Captured share of oracle gain 0.789 vs 0.792 (diff −0.002, CI [−0.07, +0.08]). Oracle gain significant | **Not supported** |
| H2 secondary | k = 4 / k = 7 / unseen device k = 1 | −0.41 % (p = 0.04, not significant after Holm) / −0.05 % / −0.29 % | Not supported |
| **H3a** vs best fixed | k = 1 / 4 / 7 | **−5.6 %** / **−4.2 %** (both Holm-significant) / −0.3 % (n.s.; 19 chains) | **Supported at k = 1, 4** |
| **H3b** stationary non-inferiority | 30 chains, 180 units | −4.9 %, CI [−7.4 %, −1.0 %] (margin +2 %) | **Supported** (non-inferior; in fact better) |
| **H4** vs CMAB-probe | k = 1 / 4 / 7 | **−15.0 % / −18.7 % / −14.5 %**, all Holm-significant | **Supported** |
| vs window-mean / ridge adaptation | k = 1, 4, 7 | within −1.1 % … +0.9 %, none significant | No difference |
| H5 (exploratory) | R3 mismatch | adaptive policies roughly tie with best fixed per circuit | See §5 |

Full numbers: `results/stage5/hypotheses_test.json`, `h3b_stationary.json`. Figure: `results/stage5/figures/hypotheses_test.pdf`.

## 4. Mean error per observable (test, development chains)

| k / regime | None | Best fixed | CMAB-probe | Persistence | Window mean | Ridge | **Transformer** | Oracle* |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 / R1 | 0.0458 | 0.0246 | 0.0277 | 0.0239 | 0.0241 | 0.0240 | 0.0239 | 0.0235 |
| 1 / R2 | 0.0467 | 0.0258 | 0.0283 | 0.0236 | 0.0240 | 0.0236 | 0.0237 | 0.0233 |
| 4 / R1 | 0.0489 | 0.0263 | 0.0309 | 0.0259 | 0.0260 | 0.0259 | 0.0258 | 0.0251 |
| 4 / R2 | 0.0496 | 0.0275 | 0.0324 | 0.0258 | 0.0259 | 0.0258 | 0.0257 | 0.0248 |
| 7 / R1 | 0.0507 | 0.0267 | 0.0316 | 0.0272 | 0.0270 | 0.0271 | 0.0270 | 0.0258 |
| 7 / R2 | 0.0514 | 0.0280 | 0.0323 | 0.0274 | 0.0271 | 0.0273 | 0.0275 | 0.0255 |

\* not realisable.

## 5. Findings

1. **Calibration-informed, model-based adaptation is the effective component.**
   - Against the best fixed action: −5.6 % (k = 1) and −4.2 % (k = 4), both significant.
   - Against the faithful contextual bandit of the base paper: −15 % to −19 % at every horizon, significant.
   - It is non-inferior (better) under stationary noise.
   - It needs no extra quantum shots. The controller's cost is classical simulation.
2. **Forecasting does not improve mitigation decisions over persistence, at any horizon tested** (H2 not supported).
   - This holds even though the pair-level forecasts are 10–23 % more accurate.
3. **The headroom grows with the horizon, but is not captured.**
   - Perfect knowledge of the execution state (oracle) beats persistence by 1.4–1.7 % (k = 1), 3.0–3.7 % (k = 4) and 5.2–7.1 % (k = 7).
   - Every forecaster captures about none of it. The gain in forecast MAE comes from shrinkage toward recent levels, which does not change the controller's *ranking* of actions.
   - **Forecast accuracy and decision value diverge.** A decision-focused forecasting objective (predict-then-optimise / decision-aware loss) is the natural next step.
4. **The contextual bandit is limited by its reward signal.**
   - On test, the Clifford-probe ranking of actions correlates weakly with the true ranking: median Spearman 0.03 (`qaoa_d2`) to 0.56 (`tfim_d6`).
   - The probe identifies the true best action in only 6–14 % of states (`figures/probe_reward_quality`).
   - With delayed feedback (k = 4, 7) it degrades further.
   - Even with an unrealisable true-error reward it only matches a well-chosen fixed action per circuit (−0.5 % to +7.6 %).
5. **Model mismatch (R3, exploratory H5):** when the controller's noise model misses a coherent error, model-based adaptation's advantage over the best fixed action per circuit largely disappears. This is the main practical risk of the model-based approach.

## 6. Answer to the research question

> *Can Transformer-based forecasting of real calibration drift improve QEM decisions compared with fixed mitigation, persistence-based adaptation and contextual-bandit adaptation under a fair budget?*

- **Against fixed mitigation and bandit adaptation: yes.** But the improvement comes from calibration-informed model-based control, which persistence achieves equally.
- **Against persistence-based adaptation: no.** Not at 1, 4 or 7 days, not on the unseen device, and not for any forecaster (Transformer, ridge, window mean).
- **The paper cannot claim Transformer superiority.** It can claim, with evidence:
  - (i) a fair benchmark on real IBM drift;
  - (ii) model-based adaptive ZNE + twirling beats the bandit approach of the base paper and fixed mitigation;
  - (iii) a quantified value-of-information ceiling for forecasting (1–7 %, growing with horizon);
  - (iv) the gap between forecast accuracy and decision value.

## 7. Limitations / threats to validity

- Simulation only. 4-qubit chains, 6 circuits (one instance each). Readout and idle errors not modelled; coherent error only as a fixed CZ phase.
- The execution state is proxied by the next published calibration.
- At k = 7 only 19 chains / 105 decisions; the unseen device is evaluated at k = 1 only.
- Best fixed and bandit settings selected on one month of validation.
- The value-of-information ceiling is specific to this controller and action set.
- Additional circuit instances and a 6-qubit check were run afterwards as pre-registered robustness analyses (`docs/experiment_reports/STAGE6_CONTRIBUTION.md` §3).

## 8. Artifacts

| Kind | Path |
|---|---|
| Code | `experiments/stage5_{decisions,simulate,policies,stationary,pair_forecast,freeze}.py`, `experiments/plot_stage5.py`, `src/forecasting/pairs.py` |
| Frozen inputs | `configs/frozen_stage5.yaml` (23 files incl. code) |
| Results | `results/stage5/hypotheses_{val,test}.json`, `h3b_stationary.json`, `selections.json`, `pair_forecast/metrics_{val,test}.csv` |
| Regenerable caches (git-ignored) | `results/stage5/sim_arrays.npy`, `policy_*.parquet`, `pair_forecast/predictions/` |
| Figures | `results/stage5/figures/*.{png,pdf}` (+ CSV): hypotheses forest plot, error by horizon, pair-forecast skill, probe-reward quality |
