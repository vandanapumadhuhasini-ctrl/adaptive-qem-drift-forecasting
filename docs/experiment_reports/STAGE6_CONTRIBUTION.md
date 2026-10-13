# Stage 6 — Contribution Validation

Every statement below cites a saved, verified result. **Simulation-based execution** (Qiskit Aer) **driven by real IBM calibration records**; no quantum-hardware execution.

## 1. The six validation questions

**Q1. Does the Transformer outperform persistence when making mitigation decisions?**
**No.**
- H2 primary (k = 1, test): +0.02 %, CI [−0.29 %, +0.54 %]; captured-share difference −0.002 (`results/stage5/hypotheses_test.json`).
- Also not at k = 4 (−0.41 %, n.s. after Holm), k = 7 (−0.05 %), or on the unseen device (−0.29 %).
- The same holds for every forecaster (window mean, ridge).

**Q2. Does it outperform the best fixed policy under non-stationary conditions?**
**Yes at k = 1 and 4** (−5.6 %, −4.2 %, Holm-significant); **no at k = 7** (−0.3 %, n.s.). Persistence-driven adaptation does equally well, so the credit belongs to calibration-informed adaptation, not to forecasting.
- Unseen device (k = 1): −0.6 % (R1) and −4.4 % (R2) in mean error.

**Q3. How does it compare with the contextual-bandit approach of the base paper?**
**Clearly better: −15.0 % / −18.7 % / −14.5 % at k = 1 / 4 / 7** (Holm-significant), at an equal shot budget. Persistence-driven adaptation matches this.
- Mechanism (measured): the bandit's feedback, a Clifford-probe reward, ranks actions poorly (median Spearman 0.03–0.56 by circuit; the true best action is identified in 6–14 % of states) and costs 12.5 % of the shots.
- Even with an unrealisable true-error reward, the bandit only matches a well-chosen fixed action per circuit.

**Q4. Does the benefit persist across horizons, devices, circuits and regimes?**
- **Over best fixed:** persists at k = 1–4 and on the unseen device; vanishes at k = 7 and under model mismatch (R3).
- **Over the bandit:** persists everywhere tested.
- **Forecasting over persistence:** absent everywhere.
- Circuits and qubit count: see §3 (robustness).

**Q5. Are the improvements large enough to matter after overhead?**
- **Shots:** identical by design. On test (R1 + R2, development chains), the adaptive policies choose *cheaper* actions than best fixed: Σ fold scales ≈ 8.0–8.5 vs 9.0, and 6.7–8.2 vs 13.5 distinct circuits on average. The bandit is costlier (Σ scales 9.7–10.7, 12.2–12.4 circuits).
- **Classical overhead:** the model-based controller simulates each action once per decision, about 0.3 s for 4 qubits. This grows exponentially with qubit count, so it needs a surrogate beyond about 10–12 qubits.
- **The Transformer adds training and inference cost with no decision benefit;** persistence needs no model.
- **A 4–6 % error reduction over best fixed at zero shot cost** is practically meaningful for NISQ expectation-value estimation (typical mitigated errors ≈ 0.024 per observable here).

**Q6. What happens under abrupt changes or inaccurate forecasts?** (exploratory; `results/stage6/h5_change_quartiles.csv`)
- **The value of information is concentrated in abrupt device changes.** In the top quartile of CZ changes (median 2.3× between decision and execution), the oracle beats persistence by 6.4 % / 9.3 % / 14.1 % (k = 1 / 4 / 7); elsewhere by < 1 %.
- In those cases persistence-driven adaptation is *worse* than best fixed (+1.0 % / +3.0 % / +5.9 %).
- The Transformer's forecast is closer than persistence in 90–97 % of those cases, but recovers only about 1.2–1.4 % (k = 1, 4) and nothing at k = 7. It is slightly worse in the small-change quartiles. The net effect is zero.
- **Model mismatch (R3):** a coherent error missing from the controller's model removes most of the advantage over best fixed.

## 2. Defensible contribution statement

1. **A leakage-safe benchmark of adaptive quantum error mitigation under *real* calibration drift:**
   - 101,891 real calibration states from 600 qubits on 4 IBM Heron devices;
   - chronological, unseen-qubit and unseen-device splits;
   - decisions timed by publication time (including a documented epoch-timing correction).
2. **Calibration-informed model-based selection of ZNE scales, extrapolator and Pauli twirling:**
   - beats the best fixed configuration (4–6 %, k ≤ 4);
   - beats a faithful, tuned contextual-bandit adaptive ZNE (the IEEE JSAC 2026 base paper's approach) by 15–19 % at an equal shot budget;
   - is non-inferior under stationary noise.
3. **A value-of-information analysis for forecast-driven QEM:**
   - perfect knowledge of the execution-time state is worth 1.4–1.7 % (1 day) to 5.2–7.1 % (7 days) over the latest calibration, concentrated in abrupt device changes;
   - Transformer, ridge and window-mean forecasts capture none of it on average, despite being 10–23 % more accurate in pair CZ error, **because forecast accuracy does not translate into action ranking**.
4. **A diagnosis of why feedback-driven (bandit) adaptation underperforms for observable estimation:** Clifford-probe rewards rank mitigation actions poorly.

**What we do *not* claim:**
- that a Transformer improves QEM decisions;
- real-time operation on hardware;
- results beyond the simulated noise model (no readout, idle or crosstalk errors beyond a fixed CZ phase).

## 3. Robustness (filled from `results/robustness/`)

Pre-registered in `ROBUSTNESS_PLAN.md`; separate freeze `configs/frozen_robustness.yaml`; test evaluated once (k = 1, development chains, R1 + R2; 95 % paired cluster CI). Results in `results/robustness/{ra,rb}/summary_test.json`.

| Transformer-driven adaptation vs … | Main study (6 circuits, 4 qubits) | R-A: 8 new circuit instances | R-B: 6-qubit chains |
|---|---|---|---|
| Persistence (H2 analogue) | +0.02 % (n.s.) | −0.42 % (n.s.) | −0.35 % (n.s.) |
| Best fixed | −5.6 % (sig.) | −5.7 % (sig.) | −3.6 % (CI touches 0) |
| Best fixed per circuit | — | −4.4 % (sig.) | −3.6 % (CI touches 0) |
| CMAB-probe | −15.0 % (sig.) | −17.8 % (sig.) | not run (by design) |
| Window mean / ridge | n.s. | −0.25 % / −0.24 %* | −0.12 % / +0.02 % |
| Oracle − persistence (ceiling) | 1.4–1.7 % | 2.5 % | 2.6 % |

\* CI excludes 0 for ridge, but the effect is practically negligible.

**Conclusion:**
- The pattern is stable across circuit instances and from 4 to 6 qubits.
- Calibration-informed adaptation beats fixed mitigation (by a smaller, borderline margin at 6 qubits) and the bandit.
- Forecasting adds nothing meaningful over persistence.
- The value-of-information ceiling at k = 1 is about 1.5–2.6 %.

## 4. Consequence for the title and framing

The approved title, *"Transformer-Based Real-Time Adaptive Quantum Error Mitigation for Non-Stationary Noise"*, is **not supported** by the evidence:
- the Transformer is not the source of the gain;
- "real-time" is not demonstrated.

Candidate titles consistent with the evidence:
- *"Does Forecasting Calibration Drift Improve Adaptive Quantum Error Mitigation? Evidence from Real IBM Heron Calibration Histories"*
- *"Calibration-Informed Adaptive Zero-Noise Extrapolation and Pauli Twirling under Real Hardware Drift: Fixed, Bandit and Forecast-Driven Policies Compared"*
- *"Forecast Accuracy Is Not Decision Value: Adaptive Quantum Error Mitigation under Real Calibration Drift"*

The decision belongs to the project owner (Stage 8).
