# Stage 5b — Decision-Focused Transformer Controller (exploratory, stopped)

Run 2026-10-13. **Validation only**: no test-period data was used, as agreed. Pre-registered in `PROTOCOL_ADDENDUM.md` (H6) before any model was trained. Time-boxed because of the 16 October submission deadline.

## Idea
The Stage 5 forecasts were more accurate but did not change decisions. The decision-focused controller (DFT) instead lets a Transformer read the chain's calibration history and **correct the model-based predicted error of every mitigation action**; the controller picks the action with the lowest corrected error. The target is the simulated true error of each action at the execution epoch.

## Setup
- **Data:** training-period decisions on development chains (k = 1, 4, 7; regimes R1, R2; 6 circuits). Early stopping on the last 20 % of the training period. Validation decisions (k = 1/4/7: 2,970 / 1,884 / 1,380 units) were never used for fitting or stopping.
- **Models, trained identically (3 seeds each):**
  - Transformer over 20 epochs of chain history;
  - MLP on summary statistics (last / mean / std);
  - no-history control (predicted errors + circuit/horizon/regime only).
- **Runtime:** generating training targets took 70 min (8,814 new simulation units); training all 9 models took 64 s.

## Results (validation, mean |error| per observable, paired cluster-bootstrap CIs)

| Policy | k = 1 | k = 4 | k = 7 |
|---|---:|---:|---:|
| Best fixed (chosen on training) | 0.02582 | 0.02741 | 0.02904 |
| Persistence (model-based) | 0.02407 | 0.02553 | 0.02696 |
| Forecast-driven Transformer (Stage 5) | 0.02407 | 0.02545 | 0.02690 |
| DFT: no-history control | **0.02402** | **0.02543** | 0.02696 |
| DFT: MLP summary | 0.02423 | 0.02568 | 0.02698 |
| DFT: Transformer | 0.02414 | 0.02552 | 0.02692 |
| Oracle state* | 0.02359 | 0.02484 | 0.02626 |

\* not realisable.

- **H6 (k = 1): DFT-Transformer − persistence = +0.28 %**, CI [+0.14 %, +0.44 %]. It is significantly *worse*, so H6 is **not met**. At k = 4 and 7: −0.05 % and −0.14 %, not significant.
- **Per circuit:** −0.26 % to +0.31 % against persistence; no consistent gain.
- **The oracle ceiling** is 2.0–2.7 %. None of the learned controllers captured a meaningful part of it.

## Is the model learning decision-relevant information or artifacts?
- The **no-history control** beats persistence slightly (−0.22 % at k = 1, CI excludes 0). It sees no drift information, so it can only learn a *static* correction to the simulator-based error predictions, for example systematic Monte-Carlo bias in the predicted versus evaluated errors of particular actions. That is a property of the simulation pipeline, **not drift information**, and it should not be claimed as a method gain.
- **Adding the history makes things worse** (Transformer vs no-history: +0.51 % at k = 1), which indicates over-fitting to non-transferable history patterns with this much data.
- **Conclusion:** with the available training data, the Transformer did not learn decision-relevant drift information.

## Decision
As agreed, the experiment is **stopped** and kept as exploratory work. The project is finalised on its verified results. **No claim** is made that a Transformer improves mitigation decisions.

**What could still work (future work):**
- more training data (the dataset grows daily) and a fresh, untouched test period;
- a loss aligned with decisions (regret or ranking) instead of per-action MSE;
- a gate that applies corrections only when an abrupt change is predicted.

**Artifacts:**
- Code: `experiments/s5b_train_targets.py`, `experiments/s5b_dft.py`, `experiments/s5b_report.py`.
- Results: `results/stage5b/dft_results.parquet`, `dft_validation_report.json`, `dft_train_log.json`.
