# Stage 3 — Multi-Horizon Forecasting of Real IBM Calibration Drift

Run on 2026-10-12. Everything below was regenerated from saved artifacts (paths in §8).
**Test sets were evaluated once**, after freezing every prediction file (`configs/frozen_stage3.yaml`, commit `1252ab0e`).

## 1. Setup

| Item | Value |
|---|---|
| Data | `data/processed/drift_v3/cce.parquet` (Stage 2; real IBM calibration records, dataset revision 539bc99) |
| Unit | One coherence-calibration event (CCE) of one qubit; median spacing 24.8 h |
| Input | 20 past CCEs × 19 calibration-state features (`src/data/windows.py`) |
| Targets | log T1, log T2, log sx error, log readout error, log mean CZ error |
| Horizons | k = 1, 2, 4, 7 CCEs ahead |
| Prediction | Change Δ = y_{n+k} − y_n (log scale); persistence = 0 |
| Train / val / test | Development qubits of ibm_fez + ibm_kingston; split by target time (T_val 2026-08-01, T_test 2026-09-01). k = 1 pairs: 30,134 / 6,162 / 9,745 |
| Extra test sets | Unseen qubits (62 qubits, 2,453 pairs), unseen device ibm_marrakesh (151 qubits, 5,677 pairs) |
| Scaling | Inputs and Δ targets standardised with **training pairs only** |
| Selection | Validation only: mean over the 20 (target, k) cells of per-series MAE ÷ persistence MAE |
| Statistics | Per-series MAE (each qubit weighted equally); 95 % cluster bootstrap (B = 10,000, qubits resampled within backend); H1 by intersection–union test + Holm over the 15 k ≥ 2 cells (`PROTOCOL.md`) |
| Environment | Python 3.14.5, numpy 2.5.1, pandas 3.0.5, torch 2.13.0 (CPU, 8 threads), scikit-learn 1.9.1 (`results/stage3/environment.json`) |

## 2. Models

| Model | Description | Selected on validation |
|---|---|---|
| Persistence | Δ = 0 (last calibration) | — |
| Rolling mean (5) | Mean of the last 5 CCEs | — |
| Window mean (20) | Mean of all 20 CCEs in the window | — |
| Ridge | Linear model on 49 lag features, one model per (k, target) | α per cell from {0.1 … 10⁴} |
| Gradient-boosted trees | `HistGradientBoostingRegressor`, same features, one per cell | lr 0.03 in all cells; 200/500 iterations; 15/31 leaves |
| LSTM | One multi-output network, masked MSE | hidden 32, 1 layer, dropout 0.3 |
| **Transformer** (improved) | Learned positional embedding, pre-norm encoder, [last token ‖ mean] read-out, 20 outputs | **d_model 32, 1 layer, 4 heads, FF 64, dropout 0.1, lr 1e-3** |

The neural models are 5-seed ensembles (seeds 0–4); 25 trained networks are in `checkpoints/stage3/`.

Changes from the pilot Transformer, each justified by a defect:
- positional information added (the pilot had none);
- feed-forward width 2048 → 64 (the pilot had 565 k parameters for 561 windows);
- multi-horizon, multi-target Δ head;
- masked loss for missing targets.

The selected Transformer is the **smallest in the grid**, which is noted as a limitation in §7.

## 3. Results — Test, development qubits (primary)

Mean over the 20 cells of MAE ÷ persistence MAE (lower is better):

| Persistence | Rolling 5 | Window mean 20 | Ridge | Gradient-boosted trees | LSTM | **Transformer** |
|---:|---:|---:|---:|---:|---:|---:|
| 1.000 | 0.827 | 0.804 | 0.791 | 0.790 | 0.799 | **0.787** |

Per-series MAE of the log value (≈ relative error), k = 1 / k = 7:

| Target | Persistence | Window mean | Ridge | Trees | LSTM | Transformer |
|---|---|---|---|---|---|---|
| T1 | 0.306 / 0.311 | **0.229** / 0.234 | 0.233 / 0.235 | 0.232 / 0.237 | 0.231 / 0.235 | **0.229** / **0.232** |
| T2 | 0.236 / 0.241 | **0.178** / **0.182** | 0.181 / 0.184 | 0.181 / 0.191 | 0.182 / 0.186 | 0.180 / 0.183 |
| sx error | 0.241 / 0.326 | 0.242 / 0.251 | 0.222 / 0.245 | **0.207** / 0.244 | 0.218 / 0.245 | 0.216 / **0.242** |
| Readout error | 0.371 / 0.389 | 0.292 / 0.309 | 0.287 / **0.303** | **0.286** / 0.305 | 0.293 / 0.308 | 0.294 / 0.314 |
| CZ error | 0.191 / 0.246 | 0.191 / 0.207 | 0.183 / 0.201 | **0.173** / 0.209 | 0.179 / 0.212 | **0.173** / **0.200** |

All 20 cells for every model, with CIs, RMSE, natural-unit MAE and large-change MAE, are in `results/stage3/metrics_test.csv`.

At k = 1, MAE in natural units:

| Property | Persistence | Window mean | Transformer |
|---|---:|---:|---:|
| T1 | 50.7 µs | 38.7 µs | 38.8 µs |
| T2 | 32.6 µs | 24.9 µs | 25.0 µs |

### H1 (pre-registered): **not supported in any of the 15 cells**

The Transformer beats persistence in every cell: by 16.5–25.7 % for k ≥ 2 and by 9.7–25.7 % including k = 1, with every CI excluding 0. But it never beats *every* baseline:

| Cell type | Strongest competitor | Transformer's relative MAE reduction R |
|---|---|---|
| T1 | Window mean | −0.1 % to +0.7 % (CI includes 0) |
| T2 | Window mean | −0.6 % to −1.3 % (significantly *worse* at k = 2, 4) |
| sx error | Trees | −1.6 % to +0.6 % |
| Readout | **Ridge** | **−3.1 % to −3.7 %** (significantly worse) |
| CZ error | Ridge | −0.1 % to +0.2 % |

The intersection–union p-value is ≥ 0.057 in every cell, no cell survives Holm correction, and no cell reaches the 5 % practical threshold. Details: `results/stage3/h1_test.json`.

## 4. Generalisation (test)

Mean ratio to persistence:

| Set | Window mean | Ridge | Trees | LSTM | Transformer |
|---|---:|---:|---:|---:|---:|
| Development qubits | 0.804 | 0.791 | 0.790 | 0.799 | 0.787 |
| Unseen qubits (62) | 0.800 | 0.784 | 0.793 | 0.807 | 0.781 |
| Unseen device, marrakesh (151) | 0.786 | 0.779 | 0.797 | 0.817 | 0.785 |

The learned models transfer to unseen qubits and to an unseen device without retraining. The ranking among the top models is unstable: on the unseen device, ridge and the window mean edge out the Transformer.

**Per qubit, k = 1** (`results/stage3/per_backend_test_primary.csv`):
- The Transformer beats persistence on 73–100 % of qubits, depending on property and backend.
- Against the window mean it is a coin flip for T1, T2 and readout (27–47 % of qubits).
- On **ibm_kingston gate errors** it is clearly better: on 93 % of qubits for sx error and 77 % for CZ error. On ibm_fez the same comparison is about 50 %.

## 5. Ablations (Transformer, test, mean MAE over k)

| Variant | T1 | T2 | sx | Readout | CZ |
|---|---:|---:|---:|---:|---:|
| Full (19 state features) | 0.2304 | 0.1814 | 0.2318 | 0.3038 | **0.1881** |
| Target history only (5 features) | 0.2325 | 0.1831 | 0.2317 | **0.2949** | 0.1888 |
| + environment (27 features) | **0.2279** | 0.1815 | **0.2303** | 0.3027 | 0.1895 |
| No positional encoding | 0.2308 | 0.1833 | 0.2335 | 0.3063 | 0.1903 |

- Effects are ≤ 1 % except readout, where the target-history-only model is 3 % better than the full model.
- Environmental covariates do not help meaningfully, and they are 22–73 % missing.
- Positional encoding helps slightly but consistently (+0.2 to +1.1 % error without it).

## 6. Decision relevance (preview for Stage 4)

Labelled **preliminary**. It used the pre-protocol pilot simulated controller (T1-only, homogeneous 4-qubit device). The pilot code is not part of the released repository; Stages 4–5 replace it. The controller picks the configuration that is best at the forecast T1. Regret = expected error at the true T1 minus the best achievable.

| Test set, regime | Persistence regret | Window mean | Ridge | Trees | Transformer |
|---|---:|---:|---:|---:|---:|
| Dev qubits, coherent | 0.00188 | 0.00123 (−35 %) | 0.00118 (−38 %) | 0.00121 (−36 %) | 0.00121 (−36 %) |
| Dev qubits, incoherent | 0.00081 | −39 % | −40 % | −38 % | −39 % |
| Unseen qubits, coherent | 0.00211 | −36 % | −36 % | −33 % | −32 % |
| Unseen device, coherent | 0.00239 | −33 % | −32 % | −28 % | −31 % |

**What this means for Stage 4:**
- **Forecasting, rather than reacting to the latest calibration, changes mitigation decisions for the better.** Regret falls by 28–40 % in every set and regime, which supports the premise of H2.
- **That gain is delivered just as well by a 20-calibration window mean.** Transformer-minus-window-mean regret differences are within ±0.0001, while the CIs are about ±0.0004.
- **Absolute stakes in the pilot controller are small:** regret is about 0.001–0.002, against mitigated errors of about 0.02–0.03 per observable. Stage 4's multi-property controller may change this, and that is exactly what has to be measured.

## 7. Interpretation

1. **Calibration drift at daily resolution is dominated by noise around a slowly moving qubit-specific level.**
   - About 51 % of next-CCE T1 values differ from the current value by more than 25 %.
   - Optimal forecasts therefore shrink toward the recent mean. Every model learns this, and a plain window mean already captures most of it.
   - This matches the known two-level-system (TLS) fluctuation physics of superconducting qubits and the measurement noise of IBM's calibration estimates.
2. **The Transformer is competitive but not superior.** It has the best average ratio on development and unseen-qubit data (0.787 / 0.781), but the margin over ridge, trees and the window mean is 0.3–2 % and not statistically reliable. It is significantly worse than ridge on readout error.
3. **Where sequence modelling may matter:** gate errors (sx, CZ) on one backend, and longer horizons for CZ error. These effects are localised.
4. **Consequence for the research question:** a fair Stage 4 must compare Transformer-driven adaptation not only with persistence but also with **window-mean-driven and ridge-driven adaptation**. Otherwise the benefit of smoothing would be wrongly credited to the Transformer. This is now pre-declared (`PROTOCOL.md` §9, before any Stage 4 work).

**Limitations:**
- The selected Transformer is at the small edge of the grid; smaller models were not tried.
- The neural models are trained on training data only, not retrained on train + validation.
- **Natural-unit metrics can rank differently:**
  - For sx error, persistence has the lowest natural-unit MAE because rare large spikes dominate absolute errors.
  - The log-scale metric was pre-registered as primary; natural units are reported for transparency.
- One unseen device.
- Calibration estimates contain measurement noise, which bounds attainable accuracy.

## 8. Artifacts (all under the project root)

| Kind | Path |
|---|---|
| Configuration | `configs/stage3.yaml`, `results/stage3/selected.yaml`, `configs/frozen_stage3.yaml` |
| Code | `src/forecasting/{data,models}.py`, `src/evaluation/{stats,plotting}.py`, `experiments/stage3_forecasting.py`, `experiments/plot_stage3.py` |
| Tuning logs | `results/stage3/tune_{lstm,transformer}.json`, `results/stage3/stage3.log`, `logs/stage3_*.out` |
| Predictions | `results/stage3/predictions/*.npz` (10 models; neural files include per-seed arrays; git-ignored, SHA-256 in the frozen file) |
| Window index | `results/stage3/windows_arrays.npz`, `results/stage3/windows_meta.parquet` |
| Metrics | `results/stage3/metrics_{val,test}.csv`, `h1_{val,test}.json`, `summary_test.json`, `per_backend_test_primary.csv` |
| Checkpoints | `checkpoints/stage3/*.pt` (25 networks with scalers, params and seed) |
| Figures | `results/stage3/figures/{val,test}/*.png` (300 dpi) + `*.pdf` (vector) + `*.csv` (plotted data) |

Figures (test): MAE vs horizon (3 sets), skill vs persistence, generalisation, ablation, forecast vs actual, error over time, residuals, per-qubit distribution, training curves.

**Reproduce:**
```
python3 -m experiments.stage3_forecasting tune
python3 -m experiments.stage3_forecasting final
python3 -m experiments.stage3_forecasting evaluate --sets val
python3 -m experiments.stage3_forecasting evaluate --sets test
python3 -m experiments.plot_stage3 --which test
```

## 9. Acceptance

| Criterion | Status |
|---|---|
| scikit-learn installs and imports; existing dependencies intact | ✅ 1.9.1, dry run showed no changes to existing packages; v2 pipeline output identical |
| Splits and preprocessing verified for leakage | ✅ deterministic rebuild; window assertions; train-only scalers; sample-set equality asserted across feature sets |
| All baselines run across all horizons | ✅ 7 models + 3 ablations × 4 horizons × 5 targets × 4 evaluation sets |
| Numerical results saved; Matplotlib figures (PNG 300 dpi + PDF) | ✅ 11 test + 7 validation figure sets, each with its CSV |
| Transformer compared with baselines and findings explained | ✅ §3–§7; H1 not supported |
| Usefulness for Stage 4 assessed | ✅ §6 (preliminary; forecasting helps, Transformer-specific benefit not shown) |
| `docs/research/PROJECT_PLAN.md` updated | ✅ |

**vs. base paper:** the base paper does no forecasting, so this stage adds a capability it lacks. The evidence says *forecasting* (shrinkage toward recent levels) improves decisions over reacting to the latest state, which also favours forecasting over a purely reactive bandit fed the latest context. It also says a Transformer is **not necessary** for that benefit on this data. A claim of Transformer superiority would currently be unsupported.
