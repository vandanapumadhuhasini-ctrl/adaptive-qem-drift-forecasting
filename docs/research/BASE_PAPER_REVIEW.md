# Base-Paper Review (verified 2026-10-12)

> The full review, hypotheses and research plan are in `docs/research/STAGE1_RESEARCH_PROPOSAL.md` and `docs/research/PROJECT_PLAN.md`; this file is the candidate survey.

Each citation below was verified against an authoritative record: arXiv pages, the IEEE TQE website, and the Crossref DOI registry. Crossref is the registry that IEEE DOIs are deposited in. doi.org returned HTTP 403 (bot protection) to automated requests, so **open each DOI in a browser once before citing it**.

## 1. The remembered reference: Placidi et al.

| Field | Verified value |
|---|---|
| Title | *Deep Learning Approaches to Quantum Error Mitigation* |
| Authors | Leonardo Placidi, Ifan Williams, Enrico Rinaldi, Daniel Mills, Cristina Cîrstoiu, Vanya Eccles, Ross Duncan (7 authors) |
| Venue | **arXiv preprint only**: arXiv:2601.14226 [quant-ph, cs.LG], v1 20 Jan 2026, 48 pp. |
| DOI | 10.48550/arXiv.2601.14226 (arXiv-issued, not a publisher DOI) |
| IEEE? | **No.** arXiv lists no journal reference, Crossref has no journal record, and a later paper (arXiv:2609.25555) cites it as a 2026 preprint |

**Verdict.** The paper is genuine and relevant: attention-based seq2seq models map noisy output distributions to ideal ones, tested on IBM QPUs with up to 5 qubits and with cross-device transfer. It is **not an IEEE journal paper**, so it should be cited as *related work* and not as the IEEE base paper. Its task, learning the mitigation map from data for a *static* device, also differs from ours, which is forecasting *drift* and adapting a physics-based mitigation protocol.

## 2. Verified 2026 IEEE journal candidates

| # | Citation (Crossref-verified) | Type | Relevance to our project |
|---|---|---|---|
| **A** | R. Rahman, D. C. Nguyen (Univ. of Alabama in Huntsville), "Quantum Noise Mitigation With Adaptive Zero-Noise Extrapolation: A Contextual Multi-Armed Bandits Approach," *IEEE J. Sel. Areas Commun.*, vol. 44, pp. 5600–5614, 2026. DOI **10.1109/JSAC.2026.3721807**, IEEE Xplore doc 11644486 (preprint arXiv:2608.06426; code github.com/Ratun11/cmab_zne) | Journal article | **Highest.** Adaptive ZNE under time-varying noise and calibration drift |
| B | M. Prodius, P. Czarnik, M. McKerns, A. T. Sornborger, L. Cincio, "Robust Design Under Uncertainty in Quantum Error Mitigation," *IEEE Trans. Quantum Eng.*, vol. 7, pp. 1–13, 2026. DOI 10.1109/TQE.2026.3680641 | Journal article | Medium. Finite-shot uncertainty in ZNE and learning-based QEM, which bears on our bias-versus-variance configuration choice |
| C | J. Pérez-Guijarro, A. Pagès-Zamora, J. R. Fonollosa, "Extension of Clifford Data Regression Methods for Quantum Error Mitigation," *IEEE Trans. Quantum Eng.*, vol. 7, pp. 1–22, 2026. DOI 10.1109/TQE.2026.3677541 | Journal article (extends their ICASSP 2024 paper) | Low–medium. Learning-based QEM (CDR), with no drift |
| D | L. Hour, M. Go, Y. Han, "Improving Zero-Noise Extrapolation for Quantum-Gate Error Mitigation Using a Noise-Aware Folding Method," *IEEE Access*, vol. 14, pp. 37362–37372, 2026. DOI 10.1109/ACCESS.2026.3672056 | Journal article | Medium. Calibration-aware folding; whether it handles drift is **not verified** (abstract not read) |

For B–D I verified the bibliographic records and read only abstracts or listing snippets, not the full texts.

## 3. Recommended base paper: A (Rahman & Nguyen, IEEE JSAC 2026)

**Why A.** It is the only verified 2026 IEEE journal paper whose stated problem is the same as ours: *fixed ZNE settings become suboptimal when hardware noise drifts over time*. JSAC is a high-impact IEEE journal. Its limitations section names, as future work, the exact ingredient our project adds: hardware telemetry (T1, T2, gate and readout errors) used as context.

### 3.1 Problem and method (from the arXiv full text, ~78% read)
- **Problem:** standard ZNE assumes static noise. Under dynamic noise, fixed or grid-searched fold scales waste circuit executions and misfit the extrapolation.
- **Noise:** a Lindblad model with a time-varying thermal photon number. The simulations use PennyLane `default.mixed`, and the drift profile η̂ₜ is **prescribed** (it rises from about 0.25 to 1.0 by epoch 40), **not measured calibration data**.
- **Controller:** cost-aware **LinUCB** contextual bandit with a discount factor for drift. Arms are fold-scale sets {1,3}, {1,3,5} and {1,3,5,7}, and the extrapolator is fixed per arm (linear, or polynomial of degree ≤ 2). Context is the training epoch, loss and accuracy at λ = 1, variance across scales, and batch size. Reward is −cross-entropy − β·variance − α·depth penalty.
- **Tasks:** hybrid VQC classifiers on CIFAR-10 and EuroSAT, with 4–32 qubits and 3–7 layers.
- **Hardware:** Rigetti Cepheus-1-108Q via Amazon Braket, frozen inference on 30 samples only.

### 3.2 Key findings
Against fixed [1,3,5] ZNE: −38.7 % / −41.2 % round trips, −34 % bytes, −30 % end-to-end cost, and +6.9 % / +6.4 % estimator fidelity. The accuracy gain over the best of 11 reimplemented baselines (Adaptive KIK) is small: 85.58 % vs 85.24 % on CIFAR-10.

### 3.3 Limitations (stated or evident)
1. **Reactive, not predictive.** No noise forecasting; adaptation follows observed feedback. The paper itself says it does not forecast.
2. **Synthetic drift.** A prescribed η̂ₜ schedule, with no real calibration time series.
3. **Exploration cost.** The bandit must spend shots on probe batches to learn.
4. **Pauli twirling appears only as a baseline**, and the extrapolator is not chosen adaptively.
5. **Limited hardware validation:** inference only, 30 samples, one device.
6. **Task-level metric** (classification accuracy), not the error of a physical observable. Some numbers differ between the abstract and Table V (40 % vs 41.2 %, and the reference baseline for "fidelity").

## 4. Research gap our project can realistically address

> **Can forecasting a device's calibration telemetry make adaptive ZNE + Pauli-twirling selection *proactive*, so that no shots are spent on online exploration, and does it beat (i) the best fixed configuration, (ii) reactive adaptation from the latest calibration, and (iii) a reactive contextual bandit, when drift comes from *real* IBM calibration histories?**

This gap is defensible because A has (1) no forecasting, (2) no real calibration time series, (3) no adaptive twirling or extrapolator choice, and (4) online exploration overhead. Our project addresses (1)–(3) directly and replaces (4) with an offline, simulator-calibrated error table.

## 5. Comparison with our framework

| Aspect | Rahman & Nguyen (JSAC 2026) | Our project (current) |
|---|---|---|
| Drift source | Prescribed synthetic schedule | **Real** IBM `ibm_fez` T1 history (calibration-drift dataset) |
| Uses telemetry | No (future work) | Yes, T1 only so far |
| Forecasting | None (reactive) | Transformer predicts next-calibration T1 |
| Decision rule | LinUCB bandit, online | Lookup of a simulated error table, offline (no exploration shots) |
| Knobs adapted | Fold-scale set | Fold set **+ extrapolator + Pauli twirling on/off** |
| Metric | Classification accuracy, cost | Mean absolute error of 12 Pauli observables vs ideal, at equal shot budget |
| Statistics | Point estimates | Paired bootstrap 95 % CIs |
| Hardware | Rigetti, inference only | **None** (Aer simulation only) |
| Circuits | VQC classifiers, 4–32 qubits | One 4-qubit, 9-CX circuit |

## 6. Does our current implementation genuinely extend A?

**Partly, and not yet at publication strength.** What it already adds: real drift data, a forecast-driven controller, joint selection of ZNE, extrapolator and twirling, an equal-shot-budget comparison, and paired statistics.

What would weaken a submission today, from our own verified results:
1. **The forecast does not yet beat persistence at the decision level.** Adaptive-Transformer and adaptive-last-calibration are statistically indistinguishable on the test split. Without that, the "proactive forecasting" claim is unsupported.
2. **No bandit baseline.** A must be reimplemented (LinUCB over the same arms and shot budget) and compared on the same drift traces. Otherwise reviewers will reject the comparison.
3. **T1 only.** A realistic noise model needs T1, T2, single- and two-qubit gate errors and readout error, all of which exist in the dataset (4 Heron backends, 133–156 qubits).
4. **Scale.** One circuit, 4 qubits, 10 qubits of one backend, about 100 days. The paper needs multiple circuit families, depths, backends, seeds, and a backend or qubit hold-out.
5. **Simulation only.** Even a small hardware validation, or replaying real calibration snapshots through `qiskit-ibm-runtime` fake backends, would strengthen the claim. Simulation results must be labelled as such.

Required methodology changes (proposals awaiting approval) are listed in the reply and are **not yet implemented**.

## Sources
- arXiv:2601.14226 — https://arxiv.org/abs/2601.14226
- arXiv:2608.06426 — https://arxiv.org/abs/2608.06426 (full text: https://arxiv.org/html/2608.06426)
- IEEE TQE quantum-computing listing — https://tqe.ieee.org/category/quantum-computing/
- Crossref REST API (api.crossref.org/works) for all four IEEE DOIs
- Citing preprint: https://arxiv.org/pdf/2609.25555
