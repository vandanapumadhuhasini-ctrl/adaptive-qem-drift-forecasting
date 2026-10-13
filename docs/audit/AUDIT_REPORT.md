# Repository and IEEE-Suitability Audit

**Date:** 2026-10-13
**Scope:** every file in the release commit `14d1350e`. The file-by-file inventory is in `FILE_INVENTORY.md`.

**Method:**
- traced imports and file reads/writes of every Python module;
- re-derived every headline number from the result files;
- verified every reference against Crossref or arXiv;
- checked the reproduction entry point by running it;
- added and ran automated tests.

Nothing in this audit used the test period to tune anything.

---

## 1. How the project fits together (traced from the code)

```text
scripts/download_ibm_data.py ── raw parquet (pinned revision, SHA-256 checked)
        │
src/data/drift_dataset.py ───── data/processed/drift_v3/cce.parquet  (101,891 calibration states)
        │
src/data/windows.py + src/forecasting/* + experiments/stage3_forecasting.py
        │      per-qubit forecasts (Stage 3; test frozen in configs/frozen_stage3.yaml)
        │
src/forecasting/pairs.py + experiments/stage5_pair_forecast.py
        │      pair-level CZ forecasts (Stage 5a)
        │
experiments/stage5_decisions.py ── chains, daily epochs, decision/execution states, forecast states
        │      (src/qem/device.py: epochs, chains, noise model)
experiments/stage5_simulate.py ─── exact density-matrix simulation of 14 actions × 6 circuits
        │      (src/qem/benchmarks.py, mitigation.py; cache frozen in configs/frozen_stage5.yaml)
experiments/stage5_policies.py ─── controller (src/qem/policies.py), bandit, fixed policies, statistics
        │      (src/evaluation/stats.py) → results/stage5/hypotheses_{val,test}.json
experiments/stage5_stationary.py ─ H3b;  experiments/stage6_change_quartiles.py ─ exploratory analysis
experiments/robust_*.py ───────── R-A (new circuit instances), R-B (6-qubit chains; src/qem/nq.py)
experiments/s5b_*.py ──────────── exploratory decision-focused Transformer (validation only)
experiments/stage4_*.py ───────── development stage (validation only), superseded by Stage 5
```

`reproduce.py` runs these in dependency order.

---

## 2. Findings and actions

Severity:
- **High** = could change a reported number or claim;
- **Medium** = traceability or reproducibility gap;
- **Low** = presentation.

| # | Finding | Severity | Evidence | Action and status |
|---|---|---|---|---|
| 1 | Simulation caches are written in worker-completion order (`imap_unordered`). The raw-byte freeze hash therefore refused a correct reproduction | Medium | reproduction run: content bit-identical after aligning rows (292,512 arrays, max diff 0.0) | **Fixed.** Order-insensitive content digest (`src/evaluation/cache_digest.py`, `configs/frozen_cache_content.yaml`). No frozen file changed. Logged in `PROTOCOL_ADDENDUM.md` |
| 2 | `reproduce.py` ran the pair-forecast test evaluation before the Stage 5 files that its freeze check needs existed | Medium | first reproduction failed at that step | **Fixed** (order changed) |
| 3 | `results/stage6/h5_change_quartiles.csv` was produced by an inline command, with no script in the repository | Medium | no producer found by tracing | **Fixed.** Code recovered verbatim into `experiments/stage6_change_quartiles.py`; output byte-identical |
| 4 | Stage 4 and Stage 5b had no entry in `reproduce.py` | Medium | `reproduce.py --list` | **Fixed.** Optional stages `stage4` and `s5b` added; `stage6` added to `all` |
| 5 | **The saved Stage 4 results predate the epoch-timing leakage fix.** The current code builds different Stage 4 decisions (validation decisions on the unseen device 69 → 58; decision times shifted) | Medium | `results/stage4/decisions.parquet` vs a fresh `stage4_decisions` run | **Disclosed** (§5 below). Stage 4 is a validation-only development stage; no confirmatory claim rests on it. The one slide that uses it is labelled "validation" |
| 6 | **Best fixed action selected on validation, while `PROTOCOL.md` §4 says "chosen on Train".** | Medium | `experiments/stage5_policies.py` lines 122–128 | **Disclosed** in `PROTOCOL_ADDENDUM.md`. The change was documented before the test (Stage 4 design and report) but not in the protocol log. It makes the baseline stronger (more recent data), so H3a is stricter, not looser |
| 7 | Rounding errors in result tables. Mean-error table: 5 cells off by 0.0001 (report, `STAGE5_RESULTS.md`, 1 slide cell). Forecast table: 4 cells off by 0.001 (report) | Low | regenerated from `hypotheses_test.json` and `metrics_test.csv` | **Fixed.** Tables regenerated from the files; a test now guards the report table |
| 8 | Report and Stage 4 report gave the *requested* chain counts (35 development, 16 unseen-device). The chains actually built are 30 (fez 12, kingston 18) and 15 | Low | `results/stage5/decisions.parquet` | **Fixed** |
| 9 | The report attributed the Transformer's 9.7–25.7 % gain over persistence to "every learned model" (true range 4.6–25.7 %) | Low | `results/stage3/metrics_test.csv` | **Fixed** |
| 10 | README gave the 1-day value-of-information ceiling as "1.4–2.6 %", mixing the main study (1.4–1.7 %) with the robustness checks (2.5–2.6 %) | Low | `hypotheses_test.json`, robustness summaries | **Fixed** (both ranges stated) |
| 11 | A slide said forecasts "gave the same decisions" as persistence. The evidence is "no significant error difference", not identical decisions | Low | — | **Fixed** |
| 12 | Reference Liao et al. 2024 lacked page numbers | Low | Crossref | **Fixed** (pp. 1478–1486) |
| 13 | `PROTOCOL.md` refers to `configs/frozen.yaml`; the implementation uses per-stage freeze files | Low | — | **Disclosed** in the addendum (`PROTOCOL.md` is frozen and cannot be edited) |
| 14 | Data report cited the parquet by its Hugging Face path, not its local path | Low | — | **Fixed** |
| 15 | No automated tests | Medium | — | **Fixed.** `tests/`: 18 tests (unit, leakage, freeze, number traceability, folding/twirling identity, shot budget), all passing |
| 16 | Pre-protocol pilot code (T1-only, 1,000-record sample) was still in the repository, unused | Low | import tracing | **Removed** before the release (34 files; see inventory §3). Its results remain described as pre-protocol evidence |
| 17 | Commit hashes cited in `docs/` (e.g. freeze commits) are from the unpublished development history | Low | — | **Disclosed** in README. The freeze files carry UTC timestamps and SHA-256 hashes |
| 18 | Title promises "Transformer-based" and "real-time"; neither is supported (§6) | High (claims) | Stage 5, 5b results | **Disclosed** in the report conclusion, README and this audit. The title was kept at the owner's request |
| 19 | The report's author block contains placeholders ("[Department, Institution]"), and figures are placeholders | — | `report/QEM_Project_Report.tex` | **Owner action required** |
| 20 | The report did not compile: in the author block, a `\\` followed by `[City, Country]` is read as `\\[<length>]` ("Missing number") | High (build) | Tectonic (XeTeX) compile | **Fixed** (placeholders wrapped in braces). Report and slides now compile without errors or overfull boxes |

No undocumented use of the test period was found:
- Stage 5b used training and validation decisions only. This is checked by a test.
- The Stage 6 change-quartile analysis uses test results, but it is post hoc and labelled exploratory.

---

## 3. Traceability of reported numbers

Every number below was recomputed from the cited file during the audit. Items marked † are guarded by `tests/test_integrity.py`.

| Claim (README / report / slides) | Value | Source file | Status |
|---|---|---|---|
| Rows / genuine events / calibration states / qubits | 16,140,252 / 2,024,027 / 101,891 / 600 | `data/processed/drift_v3/report.json` | ✓ † |
| Transformer vs persistence, per-qubit, test | 9.7–25.7 % | `results/stage3/h1_test.json` | ✓ † |
| H1 cells passing | 0 of 15 | `results/stage3/h1_test.json` (k ≥ 2, test-primary) | ✓ |
| Pair CZ forecast vs persistence (development, test) | 11.0–12.2 % (1 d); 18.0–20.0 % (4, 7 d) | `results/stage5/pair_forecast/metrics_test.csv` | ✓ |
| H2 primary | +0.02 % [−0.29, +0.54] | `results/stage5/hypotheses_test.json` | ✓ † |
| H2 secondary | −0.41 % (4 d) / −0.05 % (7 d) / −0.29 % (unseen device) | same | ✓ |
| H3a vs best fixed | −5.6 % / −4.2 % / −0.3 % (n.s.) | same | ✓ † |
| H3b stationary | −4.9 % [−7.4, −1.0] | `results/stage5/h3b_stationary.json` | ✓ † |
| H4 vs bandit | −15.0 % / −18.7 % / −14.5 % | `results/stage5/hypotheses_test.json` | ✓ † |
| Oracle vs persistence (development, R1/R2) | 1.4–1.7 % / 3.0–3.7 % / 5.2–7.1 % | `hypotheses_test.json` `by_regime` | ✓ |
| Top-quartile oracle gain | 6.4 / 9.3 / 14.1 % | `results/stage6/h5_change_quartiles.csv` | ✓ |
| Probe Spearman / best-action hit rate | 0.03–0.56 / 6–14 % | `results/stage5/figures/probe_reward_quality.csv` | ✓ |
| R-A vs persistence / best fixed / bandit | −0.42 % / −5.7 % / −17.8 % | `results/robustness/ra/summary_test.json` | ✓ † |
| R-B vs persistence / best fixed | −0.35 % / −3.6 % (CI [−8.7, +0.04], includes 0) | `results/robustness/rb/summary_test.json` | ✓ † |
| Captured share of oracle gain, validation | 61–87 % (adaptive policies; 45–87 % with the current code) | `results/stage4/figures/val/value_of_information.csv` | ✓ (Stage 4, pre-fix; see finding 5) |
| Stage 5b H6 | +0.28 % [+0.14, +0.44] | `results/stage5b/dft_validation_report.json` | ✓ (validation only) |
| Mean-error table (6 rows × 8 policies) | — | `hypotheses_test.json` `by_regime` | ✓ † (after fix) |

---

## 4. References

All 21 references in `report/QEM_Project_Report.tex` were checked on 2026-10-13:
- **17 by DOI against Crossref:** title, authors, venue, volume, pages and year match. This covers the base paper `10.1109/JSAC.2026.3721807` (Rahman & Nguyen, *IEEE JSAC* 44, pp. 5600–5614, 2026) and the other 2026 IEEE papers.
- **Placidi et al. (arXiv:2601.14226) and Qiskit (Javadi-Abhari et al., arXiv:2405.08810, 2024):** verified against the arXiv API. Placidi et al. is a preprint and is correctly cited as such; it is not an IEEE paper.
- **Vaswani et al. (NeurIPS 30, 2017):** standard record. NeurIPS proceedings have no Crossref DOI, so this one was not registry-checked.
- **The dataset:** cited by its Hugging Face identifier and pinned revision.

doi.org itself blocks automated requests (HTTP 403), so open each DOI once in a browser before submission.

---

## 5. Verification record

| Check | Result |
|---|---|
| From-scratch reproduction of the confirmatory pipeline (data → Stage 3 → pair forecasting → Stage 5 → robustness) on a clean copy | **Passed.** All 15 key result files byte-identical (`scripts/compare_reproduction.py`) |
| Freeze checks (Stage 3, Stage 5, robustness) on the project | **Passed** |
| Raw data SHA-256 | **Passed** (`6c9feb0f…752f0b`) |
| Installed package versions = `requirements.txt` | **Passed** (11/11) |
| Import smoke test of every module | **Passed** |
| `pytest tests` | **Passed** (18/18) |
| Stage 6 analysis regenerated | **Passed** (byte-identical) |
| Stage 4 regenerated with current code | **Differs by design** (finding 5). Headline comparisons change by ≤ 1.3 percentage points and every conclusion holds; captured-share range 61–87 % → 45–87 % (`docs/FINAL_TECHNICAL_DOCUMENT.md` Appendix C) |
| Stage 5b regenerated | **Passed** (byte-identical, including network training) |
| LaTeX compilation of report and slides | **Passed** with Tectonic 0.17 (XeTeX engine): no errors, no overfull boxes. Overleaf's default pdfLaTeX was not tested here |
| Older Python (compatibility check) | **Passed.** In a temporary Python 3.11.15 environment (PyTorch 2.14.1, numpy 2.4.6, Qiskit 2.5.2) all 18 tests passed, and the saved Transformer models gave bit-identical test forecasts (1,732,540 values) to the original Python 3.14.5 run. Only inference was checked; retraining under 3.11 was not tested |
| Hardware execution | **Not applicable**; the project is simulation-only by design |

---

## 6. IEEE suitability assessment

### What is scientifically strong

- **Real drift data, carefully handled:**
  - 2.0 M genuine calibration measurements from four IBM Heron processors;
  - unit harmonisation; re-timestamps removed;
  - publication-time-aware joins, with a documented leakage fix;
  - chronological, unseen-qubit and unseen-device splits.
- **Pre-registration and test discipline:**
  - hypotheses and thresholds fixed before the expanded experiments;
  - SHA-256 freezes before each test evaluation;
  - test periods evaluated once;
  - deviations logged.
- **Fair comparison:**
  - identical action set and shot budget;
  - the bandit is charged for its probe shots;
  - bandit hyperparameters tuned on validation, with fixes that only help the bandit;
  - common random numbers across policies.
- **Statistics:** paired cluster bootstrap, Holm correction, and practical-significance thresholds in addition to p-values.
- **Reproducibility:** pinned data revision, pinned packages and a single entry point. The confirmatory pipeline was reproduced byte-identically from scratch.
- **Honest negative results:** H1, H2 and H6 are reported as not supported, with diagnostics (value-of-information ceiling, change-quartile analysis, probe-reward quality).

### What is weak or incomplete

1. **Simulation only.** No circuit was run on hardware. Readout, idle and crosstalk errors are not modelled, and coherent error is a single fixed CZ phase.
2. **Proxy ground truth.** The next published calibration stands in for the true device state during execution.
3. **Small circuits.** 4-qubit chains (6 in R-B) and six circuits (eight more instances in R-A).
4. **One test period** of about six weeks (2026-09-01 to 2026-10-12). It has already been used, so any new method needs fresh data.
5. **Few long-horizon decisions:** 7-day results rest on 19 chains, and the unseen device has no 7-day test decisions.
6. **Model-based controller cost.** It needs a classical simulation per decision, which scales exponentially with qubit count. Its advantage shrinks under model mismatch (R3).
7. **Novelty.** The positive result (calibration-informed, model-based selection beats fixed settings and a bandit) is not Transformer-specific. Persistence achieves it equally.

### What can be claimed defensibly

- A leakage-safe benchmark of adaptive QEM under real IBM calibration drift.
- Calibration-informed, model-based joint selection of ZNE scales, extrapolator and twirling:
  - reduces simulated mitigated error by 5.6 % (1 day) and 4.2 % (4 days) against the best fixed setting;
  - reduces it by 15–19 % against a faithful contextual-bandit adaptive ZNE at equal shots;
  - is non-inferior under stationary noise.
- Forecasting drift with a Transformer, ridge regression or a window mean does **not** improve these decisions over the latest calibration. The value of perfect foresight is bounded at about 1.4–7 %, mostly in abrupt changes.
- The bandit's Clifford-probe reward ranks mitigation actions poorly. This explains its underperformance for observable estimation.

### What must not be claimed

- That a Transformer improves mitigation decisions.
- Real-time operation.
- Hardware validation.
- Results beyond the simulated noise model.

### What must be corrected or completed before submission

1. **Framing.** A submission should not present the Transformer as the contribution. The current project title can stay for the course report, as the owner decided. For an IEEE manuscript, a title consistent with the evidence is strongly advised (candidates in `docs/experiment_reports/STAGE6_CONTRIBUTION.md` §4).
2. Fill the author block and insert figures in `report/QEM_Project_Report.tex`, then compile and proofread.
3. State the deviations (finding 6, finding 5) in the manuscript's methods section.

### Additional experiments

- **None are essential** for a course submission.
- **For an IEEE venue,** the most valuable addition is a hardware check: a small number of circuits on a real IBM backend, comparing fixed versus calibration-informed selection. A reviewer is likely to ask for it.
- **The next most valuable** is a confirmatory evaluation on calibration data published after 2026-10-12, which is fresh and untouched.

### Overall verdict

The evidence supports a **rigorous empirical, partly negative-result paper**:
- a benchmark and analysis of adaptive QEM under real drift;
- best suited to a conference or workshop, or to an IEEE journal's letters or short-paper track after a hardware check.

It does **not** support a paper whose central claim is a Transformer-based method. Publication cannot be guaranteed, and acceptance depends on reviewers' view of the simulation-only evidence and the negative primary result.
