# Documentation index

**Start with `FINAL_TECHNICAL_DOCUMENT.md`.** It describes the whole project and cites the other documents.

## Main documents

| Document | Content |
|---|---|
| `FINAL_TECHNICAL_DOCUMENT.md` | Complete technical description: problem, data, methods, results, limitations, reproduction |
| `audit/AUDIT_REPORT.md` | Repository audit, number traceability, reference checks, verification record, IEEE suitability |
| `audit/FILE_INVENTORY.md` | Disposition of every file |

## Pre-registration (kept in this folder)

| Document | Content |
|---|---|
| `PROTOCOL.md` | Pre-registered protocol (frozen; its SHA-256 is part of the Stage 5 freeze) |
| `PROTOCOL_ADDENDUM.md` | Every change after the freeze, with date, reason and timing relative to the test |
| `ROBUSTNESS_PLAN.md` | Pre-registered robustness analyses (frozen) |

`PROTOCOL.md` and `ROBUSTNESS_PLAN.md` cannot be edited without breaking the freeze. Their bare references to other documents (for example `STAGE1_RESEARCH_PROPOSAL.md`) resolve to the subfolders below.

## `research/` — background and planning

| Document | Content |
|---|---|
| `research/BASE_PAPER_REVIEW.md` | Base-paper survey (Crossref-verified citations) and choice |
| `research/STAGE1_RESEARCH_PROPOSAL.md` | Research questions, hypotheses H1–H5, pilot evidence |
| `research/STAGE4_DESIGN.md` | Controller, baselines, regimes and metric design (written before results) |
| `research/PROJECT_PLAN.md` | Stage plan and dated progress log (historical) |

## `experiment_reports/` — results per stage

| Document | Content |
|---|---|
| `experiment_reports/STAGE2_DATA_REPORT.md` | Data |
| `experiment_reports/STAGE3_FORECASTING_REPORT.md` | Forecasting (H1) |
| `experiment_reports/STAGE4_CONTROLLER_REPORT.md` | Development stage, validation only |
| `experiment_reports/STAGE5_RESULTS.md` | Confirmatory decision experiments (H2–H4) |
| `experiment_reports/STAGE6_CONTRIBUTION.md` | Contribution validation and robustness |
| `experiment_reports/STAGE5B_DFT.md` | Exploratory decision-focused Transformer (stopped) |
