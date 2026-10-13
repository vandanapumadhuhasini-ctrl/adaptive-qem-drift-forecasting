# Stage 5 Robustness Plan (fixed 2026-10-13, before any robustness simulation)

Purpose: test whether the Stage 5 conclusions depend on the particular circuit instances or on the 4-qubit size. Robustness results can **qualify** but not overturn the pre-registered verdicts.

## R-A: circuit instances
- **New circuits:** HEA at depths 4 and 8, and QAOA at depths 2 and 4, each with circuit seeds 12 and 13: 8 circuits (main study: seed 11). TFIM is deterministic and is not re-instanced. Observables are selected exactly as before (noiseless |⟨P⟩| > 0.2).
- **Same setup as Stage 5:** chains, decisions (k = 1), regimes R1–R3, all policies, shot budget, and common-random-number seeding.
- **Selections on validation:** best fixed action (global, and per circuit) is chosen on validation for the new circuits. The bandit reuses the Stage 5 validation-tuned hyperparameters (`results/stage5/selections.json`, k1); no re-tuning.
- **Reported on test, development chains:**
  - Transformer − persistence (H2 analogue);
  - Transformer − best fixed (H3a);
  - Transformer − CMAB-probe (H4);
  - plus window mean and ridge.
  - Paired cluster bootstrap with the same settings.

## R-B: 6-qubit chains
- **Setup:** disjoint 6-qubit linear chains (development qubits; same greedy, seeded selection). Circuits: HEA depth 4 and TFIM 3 steps on 6 qubits. k = 1. Regimes R1 and R2.
- **Policies:** none; best fixed (validation); adaptive with persistence / window mean / ridge / Transformer / oracle. **No bandit** (compute; H4 is covered by the 4-qubit study and R-A).
- **Forecast states:** per-qubit Stage 3 forecasts and Stage 5a pair CZ forecasts (frozen artifacts).
- **Reported:** the same paired comparisons as R-A.

## Discipline
- New code in new modules (`src/qem/nq.py`, `experiments/robust_*.py`); frozen Stage 5 files unchanged.
- Separate freeze `configs/frozen_robustness.yaml` before any robustness test metric.
