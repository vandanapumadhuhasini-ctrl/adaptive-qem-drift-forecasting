# Stage 4 — Design: Forecast-Driven Adaptive QEM and Fair Baselines

Status: design fixed 2026-10-12, before any Stage 4 result. **Framing (approved): forecast-driven mitigation versus reactive (persistence) and bandit adaptation. The forecaster may be simple, and the Transformer is one forecaster among several.**

Stage 4 builds and validates the machinery on the **validation period only**. Test-period evaluation belongs to Stage 5, after a freeze.

## 1. Decision setting (real calibration data, simulated execution)

- **Epoch:** IBM recalibrates T1 chip-wide once a day, within seconds (93 % of backend-days cover ≥ 90 % of qubits). One decision epoch = one backend-day; the decision time is the publication of that day's calibration.
- **Qubit group:** a linear chain of 4 physically coupled qubits (CZ pairs from the dataset), chosen deterministically and disjointly per backend. Development chains use only development qubits (ibm_fez, ibm_kingston). Unseen-device chains are on ibm_marrakesh.
- **Execution state (ground truth):** the calibration published at epoch e + k for every chain member: T1, T2, sx error, and the actual pair-level CZ error from the as-of join. This is the proxy for the true device state during execution (declared threat to validity).
- **Information at decision time:** only calibrations published up to epoch e, identical for every policy.

## 2. Simulated device (labelled *simulated*; parameters from real calibrations)

Qiskit Aer, density matrix, native gate set {CZ, sx, rz, x}:

| Gate | Noise |
|---|---|
| sx, x | Thermal relaxation (T1, min(T2, 2T1)) over the real duration (24/32/36 ns per backend), composed with depolarising noise that tops up to IBM's reported sx error |
| CZ | Relaxation on both qubits over 68 ns, composed with 2-qubit depolarising noise topped up to the reported pair error; plus an optional coherent controlled-phase error δ on \|11⟩ |
| rz | Virtual, noiseless |

- Twirl Paulis are absorbed into neighbouring single-qubit gates on hardware, so they are modelled as noiseless.
- Readout error is **excluded**: all policies are assumed to share the same readout mitigation. Idle-qubit decoherence is not modelled. Both are declared limitations.

**Regimes:**

| Regime | True device | Controller's model |
|---|---|---|
| R1 incoherent drift | δ = 0 | δ = 0 |
| R2 coherent drift | δ = 0.03 rad per CZ | δ = 0.03 |
| R3 model mismatch | δ = 0.03 | δ = 0 (the controller is wrong; tests H5) |
| R4 stationary control | each chain frozen at its training-period median state, δ = 0 | same |

## 3. Circuits and observables

- **Families** (4 qubits, transpiled to the native set, linear chain), each at two depths:
  - layered hardware-efficient ansatz (HEA);
  - Trotterised transverse-field Ising model (TFIM);
  - QAOA MaxCut on a ring.
- **Observables:** weight-1 and nearest-neighbour weight-2 Z/X Paulis, plus ZZZZ, keeping those with noiseless |⟨P⟩| > 0.2, fixed before any noisy run.
- **Probe circuits** (bandit reward only): the same circuit with every rotation angle snapped to the nearest multiple of π/2. This makes it Clifford, so its ideal value is exactly known at run time, as a hardware bandit would require.

## 4. Action set (identical for every policy)

| Item | Values |
|---|---|
| Scale factors | {1,3}, {1,2,3}, {1,3,5}, **{1,3,5,7}** — the base paper's arms are {1,3}, {1,3,5}, {1,3,5,7} |
| Extrapolators | linear; Richardson; exponential; polynomial of degree ≤ 2 for {1,3,5,7} (as in the base paper) |
| Pauli twirling | off / on (8 seeded instances, local CZ folding) |
| Plus | "none" and "twirl only" |

**Budget:** S = 8,192 shots per observable per decision for every policy, split evenly over scale factors (and over twirl instances).

## 5. Policies

| Policy | Rule | Extra information or cost |
|---|---|---|
| None | No mitigation | — |
| Best fixed | The single action with the lowest mean error over **validation** decisions | Selected on validation |
| Persistence-adaptive | Model-based controller fed the latest calibration | — |
| Window-mean-adaptive | Controller fed the 20-CCE window-mean forecast | — |
| Ridge-adaptive | Controller fed the ridge forecast | — |
| **Transformer-adaptive** | Controller fed the Transformer forecast (5-seed ensemble) | — |
| Oracle | Controller fed the true execution state | Upper bound; not realisable |
| **CMAB-probe** | LinUCB, discounted, cost-aware (base paper Eqs. 16–20). Context = latest calibration summary of the chain. Reward = −(error on the Clifford probe circuit) − β·Var − λ_c·cost, observed for the chosen arm only | **1/8 of S is spent on probes**, so the main estimate uses 7/8 of S |
| CMAB-warm | As CMAB-probe, but initialised with pseudo-rewards from the same simulator model the controller uses | Same model knowledge as our controller |
| CMAB-oracle-reward | Reward = true error, no probe shots | Optimistic reference only, not realisable |

**Controller (model-based, predictive):**
1. Build a noise model from the *forecast* state of the chain for the execution epoch.
2. Simulate every action exactly.
3. Estimate each action's expected |error| including shot noise (Monte-Carlo trials).
4. Pick the action with the lowest expected error.

It updates once per epoch, it is predictive because it acts on the forecast for epoch e + k, and it needs **no extra quantum shots**. The classical cost is the simulation, which is reported as a cost.

**Pair-level CZ forecast:** latest pair error × exp(mean of the two qubits' forecast Δ log CZ-mean). This uses only Stage 3 outputs.

## 6. Metric and statistics (`PROTOCOL.md` §4–§5)

- **Per decision:** error = mean over observables of E_shots|Ê − E_ideal|, with 200 Monte-Carlo shot trials using **common random numbers** across policies.
- **Statistics:** paired differences; cluster bootstrap over chains within backend; Holm correction within each secondary family.
- **H2 primary cell:** R1 / R2 pooled real drift, k = 1, development chains, all circuits pooled. Evaluated on test in Stage 5; in Stage 4 only validation numbers are produced.

## 7. Stage 4 acceptance

1. Device noise model reproduces the reported sx/CZ errors: unit test, simulated average gate infidelity within 5 % of target.
2. Folded and twirled circuits are noiseless-equivalent to the original (exact check).
3. Every policy spends exactly S shots per observable per decision: assertion, with probe shots counted.
4. Information parity table (above) enforced in code: policies receive only arrays computed from data published ≤ epoch e.
5. CMAB sanity: in a stationary toy stream with one clearly best arm, it converges to that arm.
6. Validation-period results for all policies are produced and documented. **No test-period numbers.**

## 8. Compute plan

- **Exact simulations are cached** per (decision, circuit, regime, state source), so policies are evaluated offline from the cache.
- **Size:** 16 chains × 2 backends × ~31 validation epochs × 6 circuits × 3 regimes.
- **Parallelism:** a process pool, with a fixed seed per unit.
