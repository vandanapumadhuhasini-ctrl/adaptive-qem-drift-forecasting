"""
Stage 4 device layer: decision epochs, qubit chains, pair-level CZ states, and
an Aer noise model built from one calibration snapshot of a 4-qubit chain.

Calibration values are real IBM records (Stage 2). Execution is simulated.
"""

import numpy as np
import pandas as pd
from qiskit.quantum_info import Kraus, SuperOp, average_gate_fidelity
from qiskit_aer.noise import (NoiseModel, QuantumError, coherent_unitary_error, depolarizing_error,
                              thermal_relaxation_error)

from src.data.drift_dataset import genuine_events, load_config, read_events

BASIS = ["cz", "sx", "rz", "x"]
SX_NS = {"ibm_fez": 24.0, "ibm_kingston": 32.0, "ibm_marrakesh": 36.0}   # dataset medians
CZ_NS = 68.0                                                              # all three backends
CHAIN_LEN = 4


# ------------------------------------------------------------------ epochs

def build_epochs(cce, min_frac=0.9, max_spread_h=6.0):
    """
    One epoch per backend-day with a chip-wide T1 calibration.

    Epoch (decision) time = the moment by which `min_frac` of that day's calibrations
    had been published (quantile of publication times). A day is a valid epoch only if
    (a) it covers at least `min_frac` of the backend's qubits, and (b) that moment lies
    within `max_spread_h` of the day's median publication, which rejects days whose
    calibrations were only published much later (initial snapshots, collection
    outages). Calibrations published after the epoch time are treated as unavailable
    at that epoch (see qubit_epoch_rows).
    """
    c = cce.assign(day=cce["t1_calibrated"].dt.floor("D"))
    nq = c.groupby("backend")["qubit"].nunique()
    g = c.groupby(["backend", "day"]).agg(
        n=("qubit", "nunique"),
        epoch_time=("decision_time", lambda s: s.quantile(min_frac)),
        median_pub=("decision_time", "median")).reset_index()
    ok = (g["n"] >= min_frac * nq.reindex(g["backend"]).to_numpy()) & \
         ((g["epoch_time"] - g["median_pub"]).dt.total_seconds() / 3600 <= max_spread_h)
    g = g[ok].sort_values(["backend", "day"])
    g["epoch"] = g.groupby("backend").cumcount()
    for b, d in g.groupby("backend"):
        assert d["epoch_time"].is_monotonic_increasing and d["epoch_time"].is_unique, f"{b}: epochs out of order"
    return g[["backend", "day", "epoch", "epoch_time", "n"]].reset_index(drop=True)


def qubit_epoch_rows(cce, epochs):
    """(backend, qubit, epoch) -> the qubit's CCE row in that epoch's day (last one)."""
    c = cce.assign(day=cce["t1_calibrated"].dt.floor("D")).reset_index(names="cce_row")
    c = c.merge(epochs[["backend", "day", "epoch", "epoch_time"]], on=["backend", "day"])
    c = c[c["decision_time"] <= c["epoch_time"]]          # published by the epoch time
    return c.sort_values("t1_calibrated").groupby(["backend", "qubit", "epoch"]).tail(1)


# ------------------------------------------------------------------ chains

def coupling_pairs(cfg=None):
    cfg = cfg or load_config()
    ev = genuine_events(read_events(cfg))
    cz = ev[ev["property_family"] == "cz_error"]
    return cz, cz[["backend", "qubit_a", "qubit_b"]].drop_duplicates()


def select_chains(pairs, backend, allowed, n_chains, seed=0):
    """Disjoint simple paths of CHAIN_LEN coupled qubits, found greedily in a seeded order."""
    adj = {}
    for a, b in pairs.loc[pairs["backend"] == backend, ["qubit_a", "qubit_b"]].itertuples(index=False):
        if a in allowed and b in allowed:
            adj.setdefault(int(a), set()).add(int(b))
            adj.setdefault(int(b), set()).add(int(a))
    rng = np.random.default_rng(seed)
    starts = sorted(adj)
    rng.shuffle(starts)
    used, chains = set(), []

    def extend(path):
        if len(path) == CHAIN_LEN:
            return path
        for nb in sorted(adj[path[-1]]):
            if nb not in used and nb not in path:
                out = extend(path + [nb])
                if out:
                    return out
        return None

    for s in starts:
        if s in used:
            continue
        p = extend([s])
        if p:
            chains.append(p)
            used.update(p)
        if len(chains) == n_chains:
            break
    return chains


def pair_states(cz_events, backend, pairs, times):
    """Latest valid CZ error of each (a, b) pair published at or before each time."""
    ev = cz_events[(cz_events["backend"] == backend) & ~cz_events["is_failure_ceiling"]]
    ev = ev.sort_values("observed_time")
    out = {}
    tt = pd.Series(np.sort(np.asarray(times, dtype="datetime64[ns]")))
    for a, b in pairs:
        e = ev[((ev["qubit_a"] == a) & (ev["qubit_b"] == b)) | ((ev["qubit_a"] == b) & (ev["qubit_b"] == a))]
        if e.empty:                                   # only failure ceilings: no valid value
            out[(a, b)] = dict(zip(tt.to_numpy(), np.full(len(tt), np.nan)))
            continue
        idx = np.searchsorted(e["observed_time"].to_numpy("datetime64[ns]"), tt.to_numpy(), side="right") - 1
        vals = np.where(idx >= 0, e["value"].to_numpy()[np.clip(idx, 0, None)], np.nan)
        out[(a, b)] = dict(zip(tt.to_numpy(), vals))
    return out


# ------------------------------------------------------------------ noise model

def _depol_fill(reported, relax_infidelity, dim):
    """Depolarising parameter that tops the average gate infidelity up to `reported`."""
    extra = max(reported - relax_infidelity, 0.0)
    return min(extra * dim / (dim - 1), 1.0)


def _relax(t1_us, t2_us, ns):
    t2 = min(t2_us, 2 * t1_us)
    return thermal_relaxation_error(t1_us * 1e3, t2 * 1e3, ns)


MAX_GATE_ERROR = 0.5     # clip for unphysical *forecast* values (no real state exceeds 0.18)


def _compose(*errors):
    """Compose quantum errors as channels (superoperators) and return one Kraus error.
    Numerically identical to QuantumError.compose for valid channels, without Aer's strict
    probability-normalisation check that can fail on round-off."""
    ch = SuperOp(errors[0].to_quantumchannel())
    for e in errors[1:]:
        ch = ch.compose(SuperOp(e.to_quantumchannel()))
    return QuantumError(Kraus(ch))


def noise_model(state, backend, delta=0.0):
    """
    state: dict with arrays t1[4], t2[4] (µs), sx[4], cz[3] (pairs (0,1),(1,2),(2,3) of the chain),
    indexed by *circuit* qubit. delta: coherent controlled-phase error (rad) per CZ.
    """
    state = {k: np.asarray(v, dtype=np.float64) for k, v in state.items()}   # float32 inputs break Aer's checks
    nm = NoiseModel(basis_gates=BASIS)
    sx_ns = SX_NS[backend]
    for q in range(CHAIN_LEN):
        tr = _relax(state["t1"][q], state["t2"][q], sx_ns)
        p = _depol_fill(min(state["sx"][q], MAX_GATE_ERROR), 1 - average_gate_fidelity(tr), 2)
        err = _compose(tr, depolarizing_error(p, 1)) if p > 0 else tr
        nm.add_quantum_error(err, ["sx", "x"], [q])
    phase = coherent_unitary_error(np.diag([1, 1, 1, np.exp(1j * delta)])) if delta else None
    for i in range(CHAIN_LEN - 1):
        a, b = i, i + 1
        tr = _relax(state["t1"][a], state["t2"][a], CZ_NS).tensor(_relax(state["t1"][b], state["t2"][b], CZ_NS))
        p = _depol_fill(min(state["cz"][i], MAX_GATE_ERROR), 1 - average_gate_fidelity(tr), 4)
        parts = [tr] + ([depolarizing_error(p, 2)] if p > 0 else [])
        err = _compose(phase, *parts) if phase is not None else (_compose(*parts) if len(parts) > 1 else tr)
        nm.add_quantum_error(err, ["cz"], [a, b])
        nm.add_quantum_error(err, ["cz"], [b, a])
    return nm


def gate_infidelities(state, backend, delta=0.0):
    """Average gate infidelity of each modelled sx and CZ error (for the unit test)."""
    nm = noise_model(state, backend, delta)
    out = {}
    for name, qubits in [("sx", (q,)) for q in range(CHAIN_LEN)] + [("cz", (i, i + 1)) for i in range(CHAIN_LEN - 1)]:
        err = nm._local_quantum_errors[name][qubits]
        out[(name, qubits)] = 1 - average_gate_fidelity(err)
    return out
