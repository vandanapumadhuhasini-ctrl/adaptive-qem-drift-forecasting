"""
n-qubit generalisation for the robustness analyses (docs/ROBUSTNESS_PLAN.md).

Reuses the frozen Stage 4/5 helpers unchanged; for n = 4 and seed 11 the circuits,
observables and noise models are identical to src/qem/benchmarks.py and
src/qem/device.py (checked in experiments/robust_check.py).
"""

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Pauli, average_gate_fidelity
from qiskit.transpiler import CouplingMap
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, coherent_unitary_error, depolarizing_error

from src.qem.benchmarks import OBS_THRESHOLD, clifford_probe
from src.qem.device import BASIS, CZ_NS, MAX_GATE_ERROR, SX_NS, _compose, _depol_fill, _relax


def hea(n, layers, seed):
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(n)
    for _ in range(layers):
        for q in range(n):
            qc.ry(rng.uniform(-0.6, 0.6), q)
            qc.rz(rng.uniform(-np.pi, np.pi), q)
        for q in range(n - 1):
            qc.cz(q, q + 1)
    return qc


def tfim(n, steps, seed, J=1.0, h=0.9, dt=0.25):
    qc = QuantumCircuit(n)
    for q in range(n):
        qc.ry(np.pi / 3, q)
    for _ in range(steps):
        for q in range(n - 1):
            qc.rzz(2 * J * dt, q, q + 1)
        for q in range(n):
            qc.rx(2 * h * dt, q)
    return qc


def qaoa(n, p, seed):
    rng = np.random.default_rng(seed)
    gammas, betas = rng.uniform(0.2, 0.8, p), rng.uniform(0.1, 0.5, p)
    qc = QuantumCircuit(n)
    qc.h(range(n))
    for g, b in zip(gammas, betas):
        for q in range(n - 1):
            qc.rzz(2 * g, q, q + 1)
        for q in range(n):
            qc.rx(2 * b, q)
    return qc


BUILDERS = {"hea": hea, "tfim": tfim, "qaoa": qaoa}


def candidate_observables(n):
    obs = []
    for q in range(n):
        for P in "ZX":
            obs.append("".join(P if i == q else "I" for i in reversed(range(n))))
    for q in range(n - 1):
        for P in "ZX":
            obs.append("".join(P if i in (q, q + 1) else "I" for i in reversed(range(n))))
    return obs + ["Z" * n, "X" * n]


def _ideal(qc, observables):
    c = qc.copy()
    for o in observables:
        c.save_expectation_value(Pauli(o), range(qc.num_qubits), label=o)
    d = AerSimulator(method="density_matrix").run(c).result().data(0)
    return np.array([d[o] for o in observables])


def benchmark_n(family, depth, n=4, seed=11):
    raw = BUILDERS[family](n, depth, seed)
    native = transpile(raw, basis_gates=["cz", "sx", "rz", "x"], coupling_map=CouplingMap.from_line(n),
                       initial_layout=list(range(n)), optimization_level=1, seed_transpiler=0)
    cands = candidate_observables(n)
    ideal_all = _ideal(native, cands)
    keep = [i for i, v in enumerate(ideal_all) if abs(v) > OBS_THRESHOLD]
    observables = [cands[i] for i in keep]
    probe = clifford_probe(native)
    suffix = "" if (n, seed) == (4, 11) else (f"_s{seed}" if n == 4 else f"_n{n}")
    return dict(name=f"{family}_d{depth}{suffix}", native=native, probe=probe, observables=observables,
                ideal=ideal_all[keep], probe_ideal=_ideal(probe, observables),
                n_cz=int(native.count_ops().get("cz", 0)))


def noise_model_n(state, backend, delta=0.0):
    """Same construction as src.qem.device.noise_model for any chain length
    (state: t1[n], t2[n], sx[n], cz[n-1])."""
    state = {k: np.asarray(v, dtype=np.float64) for k, v in state.items()}
    n = len(state["t1"])
    nm = NoiseModel(basis_gates=BASIS)
    for q in range(n):
        tr = _relax(state["t1"][q], state["t2"][q], SX_NS[backend])
        p = _depol_fill(min(state["sx"][q], MAX_GATE_ERROR), 1 - average_gate_fidelity(tr), 2)
        err = _compose(tr, depolarizing_error(p, 1)) if p > 0 else tr
        nm.add_quantum_error(err, ["sx", "x"], [q])
    phase = coherent_unitary_error(np.diag([1, 1, 1, np.exp(1j * delta)])) if delta else None
    for i in range(n - 1):
        a, b = i, i + 1
        tr = _relax(state["t1"][a], state["t2"][a], CZ_NS).tensor(_relax(state["t1"][b], state["t2"][b], CZ_NS))
        p = _depol_fill(min(state["cz"][i], MAX_GATE_ERROR), 1 - average_gate_fidelity(tr), 4)
        parts = [tr] + ([depolarizing_error(p, 2)] if p > 0 else [])
        err = _compose(phase, *parts) if phase is not None else (_compose(*parts) if len(parts) > 1 else tr)
        nm.add_quantum_error(err, ["cz"], [a, b])
        nm.add_quantum_error(err, ["cz"], [b, a])
    return nm


def select_chains_n(pairs, backend, allowed, n_chains, length, seed=0):
    """Same greedy, seeded disjoint-path search as device.select_chains, any length."""
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
        if len(path) == length:
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
