"""
Stage 4 benchmark circuits on a 4-qubit linear chain, transpiled to the IBM Heron
native set {cz, sx, rz, x}. Each family is used at two depths. Clifford probe
versions (rz angles snapped to multiples of π/2) supply the bandit's reward.
"""

from functools import lru_cache

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Pauli
from qiskit.transpiler import CouplingMap
from qiskit_aer import AerSimulator

N = 4
OBS_THRESHOLD = 0.2
FAMILIES = {           # name -> (builder, depth settings)
    "hea": ("hea", (4, 8)),
    "tfim": ("tfim", (3, 6)),
    "qaoa": ("qaoa", (2, 4)),
}


def _hea(layers, seed):
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(N)
    for _ in range(layers):
        for q in range(N):
            qc.ry(rng.uniform(-0.6, 0.6), q)
            qc.rz(rng.uniform(-np.pi, np.pi), q)
        for q in range(N - 1):
            qc.cz(q, q + 1)
    return qc


def _tfim(steps, seed, J=1.0, h=0.9, dt=0.25):
    qc = QuantumCircuit(N)
    for q in range(N):
        qc.ry(np.pi / 3, q)
    for _ in range(steps):
        for q in range(N - 1):
            qc.rzz(2 * J * dt, q, q + 1)
        for q in range(N):
            qc.rx(2 * h * dt, q)
    return qc


def _qaoa(p, seed):
    rng = np.random.default_rng(seed)
    gammas, betas = rng.uniform(0.2, 0.8, p), rng.uniform(0.1, 0.5, p)
    qc = QuantumCircuit(N)
    qc.h(range(N))
    for g, b in zip(gammas, betas):
        for q in range(N - 1):
            qc.rzz(2 * g, q, q + 1)
        for q in range(N):
            qc.rx(2 * b, q)
    return qc


BUILDERS = {"hea": _hea, "tfim": _tfim, "qaoa": _qaoa}


def candidate_observables():
    obs = []
    for q in range(N):
        for P in "ZX":
            obs.append("".join(P if i == q else "I" for i in reversed(range(N))))
    for q in range(N - 1):
        for P in "ZX":
            obs.append("".join(P if i in (q, q + 1) else "I" for i in reversed(range(N))))
    return obs + ["ZZZZ", "XXXX"]


def _ideal(qc, observables):
    c = qc.copy()
    for o in observables:
        c.save_expectation_value(Pauli(o), range(N), label=o)
    d = AerSimulator(method="density_matrix").run(c).result().data(0)
    return np.array([d[o] for o in observables])


def to_native(qc):
    return transpile(qc, basis_gates=["cz", "sx", "rz", "x"], coupling_map=CouplingMap.from_line(N),
                     initial_layout=list(range(N)), optimization_level=1, seed_transpiler=0)


def clifford_probe(native):
    """Snap every rz angle to the nearest multiple of π/2 (sx, x, cz are Clifford)."""
    out = native.copy_empty_like()
    for inst in native.data:
        if inst.operation.name == "rz":
            ang = float(inst.operation.params[0])
            out.rz(np.round(ang / (np.pi / 2)) * (np.pi / 2), inst.qubits[0])
        else:
            out.append(inst)
    return out


@lru_cache(maxsize=None)
def benchmark(family, depth, seed=11):
    """Native circuit, its Clifford probe, the selected observables and their ideal values."""
    native = to_native(BUILDERS[family](depth, seed))
    cands = candidate_observables()
    ideal_all = _ideal(native, cands)
    keep = [i for i, v in enumerate(ideal_all) if abs(v) > OBS_THRESHOLD]
    observables = [cands[i] for i in keep]
    probe = clifford_probe(native)
    return dict(name=f"{family}_d{depth}", native=native, probe=probe, observables=observables,
                ideal=ideal_all[keep], probe_ideal=_ideal(probe, observables),
                n_cz=int(native.count_ops().get("cz", 0)))


def all_benchmarks():
    return [benchmark(f, d) for f, (_, depths) in FAMILIES.items() for d in depths]
