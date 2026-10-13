"""
Stage 4 mitigation layer on native {cz, sx, rz, x} circuits:
local CZ folding (ZNE), CZ Pauli twirling, the shared action set, exact
expectation values, and shot-noise-aware expected error of each action.
"""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli
from qiskit_aer import AerSimulator

from src.qem.zne import extrapolate

SCALES = (1, 2, 3, 5, 7)
N_TWIRL = 8
TWIRL_SEED = 1234
SHOTS = 8192                  # per observable per decision, identical for every policy
PROBE_FRACTION = 1 / 8        # bandit probe budget, deducted from SHOTS
MC_TRIALS = 200


@dataclass(frozen=True)
class Action:
    name: str
    scales: tuple
    twirl: bool
    method: str


def _actions():
    acts = [Action("none", (1,), False, "none"), Action("twirl", (1,), True, "none")]
    zne = [((1, 3), "linear"), ((1, 2, 3), "linear"), ((1, 3, 5), "richardson"), ((1, 3, 5), "exp"),
           ((1, 3, 5, 7), "poly2"), ((1, 3, 5, 7), "exp")]
    for tw in (False, True):
        for sc, m in zne:
            acts.append(Action(("tw_" if tw else "") + f"zne_{m}_{''.join(map(str, sc))}", sc, tw, m))
    return acts


ACTIONS = _actions()
ACTION_NAMES = [a.name for a in ACTIONS]


def action_cost(a):
    """Relative execution cost: total folded circuit length summed over scale factors
    (twirling changes the number of distinct circuits, not the shots)."""
    return float(sum(a.scales))


# ------------------------------------------------------------------ circuits

def _ops(native):
    return [(i.operation.name, [native.find_bit(q).index for q in i.qubits], list(i.operation.params))
            for i in native.data]


def fold(ops, scale):
    """CZ -> CZ (CZ CZ)^n. Partial scales fold the first CZs. Returns ops and achieved scale."""
    n_cz = sum(o[0] == "cz" for o in ops)
    extra = int(round((scale - 1) * n_cz / 2))
    full, part = divmod(extra, n_cz)
    out, seen = [], 0
    for o in ops:
        out.append(o)
        if o[0] == "cz":
            out.extend([o, o] * (full + (1 if seen < part else 0)))
            seen += 1
    return out, (n_cz + 2 * extra) / n_cz


@lru_cache(maxsize=None)
def _cz_table():
    cz = QuantumCircuit(2)
    cz.cz(0, 1)
    tab = []
    for a in "IXYZ":
        for b in "IXYZ":
            after = Pauli(b + a).evolve(cz, frame="s").to_label().lstrip("-i")
            tab.append(((a, b), (after[1], after[0])))
    return tab


def twirl(ops, rng):
    tab = _cz_table()
    out = []
    for o in ops:
        if o[0] != "cz":
            out.append(o)
            continue
        (pa, pb), (qa, qb) = tab[rng.integers(16)]
        a, b = o[1]
        out += [("pauli", [a], [pa]), ("pauli", [b], [pb]), o, ("pauli", [a], [qa]), ("pauli", [b], [qb])]
    return out


def build(ops, n, observables):
    qc = QuantumCircuit(n)
    for name, qs, params in ops:
        if name == "pauli":
            if params[0] != "I":
                getattr(qc, params[0].lower())(qs[0])
        elif name == "rz":
            qc.rz(params[0], qs[0])
        else:
            getattr(qc, name)(*qs)
    for o in observables:
        qc.save_expectation_value(Pauli(o), range(n), label=o)
    return qc


def circuit_set(native, observables):
    """All circuits needed for every action: SCALES × (1 untwirled + N_TWIRL twirled)."""
    ops, n = _ops(native), native.num_qubits
    circuits, achieved = [], []
    for s in SCALES:
        f, a = fold(ops, s)
        achieved.append(a)
        circuits.append(build(f, n, observables))
        rng = np.random.default_rng(TWIRL_SEED)
        circuits += [build(twirl(f, rng), n, observables) for _ in range(N_TWIRL)]
    return circuits, np.array(achieved)


def exact_values(circuits, noise_model, observables, threads=0):
    """Returns (n_obs, n_scales, 1 + N_TWIRL); column 0 is untwirled."""
    sim = AerSimulator(method="density_matrix", noise_model=noise_model, max_parallel_threads=threads)
    res = sim.run(circuits).result()
    v = np.array([[res.data(i)[o] for i in range(len(circuits))] for o in observables])
    return v.reshape(len(observables), len(SCALES), 1 + N_TWIRL)


# ------------------------------------------------------------------ estimators

def _sample(vals, shots_per_scale, rng, trials):
    """vals (n_scales_used, n_inst): exact values; returns (trials, n_scales_used) estimates."""
    n_sc, n_inst = vals.shape
    per = max(1, shots_per_scale // n_inst)
    p = (1 + np.clip(vals, -1, 1)) / 2
    k = rng.binomial(per, p, size=(trials, n_sc, n_inst))
    return (2 * k / per - 1).mean(axis=2)


def action_values(values, a):
    idx = [SCALES.index(s) for s in a.scales]
    cols = slice(1, None) if a.twirl else slice(0, 1)
    return values[:, idx, cols]


def _fit(scales, est, method):
    if method == "poly2":
        deg = min(len(scales) - 1, 2)
        return float(np.clip(np.polyfit(scales, est, deg)[-1], -1, 1))
    return extrapolate(scales, est, method)


def _intercept_weights(scales, deg):
    """Least-squares intercept as a fixed linear combination of the scale values."""
    V = np.vander(np.asarray(scales, float), deg + 1, increasing=True)
    return np.linalg.pinv(V)[0]


def _fit_vec(scales, est, method):
    """Vectorised extrapolation; est (..., n_scales). Identical to zne.extrapolate / _fit."""
    n = len(scales)
    if method == "none":
        out = est[..., 0]
    elif method == "exp":
        w1 = _intercept_weights(scales, 1)
        sign = np.sign(est[..., 0])
        ok = np.all(np.sign(est) == sign[..., None], axis=-1) & np.all(np.abs(est) > 1e-3, axis=-1)
        logv = np.log(np.clip(np.abs(est), 1e-12, None))
        out = np.where(ok, sign * np.exp(logv @ w1), est @ w1)
    else:
        deg = {"linear": 1, "richardson": n - 1, "poly2": min(n - 1, 2)}[method]
        out = est @ _intercept_weights(scales, deg)
    return np.clip(out, -1, 1)


def expected_errors(values, achieved, ideal, rng, shots=SHOTS, trials=MC_TRIALS, obs_idx=None,
                    single_draw=False, return_var=False):
    """
    Mean over observables of E_shots |Ê − ideal| for every action (vectorised Monte Carlo).
    single_draw=True returns one shot-noise realisation (bandit feedback) instead.
    """
    obs_idx = np.arange(values.shape[0]) if obs_idx is None else np.asarray(obs_idx)
    T = 1 if single_draw else trials
    out = np.zeros(len(ACTIONS))
    var = np.zeros(len(ACTIONS))
    for j, a in enumerate(ACTIONS):
        v = action_values(values, a)[obs_idx]                       # (O, S, I)
        per = max(1, (shots // len(a.scales)) // v.shape[2])
        p = (1 + np.clip(v, -1, 1)) / 2
        k = rng.binomial(per, p[None], size=(T,) + p.shape)       # (T, O, S, I)
        est = (2 * k / per - 1).mean(axis=3)                       # (T, O, S)
        idx = [SCALES.index(s) for s in a.scales]
        fit = _fit_vec(achieved[idx], est, a.method)               # (T, O)
        out[j] = np.abs(fit - ideal[obs_idx][None]).mean()
        if return_var:
            var[j] = fit.var(axis=0).mean()
    return (out, var) if return_var else out
