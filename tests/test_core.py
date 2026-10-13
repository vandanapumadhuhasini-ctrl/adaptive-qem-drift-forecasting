"""Unit tests of the numerical building blocks (fast; no project outputs needed)."""

import numpy as np
import pandas as pd

from src.evaluation.stats import bootstrap_weights, holm, per_series_mae
from src.qem.zne import extrapolate

STATE = dict(t1=[150.0, 120.0, 180.0, 90.0], t2=[100.0, 80.0, 140.0, 60.0],
             sx=[3e-4, 2.5e-4, 4e-4, 3.5e-4], cz=[3e-3, 2.5e-3, 4e-3])


def test_zne_fits_recover_the_zero_noise_value():
    s = np.array([1, 3, 5])
    assert abs(extrapolate(s, 0.8 - 0.05 * s, "linear") - 0.8) < 1e-9
    assert abs(extrapolate(s, 0.8 - 0.05 * s + 0.004 * s ** 2, "richardson") - 0.8) < 1e-9
    assert abs(extrapolate(s, 0.8 * np.exp(-0.1 * s), "exp") - 0.8) < 1e-9
    assert extrapolate(s, [0.5, 0.4, 0.3], "none") == 0.5
    assert extrapolate([1, 2], [0.9, -0.5], "linear") == 1.0   # clipped to the physical range


def test_holm_and_bootstrap():
    assert holm([0.01, 0.04, 0.03]).tolist() == [True, False, False]
    assert holm([0.001, 0.02, 0.04]).tolist() == [True, True, True]
    backends = np.array(["a", "a", "a", "b", "b"])
    W = bootstrap_weights(backends, B=200, seed=0)
    assert W.shape == (200, 5)
    assert np.all(W[:, :3].sum(1) == 3) and np.all(W[:, 3:].sum(1) == 2)   # resampling within backend
    assert abs(per_series_mae(np.array([2.0, 6.0]), np.array([2, 3])) - 1.5) < 1e-12


def test_noise_model_matches_reported_gate_errors():
    """Modelled gate infidelity equals the reported error (relaxation + depolarising fill)."""
    from src.qem.device import gate_infidelities
    inf = gate_infidelities(STATE, "ibm_fez")
    for q in range(4):
        assert abs(inf[("sx", (q,))] - STATE["sx"][q]) / STATE["sx"][q] < 0.01
    for i in range(3):
        assert abs(inf[("cz", (i, i + 1))] - STATE["cz"][i]) / STATE["cz"][i] < 0.01


def test_n_qubit_code_equals_4_qubit_code():
    """The 6-qubit robustness check relies on src.qem.nq reducing to src.qem.device at n = 4."""
    from qiskit.quantum_info import average_gate_fidelity
    from src.qem.benchmarks import benchmark
    from src.qem.device import noise_model
    from src.qem.nq import benchmark_n, noise_model_n
    a, b = noise_model(STATE, "ibm_kingston", 0.03), noise_model_n(STATE, "ibm_kingston", 0.03)
    for name, qs in [("sx", (0,)), ("cz", (1, 2))]:
        fa = average_gate_fidelity(a._local_quantum_errors[name][qs])
        fb = average_gate_fidelity(b._local_quantum_errors[name][qs])
        assert abs(fa - fb) < 1e-12
    x, y = benchmark("tfim", 3), benchmark_n("tfim", 3, n=4)
    assert x["name"] == y["name"] and np.allclose(x["ideal"], y["ideal"])


def test_action_space_and_shot_budget():
    from src.qem.mitigation import ACTIONS, PROBE_FRACTION, SHOTS
    from src.qem.policies import SHOTS_BANDIT_MAIN, SHOTS_BANDIT_PROBE, controller_choice
    assert len(ACTIONS) == 14
    assert SHOTS == 8192 and SHOTS_BANDIT_MAIN + SHOTS_BANDIT_PROBE == SHOTS
    assert SHOTS_BANDIT_PROBE == int(SHOTS * PROBE_FRACTION)
    assert controller_choice(np.array([[0.3, 0.1, 0.2], [0.0, 0.5, 0.4]])).tolist() == [1, 0]


def test_cache_digest_ignores_row_order(tmp_path):
    from src.evaluation.cache_digest import digest
    rng = np.random.default_rng(0)
    idx = pd.DataFrame(dict(backend=["f"] * 4, chain=[0, 0, 1, 1], epoch=[1, 2, 1, 2], circuit=["c"] * 4,
                            quantity=["q"] * 4, delta=[0.0] * 4, row=range(4)))
    arr = rng.random((4, 3)).astype(np.float32)
    for d, perm in [("a", [0, 1, 2, 3]), ("b", [2, 0, 3, 1])]:
        p = tmp_path / d
        p.mkdir()
        sub = idx.iloc[perm].assign(row=range(4))
        sub.to_parquet(p / "sim_index.parquet")
        np.save(p / "sim_arrays.npy", arr[perm])
    assert digest(str(tmp_path / "a")) == digest(str(tmp_path / "b"))
    np.save(tmp_path / "b" / "sim_arrays.npy", arr[[2, 0, 3, 1]] + 1e-6)
    assert digest(str(tmp_path / "a")) != digest(str(tmp_path / "b"))
