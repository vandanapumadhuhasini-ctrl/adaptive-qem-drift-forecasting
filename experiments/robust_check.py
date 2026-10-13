"""Equivalence check: src/qem/nq.py reproduces the frozen 4-qubit objects exactly."""
import numpy as np

from src.qem.benchmarks import all_benchmarks
from src.qem.device import noise_model
from src.qem.mitigation import circuit_set, exact_values
from src.qem.nq import benchmark_n, noise_model_n

st = dict(t1=np.array([170, 150, 200, 120.]), t2=np.array([110, 90, 150, 100.]),
          sx=np.array([3e-4, 2.5e-4, 4e-4, 3e-4]), cz=np.array([2.7e-3, 3e-3, 2.5e-3]))
for b in all_benchmarks():
    fam, depth = b["name"].split("_d")
    m = benchmark_n(fam, int(depth), 4, 11)
    assert m["name"] == b["name"] and m["observables"] == b["observables"]
    assert m["native"] == b["native"], b["name"]
    assert np.allclose(m["ideal"], b["ideal"]) and np.allclose(m["probe_ideal"], b["probe_ideal"])
    c, _ = circuit_set(b["native"], b["observables"])
    for d in (0.0, 0.03):
        v1 = exact_values(c, noise_model(st, "ibm_fez", d), b["observables"])
        v2 = exact_values(c, noise_model_n(st, "ibm_fez", d), b["observables"])
        assert np.array_equal(v1, v2), b["name"]
print("nq.py reproduces all 6 frozen 4-qubit benchmarks and the noise model exactly")
