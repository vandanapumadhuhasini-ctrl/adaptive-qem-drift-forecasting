"""
T1-driven noise model for the simulated device (Qiskit Aer).

Assumptions (the calibration sample only contains T1, so these are modelling
choices, not measured values):
  - Every CX carries thermal relaxation on both qubits for an effective duration
    of CX_TIME_US. This is longer than a physical gate and stands in for the gate
    plus surrounding idle time, so that T1 drift visibly changes the result.
  - T2 = T1 (inside the physical bound T2 <= 2*T1).
  - A stationary coherent ZZ over-rotation of COHERENT_ZZ_RAD on every CX
    (the error type Pauli twirling is designed to tame).
  - Single-qubit gates and readout are noiseless.
"""

from qiskit.circuit.library import RZZGate
from qiskit_aer.noise import NoiseModel, coherent_unitary_error, thermal_relaxation_error

CX_TIME_US = 2.0
COHERENT_ZZ_RAD = 0.06

# Two pre-declared regimes, reported side by side: the sample data says nothing
# about coherent error, so the conclusions are tested under both.
NOISE_REGIMES = {
    "incoherent": 0.06,   # T1 relaxation dominates
    "coherent": 0.20,     # strong stationary ZZ over-rotation
}


def cx_error(t1_us, cx_time_us=CX_TIME_US, zz_rad=COHERENT_ZZ_RAD):
    relax = thermal_relaxation_error(t1_us, t1_us, cx_time_us)
    coherent = coherent_unitary_error(RZZGate(zz_rad).to_matrix())
    return coherent.compose(relax.tensor(relax))


def noise_model(t1_us, **kwargs):
    nm = NoiseModel(basis_gates=["cx", "rx", "ry", "rz", "x", "y", "z", "id"])
    nm.add_all_qubit_quantum_error(cx_error(t1_us, **kwargs), ["cx"])
    return nm
