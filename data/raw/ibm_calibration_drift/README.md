---
license: cc-by-4.0
task_categories:
- time-series-forecasting
- tabular-classification
tags:
- quantum-computing
- IBM-Quantum
- calibration
- hardware-characterization
- drift-analysis
- qiskit
- space-weather
size_categories:
- 1M<n<10M
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
---

# Qiskit Calibration Drift

Calibration parameters from IBM Quantum Heron processors joined to ambient and space-weather conditions at the time of each measurement. Designed for time-series forecasting of qubit drift and for studying environmental coupling to superconducting calibrations.

A GitHub Action ([source](https://github.com/CharlesCNorton/qiskit-calibration-drift)) polls `backend.properties()` on every available Heron device every 30 minutes and appends new calibration events keyed on `(backend, property, qubit_a, qubit_b, calibrated_time)`.

## Backends

| Backend | Processor | Qubits | Active |
|---|---|---|---|
| `ibm_fez` | Heron r2 | 156 | yes |
| `ibm_kingston` | Heron r2 | 156 | yes |
| `ibm_marrakesh` | Heron r2 | 156 | yes |
| `ibm_torino` | Heron r1 | 133 | retired 2026-04-27 |

All backends are at IBM's Yorktown Heights, NY facility (`lat=41.27, lon=-73.78`).

## Schema

| Column | Type | Description |
|---|---|---|
| `backend` | str | backend name |
| `property_family` | str | normalized property group (e.g. `sx_error`, `cz_error`, `T1`, `readout_error`) |
| `property` | str | IBM API canonical name (e.g. `sx_gate_error`, `cz_gate_length`) |
| `qubit_a` | Int16 | qubit index, or first qubit of a coupled pair |
| `qubit_b` | Int16 | second qubit of a coupled pair; NA for single-qubit and qubit-level properties |
| `value` | float64 | the calibration value |
| `unit` | str | unit string from the IBM API (`us`, `ns`, empty for dimensionless) |
| `scope` | str | one of `qubit`, `gate1q`, `gate2q` |
| `is_failure_ceiling` | bool | true when `value ≥ 0.999` for an error or probability property — the calibration saturated at the ceiling and the underlying measurement is unreliable |
| `is_new_measurement` | bool | true when `value` differs from the prior row's value for the same `(backend, property, qubit_a, qubit_b)`; filter on this to drop heartbeat replicas |
| `observed_time` | timestamp UTC | snapshot pull time |
| `calibrated_time` | timestamp UTC | IBM's published freshness timestamp for this property |
| `snapshot_update_time` | timestamp UTC | the `last_update_date` of the BackendProperties at the pull moment |
| `calibration_age_seconds` | float64 | `observed_time − calibrated_time` |
| `chipwide_recal_event_id` | str | identifier shared across rows that participate in a chip-wide recalibration event (≥20% of qubits updated simultaneously); NA otherwise |
| `latitude`, `longitude` | float64 | IBM facility coordinate |
| `solar_zenith_deg` | float64 | solar zenith angle at the site (>90 means night) |
| `temperature_c`, `pressure_hpa`, `humidity_pct` | float64 | ambient weather |
| `bz_gsm_nt` | float64 | interplanetary magnetic field Bz in GSM coordinates, nT |
| `neutron_flux` | float64 | cosmic-ray neutron monitor count rate |
| `kp_index` | float64 | planetary K index (3-hour), 0..9 in 1/3 steps |
| `ap_index` | float64 | planetary equivalent-amplitude (3-hour) |
| `Ap_daily` | float64 | daily mean ap |
| `SN` | float64 | International Sunspot Number v2.0 (daily) |
| `f107_observed_sfu` | float64 | observed F10.7 cm solar radio flux, s.f.u. |
| `f107_adjusted_sfu` | float64 | distance-adjusted F10.7 |
| `solar_flux_sfu` | float64 | NOAA SWPC 10cm flux summary |
| `dst_nt` | float64 | hourly Dst index, nT |

## Property families

**Coherence.** `T1`, `T2` per qubit, in microseconds.

**Single-qubit gate errors.** `sx_error`, `x_error`, `id_error`, `rx_error`, `rz_error`. Reported separately by the IBM API; on a given measurement the four basis gates of the same qubit are typically bit-identical because IBM derives them from one calibration.

**Slow single-qubit gate.** `xslow_error` and its `_length`, on `ibm_fez` and `ibm_kingston`.

**Two-qubit gates.** `cz_error` (controlled-Z) and `rzz_error` (parametric ZZ rotation), stored once per ordered directional pair `(qubit_a, qubit_b)`. For an undirected pair `{a, b}` both `(a, b)` and `(b, a)` rows appear and carry different values.

**Readout.** `readout_error`, `prob_meas0_prep1`, `prob_meas1_prep0`, `readout_length`, `measure_gate_error`, `measure_gate_length`. Readout-error values are quantized to multiples of 1/8192. `measure_2_error` is an alternative readout scheme present on `ibm_kingston` only.

**Durations.** `sx_length`, `x_length`, `id_length`, `rx_length`, `rz_length`, `cz_length`, `rzz_length`, `readout_length`, `reset_length`, `measure_length`. Mostly constant during the observation window.

## Working with the data

```python
from datasets import load_dataset
ds = load_dataset("phanerozoic/qiskit-calibration-drift", split="train")
```

For drift analysis, filter to `is_new_measurement == True` to drop heartbeat replicas. For honest aggregations, also exclude rows where `is_failure_ceiling == True`. Property families are calibrated on independent schedules — `cz_error` and the single-qubit basis gates cycle together on a chip-wide cadence (look for `chipwide_recal_event_id`), `readout_error` refreshes more often, `T1` / `T2` less often, so the `calibrated_time` for two different families on the same qubit will generally not coincide.

## Sources

- Calibration data: IBM Quantum `backend.properties()` via Qiskit Runtime.
- Kp (definitive), ap, Ap, F10.7, SN: GFZ Helmholtz Centre for Geosciences (Matzka et al. 2021, *Space Weather*, doi:10.1029/2020SW002641); SN from WDC-SILSO, Royal Observatory of Belgium.
- Kp (realtime fallback), Bz, Dst, 10cm flux summary: NOAA SWPC.
- Solar wind Bz: DSCOVR L2 magnetometer.
- Cosmic-ray neutron monitor: NMDB Newark station.
- Ambient weather: `api.weather.gov` (KHPN, Westchester County).

## License

CC-BY-4.0, except for the Sunspot Number column (`SN`) which is CC-BY-NC-4.0 per WDC-SILSO licensing.
