"""
Single entry point for reproducing the study (system Python 3.14, see requirements.txt).

    python3 reproduce.py --list              # show the steps
    python3 reproduce.py data forecast       # run selected stages
    python3 reproduce.py all                 # everything (several hours on a 14-core laptop)

Test-set evaluations are guarded: they only run if the frozen inputs are unchanged
(configs/frozen_stage3.yaml, frozen_stage5.yaml, frozen_robustness.yaml).
Simulation caches are deterministic (seeded); figures are regenerated from saved results.
"""

import subprocess
import sys
import time

PY = sys.executable
STEPS = {
    "data": [  # Stage 2 (~1 min). Needs the raw parquet: python3 scripts/download_ibm_data.py
        "-m src.data.drift_dataset",
    ],
    "forecast": [  # Stage 3 (~45 min): per-qubit multi-horizon forecasting
        "-m experiments.stage3_forecasting tune",
        "-m experiments.stage3_forecasting final",
        "-m experiments.stage3_forecasting evaluate --sets val",
        "-m experiments.stage3_forecasting evaluate --sets test",
        "-m experiments.plot_stage3 --which val",
        "-m experiments.plot_stage3 --which test",
    ],
    "pair_forecast": [  # Stage 5a (~10 min): pair-level CZ forecasting
        "-c \"from src.forecasting.pairs import build_pair_table as b; b().to_parquet('data/processed/drift_v3/pair_epochs.parquet')\"",
        "-m experiments.stage5_pair_forecast run",
        "-m experiments.stage5_pair_forecast evaluate --sets val",
    ],
    "qem": [  # Stage 5 (~75 min): decisions, simulation cache, policies, hypotheses, figures
        "-m experiments.stage5_decisions",
        "-m experiments.stage5_simulate --workers 12 --history 60",
        "-m experiments.stage5_policies --split val",
        # test evaluations check the Stage 5 freeze, which covers the files created above
        "-m experiments.stage5_pair_forecast evaluate --sets test",
        "-m experiments.stage5_policies --split test",
        "-m experiments.stage5_stationary",
        "-m experiments.plot_stage5",
    ],
    "robustness": [  # pre-registered robustness (~80 min)
        "-m experiments.robust_check",
        "-m experiments.robust_decisions_6q",
        "-m experiments.robust_sim rb",
        "-m experiments.robust_sim ra",
        "-m experiments.robust_policies rb --split val",
        "-m experiments.robust_policies ra --split val",
        "-m experiments.robust_policies ra --split test",
        "-m experiments.robust_policies rb --split test",
    ],
}
STEPS["all"] = [s for k in ["data", "forecast", "pair_forecast", "qem", "robustness"] for s in STEPS[k]]
STEPS["resume_from_qem"] = [s for k in ["qem", "robustness"] for s in STEPS[k]]


def main(args):
    if not args or "--list" in args:
        for k, v in STEPS.items():
            if k != "all":
                print(f"[{k}]")
                for s in v:
                    print(f"   python3 {s}")
        return
    for stage in args:
        for s in STEPS[stage]:
            print(f"\n>>> python3 {s}", flush=True)
            t = time.time()
            r = subprocess.run(f"{PY} -B {s}", shell=True)
            if r.returncode:
                sys.exit(f"step failed ({r.returncode}): {s}")
            print(f"    done in {time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
