"""
Collect the figures used by report/QEM_Project_Report.tex into report/figures/ and draw the
six benchmark circuits (native gate set, after transpilation) with Qiskit's Matplotlib drawer.

    python3 scripts/make_report_figures.py

Writes PDF (vector) and PNG (300 dpi) versions of every figure. Needs pylatexenc for Qiskit's
"mpl" drawer.
"""

import os
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.qem.benchmarks import all_benchmarks

OUT = "report/figures"
RESULT_FIGURES = [
    "results/stage2/data_coverage",
    "results/stage3/figures/test/mae_vs_horizon_test_primary",
    "results/stage3/figures/test/skill_vs_persistence_test_primary",
    "results/stage3/figures/test/examples_log_t1_k1_test_primary",
    "results/stage3/figures/test/generalisation_test",
    "results/stage5/figures/pair_forecast_skill_test",
    "results/stage5/figures/hypotheses_test",
    "results/stage5/figures/error_by_horizon_test",
    "results/stage5/figures/probe_reward_quality",
    "results/stage4/figures/val/value_of_information",
]


def main():
    os.makedirs(OUT, exist_ok=True)
    for f in RESULT_FIGURES:
        for ext in ("pdf", "png"):
            shutil.copy(f"{f}.{ext}", f"{OUT}/{os.path.basename(f)}.{ext}")
    for b in all_benchmarks():
        fig = b["native"].draw("mpl", fold=40, idle_wires=False)
        fig.suptitle(f"{b['name']}: 4 qubits, {b['n_cz']} CZ gates (native gate set CZ, sx, rz, x)", fontsize=11)
        for ext, kw in (("pdf", {}), ("png", {"dpi": 300})):
            fig.savefig(f"{OUT}/circuit_{b['name']}.{ext}", bbox_inches="tight", **kw)
        plt.close(fig)
    print(f"wrote {len(os.listdir(OUT))} files to {OUT}/")


if __name__ == "__main__":
    main()
