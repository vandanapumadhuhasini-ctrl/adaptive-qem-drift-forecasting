"""Shared Matplotlib style for publication figures (300-dpi PNG + vector PDF)."""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

# Colour follows the model, never its rank (categorical slots in fixed order)
MODEL_COLORS = {
    "transformer": "#2a78d6",
    "persistence": "#eb6834",
    "window_mean": "#1baf7a",
    "rolling5": "#eda100",
    "ridge": "#e87ba4",
    "hgb": "#008300",
    "lstm": "#4a3aa7",
    "tf_targets_only": "#e34948",
    "tf_state_env": "#e87ba4",
    "tf_no_positional": "#eda100",
}
MODEL_LABELS = {
    "transformer": "Transformer",
    "persistence": "Persistence",
    "window_mean": "Window mean (20)",
    "rolling5": "Rolling mean (5)",
    "ridge": "Ridge (lags)",
    "hgb": "Gradient-boosted trees",
    "lstm": "LSTM",
    "tf_targets_only": "TF: target history only",
    "tf_state_env": "TF: + environment",
    "tf_no_positional": "TF: no positional enc.",
}
TARGET_LABELS = {
    "log_t1": "T1",
    "log_t2": "T2",
    "log_sx_err": "1q (sx) error",
    "log_ro_err": "Readout error",
    "log_cz_err_mean": "CZ error (mean)",
}
SET_LABELS = {
    "val": "Validation",
    "test_primary": "Test: development qubits",
    "test_unseen_qubit": "Test: unseen qubits",
    "test_unseen_device": "Test: unseen device",
}


def setup():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 11,
        "axes.labelsize": 10, "legend.fontsize": 8.5, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "axes.edgecolor": INK_2, "axes.labelcolor": INK, "xtick.color": INK_2, "ytick.color": INK_2,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "lines.linewidth": 1.8, "lines.markersize": 5, "savefig.bbox": "tight",
        "pdf.fonttype": 42,
    })


def save(fig, path_no_ext):
    os.makedirs(os.path.dirname(path_no_ext), exist_ok=True)
    fig.savefig(path_no_ext + ".pdf")
    fig.savefig(path_no_ext + ".png", dpi=300)
    plt.close(fig)
