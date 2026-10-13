"""Zero-noise extrapolation fits."""

import numpy as np


def extrapolate(scales, values, method):
    scales = np.asarray(scales, dtype=float)
    values = np.asarray(values, dtype=float)

    if method == "none":
        return float(values[0])
    if method == "linear":
        est = np.polyfit(scales, values, 1)[1]
    elif method == "richardson":
        est = np.polyfit(scales, values, len(scales) - 1)[-1]
    elif method == "exp":
        # E(s) = A exp(-b s): linear fit of log|E|; valid when the decay is to 0
        # (Pauli noise), falls back to linear if the signs are inconsistent.
        sign = np.sign(values[0])
        if np.all(np.sign(values) == sign) and np.all(np.abs(values) > 1e-3):
            est = sign * np.exp(np.polyfit(scales, np.log(np.abs(values)), 1)[1])
        else:
            est = np.polyfit(scales, values, 1)[1]
    else:
        raise ValueError(method)
    return float(np.clip(est, -1.0, 1.0))
