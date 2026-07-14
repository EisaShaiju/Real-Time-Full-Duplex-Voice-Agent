"""Add noise to a clean signal at a target SNR (dB).

SNR is computed from the actual signal power, so a given dB level means the same
thing across clips of different loudness. Supports white Gaussian noise or a
supplied background-noise track (tiled/cropped to length).
"""

import numpy as np


def add_noise(clean, snr_db, kind="gaussian", noise=None, rng=None):
    rng = rng or np.random.default_rng(0)
    clean = np.asarray(clean, dtype="float64")
    sig_power = float(np.mean(clean ** 2)) + 1e-12

    if kind == "gaussian" or noise is None:
        n = rng.standard_normal(len(clean))
    else:
        n = np.asarray(noise, dtype="float64")
        if len(n) < len(clean):
            n = np.tile(n, int(np.ceil(len(clean) / len(n))))
        n = n[: len(clean)]

    noise_power = float(np.mean(n ** 2)) + 1e-12
    target_power = sig_power / (10 ** (snr_db / 10))
    n *= np.sqrt(target_power / noise_power)

    # Clip (not peak-normalize) so the target SNR is preserved: rescaling the
    # whole mix would change the signal-to-noise ratio relative to the clean ref.
    out = np.clip(clean + n, -1.0, 1.0)
    return out.astype("float32")
