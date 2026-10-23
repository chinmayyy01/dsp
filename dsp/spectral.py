"""Adaptive spectral gain (decision-directed Wiener filter).

For every STFT bin k and frame t the a posteriori SNR is

    gamma(k,t) = |X(k,t)|^2 / N(k,t)

with N the tracked noise PSD.  The a priori SNR is smoothed over time using the
previous frame's clean estimate (decision-directed, Ephraim-Malah):

    xi(k,t) = dd * |G(k,t-1) X(k,t-1)|^2 / N(k,t) + (1 - dd) * max(gamma - 1, 0)

The Wiener gain is g = xi / (1 + xi), floored so it never reaches zero (a hard
zero produces "musical noise"), and smoothed across frames for the same reason.
The noisy phase is kept, so no phase reconstruction is needed.
"""

import numpy as np

DEFAULT_GAIN_FLOOR = 0.1
DEFAULT_SMOOTH = 0.0
DEFAULT_DD = 0.9


def adaptive_wiener_gain(power: np.ndarray, noise_psd: np.ndarray,
                         gain_floor: float = DEFAULT_GAIN_FLOOR,
                         smooth: float = DEFAULT_SMOOTH,
                         dd: float = DEFAULT_DD) -> np.ndarray:
    """Per-bin gain matrix, shape (n_frames, n_bins), values in [floor, 1]."""
    noise = np.maximum(noise_psd, 1e-20)
    gamma = power / noise

    gains = np.ones_like(power)
    previous_clean = np.zeros(power.shape[1])
    previous_gain = np.ones(power.shape[1])

    for t in range(power.shape[0]):
        frame_noise = noise[t] if noise.ndim > 1 else noise
        prior = dd * previous_clean / frame_noise
        prior += (1.0 - dd) * np.maximum(gamma[t] - 1.0, 0.0)
        gain = np.clip(prior / (1.0 + prior), gain_floor, 1.0)
        if t > 0 and smooth > 0:
            gain = np.clip(smooth * previous_gain + (1.0 - smooth) * gain, gain_floor, 1.0)

        gains[t] = gain
        previous_clean = gain ** 2 * power[t]
        previous_gain = gain

    return gains


def apply_gain(spec: np.ndarray, gains: np.ndarray) -> np.ndarray:
    return spec * gains


def spectral_subtraction(power: np.ndarray, noise_psd: np.ndarray,
                         over_subtraction: float = 1.5,
                         floor: float = 0.02) -> np.ndarray:
    """Classic magnitude spectral subtraction, kept for the comparison section.

    g = max(1 - alpha * N / P, floor)
    """
    return np.clip(1.0 - over_subtraction * noise_psd / (power + 1e-20), floor, 1.0)
