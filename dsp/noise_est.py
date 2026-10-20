"""Voice activity detection and adaptive noise-PSD tracking.

The denoiser needs to know the noise spectrum so it can compute a gain per
frequency bin.  It cannot know it in advance, so it estimates it from frames
the VAD marks as silence and keeps updating it with an exponential average:

    N_new(k) = alpha * N_old(k) + (1 - alpha) * P_silent(k)

Speech frames never touch the estimate, so the noise profile follows a fan
speeding up or down but is not corrupted by the talker.  If a recording has no
silence at all, the quietest decile of frames is used instead.
"""

import numpy as np

DEFAULT_ALPHA = 0.95
DEFAULT_MARGIN_DB = 6.0


def frame_power(spec: np.ndarray) -> np.ndarray:
    return np.abs(spec) ** 2


def frame_energy_db(power: np.ndarray) -> np.ndarray:
    energy = power.sum(axis=1) + 1e-20
    return 10.0 * np.log10(energy / power.shape[1])


def spectral_flatness(power: np.ndarray) -> np.ndarray:
    """Geometric mean over arithmetic mean, per frame; 1.0 for white noise."""
    p = power + 1e-20
    geo = np.exp(np.mean(np.log(p), axis=1))
    arith = np.mean(p, axis=1)
    return geo / (arith + 1e-20)


def detect_silence(power: np.ndarray, margin_db: float = DEFAULT_MARGIN_DB,
                   min_fraction: float = 0.05) -> np.ndarray:
    """Mark frames whose energy sits close to the recording's noise floor."""
    energy_db = frame_energy_db(power)
    floor = np.percentile(energy_db, 20)
    silence = energy_db < (floor + margin_db)

    if silence.mean() < min_fraction:
        cutoff = np.percentile(energy_db, min_fraction * 100)
        silence = energy_db <= cutoff
    return silence


class NoisePsdTracker:
    """Exponential noise-PSD estimate, updated only on silence frames."""

    def __init__(self, alpha: float = DEFAULT_ALPHA):
        self.alpha = alpha
        self.psd: np.ndarray | None = None
        self.updates = 0

    def update(self, power: np.ndarray, silence: np.ndarray) -> np.ndarray:
        if self.psd is None:
            self.psd = power[silence].mean(axis=0) if silence.any() else power.mean(axis=0)

        for frame, is_silent in zip(power, silence):
            if is_silent:
                self.psd = self.alpha * self.psd + (1.0 - self.alpha) * frame
                self.updates += 1
        return self.psd


def estimate_noise_psd(power: np.ndarray, alpha: float = DEFAULT_ALPHA) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    silence = detect_silence(power)
    tracker = NoisePsdTracker(alpha=alpha)
    psd = tracker.update(power, silence)
    return psd, silence, spectral_flatness(power)
