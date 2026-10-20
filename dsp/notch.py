"""Adaptive notch filter and adaptive line enhancer.

The notch is a two-tap LMS filter whose reference is a unit-amplitude sinusoid
pair at the frequency to remove.  Adapting the two taps estimates the complex
amplitude of that tone, and subtracting it from the input cancels the tone
without touching the rest of the spectrum.  A cascade handles mains harmonics.

The adaptive line enhancer (ALE) predicts the *periodic* part of a signal from
a delayed copy of itself.  The residual (input minus prediction) contains the
unpredictable part, i.e. broadband noise.
"""

import numpy as np
from scipy.signal import lfilter

from .lms import nlms


def _tone_reference(n: int, freq: float, fs: float) -> np.ndarray:
    t = np.arange(n) / fs
    return np.stack([np.cos(2 * np.pi * freq * t), np.sin(2 * np.pi * freq * t)], axis=1)


def adaptive_notch(x: np.ndarray, freq: float, fs: float, mu: float = 0.01,
                   order: int = 2) -> dict:
    """Cancel a single tone at ``freq`` Hz with a 2-tap LMS filter."""
    x = np.asarray(x, dtype=np.float64)
    ref = _tone_reference(len(x), freq, fs)

    w = np.zeros(2)
    y = np.zeros(len(x))
    e = np.zeros(len(x))

    for n in range(len(x)):
        r = ref[n]
        y[n] = w @ r
        e[n] = x[n] - y[n]
        w = w + mu * e[n] * r

    return {"output": e, "estimated_tone": y, "weights": w}


def adaptive_notch_harmonics(x: np.ndarray, f0: float, fs: float,
                             n_harmonics: int = 4, mu: float = 0.01) -> dict:
    """Cascade of notches at f0, 2*f0, ... n_harmonics*f0."""
    out = np.asarray(x, dtype=np.float64)
    stages = []
    for k in range(1, n_harmonics + 1):
        freq = k * f0
        if freq >= fs / 2:
            break
        stage = adaptive_notch(out, freq, fs, mu=mu)
        out = stage["output"]
        stages.append({"freq": freq, "removed": stage["estimated_tone"]})
    return {"output": out, "stages": stages}


def adaptive_line_enhancer(x: np.ndarray, delay: int = 8, order: int = 32,
                           mu: float = 0.5) -> dict:
    """Remove predictable (tonal) content by predicting it from a delayed copy.

    ``delay`` must exceed the correlation time of the broadband noise, otherwise
    the filter just learns to copy the noise.
    """
    x = np.asarray(x, dtype=np.float64)
    reference = np.concatenate([np.zeros(delay), x[:-delay]]) if delay > 0 else x
    result = nlms(reference, x, mu=mu, order=order)
    return {
        "output": result["output"],
        "prediction": result["prediction"],
        "mse": result["mse"],
    }


def is_tonal(x: np.ndarray, freq: float, fs: float, bandwidth_hz: float = 5.0) -> bool:
    """True when the signal carries more than 10% of its energy in ``freq`` band."""
    n = len(x)
    spec = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    band = np.abs(freqs - freq) <= bandwidth_hz
    total = spec.sum() + 1e-12
    return bool(spec[band].sum() / total > 0.1)


def highpass(x: np.ndarray, fs: float, cutoff: float = 80.0) -> np.ndarray:
    """Simple DC/rumble guard used before the adaptive stages."""
    b, a = _butter_highpass(cutoff, fs)
    return lfilter(b, a, x)


def _butter_highpass(cutoff: float, fs: float, order: int = 2):
    from scipy.signal import butter
    return butter(order, cutoff / (fs / 2), btype="highpass")
