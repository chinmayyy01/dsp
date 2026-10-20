"""Synthetic test material, so the benchmark runs without external audio.

The "speech" signal is a harmonic stack with vibrato, shaped by three formant
resonators and gated by a syllable envelope with real pauses.  Pauses matter:
the noise-PSD tracker only learns during silence.
"""

import numpy as np
from scipy.signal import lfilter

DEFAULT_FS = 16000


def speech_like(duration: float = 6.0, fs: int = DEFAULT_FS, f0: float = 120.0,
                seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(duration * fs)
    t = np.arange(n) / fs

    vibrato = 1.0 + 0.02 * np.sin(2 * np.pi * 5.0 * t)
    phase = 2 * np.pi * f0 * np.cumsum(vibrato) / fs
    harmonics = sum(np.sin(h * phase) / h for h in range(1, 24))

    formants = [(600, 80, 1.0), (1100, 120, 0.6), (2600, 200, 0.35)]
    voiced = np.zeros(n)
    for freq, bandwidth, weight in formants:
        r = np.exp(-np.pi * bandwidth / fs)
        b, a = [1 - r], [1, -2 * r * np.cos(2 * np.pi * freq / fs), r * r]
        voiced += weight * lfilter(b, a, harmonics)
    voiced /= np.max(np.abs(voiced)) + 1e-20

    envelope = np.zeros(n)
    pos = 0
    while pos < n:
        speech_len = int(rng.uniform(0.18, 0.45) * fs)
        gap_len = int(rng.uniform(0.12, 0.35) * fs)
        end = min(n, pos + speech_len)
        envelope[pos:end] = np.hanning(max(2, end - pos)) ** 0.5
        pos = end + gap_len

    fricative = 0.06 * rng.standard_normal(n) * (envelope > 0)
    signal = voiced * envelope + fricative
    return signal / (np.max(np.abs(signal)) + 1e-20) * 0.8


def white_noise(n: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(n)


def pink_noise(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    white = rng.standard_normal(n)
    spectrum = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n)
    freqs[0] = freqs[1]
    spectrum /= np.sqrt(freqs)
    out = np.fft.irfft(spectrum, n=n)
    return out / (np.std(out) + 1e-20)


def brown_noise(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.cumsum(rng.standard_normal(n))
    out = lfilter([1, -0.995], [1], out)
    return out / (np.std(out) + 1e-20)


def hum_noise(n: int, fs: int = DEFAULT_FS, f0: float = 50.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(n) / fs
    out = sum((1.0 / k) * np.sin(2 * np.pi * k * f0 * t + rng.uniform(0, 2 * np.pi))
              for k in range(1, 7))
    return out / (np.std(out) + 1e-20)


def fan_noise(n: int, fs: int = DEFAULT_FS, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(n) / fs
    broadband = lfilter([1], [1, -0.7], rng.standard_normal(n))
    tone = 0.4 * np.sin(2 * np.pi * 180 * t)
    out = broadband / (np.std(broadband) + 1e-20) + tone
    return out / (np.std(out) + 1e-20)


def babble_noise(n: int, fs: int = DEFAULT_FS, voices: int = 8, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.zeros(n)
    for v in range(voices):
        f0 = rng.uniform(90, 220)
        t = np.arange(n) / fs
        env = lfilter([1], [1, -0.999], np.abs(rng.standard_normal(n))) + 0.05
        out += env * sum(np.sin(2 * np.pi * h * f0 * t + rng.uniform(0, 6.28)) / h
                         for h in range(1, 10))
    return out / (np.std(out) + 1e-20)


NOISE_BANK = {
    "white": lambda n, fs, seed: white_noise(n, seed),
    "pink": lambda n, fs, seed: pink_noise(n, seed),
    "brown": lambda n, fs, seed: brown_noise(n, seed),
    "hum": lambda n, fs, seed: hum_noise(n, fs, 50.0, seed),
    "fan": lambda n, fs, seed: fan_noise(n, fs, seed),
    "babble": lambda n, fs, seed: babble_noise(n, fs, 8, seed),
}


def mix_at_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    clean_rms = np.sqrt(np.mean(clean ** 2) + 1e-20)
    noise_rms = np.sqrt(np.mean(noise ** 2) + 1e-20)
    scaled = noise / noise_rms * clean_rms * 10 ** (-snr_db / 20.0)
    return clean + scaled


def make_mixture(noise_type: str = "white", snr_db: float = 5.0,
                 duration: float = 6.0, fs: int = DEFAULT_FS, seed: int = 0) -> dict:
    clean = speech_like(duration=duration, fs=fs, seed=seed)
    noise = NOISE_BANK[noise_type](len(clean), fs, seed)
    noisy = mix_at_snr(clean, noise, snr_db)
    return {"clean": clean, "noise": noise, "noisy": noisy,
            "noise_type": noise_type, "snr_db": snr_db, "fs": fs}
