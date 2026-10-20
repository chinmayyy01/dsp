"""Objective quality metrics used by the benchmark and the web UI."""

import numpy as np


def rms(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    return float(np.sqrt(np.mean(x ** 2) + 1e-20))


def snr_db(reference: np.ndarray, estimate: np.ndarray) -> float:
    """SNR of ``estimate`` treated as an estimate of ``reference``."""
    reference = np.asarray(reference, dtype=np.float64)
    estimate = np.asarray(estimate, dtype=np.float64)
    n = min(len(reference), len(estimate))
    reference, estimate = reference[:n], estimate[:n]
    return 10.0 * np.log10(np.sum(reference ** 2) / (np.sum((reference - estimate) ** 2) + 1e-20) + 1e-20)


def si_sdr(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Scale-invariant SDR in dB.

    alpha = <s, s_hat> / <s, s>, target = alpha s, residual = s_hat - target.
    """
    s = np.asarray(reference, dtype=np.float64)
    s_hat = np.asarray(estimate, dtype=np.float64)
    n = min(len(s), len(s_hat))
    s, s_hat = s[:n] - s[:n].mean(), s_hat[:n] - s_hat[:n].mean()
    alpha = np.dot(s, s_hat) / (np.dot(s, s) + 1e-20)
    target = alpha * s
    residual = s_hat - target
    return float(10.0 * np.log10(np.sum(target ** 2) / (np.sum(residual ** 2) + 1e-20) + 1e-20))


def erle_db(primary: np.ndarray, output: np.ndarray) -> float:
    """Echo/Noise return loss enhancement: 10 log10( sum d^2 / sum e^2 )."""
    d = np.asarray(primary, dtype=np.float64)
    e = np.asarray(output, dtype=np.float64)
    n = min(len(d), len(e))
    return float(10.0 * np.log10(np.sum(d[:n] ** 2) / (np.sum(e[:n] ** 2) + 1e-20) + 1e-20))


def snr_improvement(clean: np.ndarray, noisy: np.ndarray, denoised: np.ndarray) -> dict:
    return {
        "noisy_snr_db": snr_db(clean, noisy),
        "denoised_snr_db": snr_db(clean, denoised),
        "delta_db": snr_db(clean, denoised) - snr_db(clean, noisy),
    }


def segmental_snr(clean: np.ndarray, estimate: np.ndarray, frame: int = 512,
                  hop: int = 256, floor_db: float = -10.0, ceil_db: float = 35.0) -> float:
    """Frame-wise SNR with clipping, averaged over frames (nulls excluded)."""
    clean = np.asarray(clean, dtype=np.float64)
    estimate = np.asarray(estimate, dtype=np.float64)
    n = min(len(clean), len(estimate))
    values = []
    for start in range(0, n - frame + 1, hop):
        s = clean[start:start + frame]
        e = estimate[start:start + frame]
        if np.sum(s ** 2) < 1e-8:
            continue
        val = 10.0 * np.log10(np.sum(s ** 2) / (np.sum((s - e) ** 2) + 1e-20) + 1e-20)
        values.append(np.clip(val, floor_db, ceil_db))
    return float(np.mean(values)) if values else float("nan")
