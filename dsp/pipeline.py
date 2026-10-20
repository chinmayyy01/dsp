"""End-to-end single-input denoiser: notch -> ALE -> adaptive spectral gain.

Every stage is optional so the benchmark can ablate them, and the result keeps
the intermediate signals so the web UI can draw what each stage did.
"""

from dataclasses import dataclass, field, asdict

import numpy as np

from . import framing, noise_est, notch, spectral


@dataclass
class DenoiseConfig:
    frame_size: int = 1024
    hop_ratio: float = 0.25
    use_notch: bool = True
    notch_f0: float = 50.0
    notch_harmonics: int = 2
    notch_mu: float = 0.001
    use_ale: bool = False
    ale_delay: int = 8
    ale_order: int = 32
    ale_mu: float = 0.5
    noise_alpha: float = 0.95
    gain_floor: float = 0.1
    gain_smooth: float = 0.0
    dd: float = 0.9

    def hop(self) -> int:
        return max(1, int(self.frame_size * self.hop_ratio))


def _auto_notch(x: np.ndarray, fs: float, config: DenoiseConfig) -> dict:
    """Only notch when a mains tone is actually present, to protect speech."""
    if not config.use_notch:
        return {"output": x, "applied": False, "stages": []}
    if not notch.is_tonal(x, config.notch_f0, fs, bandwidth_hz=4.0):
        return {"output": x, "applied": False, "stages": []}
    result = notch.adaptive_notch_harmonics(
        x, config.notch_f0, fs, n_harmonics=config.notch_harmonics, mu=config.notch_mu)
    return {"output": result["output"], "applied": True, "stages": result["stages"]}


def denoise(x: np.ndarray, fs: int = 16000, config: DenoiseConfig | None = None,
            method: str = "adaptive_wiener") -> dict:
    """Run the full chain.

    ``method`` selects the spectral rule: ``adaptive_wiener`` (default) or
    ``spectral_subtraction`` (baseline used in the report comparison).
    """
    config = config or DenoiseConfig()
    x = np.asarray(x, dtype=np.float64)
    if x.ndim > 1:
        x = x.mean(axis=1)
    x = x - np.mean(x)
    peak = np.max(np.abs(x)) or 1.0
    x_norm = x / peak

    notch_result = _auto_notch(x_norm, fs, config)
    stage1 = notch_result["output"]

    if config.use_ale:
        ale_result = notch.adaptive_line_enhancer(
            stage1, delay=config.ale_delay, order=config.ale_order, mu=config.ale_mu)
        stage2 = ale_result["output"]
        ale_mse = ale_result["mse"]
    else:
        stage2 = stage1
        ale_mse = np.zeros(1)

    hop = config.hop()
    spec = framing.stft(stage2, config.frame_size, hop)
    power = noise_est.frame_power(spec)
    noise_psd, silence, flatness = noise_est.estimate_noise_psd(power, alpha=config.noise_alpha)

    if method == "spectral_subtraction":
        gains = spectral.spectral_subtraction(power, noise_psd)
    else:
        gains = spectral.adaptive_wiener_gain(
            power, noise_psd, gain_floor=config.gain_floor,
            smooth=config.gain_smooth, dd=config.dd)

    cleaned_spec = spectral.apply_gain(spec, gains)
    y = framing.istft(cleaned_spec, config.frame_size, hop, length=len(x_norm))
    y = y * peak

    return {
        "audio": y,
        "fs": fs,
        "stages": {
            "input": x_norm * peak,
            "notch": stage1 * peak,
            "ale": stage2 * peak,
            "spectral": y,
        },
        "notch_applied": notch_result["applied"],
        "spectrogram": {
            "input": framing.spectrogram_db(spec),
            "output": framing.spectrogram_db(cleaned_spec),
        },
        "noise_psd": noise_psd,
        "silence": silence,
        "flatness": flatness,
        "gains": gains,
        "ale_mse": ale_mse,
        "config": asdict(config),
        "method": method,
    }
