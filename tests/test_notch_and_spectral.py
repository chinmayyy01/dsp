import numpy as np
import pytest

from dsp import notch, synth
from dsp.metrics import si_sdr
from dsp.pipeline import DenoiseConfig, denoise


def tone_energy(x, freq, fs=16000, bandwidth=3.0):
    spec = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(len(x), 1 / fs)
    return spec[np.abs(freqs - freq) <= bandwidth].sum()


def test_notch_removes_mains_and_keeps_speech_band():
    fs, n = 16000, 16000
    t = np.arange(n) / fs
    x = 0.5 * np.sin(2 * np.pi * 50 * t) + 0.4 * np.sin(2 * np.pi * 1000 * t)

    out = notch.adaptive_notch_harmonics(x, 50.0, fs, n_harmonics=3, mu=0.01)
    y = out["output"]

    ratio_50 = tone_energy(y, 50.0) / (tone_energy(x, 50.0) + 1e-20)
    ratio_1k = tone_energy(y, 1000.0) / (tone_energy(x, 1000.0) + 1e-20)
    assert 10 * np.log10(ratio_50 + 1e-20) < -20
    assert 10 * np.log10(ratio_1k + 1e-20) > -1


def test_tonal_detection():
    fs, n = 16000, 16000
    t = np.arange(n) / fs
    hum = 0.5 * np.sin(2 * np.pi * 50 * t) + 0.1 * np.random.default_rng(0).standard_normal(n)
    assert notch.is_tonal(hum, 50.0, fs)
    assert not notch.is_tonal(np.random.default_rng(0).standard_normal(n), 50.0, fs)


def test_ale_prediction_tracks_tone():
    fs, n = 16000, 16000
    t = np.arange(n) / fs
    tone = np.sin(2 * np.pi * 300 * t)
    x = tone + 0.5 * np.random.default_rng(0).standard_normal(n)

    out = notch.adaptive_line_enhancer(x, delay=8, order=64, mu=0.5)
    corr = np.corrcoef(out["prediction"], tone)[0, 1]
    assert corr > 0.85
    assert si_sdr(tone, out["prediction"]) > si_sdr(tone, x)


@pytest.mark.parametrize("noise_type", ["white", "pink", "hum"])
def test_pipeline_improves_si_sdr(noise_type):
    mix = synth.make_mixture(noise_type=noise_type, snr_db=5.0, duration=6.0, seed=1)
    result = denoise(mix["noisy"], fs=mix["fs"], config=DenoiseConfig(use_notch=False))

    before = si_sdr(mix["clean"], mix["noisy"])
    after = si_sdr(mix["clean"], result["audio"])
    assert after > before + 1.0


def test_pipeline_handles_silent_input():
    x = np.zeros(16000)
    result = denoise(x, fs=16000)
    assert result["audio"].shape == x.shape
    assert np.all(np.isfinite(result["audio"]))
