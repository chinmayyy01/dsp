"""Compare the classical pipeline with RNNoise, a pretrained neural denoiser.

RNNoise is a small GRU trained for speech enhancement; it is the cheapest
credible learned baseline.  It runs at 48 kHz in 10 ms frames, so the clip is
resampled up, processed, resampled back and delay-compensated before scoring.

Optional dependency (not in requirements.txt, not deployed):

    uv pip install pyrnnoise

Run:  .venv/bin/python benchmarks/neural_baseline.py
"""

import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import correlate, resample_poly

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dsp import metrics, synth  # noqa: E402
from dsp.pipeline import denoise  # noqa: E402

NOISES = ("white", "fan", "babble")
SNR_DB = 5.0
SECONDS = 6.0
RATE = 16000


def load_clean() -> np.ndarray:
    path = ROOT / "data" / "clean" / "speech.wav"
    if path.exists():
        audio, rate = sf.read(path)
        audio = audio[: int(SECONDS * rate)]
        if rate != RATE:
            factor = rate // RATE
            audio = resample_poly(audio, 1, factor)
        return audio
    return synth.speech_like(SECONDS, RATE, seed=1)


def denoise_with_rnnoise(noisy: np.ndarray, rate: int = RATE) -> np.ndarray:
    from pyrnnoise import RNNoise

    up = resample_poly(noisy, 3, 1).astype(np.float32)
    model = RNNoise(sample_rate=48_000)
    frames = [frame for _, frame in model.denoise_chunk(up, partial=False)]
    if not frames:
        raise RuntimeError("RNNoise returned no frames")

    output = np.concatenate([np.atleast_1d(f).reshape(-1) for f in frames]).astype(np.float64)
    output /= np.max(np.abs(output)) + 1e-12

    down = resample_poly(output, 1, 3)
    n = min(len(down), len(noisy))
    correlation = correlate(down[:n], noisy[:n], mode="full")
    lag = int(np.argmax(np.abs(correlation)) - (n - 1))
    down = np.roll(down, -lag)
    if lag > 0:
        down[-lag:] = 0
    return down[: len(noisy)]


def main() -> None:
    try:
        import pyrnnoise  # noqa: F401
    except ImportError:
        raise SystemExit("pyrnnoise is not installed. Run: uv pip install pyrnnoise")

    clean = load_clean()
    rows = []
    for noise_type in NOISES:
        noise = synth.NOISE_BANK[noise_type](len(clean), RATE, seed=7)
        noisy = synth.mix_at_snr(clean, noise, SNR_DB)
        ours = denoise(noisy, RATE)["audio"]
        neural = denoise_with_rnnoise(noisy)

        for label, signal in (("noisy", noisy), ("classical", ours), ("rnnoise", neural)):
            rows.append({
                "noise": noise_type,
                "method": label,
                "si_sdr": metrics.si_sdr(clean, signal),
                "pesq": metrics.pesq_score(clean, signal, RATE),
                "stoi": metrics.stoi_score(clean, signal, RATE),
            })

    print(f"\n{SNR_DB:+.0f} dB input, {SECONDS:.0f} s speech, 16 kHz\n")
    print(f"{'noise':8s} {'method':10s} {'SI-SDR':>7s} {'PESQ':>6s} {'STOI':>6s}")
    for row in rows:
        print(f"{row['noise']:8s} {row['method']:10s} {row['si_sdr']:7.2f} "
              f"{row['pesq']:6.2f} {row['stoi']:6.3f}")

    lines = [
        "# Classical adaptive pipeline vs RNNoise",
        "",
        f"{SNR_DB:+.0f} dB input SNR, {SECONDS:.0f} s of speech, 16 kHz. RNNoise runs at",
        "48 kHz and is delay-compensated before scoring.",
        "",
        "| noise | method | SI-SDR | PESQ | STOI |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(f"| {row['noise']} | {row['method']} | {row['si_sdr']:.2f} "
                     f"| {row['pesq']:.2f} | {row['stoi']:.3f} |")
    lines.append("")
    (ROOT / "results" / "summary-neural.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
