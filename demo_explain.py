"""Walk one recording through every stage and print what each stage did.

Useful in a viva: every number quoted here can be recomputed live.

    .venv/bin/python demo_explain.py
    .venv/bin/python demo_explain.py path/to/noisy.wav --reference path/to/clean.wav
    .venv/bin/python demo_explain.py --snr 0 --noise pink

Sections mirror docs/equations.md.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from dsp import framing, metrics, noise_est, notch, synth
from dsp.lms import lms, nlms, rls
from dsp.pipeline import DenoiseConfig, denoise

ROOT = Path(__file__).resolve().parent


def rule(title: str) -> None:
    print(f"\n{title}")
    print("-" * len(title))


def band_share(audio: np.ndarray, freq: float, fs: int, bandwidth: float = 4.0) -> float:
    power = np.abs(np.fft.rfft(audio)) ** 2
    freqs = np.fft.rfftfreq(len(audio), 1.0 / fs)
    return float(power[np.abs(freqs - freq) <= bandwidth].sum() / (power.sum() + 1e-20))


def erle_curve(desired: np.ndarray, error: np.ndarray, block: int = 256,
               width: int = 31) -> tuple[np.ndarray, int | None]:
    n = min(len(desired), len(error)) // block
    d = desired[:n * block].reshape(n, block)
    e = error[:n * block].reshape(n, block)
    curve = 10.0 * np.log10(np.mean(d ** 2, axis=1) / (np.mean(e ** 2, axis=1) + 1e-20) + 1e-20)
    pad = width // 2
    curve = np.convolve(np.pad(curve, (pad, pad), mode="edge"), np.ones(width) / width, mode="valid")
    converged = next((i for i, v in enumerate(curve) if v >= 20.0), None)
    return curve, converged


def load_material(args) -> tuple[np.ndarray, np.ndarray, int]:
    if args.audio:
        noisy, fs = sf.read(args.audio, dtype="float64")
        if noisy.ndim > 1:
            noisy = noisy.mean(axis=1)
        clean = None
        if args.reference:
            clean, ref_fs = sf.read(args.reference, dtype="float64")
            if ref_fs != fs:
                raise SystemExit("reference must use the same sample rate")
            n = min(len(clean), len(noisy))
            clean, noisy = clean[:n], noisy[:n]
        return noisy, clean, fs

    mixture = synth.make_mixture(args.noise, args.snr, duration=args.seconds)
    print(f"generated mixture: {args.noise} noise at {args.snr:+.0f} dB SNR")
    return mixture["noisy"], mixture["clean"], mixture["fs"]


def single_channel(noisy: np.ndarray, clean: np.ndarray | None, fs: int, save: Path | None) -> None:
    config = DenoiseConfig(frame_size=audio_frame(fs), hop_ratio=0.25)
    hop = config.hop()

    rule("1. Input")
    print(f"  samples        {len(noisy)}  ({len(noisy) / fs:.2f} s at {fs} Hz)")
    print(f"  peak / rms     {np.max(np.abs(noisy)):.3f} / {metrics.rms(noisy):.3f}")

    rule("2. STFT")
    spec = framing.stft(noisy, config.frame_size, hop)
    print(f"  frame / hop    {config.frame_size} / {hop} samples "
          f"({1000 * config.frame_size / fs:.0f} ms, {100 * (1 - hop / config.frame_size):.0f}% overlap)")
    print(f"  bins / frames  {spec.shape[1]} / {spec.shape[0]}")

    rule("3. Adaptive notch")
    share = band_share(noisy, config.notch_f0, fs)
    print(f"  50 Hz energy   {100 * share:.2f}% of total")
    print(f"  applied        {notch.is_tonal(noisy, config.notch_f0, fs)} "
          f"(needs > 10%, harmonics={config.notch_harmonics}, mu={config.notch_mu})")
    if notch.is_tonal(noisy, config.notch_f0, fs):
        notched = notch.adaptive_notch_harmonics(
            noisy, config.notch_f0, fs, config.notch_harmonics, config.notch_mu)["output"]
        print(f"  50 Hz after    {100 * band_share(notched, config.notch_f0, fs):.3f}%")

    rule("4. VAD + noise PSD")
    result = denoise(noisy, fs=fs, config=config)
    power = noise_est.frame_power(spec)
    silence = noise_est.detect_silence(power)
    alpha = config.noise_alpha
    print(f"  silent frames  {100 * silence.mean():.1f}%")
    print(f"  tracker alpha  {alpha}  (time constant ~{config.frame_size * 0.25 / fs / (1 - alpha):.2f} s)")
    peak_bin = int(np.argmax(result["noise_psd"]))
    print(f"  noise peak     {peak_bin * fs / config.frame_size:.0f} Hz "
          f"({100 * result['noise_psd'][peak_bin] / result['noise_psd'].sum():.1f}% of PSD)")

    rule("5. Adaptive spectral gain")
    gains = result["gains"]
    print(f"  gain range     {20 * np.log10(gains.max() + 1e-12):.1f} .. "
          f"{20 * np.log10(gains.min() + 1e-12):.1f} dB")
    print(f"  mean gain      {20 * np.log10(gains.mean() + 1e-12):.1f} dB")
    print(f"  gain floor     {config.gain_floor:.2f}, decision-directed delta {config.dd}")

    rule("6. Reconstruction + metrics")
    cleaned = result["audio"]

    spec_out = framing.stft(cleaned, config.frame_size, hop)
    power_out = np.abs(spec_out) ** 2
    mask = result["silence"][:min(len(power), len(power_out))]
    floor_drop = 10 * np.log10(
        (np.mean(power[:len(mask)][mask]) + 1e-20) / (np.mean(power_out[:len(mask)][mask]) + 1e-20)
    ) if mask.any() else None
    if clean is not None:
        before = metrics.si_sdr(clean, noisy)
        after = metrics.si_sdr(clean, cleaned)
        print(f"  SI-SDR         {before:.2f} -> {after:.2f} dB  ({after - before:+.2f} dB)")
        print(f"  segmental SNR  {metrics.segmental_snr(clean, noisy):.2f} -> "
              f"{metrics.segmental_snr(clean, cleaned):.2f} dB")
    print(f"  noise floor    {f'{floor_drop:.2f}' if floor_drop is not None else 'n/a'} dB (silence frames)")
    print(f"  output peak    {np.max(np.abs(cleaned)):.3f}")

    if save:
        save.mkdir(parents=True, exist_ok=True)
        sf.write(save / "input.wav", noisy, fs)
        sf.write(save / "denoised.wav", cleaned, fs)
        print(f"  wrote          {save / 'input.wav'}, {save / 'denoised.wav'}")


def audio_frame(fs: int) -> int:
    return 1024 if fs <= 16000 else 2048


def two_channel(clean: np.ndarray, fs: int) -> None:
    rule("7. Two-channel ANC (simulated second mic)")
    noise = synth.fan_noise(len(clean), fs, seed=2)
    h_reference = np.array([1.0, 0.35, 0.12])
    h_primary = np.concatenate([np.zeros(4), np.array([1.0, 0.30, 0.10])])
    rng = np.random.default_rng(5)

    reference = np.convolve(noise, h_reference)[:len(clean)]
    reference += 1e-3 * rng.standard_normal(len(clean))
    primary_noise = np.convolve(noise, h_primary)[:len(clean)]
    primary = clean + 0.6 * primary_noise / (np.std(primary_noise) + 1e-20)
    primary += 3e-3 * rng.standard_normal(len(clean))

    print(f"  paths           reference = h_ref * noise, primary = speech + h_pri * noise")
    print(f"  mic offset      4 samples ({4000 / fs:.2f} ms)")

    for name, run in (
        ("LMS  mu=0.002", lambda: lms(reference, primary, mu=0.002, order=32)),
        ("NLMS mu=0.5", lambda: nlms(reference, primary, mu=0.5, order=32)),
        ("RLS  lambda=0.995", lambda: rls(reference, primary, lam=0.995, order=32)),
    ):
        out = run()
        curve, converged = erle_curve(primary, out["output"])
        print(f"  {name:18s} ERLE {metrics.erle_db(primary, out['output']):5.1f} dB   "
              f"converged at block {converged if converged is not None else '-'}"
              f"   steady {np.mean(curve[-40:]):.1f} dB")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", nargs="?", help="noisy WAV to analyse")
    parser.add_argument("--reference", help="clean reference for SI-SDR")
    parser.add_argument("--noise", default="fan", choices=sorted(synth.NOISE_BANK))
    parser.add_argument("--snr", type=float, default=-5.0)
    parser.add_argument("--seconds", type=float, default=6.0)
    parser.add_argument("--save", type=Path, help="directory for input/denoised WAVs")
    parser.add_argument("--no-anc", action="store_true", help="skip the ANC demo")
    args = parser.parse_args()

    noisy, clean, fs = load_material(args)
    single_channel(noisy, clean, fs, args.save)
    if clean is not None and not args.no_anc:
        two_channel(clean, fs)
    elif not args.no_anc:
        print("\n(no clean reference: skipping the two-channel ANC section)")


if __name__ == "__main__":
    sys.exit(main())
