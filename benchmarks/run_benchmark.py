"""Benchmark the denoiser and produce every figure/table the report needs.

Run:  .venv/bin/python benchmarks/run_benchmark.py

Outputs (results/):
  results.csv            SI-SDR / segmental SNR for every noise type, SNR, method
  figures/learning.png   LMS vs NLMS vs RLS convergence (two-channel ANC)
  figures/sisdr.png      SI-SDR improvement per noise type and method
  figures/spectrograms.png  before/after spectrograms
  figures/mu_sweep.png   step-size vs convergence (LMS/NLMS)
  figures/snr_sweep.png  SI-SDR improvement vs input SNR
"""

import argparse
import csv
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy.signal import lfilter, resample_poly

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dsp import framing, metrics, synth  # noqa: E402
from dsp.lms import lms, nlms, rls  # noqa: E402
from dsp.pipeline import DenoiseConfig, denoise  # noqa: E402
from dsp.synth import NOISE_BANK  # noqa: E402

RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
DATA = ROOT / "data" / "clean"
FS = 16000

METHODS = {
    "spectral_subtraction": {"method": "spectral_subtraction"},
    "adaptive_wiener": {"method": "adaptive_wiener"},
}
NOISES = ["white", "pink", "brown", "hum", "fan", "babble"]
INPUT_SNRS = [0.0, 5.0, 10.0]


# --------------------------------------------------------------------------- #
# clean speech
# --------------------------------------------------------------------------- #
def load_clean_speech(force: bool = False) -> np.ndarray:
    """Use macOS `say` to render real speech once, then cache it as 16 kHz WAV."""
    DATA.mkdir(parents=True, exist_ok=True)
    target = DATA / "speech.wav"
    if target.exists() and not force:
        x, sr = sf.read(target)
        if sr == FS:
            return x / (np.max(np.abs(x)) + 1e-20) * 0.8

    text = ("The quick brown fox jumps over the lazy dog. "
            "Adaptive filters learn the noise from the signal itself. "
            "Noise reduction works best when the noise is stationary. "
            "We compare the least mean squares filter with recursive least squares.")
    tmp = DATA / "speech.aiff"
    try:
        subprocess.run(["say", "-v", "Samantha", "-o", str(tmp), text],
                       check=True, capture_output=True)
        x, sr = sf.read(tmp)
        if x.ndim > 1:
            x = x.mean(axis=1)
        if sr != FS:
            x = resample_poly(x, FS, sr)
        x = x / (np.max(np.abs(x)) + 1e-20) * 0.8
        sf.write(target, x, FS)
        return x
    except Exception as exc:  # noqa: BLE001 - fallback is intentional
        print(f"[warn] `say` unavailable ({exc}); using synthetic speech-like signal")
        x = synth.speech_like(duration=8.0, fs=FS)
        sf.write(target, x, FS)
        return x


# --------------------------------------------------------------------------- #
# two-channel adaptive noise canceller (the adaptive-filter showcase)
# --------------------------------------------------------------------------- #
def acoustic_paths(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Direct path plus two reflections for each microphone.

    The primary path is delayed by four samples, as if its mic sits further
    from the noise source.  Both paths stay correlated, which is the physical
    assumption every adaptive noise canceller relies on.
    """
    rng = np.random.default_rng(seed)
    h_reference = np.array([1.0, 0.35, 0.12]) + 0.03 * rng.standard_normal(3)
    h_primary = np.concatenate([np.zeros(4), np.array([1.0, 0.30, 0.10])
                                + 0.03 * rng.standard_normal(3)])
    return h_reference, h_primary


def make_anc_scenario(clean: np.ndarray, seed: int = 0) -> dict:
    """Simulate two microphones: reference hears the noise, primary hears noise+speech."""
    noise = synth.fan_noise(len(clean), FS, seed)
    h_reference, h_primary = acoustic_paths(seed)

    rng = np.random.default_rng(seed + 100)
    reference = lfilter(h_reference, [1.0], noise)
    noise_at_primary = lfilter(h_primary, [1.0], noise)
    primary = clean + 0.6 * noise_at_primary / (np.std(noise_at_primary) + 1e-20)

    # Independent microphone self-noise: without it the simulation is exactly
    # invertible and ERLE runs to numerical infinity, which is not physical.
    reference = reference + 1e-3 * rng.standard_normal(len(reference))
    primary = primary + 3e-3 * rng.standard_normal(len(primary))
    return {"reference": reference, "noise": noise_at_primary, "primary": primary,
            "clean": clean}


def anc_learning_curves(seconds: float = 4.0, order: int = 32) -> dict:
    """Noise-only ANC so the learning curve measures adaptation, not double-talk."""
    n = int(seconds * FS)
    noise = synth.fan_noise(n, FS, 2)
    h_reference, h_primary = acoustic_paths(1)
    rng = np.random.default_rng(101)
    reference = lfilter(h_reference, [1.0], noise) + 1e-3 * rng.standard_normal(n)
    desired = lfilter(h_primary, [1.0], noise) + 3e-3 * rng.standard_normal(n)

    curves = {
        "LMS (mu=0.002)": lms(reference, desired, mu=0.002, order=order),
        "NLMS (mu=0.5)": nlms(reference, desired, mu=0.5, order=order),
        "RLS (lambda=0.995)": rls(reference, desired, lam=0.995, order=order),
    }
    return curves, desired


def erle_curve(desired: np.ndarray, error: np.ndarray, block: int = 256) -> np.ndarray:
    """ERLE per block: 10 log10( mean d^2 / mean e^2 )."""
    n = min(len(desired), len(error)) // block
    d = desired[:n * block].reshape(n, block)
    e = error[:n * block].reshape(n, block)
    return 10.0 * np.log10(np.mean(d ** 2, axis=1) / (np.mean(e ** 2, axis=1) + 1e-20) + 1e-20)


def smooth(values: np.ndarray, width: int = 5) -> np.ndarray:
    if len(values) < width:
        return values
    pad = width // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    return np.convolve(padded, np.ones(width) / width, mode="valid")


# --------------------------------------------------------------------------- #
# experiments
# --------------------------------------------------------------------------- #
def run_grid(clean: np.ndarray) -> list[dict]:
    rows = []
    for noise_type in NOISES:
        for snr in INPUT_SNRS:
            noise = NOISE_BANK[noise_type](len(clean), FS, seed=7)
            noisy = synth.mix_at_snr(clean, noise, snr)

            baseline_si = metrics.si_sdr(clean, noisy)
            rows.append({
                "noise": noise_type, "input_snr": snr, "method": "noisy",
                "si_sdr": baseline_si,
                "seg_snr": metrics.segmental_snr(clean, noisy),
                "delta_si_sdr": 0.0,
            })

            for name, kwargs in METHODS.items():
                result = denoise(noisy, fs=FS, config=DenoiseConfig(), **kwargs)
                si = metrics.si_sdr(clean, result["audio"])
                rows.append({
                    "noise": noise_type, "input_snr": snr, "method": name,
                    "si_sdr": si,
                    "seg_snr": metrics.segmental_snr(clean, result["audio"]),
                    "delta_si_sdr": si - baseline_si,
                })
    return rows


def convergence_block(erle: np.ndarray, target_db: float = 20.0) -> int | None:
    above = np.where(erle >= target_db)[0]
    return int(above[0]) if len(above) else None


def run_anc_with_speech(clean: np.ndarray, order: int = 32) -> dict:
    scenario = make_anc_scenario(clean[: int(6 * FS)])
    out = {}
    for name, fn in (
        ("LMS mu=0.002", lambda: lms(scenario["reference"], scenario["primary"], mu=0.002, order=order)),
        ("NLMS mu=0.5", lambda: nlms(scenario["reference"], scenario["primary"], mu=0.5, order=order)),
        ("NLMS mu=0.1", lambda: nlms(scenario["reference"], scenario["primary"], mu=0.1, order=order)),
        ("RLS lambda=0.995", lambda: rls(scenario["reference"], scenario["primary"], lam=0.995, order=order)),
    ):
        result = fn()
        out[name] = {
            "si_sdr": metrics.si_sdr(scenario["clean"], result["output"]),
            "noisy_si_sdr": metrics.si_sdr(scenario["clean"], scenario["primary"]),
            "erle": metrics.erle_db(scenario["primary"], result["output"]),
            "mse": result["mse"],
            "erle_curve": erle_curve(scenario["primary"], result["output"]),
            "conv_blocks": convergence_block(erle_curve(scenario["primary"], result["output"])),
        }
    return out


def write_csv(rows: list[dict]) -> None:
    path = RESULTS / "results.csv"
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[ok] {path.relative_to(ROOT)}")


def plot_learning(curves: dict, desired: np.ndarray) -> None:
    plt.figure(figsize=(7, 4.2))
    for name, result in curves.items():
        plt.plot(smooth(erle_curve(desired, result["output"]), 5), label=name)
    plt.axhline(20, color="gray", linestyle="--", linewidth=0.8)
    plt.xlabel("block (256 samples)")
    plt.ylabel("ERLE (dB)")
    plt.title("Two-channel ANC convergence (dashed = 20 dB)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "learning.png", dpi=160)
    plt.close()


def plot_sisdr(rows: list[dict]) -> None:
    methods = ["spectral_subtraction", "adaptive_wiener"]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    width = 0.8 / (len(methods) + 1)
    positions = np.arange(len(NOISES))

    noisy = [np.mean([r["si_sdr"] for r in rows if r["noise"] == n and r["method"] == "noisy"])
             for n in NOISES]
    ax.bar(positions - 1.5 * width, noisy, width, label="noisy input")

    for i, method in enumerate(methods):
        values = [np.mean([r["si_sdr"] for r in rows if r["noise"] == n and r["method"] == method])
                  for n in NOISES]
        ax.bar(positions + (i - 0.5) * width, values, width, label=method.replace("_", " "))

    ax.set_xticks(positions, NOISES)
    ax.set_ylabel("SI-SDR (dB), averaged over 0/5/10 dB input")
    ax.set_title("Denoising performance by noise type")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "sisdr.png", dpi=160)
    plt.close()


def plot_spectrograms(clean: np.ndarray, noise_type: str = "white", snr: float = 5.0) -> None:
    noise = NOISE_BANK[noise_type](len(clean), FS, 7)
    noisy = synth.mix_at_snr(clean, noise, snr)
    result = denoise(noisy, fs=FS, config=DenoiseConfig())

    frame, hop = 1024, 256
    specs = [
        ("noisy input", framing.spectrogram_db(framing.stft(noisy, frame, hop))),
        ("denoised", framing.spectrogram_db(framing.stft(result["audio"], frame, hop))),
        ("clean reference", framing.spectrogram_db(framing.stft(clean, frame, hop))),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
    for ax, (title, db) in zip(axes, specs):
        ax.imshow(np.clip(db, -60, 0).T, origin="lower", aspect="auto",
                  extent=[0, len(noisy) / FS, 0, FS / 2000], cmap="magma")
        ax.set_title(title)
        ax.set_xlabel("time (s)")
    axes[0].set_ylabel("frequency (kHz)")
    fig.suptitle(f"Spectrograms, {noise_type} noise at {snr:.0f} dB input SNR")
    plt.tight_layout()
    plt.savefig(FIGURES / "spectrograms.png", dpi=160)
    plt.close()


def plot_mu_sweep() -> None:
    n = int(3 * FS)
    rng = np.random.default_rng(3)
    noise = synth.white_noise(n, 4)
    h = rng.standard_normal(12) * 0.3
    desired = np.concatenate([np.zeros(4), lfilter(h, [1.0], noise)[:-4]])
    desired = desired + 3e-3 * rng.standard_normal(n)
    reference = noise + 1e-3 * rng.standard_normal(n)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, name, fn, mus in (
        (axes[0], "LMS", lambda mu: lms(reference, desired, mu=mu, order=32), [0.0005, 0.002, 0.008]),
        (axes[1], "NLMS", lambda mu: nlms(reference, desired, mu=mu, order=32), [0.02, 0.1, 0.5]),
    ):
        for mu in mus:
            ax.semilogy(smooth(fn(mu)["mse"], 5), label=f"mu={mu}")
        ax.set_title(name)
        ax.set_xlabel("block")
        ax.set_ylim(1e-5, 1e1)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
    axes[0].set_ylabel("MSE")
    plt.tight_layout()
    plt.savefig(FIGURES / "mu_sweep.png", dpi=160)
    plt.close()


def plot_snr_sweep(rows: list[dict]) -> None:
    plt.figure(figsize=(6.5, 4))
    for method in ["spectral_subtraction", "adaptive_wiener"]:
        means = []
        for snr in INPUT_SNRS:
            values = [r["delta_si_sdr"] for r in rows
                      if r["method"] == method and r["input_snr"] == snr]
            means.append(np.mean(values))
        plt.plot(INPUT_SNRS, means, marker="o", label=method.replace("_", " "))
    plt.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    plt.xlabel("input SNR (dB)")
    plt.ylabel("SI-SDR improvement (dB)")
    plt.title("Improvement vs input SNR (mean over noise types)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "snr_sweep.png", dpi=160)
    plt.close()


def print_summary(rows: list[dict], anc: dict) -> None:
    print("\nSingle-channel denoising (mean SI-SDR dB over 0/5/10 dB input)")
    print(f"{'noise':8s} {'noisy':>8s} {'spec-sub':>9s} {'adaptive':>9s} {'gain':>7s}")
    for noise_type in NOISES:
        def mean(method):
            return np.mean([r["si_sdr"] for r in rows
                            if r["noise"] == noise_type and r["method"] == method])
        base = mean("noisy")
        sub = mean("spectral_subtraction")
        wit = mean("adaptive_wiener")
        print(f"{noise_type:8s} {base:8.2f} {sub:9.2f} {wit:9.2f} {wit - base:+7.2f}")

    print("\nTwo-channel ANC with speech present (6 s, fan noise)")
    print(f"{'filter':18s} {'noisy SI-SDR':>13s} {'clean SI-SDR':>13s} {'ERLE dB':>9s} {'conv. blocks':>13s}")
    for name, result in anc.items():
        conv = result["conv_blocks"] if result["conv_blocks"] is not None else -1
        print(f"{name:18s} {result['noisy_si_sdr']:13.2f} {result['si_sdr']:13.2f} "
              f"{result['erle']:9.2f} {conv:13d}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-speech", action="store_true")
    args = parser.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    clean = load_clean_speech(force=args.force_speech)
    if len(clean) < 6 * FS:
        clean = np.tile(clean, int(np.ceil(6 * FS / len(clean))))
    print(f"[ok] clean speech: {len(clean) / FS:.2f} s")

    rows = run_grid(clean)
    write_csv(rows)

    print("[..] ANC learning curves")
    curves, desired = anc_learning_curves()
    plot_learning(curves, desired)
    anc = run_anc_with_speech(clean)
    print("[..] figures")
    plot_sisdr(rows)
    plot_spectrograms(clean)
    plot_mu_sweep()
    plot_snr_sweep(rows)
    print_summary(rows, anc)
    print(f"[ok] figures in {FIGURES.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
