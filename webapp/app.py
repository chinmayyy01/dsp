"""FastAPI backend for the adaptive noise reduction demo.

Endpoints
  GET  /              -> the single page UI
  POST /api/denoise   -> single-file adaptive spectral denoising
  POST /api/anc       -> two-channel LMS/NLMS/RLS adaptive noise canceller
  GET  /api/health    -> liveness probe
"""

import hashlib
import io
import json
import math
import subprocess
import threading
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
import soundfile as sf
from scipy.signal import correlate, correlation_lags, resample_poly

from dsp import framing, metrics, noise_est, synth
from dsp.lms import lms, nlms, rls
from dsp.pipeline import DenoiseConfig, denoise
from webapp import audio_io

STATIC_DIR = Path(__file__).parent / "static"
ANC_MAX_SECONDS = 20.0
BLOCK = 256
ANC_CACHE_PATH = Path(__file__).resolve().parents[1] / "assets" / "demo-anc.json"
_ANC_CACHE: dict | None = None

app = FastAPI(title="Adaptive Noise Reduction", version="1.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> RedirectResponse:
    return RedirectResponse("/static/favicon.svg")


@app.get("/api/health")
def health() -> dict:
    return {"ok": True}


@app.get("/api/debug")
def debug() -> dict:
    import dsp
    from dsp.lms import NUMBA_AVAILABLE

    version = None
    if NUMBA_AVAILABLE:
        import numba

        version = numba.__version__

    cache_dir = Path(dsp.__file__).parent / "__pycache__"
    cache_files = sorted(p.name for p in cache_dir.glob("lms.*.nb*")) if cache_dir.exists() else []
    return {"numba": NUMBA_AVAILABLE, "numba_version": version,
            "numba_cache": cache_files}


_DEMO_CACHE: dict[str, np.ndarray] = {}
DEMO_SECONDS = 10.0
DEMO_RATE = 16000
DEMO_TEXT = ("Adaptive filters learn the noise from the signal itself. "
             "This clip was recorded next to a noisy fan. "
             "Watch the background hiss disappear while the voice stays clear.")
BUNDLED_DEMO = Path(__file__).resolve().parents[1] / "assets" / "demo-speech.wav"
DEMO_PATH = Path(__file__).resolve().parents[1] / "data" / "demo" / "speech.wav"


def _trim_to_duration(audio: np.ndarray, rate: int, seconds: float) -> np.ndarray:
    """Trim at a quiet point near the target length, then pad with silence."""
    target = int(rate * seconds)
    if len(audio) < target:
        return np.pad(audio, (0, target - len(audio)))

    window = int(0.05 * rate)
    search_from = int(0.85 * target)
    best, best_energy = target, np.inf
    for start in range(search_from, target - window, window // 2):
        energy = float(np.mean(audio[start:start + window] ** 2))
        if energy < best_energy:
            best, best_energy = start, energy
    return np.pad(audio[:best], (0, target - best))


def _demo_speech(rate: int = DEMO_RATE) -> np.ndarray:
    """Real speech for the demo: rendered once with macOS `say`, then cached."""
    if BUNDLED_DEMO.exists():
        audio, sample_rate = sf.read(BUNDLED_DEMO, dtype="float64")
    elif DEMO_PATH.exists():
        audio, sample_rate = sf.read(DEMO_PATH, dtype="float64")
    else:
        DEMO_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = DEMO_PATH.with_suffix(".aiff")
        try:
            try:
                subprocess.run(["say", "-v", "Samantha", "-o", str(tmp), DEMO_TEXT],
                               check=True, capture_output=True)
            except subprocess.CalledProcessError:
                subprocess.run(["say", "-o", str(tmp), DEMO_TEXT],
                               check=True, capture_output=True)
            audio, sample_rate = sf.read(tmp, dtype="float64")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sample_rate != rate:
                audio = resample_poly(audio, rate, sample_rate)
            audio = audio / (np.max(np.abs(audio)) + 1e-20) * 0.9
            audio = _trim_to_duration(audio, rate, DEMO_SECONDS)
            sf.write(DEMO_PATH, audio, rate, subtype="PCM_16")
        except Exception:
            audio = synth.speech_like(DEMO_SECONDS, rate, seed=11)
            sf.write(DEMO_PATH, audio, rate, subtype="PCM_16")
            sample_rate = rate
        finally:
            if 'tmp' in locals():
                tmp.unlink(missing_ok=True)

    if sample_rate != rate:
        audio = resample_poly(audio, rate, sample_rate)
    return _trim_to_duration(np.asarray(audio, dtype=np.float64), rate, DEMO_SECONDS)


def _demo_audio() -> dict[str, np.ndarray]:
    if _DEMO_CACHE:
        return _DEMO_CACHE

    clean = _demo_speech()
    noise = synth.fan_noise(len(clean), DEMO_RATE, seed=12)

    def scaled(x: np.ndarray) -> np.ndarray:
        return x / (np.max(np.abs(x)) + 1e-20) * 0.9

    _DEMO_CACHE["noisy"] = scaled(synth.mix_at_snr(clean, noise, 4.0))
    rng = np.random.default_rng(12)
    primary = synth.mix_at_snr(clean, noise, -10.0)
    reference = np.concatenate([np.zeros(5), noise[:-5]])
    primary = primary + 3e-3 * rng.standard_normal(len(primary))
    reference = reference + 1e-3 * rng.standard_normal(len(reference))
    _DEMO_CACHE["anc_primary"] = scaled(primary)
    _DEMO_CACHE["anc_reference"] = scaled(reference)
    return _DEMO_CACHE


@app.get("/api/demo/{kind}")
def demo(kind: str) -> Response:
    available = _demo_audio()
    if kind not in available:
        raise HTTPException(status_code=404, detail="Unknown demo clip")
    buffer = io.BytesIO()
    import soundfile as sf

    sf.write(buffer, available[kind], 16000, format="WAV", subtype="PCM_16")
    return Response(content=buffer.getvalue(), media_type="audio/wav",
                    headers={"Content-Disposition": f'inline; filename="{kind}.wav"'})


def _parse_settings(raw: str) -> dict:
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="settings must be valid JSON") from exc
    return data if isinstance(data, dict) else {}


def _read_upload(upload: UploadFile) -> bytes:
    return upload.file.read()


def _float(value, default: float, low: float, high: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(parsed):
        return default
    return float(min(high, max(low, parsed)))


def _int(value, default: int, low: int, high: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return int(min(high, max(low, parsed)))


def _bool(value, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _speech_free_frames(audio: np.ndarray, frame: int = 512,
                        hop: int = 128) -> tuple[np.ndarray, int, int]:
    """Per-frame mask of frames without speech.

    Energy cannot separate speech from noise when the noise dominates, so use
    spectral flatness: harmonic speech is less flat than broadband noise.
    """
    spec = framing.stft(audio, frame, hop)
    power = np.abs(spec) ** 2
    flatness = noise_est.spectral_flatness(power)
    mask = flatness >= np.median(flatness)
    if mask.mean() < 0.1:
        mask[:] = True
    return mask, frame, hop


def _noise_floor_reduction(input_audio: np.ndarray, output_audio: np.ndarray,
                           frame: int, hop: int) -> float | None:
    """Power drop, in dB, measured only on speech-free frames."""
    n = min(len(input_audio), len(output_audio))
    spec_in = framing.stft(input_audio[:n], frame, hop)
    spec_out = framing.stft(output_audio[:n], frame, hop)
    power_in = np.abs(spec_in) ** 2
    power_out = np.abs(spec_out) ** 2
    speech_free, _, _ = _speech_free_frames(input_audio[:n], frame, hop)

    frames = min(len(power_in), len(power_out), len(speech_free))
    mask = speech_free[:frames]
    if mask.sum() < 2:
        return None
    before = float(np.mean(power_in[:frames][mask]))
    after = float(np.mean(power_out[:frames][mask]))
    return float(10.0 * np.log10((before + 1e-20) / (after + 1e-20)))


@app.post("/api/denoise")
def denoise_endpoint(
    audio: UploadFile = File(...),
    reference: UploadFile | None = File(None),
    settings: str = Form("{}"),
) -> dict:
    options = _parse_settings(settings)
    try:
        signal, rate, messages = audio_io.read_audio(_read_upload(audio))
    except audio_io.AudioError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    frame = _int(options.get("frame_size"), audio_io.frame_size_for(rate), 256, 4096)
    frame = 2 ** int(round(np.log2(frame)))
    hop = max(1, frame // 4)

    config = DenoiseConfig(
        frame_size=frame,
        hop_ratio=0.25,
        use_notch=_bool(options.get("use_notch"), True),
        notch_f0=_float(options.get("notch_f0"), 50.0, 40.0, 70.0),
        notch_harmonics=_int(options.get("notch_harmonics"), 2, 1, 6),
        notch_mu=_float(options.get("notch_mu"), 0.001, 0.00005, 0.02),
        use_ale=_bool(options.get("use_ale"), False),
        ale_delay=_int(options.get("ale_delay"), 8, 1, 64),
        ale_order=_int(options.get("ale_order"), 32, 4, 128),
        ale_mu=_float(options.get("ale_mu"), 0.5, 0.01, 1.0),
        noise_alpha=_float(options.get("noise_alpha"), 0.95, 0.5, 0.999),
        gain_floor=_float(options.get("gain_floor"), 0.1, 0.0, 0.8),
        gain_smooth=_float(options.get("gain_smooth"), 0.0, 0.0, 0.95),
        dd=_float(options.get("dd"), 0.9, 0.0, 0.99),
    )
    method = options.get("method", "adaptive_wiener")
    if method not in {"adaptive_wiener", "spectral_subtraction"}:
        method = "adaptive_wiener"

    result = denoise(signal, fs=rate, config=config, method=method)
    cleaned = result["audio"]

    spec_in = audio_io.spectrogram_payload(signal, rate, frame, hop)
    spec_out = audio_io.spectrogram_payload(
        cleaned, rate, frame, hop, reference_db=spec_in["peak_db"])

    response = {
        "sample_rate": rate,
        "duration": len(signal) / rate,
        "input_audio": audio_io.encode_wav(signal, rate),
        "output_audio": audio_io.encode_wav(cleaned, rate),
        "waveform": {
            "input": audio_io.waveform_payload(signal),
            "output": audio_io.waveform_payload(cleaned),
        },
        "spectrogram": {"input": spec_in, "output": spec_out},
        "config": {
            "frame_size": frame,
            "hop": hop,
            "method": method,
            "notch_applied": bool(result["notch_applied"]),
            "silence_fraction": float(np.mean(result["silence"])),
        },
        "metrics": {
            "noise_floor_reduction_db": _noise_floor_reduction(signal, cleaned, frame, hop),
            "has_reference": False,
        },
        "messages": messages,
    }

    if reference is not None and reference.filename:
        try:
            ref_signal, ref_rate, _ = audio_io.read_audio(_read_upload(reference))
        except audio_io.AudioError as exc:
            raise HTTPException(status_code=400, detail=f"Reference file: {exc}") from exc
        if ref_rate != rate:
            raise HTTPException(status_code=400, detail="Reference must use the same sample rate.")
        n = min(len(ref_signal), len(signal))
        noisy_si = metrics.si_sdr(ref_signal[:n], signal[:n])
        clean_si = metrics.si_sdr(ref_signal[:n], cleaned[:n])
        response["metrics"].update({
            "has_reference": True,
            "noisy_si_sdr": noisy_si,
            "denoised_si_sdr": clean_si,
            "delta_si_sdr": clean_si - noisy_si,
            "noisy_seg_snr": metrics.segmental_snr(ref_signal[:n], signal[:n]),
            "denoised_seg_snr": metrics.segmental_snr(ref_signal[:n], cleaned[:n]),
        })

    return response


def align_reference(primary: np.ndarray, reference: np.ndarray, rate: int,
                    max_lag_ms: float = 500.0) -> tuple[np.ndarray, int]:
    """Delay the reference so that its direct path lines up with the primary.

    Cross-correlation peak location gives the bulk delay between microphones.
    Returns the shifted reference and the applied lag in samples.
    """
    n = min(len(primary), len(reference))
    primary, reference = primary[:n], reference[:n]
    window = min(n, int(2.0 * rate))
    if window < 64:
        return reference, 0

    energies = np.convolve(primary ** 2, np.ones(window) / window, mode="valid")
    start = int(np.argmax(energies)) if len(energies) else 0
    a = primary[start:start + window]
    b = reference[start:start + window]

    corr = correlate(a, b, mode="full")
    lags = correlation_lags(len(a), len(b), mode="full")
    max_lag = int(max_lag_ms * rate / 1000.0)
    mask = np.abs(lags) <= max_lag
    if not mask.any():
        return reference, 0

    lag = int(lags[mask][int(np.argmax(corr[mask]))])
    if lag == 0:
        return reference, 0

    if lag > 0:
        padded = np.concatenate([np.zeros(lag), reference[:-lag]])
    else:
        padded = np.concatenate([reference[-lag:], np.zeros(-lag)])
    padded = padded[:n]

    check = min(len(primary), 2 * rate)
    score = float(np.corrcoef(primary[:check], padded[:check])[0, 1])
    if not np.isfinite(score) or score < 0.2:
        return reference, 0
    return padded, lag


def _smooth_series(values: list[float | None], width: int) -> list[float | None]:
    """Moving average that ignores gaps and never fills them."""
    half = width // 2
    out: list[float | None] = []
    for i in range(len(values)):
        window = [v for v in values[max(0, i - half):i + half + 1] if v is not None]
        out.append(round(float(np.mean(window)), 3) if window else None)
    return out


def _erle_curve_smoothed(desired: np.ndarray, error: np.ndarray, block: int = BLOCK,
                         width: int = 31) -> tuple[list[float], float | None, int | None]:
    """ERLE per block with a 0.5 s moving average.

    ERLE is the total power reduction, which with double-talk follows the speech
    envelope; averaging over ~0.5 s keeps the trend visible and is the standard
    way to present an ANC learning curve.
    """
    n = min(len(desired), len(error)) // block
    if n == 0:
        return [], None, None

    d = desired[:n * block].reshape(n, block)
    e = error[:n * block].reshape(n, block)
    power = np.mean(d ** 2, axis=1)
    error_power = np.mean(e ** 2, axis=1)
    curve = 10.0 * np.log10(power / (error_power + 1e-20) + 1e-20)

    pad = width // 2
    smoothed = np.convolve(np.pad(curve, (pad, pad), mode="edge"),
                           np.ones(width) / width, mode="valid")
    values = [round(float(np.clip(v, -40, 80)), 3) for v in smoothed]

    tail = values[int(n * 0.75):]
    steady = float(np.mean(tail)) if tail else None
    converged = next((i for i, v in enumerate(values) if v >= 20.0), None)
    return values, steady, converged


def _anc_settings(options: dict) -> dict:
    filters = options.get("filters", ["lms", "nlms", "rls"])
    if not isinstance(filters, list):
        filters = ["lms", "nlms", "rls"]
    return {
        "filters": sorted(filters),
        "order": _int(options.get("order"), 32, 4, 128),
        "mu_lms": _float(options.get("mu_lms"), 0.004, 1e-5, 0.05),
        "mu_nlms": _float(options.get("mu_nlms"), 0.1, 0.001, 1.0),
        "lambda_rls": _float(options.get("lambda_rls"), 0.995, 0.9, 0.99999),
        "align": _bool(options.get("align"), True),
    }


def _anc_cache_key(primary_signal: np.ndarray, reference_signal: np.ndarray,
                   settings: dict) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(primary_signal, dtype=np.float64).tobytes())
    digest.update(np.ascontiguousarray(reference_signal, dtype=np.float64).tobytes())
    digest.update(json.dumps(settings, sort_keys=True).encode())
    return digest.hexdigest()


def _load_anc_cache() -> dict:
    global _ANC_CACHE
    if _ANC_CACHE is None:
        try:
            _ANC_CACHE = json.loads(ANC_CACHE_PATH.read_text())
        except (OSError, json.JSONDecodeError):
            _ANC_CACHE = {}
    return _ANC_CACHE


def _anc_process(primary_bytes: bytes, reference_bytes: bytes, options: dict) -> dict:
    try:
        primary_signal, rate, messages = audio_io.read_audio(primary_bytes)
        reference_signal, ref_rate, _ = audio_io.read_audio(reference_bytes)
    except audio_io.AudioError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    settings = _anc_settings(options)
    cached = _load_anc_cache()
    key = _anc_cache_key(primary_signal, reference_signal, settings)
    if isinstance(cached, dict) and cached.get("key") == key:
        return cached["payload"]

    if ref_rate != rate:
        factor = math.gcd(int(ref_rate), int(rate))
        reference_signal = resample_poly(reference_signal, rate // factor, ref_rate // factor)
        messages.append(f"Reference resampled from {ref_rate} Hz to {rate} Hz.")

    n = min(len(primary_signal), len(reference_signal), int(ANC_MAX_SECONDS * rate))
    primary_signal = primary_signal[:n]
    reference_signal = reference_signal[:n]

    align = settings["align"]
    lag = 0
    if align:
        reference_signal, lag = align_reference(primary_signal, reference_signal, rate)

    window = min(len(primary_signal), 2 * rate)
    correlation = float(np.corrcoef(primary_signal[:window], reference_signal[:window])[0, 1])
    if not np.isfinite(correlation):
        correlation = 0.0
    if correlation < 0.2:
        messages.append("The two recordings look uncorrelated. ANC can only cancel noise "
                        "that appears in both microphones.")

    order = settings["order"]
    runners = {
        "lms": ("LMS", lambda: lms(reference_signal, primary_signal, mu=settings["mu_lms"],
                                   order=order, block=BLOCK)),
        "nlms": ("NLMS", lambda: nlms(reference_signal, primary_signal, mu=settings["mu_nlms"],
                                      order=order, block=BLOCK)),
        "rls": ("RLS", lambda: rls(reference_signal, primary_signal, lam=settings["lambda_rls"],
                                   order=order, block=BLOCK)),
    }

    results = {}
    for name in ("lms", "nlms", "rls"):
        if name not in settings["filters"]:
            continue
        label, run = runners[name]
        try:
            out = run()
        except Exception as exc:
            results[label] = {"error": str(exc)}
            continue

        if not np.all(np.isfinite(out["output"])):
            results[label] = {"error": "Filter diverged. Reduce the step size."}
            continue

        curve, _, converged = _erle_curve_smoothed(primary_signal, out["output"])
        frame = audio_io.frame_size_for(rate)
        results[label] = {
            "audio": audio_io.encode_wav(out["output"], rate),
            "erle_db": metrics.erle_db(primary_signal, out["output"]),
            "noise_floor_reduction_db": _noise_floor_reduction(
                primary_signal, out["output"], frame, frame // 4),
            "erle_curve": curve,
            "conv_block": converged,
        }

    if not results:
        raise HTTPException(status_code=400, detail="Select at least one filter.")

    best = max(
        (label for label, value in results.items() if "erle_db" in value),
        key=lambda label: results[label]["erle_db"],
        default=next(iter(results)),
    )

    return {
        "sample_rate": rate,
        "duration": n / rate,
        "lag_samples": lag,
        "lag_ms": lag / rate * 1000.0,
        "correlation": correlation,
        "input_audio": audio_io.encode_wav(primary_signal, rate),
        "results": results,
        "best": best,
        "order": order,
        "messages": messages,
    }


@app.post("/api/anc")
def anc_endpoint(
    primary: UploadFile = File(...),
    reference: UploadFile = File(...),
    settings: str = Form("{}"),
) -> dict:
    return _anc_process(_read_upload(primary), _read_upload(reference), _parse_settings(settings))


@app.on_event("startup")
def _warm_filters() -> None:
    def warm() -> None:
        try:
            from dsp.warmup import main as warm_main

            warm_main()
        except Exception:
            pass

    threading.Thread(target=warm, daemon=True).start()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("webapp.app:app", host="127.0.0.1", port=8000, reload=False)
