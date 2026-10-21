"""Audio decoding, encoding and plot-ready payload helpers for the web app."""

import base64
import io
import math
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from dsp import framing

MAX_SECONDS = 120.0
MAX_BYTES = 40 * 1024 * 1024
MIN_RATE = 8000
MAX_RATE = 48000
DEFAULT_RATE = 16000

SPEC_DB_MIN = -80.0
SPEC_DB_MAX = 0.0
SPEC_MAX_TIME_BINS = 600
SPEC_MAX_FREQ_BINS = 200
WAVE_BUCKETS = 900


class AudioError(ValueError):
    pass


def read_audio(data: bytes) -> tuple[np.ndarray, int, list[str]]:
    """Decode any libsndfile-supported container to mono float64 + sample rate."""
    if not data:
        raise AudioError("Empty upload")
    if len(data) > MAX_BYTES:
        raise AudioError(f"File larger than {MAX_BYTES // (1024 * 1024)} MB")

    try:
        audio, rate = sf.read(io.BytesIO(data), dtype="float64", always_2d=False)
    except Exception as exc:
        raise AudioError("Could not decode this audio file. Try WAV, FLAC or OGG.") from exc

    messages = []
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
        messages.append("Stereo input was mixed down to mono.")

    if not np.all(np.isfinite(audio)):
        audio = np.nan_to_num(audio)
        messages.append("Non-finite samples were replaced with zeros.")

    if rate < MIN_RATE or rate > MAX_RATE:
        target = DEFAULT_RATE
        factor = math.gcd(int(rate), target)
        audio = resample_poly(audio, target // factor, int(rate) // factor)
        messages.append(f"Resampled from {rate} Hz to {target} Hz.")
        rate = target

    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    if peak < 1e-6:
        raise AudioError("The file looks silent.")
    audio = audio / peak * 0.9

    max_samples = int(MAX_SECONDS * rate)
    if len(audio) > max_samples:
        audio = audio[:max_samples]
        messages.append(f"Trimmed to the first {int(MAX_SECONDS)} s.")

    return audio, int(rate), messages


def encode_wav(audio: np.ndarray, rate: int) -> str:
    buffer = io.BytesIO()
    clipped = np.clip(audio, -1.0, 1.0)
    sf.write(buffer, clipped, rate, format="WAV", subtype="PCM_16")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def frame_size_for(rate: int) -> int:
    """~32 ms analysis window, rounded to a power of two."""
    target = 0.032 * rate
    size = 2 ** int(round(math.log2(target)))
    return int(min(4096, max(512, size)))


def _downsample_max(matrix: np.ndarray, target_rows: int, target_cols: int) -> np.ndarray:
    """Fold a 2-D array by taking the maximum, preserving spectral peaks."""
    rows, cols = matrix.shape
    row_edges = np.linspace(0, rows, min(target_rows, rows) + 1).astype(int)
    col_edges = np.linspace(0, cols, min(target_cols, cols) + 1).astype(int)

    reduced_rows = np.maximum.reduceat(matrix, row_edges[:-1], axis=0)
    reduced = np.maximum.reduceat(reduced_rows, col_edges[:-1], axis=1)
    return reduced


def spectrogram_payload(audio: np.ndarray, rate: int, frame: int,
                        hop: int, reference_db: float | None = None) -> dict:
    """Quantise a log-magnitude spectrogram to uint8 and base64 it."""
    spec = framing.stft(audio, frame, hop)
    magnitude = np.abs(spec)
    db = 20.0 * np.log10(magnitude + 1e-12)

    peak = reference_db if reference_db is not None else db.max()
    db = db - peak
    db = np.clip(db, SPEC_DB_MIN, SPEC_DB_MAX)

    matrix = db.T
    matrix = _downsample_max(matrix, SPEC_MAX_FREQ_BINS, SPEC_MAX_TIME_BINS)

    scaled = (matrix - SPEC_DB_MIN) / (SPEC_DB_MAX - SPEC_DB_MIN)
    quantised = np.clip(scaled * 255.0, 0, 255).astype(np.uint8)

    return {
        "data": base64.b64encode(quantised.tobytes()).decode("ascii"),
        "width": int(quantised.shape[1]),
        "height": int(quantised.shape[0]),
        "duration": len(audio) / rate,
        "max_frequency": rate / 2,
        "db_min": SPEC_DB_MIN,
        "db_max": SPEC_DB_MAX,
        "peak_db": float(peak),
    }


def waveform_payload(audio: np.ndarray, buckets: int = WAVE_BUCKETS) -> list[float]:
    """Peak envelope, one value per bucket, for a compact waveform drawing."""
    n = len(audio)
    if n == 0:
        return []
    count = min(buckets, n)
    edges = np.linspace(0, n, count + 1).astype(int)
    envelope = np.array([
        np.max(np.abs(audio[edges[i]:max(edges[i] + 1, edges[i + 1])]))
        for i in range(count)
    ])
    peak = envelope.max() or 1.0
    return [round(float(v / peak), 4) for v in envelope]


def read_uploaded_file(path: Path) -> bytes:
    return path.read_bytes()
