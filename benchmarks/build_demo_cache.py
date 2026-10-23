"""Precompute the two-channel demo so it loads instantly on a slow host.

The free Render instance compiles the Numba kernels on first use, which takes
over a minute on a 0.1-CPU machine.  The bundled demo is deterministic, so its
result is computed once here and shipped with the app.  The API still processes
every uploaded pair live; this only short-circuits the demo clip.

Run:  .venv/bin/python benchmarks/build_demo_cache.py
"""

import io
import json
import sys
from pathlib import Path

import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from webapp import audio_io  # noqa: E402
from webapp.app import (  # noqa: E402
    ANC_CACHE_PATH,
    _anc_cache_key,
    _anc_process,
    _anc_settings,
    _demo_audio,
)

DEFAULTS = {
    "filters": ["lms", "nlms", "rls"],
    "order": 32,
    "mu_lms": 0.004,
    "mu_nlms": 0.1,
    "lambda_rls": 0.995,
    "align": True,
}


def wav_bytes(audio, rate: int = 16000) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, audio, rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def main() -> None:
    ANC_CACHE_PATH.unlink(missing_ok=True)

    demo = _demo_audio()
    primary = wav_bytes(demo["anc_primary"])
    reference = wav_bytes(demo["anc_reference"])

    primary_signal, rate, _ = audio_io.read_audio(primary)
    reference_signal, _, _ = audio_io.read_audio(reference)
    key = _anc_cache_key(primary_signal, reference_signal, _anc_settings(DEFAULTS))
    payload = _anc_process(primary, reference, DEFAULTS)

    ANC_CACHE_PATH.write_text(json.dumps({"key": key, "payload": payload}))
    size = ANC_CACHE_PATH.stat().st_size / (1024 * 1024)
    print(f"[ok] {ANC_CACHE_PATH.relative_to(ROOT)} ({size:.1f} MB)")
    for label, result in payload["results"].items():
        print(f"  {label}: ERLE {result['erle_db']:.1f} dB, converged {result['conv_block']}")


if __name__ == "__main__":
    main()
