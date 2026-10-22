import io
import json

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

from dsp import synth
from webapp.app import align_reference, app

client = TestClient(app)


def wav_bytes(audio: np.ndarray, rate: int = 16000) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, audio, rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def decode(base64_audio: str) -> tuple[np.ndarray, int]:
    import base64

    data, rate = sf.read(io.BytesIO(base64.b64decode(base64_audio)))
    return data, rate


def test_health():
    assert client.get("/api/health").json() == {"ok": True}


def test_index_serves_html():
    response = client.get("/")
    assert response.status_code == 200
    assert "Anechoic" in response.text


def test_denoise_returns_audio_and_spectrograms():
    mixture = synth.make_mixture("white", 5.0, duration=2.0, seed=3)
    response = client.post(
        "/api/denoise",
        files={"audio": ("noisy.wav", wav_bytes(mixture["noisy"]), "audio/wav")},
        data={"settings": json.dumps({"use_notch": False})},
    )
    assert response.status_code == 200, response.text
    payload = response.json()

    audio, rate = decode(payload["output_audio"])
    assert rate == 16000
    assert len(audio) == len(mixture["noisy"])

    spec = payload["spectrogram"]
    assert spec["input"]["width"] > 0 and spec["input"]["height"] > 0
    assert len(spec["output"]["data"]) > 0
    assert payload["waveform"]["output"]
    assert payload["metrics"]["noise_floor_reduction_db"] is not None


def test_denoise_with_reference_reports_si_sdr():
    mixture = synth.make_mixture("pink", 5.0, duration=2.0, seed=4)
    response = client.post(
        "/api/denoise",
        files={
            "audio": ("noisy.wav", wav_bytes(mixture["noisy"]), "audio/wav"),
            "reference": ("clean.wav", wav_bytes(mixture["clean"]), "audio/wav"),
        },
        data={"settings": json.dumps({"use_notch": False})},
    )
    payload = response.json()
    assert payload["metrics"]["has_reference"] is True
    assert payload["metrics"]["delta_si_sdr"] > 1.0


def test_denoise_rejects_non_audio():
    response = client.post(
        "/api/denoise",
        files={"audio": ("noise.txt", b"definitely not audio", "text/plain")},
    )
    assert response.status_code == 400


def test_align_reference_recovers_known_delay():
    rng = np.random.default_rng(0)
    noise = rng.standard_normal(16000)
    delay = 7
    reference = np.concatenate([np.zeros(delay), noise[:-delay]])
    aligned, lag = align_reference(noise, reference, 16000, max_lag_ms=5.0)
    assert lag == -delay
    assert np.corrcoef(aligned[1000:5000], noise[1000:5000])[0, 1] > 0.99


def test_anc_returns_filters_and_curves():
    rng = np.random.default_rng(1)
    rate = 16000
    n = 2 * rate
    noise = synth.fan_noise(n, rate, seed=2)
    clean = synth.speech_like(n / rate, rate, seed=2)
    reference = noise + 1e-3 * rng.standard_normal(n)
    primary = clean + 0.5 * noise + 3e-3 * rng.standard_normal(n)

    response = client.post(
        "/api/anc",
        files={
            "primary": ("primary.wav", wav_bytes(primary, rate), "audio/wav"),
            "reference": ("reference.wav", wav_bytes(reference, rate), "audio/wav"),
        },
        data={"settings": json.dumps({
            "filters": ["lms", "nlms"],
            "order": 16,
            "align": False,
        })},
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert set(payload["results"]) == {"LMS", "NLMS"}
    assert payload["results"]["NLMS"]["erle_db"] > 5.0
    assert len(payload["results"]["NLMS"]["erle_curve"]) > 0
    assert payload["best"] in payload["results"]


def test_demo_clips_are_available_audio():
    for kind in ("noisy", "anc_primary", "anc_reference"):
        response = client.get(f"/api/demo/{kind}")
        assert response.status_code == 200
        audio, rate = sf.read(io.BytesIO(response.content))
        assert rate == 16000
        assert np.max(np.abs(audio)) <= 1.0

    assert client.get("/api/demo/nope").status_code == 404
