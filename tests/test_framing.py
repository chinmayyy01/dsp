import numpy as np
import pytest

from dsp import framing


@pytest.mark.parametrize("frame_size,hop_ratio", [(1024, 0.25), (512, 0.5), (256, 0.25)])
def test_stft_istft_reconstruction(frame_size, hop_ratio):
    rng = np.random.default_rng(0)
    x = rng.standard_normal(16000)
    hop = int(frame_size * hop_ratio)
    spec = framing.stft(x, frame_size, hop)
    y = framing.istft(spec, frame_size, hop, length=len(x))
    assert y.shape == x.shape
    assert np.max(np.abs(y - x)) < 1e-10


def test_short_signal_roundtrip():
    x = np.array([0.1, -0.2, 0.3])
    spec = framing.stft(x, frame_size=64, hop=16)
    y = framing.istft(spec, frame_size=64, hop=16, length=len(x))
    np.testing.assert_allclose(y, x, atol=1e-10)


def test_spectrogram_shape():
    x = np.random.default_rng(1).standard_normal(4096)
    spec = framing.stft(x, 512, 128)
    expected = framing.frame_count(len(x) + 2 * (512 - 128), 512, 128)
    assert spec.shape == (expected, 257)
    assert np.iscomplexobj(spec)


def test_spectrogram_db_is_normalised():
    x = np.random.default_rng(2).standard_normal(4096)
    db = framing.spectrogram_db(framing.stft(x, 512, 128))
    assert db.max() == pytest.approx(0.0, abs=1e-6)
