"""STFT / ISTFT with explicit overlap-add.

The analysis window is the square root of a Hann window, and the synthesis
stage multiplies by the same window.  Analysis x synthesis therefore applies
w^2 = Hann to each sample, and Hann satisfies the constant-overlap-add (COLA)
condition at 75% overlap.  The overlap-add denominator is accumulated from the
actual frames, so reconstruction is exact even at the signal edges.
"""

import numpy as np

DEFAULT_FRAME = 1024
DEFAULT_HOP_RATIO = 0.25


def sqrt_hann(frame_size: int) -> np.ndarray:
    return np.sqrt(np.hanning(frame_size))


def resolve_hop(frame_size: int, hop: int | None) -> int:
    return hop if hop is not None else int(frame_size * DEFAULT_HOP_RATIO)


def frame_count(length: int, frame_size: int, hop: int) -> int:
    if length <= frame_size:
        return 1
    return 1 + int(np.ceil((length - frame_size) / hop))


def stft(x: np.ndarray, frame_size: int = DEFAULT_FRAME, hop: int | None = None,
         window: np.ndarray | None = None) -> np.ndarray:
    """Complex spectrogram, shape (n_frames, frame_size // 2 + 1).

    The signal is zero-padded by ``frame_size - hop`` samples at the front so
    that the first real sample is covered by two overlapping windows.  Without
    this, the zero at the edge of a Hann window makes it unrecoverable.
    """
    x = np.asarray(x, dtype=np.float64)
    hop = resolve_hop(frame_size, hop)
    if window is None:
        window = sqrt_hann(frame_size)

    edge = frame_size - hop
    padded = np.concatenate([np.zeros(edge), x, np.zeros(edge)])
    n_frames = frame_count(len(padded), frame_size, hop)
    pad = (n_frames - 1) * hop + frame_size - len(padded)
    padded = np.pad(padded, (0, max(0, pad)))

    frames = np.lib.stride_tricks.sliding_window_view(padded, frame_size)[::hop]
    frames = frames[:n_frames]
    return np.fft.rfft(frames * window, axis=1)


def istft(spec: np.ndarray, frame_size: int = DEFAULT_FRAME, hop: int | None = None,
          window: np.ndarray | None = None, length: int | None = None) -> np.ndarray:
    """Inverse of :func:`stft`, trimmed to ``length`` samples when given."""
    hop = resolve_hop(frame_size, hop)
    if window is None:
        window = sqrt_hann(frame_size)

    frames = np.fft.irfft(spec, n=frame_size, axis=1) * window
    total = (frames.shape[0] - 1) * hop + frame_size
    if length is not None:
        total = max(total, length + frame_size - hop)

    out = np.zeros(total)
    norm = np.zeros(total)
    w_sq = window ** 2
    for i, frame in enumerate(frames):
        start = i * hop
        out[start:start + frame_size] += frame
        norm[start:start + frame_size] += w_sq

    out /= np.maximum(norm, 1e-10)
    out = out[frame_size - hop:]
    return out[:length] if length is not None else out


def spectrogram_db(spec: np.ndarray, reference: float | None = None) -> np.ndarray:
    """|STFT|^2 in dB, normalised to its own maximum when no reference given."""
    power = np.abs(spec) ** 2
    ref = reference if reference is not None else power.max()
    return 10.0 * np.log10(power + 1e-12) - 10.0 * np.log10(ref + 1e-12)
