"""Time-domain adaptive filters: LMS, NLMS and RLS.

All three solve the same problem: given a reference signal ``x`` and a desired
signal ``d``, find filter taps ``w`` that predict ``d`` from ``x``.  The
prediction ``y = w . x`` is subtracted, and the residual ``e = d - y`` is both
the output and the error that drives adaptation.
"""

import numpy as np

DEFAULT_ORDER = 32
DEFAULT_BLOCK = 256


def _taps(reference: np.ndarray, order: int) -> np.ndarray:
    """Rows of shape (N, order) where row n is [x[n], x[n-1], ..., x[n-order+1]]."""
    padded = np.concatenate([np.zeros(order - 1), np.asarray(reference, dtype=np.float64)])
    return np.lib.stride_tricks.sliding_window_view(padded, order)[:, ::-1]


def _mse_curve(error: np.ndarray, block: int) -> np.ndarray:
    n_blocks = max(1, len(error) // block)
    trimmed = error[:n_blocks * block].reshape(n_blocks, block)
    return np.mean(trimmed ** 2, axis=1)


def lms(reference: np.ndarray, desired: np.ndarray, mu: float = 0.01,
        order: int = DEFAULT_ORDER, block: int = DEFAULT_BLOCK) -> dict:
    """w(n+1) = w(n) + mu * e(n) * x(n)."""
    x_mat = _taps(reference, order)
    d = np.asarray(desired, dtype=np.float64)

    w = np.zeros(order)
    y = np.zeros(len(d))
    e = np.zeros(len(d))

    for n in range(len(d)):
        x = x_mat[n]
        y[n] = w @ x
        e[n] = d[n] - y[n]
        w = w + mu * e[n] * x

    return {"output": e, "prediction": y, "error": e, "weights": w,
            "mse": _mse_curve(e, block)}


def nlms(reference: np.ndarray, desired: np.ndarray, mu: float = 0.5,
         order: int = DEFAULT_ORDER, block: int = DEFAULT_BLOCK,
         eps: float = 1e-8) -> dict:
    """w(n+1) = w(n) + mu * e(n) * x(n) / (||x(n)||^2 + eps)."""
    x_mat = _taps(reference, order)
    d = np.asarray(desired, dtype=np.float64)

    w = np.zeros(order)
    y = np.zeros(len(d))
    e = np.zeros(len(d))

    for n in range(len(d)):
        x = x_mat[n]
        y[n] = w @ x
        e[n] = d[n] - y[n]
        w = w + mu * e[n] * x / (x @ x + eps)

    return {"output": e, "prediction": y, "error": e, "weights": w,
            "mse": _mse_curve(e, block)}


def rls(reference: np.ndarray, desired: np.ndarray, lam: float = 0.99,
        order: int = DEFAULT_ORDER, block: int = DEFAULT_BLOCK,
        delta: float = 1.0) -> dict:
    """Recursive least squares with forgetting factor ``lam``.

    k(n) = P x / (lam + x^T P x)
    w(n+1) = w(n) + k(n) e(n)
    P(n+1) = (P(n) - k(n) x^T P(n)) / lam
    """
    x_mat = _taps(reference, order)
    d = np.asarray(desired, dtype=np.float64)

    w = np.zeros(order)
    p_inv = np.eye(order) / delta
    y = np.zeros(len(d))
    e = np.zeros(len(d))

    for n in range(len(d)):
        x = x_mat[n]
        px = p_inv @ x
        gain = px / (lam + x @ px)
        y[n] = w @ x
        e[n] = d[n] - y[n]
        w = w + gain * e[n]
        p_inv = (p_inv - np.outer(gain, px)) / lam
        # P must stay symmetric; rounding drift otherwise makes it explode.
        p_inv = 0.5 * (p_inv + p_inv.T)

    return {"output": e, "prediction": y, "error": e, "weights": w,
            "mse": _mse_curve(e, block)}
